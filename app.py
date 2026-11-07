"""Local web control for a Tapo C211. Talks only to the camera on the LAN."""
import base64
import collections
import datetime
import hashlib
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

import requests
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

ENV_FILE = Path(__file__).parent / ".env"
STREAM_PATHS = {"low": "stream2", "high": "stream1"}
MOVE_FAILSAFE = "PT5S"

app = Flask(__name__)


class CameraError(Exception):
    pass


def load_settings():
    settings = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                settings[key.strip()] = value.strip()
    return settings


PORT = int(load_settings().get("TAPO_PORT") or 8080)
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"} | {
    h.strip() for h in (load_settings().get("TAPO_EXTRA_HOSTS") or "").split(",") if h.strip()
}


def require_settings():
    settings = load_settings()
    if not all(settings.get(k) for k in ("TAPO_HOST", "TAPO_USER", "TAPO_PASSWORD")):
        raise CameraError(f"Add the camera account login to {ENV_FILE}")
    return settings


def redact(text, settings):
    text = re.sub(r"rtsp://[^@\s]+@", "rtsp://***@", text)
    for key in ("TAPO_PASSWORD", "TAPO_CLOUD_PASSWORD", "OPENAI_API_KEY"):
        secret = settings.get(key)
        if secret:
            text = text.replace(secret, "***").replace(urllib.parse.quote(secret, safe=""), "***")
    return text


# --- Video ---------------------------------------------------------------


def ffmpeg_input_args(settings, path):
    user = urllib.parse.quote(settings["TAPO_USER"], safe="")
    password = urllib.parse.quote(settings["TAPO_PASSWORD"], safe="")
    url = f"rtsp://{user}:{password}@{settings['TAPO_HOST']}:554/{path}"
    return ["-fflags", "nobuffer", "-flags", "low_delay", "-rtsp_transport", "tcp", "-timeout", "5000000", "-i", url]


def die_with_parent():
    """Linux only: the child gets SIGTERM when the thread that started it ends."""
    import ctypes
    import signal

    ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def start_ffmpeg(settings, path, output_args, linked=False):
    # linked: for an ffmpeg that writes files rather than to us, which would otherwise
    # carry on alone if the server were killed. It must be started from a thread that
    # outlives it.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += ffmpeg_input_args(settings, path) + output_args
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=die_with_parent if linked and os.name == "posix" else None,
    )
    errors = collections.deque(maxlen=5)
    threading.Thread(
        target=lambda: errors.extend(l.decode("utf-8", "replace").strip() for l in proc.stderr),
        daemon=True,
    ).start()
    return proc, errors


def stop_ffmpeg(proc):
    # A launcher such as Chocolatey's ffmpeg shim can outlive terminate(),
    # leaving the real ffmpeg running. Closing the pipe makes that one exit.
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    proc.stdout.close()


def camera_full(errors):
    return "Operation not permitted" in " ".join(errors)


def describe_failure(errors, settings):
    text = redact(" ".join(errors), settings)
    if "401" in text:
        return "The camera rejected the login. Check the camera account in .env."
    if camera_full([text]):
        # Tested: the camera takes 2 connections per quality. The Tapo app counts too.
        return ("The camera is already streaming to as many viewers as it allows. Close the Tapo app or "
                "another IridaCare page and the video will come back by itself.")
    return text or "The video stream stopped."


class Stream:
    """One ffmpeg process per stream, running only while someone is watching."""

    def __init__(self, quality):
        self.quality = quality
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.clients = 0
        self.proc = None
        self.thread = None
        self.error = None

    def acquire(self):
        with self.cond:
            self.clients += 1
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            return self.seq

    def release(self):
        with self.cond:
            self.clients -= 1
            if self.clients == 0 and self.proc:
                self.proc.terminate()

    def next_frame(self, after, timeout):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != after, timeout=timeout)
            return (self.frame if self.seq != after else None), self.seq

    def _run(self):
        while True:
            with self.cond:
                if self.clients == 0:
                    self.thread = None
                    self.frame = None
                    return
            try:
                self._read_ffmpeg(require_settings())
            except (CameraError, OSError) as exc:
                self.error = str(exc)
            with self.cond:
                self.cond.wait_for(lambda: self.clients == 0, timeout=3)

    def _read_ffmpeg(self, settings):
        # A fixed output frame rate here delayed the video by over a second.
        output = ["-an", "-fps_mode", "passthrough", "-q:v", "5", "-f", "mjpeg", "pipe:1"]
        proc, errors = start_ffmpeg(settings, STREAM_PATHS[self.quality], output)
        with self.cond:
            self.proc = proc
            if self.clients == 0:
                proc.terminate()

        buffer = bytearray()
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk or self.clients == 0:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if end == -1:
                    if start > 0:
                        del buffer[:start]
                    break
                with self.cond:
                    self.frame = bytes(buffer[start:end + 2])
                    self.seq += 1
                    self.error = None
                    self.cond.notify_all()
                del buffer[:end + 2]

        stop_ffmpeg(proc)
        with self.cond:
            self.proc = None
            stopped = self.clients == 0
        if not stopped:
            self.error = describe_failure(errors, settings)


streams = {quality: Stream(quality) for quality in STREAM_PATHS}


def get_stream():
    return streams.get(request.args.get("quality", "low")) or abort(400)


@app.get("/video.h264")
def video_h264():
    """Sends the camera's H.264 to the browser as it is, with no re-encoding.

    Access unit delimiters mark where each frame starts, and repeating the
    stream settings at every keyframe lets the browser's decoder start cleanly.
    """
    quality = request.args.get("quality", "high")
    if quality not in STREAM_PATHS:
        abort(400)
    settings = require_settings()
    output = [
        "-an", "-c:v", "copy", "-bsf:v", "dump_extra=freq=keyframe,h264_metadata=aud=insert",
        "-flush_packets", "1", "-f", "h264", "pipe:1",
    ]
    # The camera takes 2 connections per quality. The cat watcher holds one HD one, so
    # when the Tapo app or another page has the other, fall back to SD.
    tries = [quality, "low"] if quality == "high" else [quality]
    for served in tries:
        proc, errors = start_ffmpeg(settings, STREAM_PATHS[served], output)
        first = proc.stdout.read1(65536)
        if first:
            break
        stop_ffmpeg(proc)
        time.sleep(0.1)  # Lets the last error lines arrive.
        if not (camera_full(errors) and served != tries[-1]):
            raise CameraError(describe_failure(errors, settings))

    def generate():
        try:
            chunk = first
            while chunk:
                yield chunk
                chunk = proc.stdout.read1(65536)
        finally:
            stop_ffmpeg(proc)

    return Response(generate(), mimetype="video/h264", headers={"Cache-Control": "no-store", "X-Quality": served})


@app.get("/video.mjpg")
def video():
    stream = get_stream()

    def generate():
        last = stream.acquire()
        try:
            while True:
                frame, last = stream.next_frame(last, timeout=5)
                if frame is None:
                    # Lets the server notice a closed page while no frames arrive.
                    yield b"\r\n"
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
        finally:
            stream.release()

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/snapshot.jpg")
def snapshot():
    stream = get_stream()
    last = stream.acquire()
    try:
        frame, _ = stream.next_frame(last, timeout=12)
    finally:
        stream.release()
    if frame is None:
        return jsonify(error=stream.error or "No frame from the camera."), 502
    name = datetime.datetime.now().strftime("tapo-%Y%m%d-%H%M%S.jpg")
    return Response(frame, mimetype="image/jpeg", headers={"Content-Disposition": f'attachment; filename="{name}"'})


# --- Pan and tilt (ONVIF) ------------------------------------------------


NAMESPACES = (
    'xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
    'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
    'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
    'xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" '
    'xmlns:tt="http://www.onvif.org/ver10/schema"'
)


def elements(root, name):
    return [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == name]


def child_int(parent, name):
    return int(elements(parent, name)[0].text)


def security_header(settings, clock_offset):
    now = datetime.datetime.now(datetime.timezone.utc) + clock_offset
    created = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce = os.urandom(16)
    digest = hashlib.sha1(nonce + created.encode() + settings["TAPO_PASSWORD"].encode()).digest()
    return (
        '<s:Header><Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
        f"<UsernameToken><Username>{escape(settings['TAPO_USER'])}</Username>"
        '<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{base64.b64encode(digest).decode()}</Password>"
        f'<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>'
        "</UsernameToken></Security></s:Header>"
    )


def soap(url, body, header=""):
    envelope = f"<s:Envelope {NAMESPACES}>{header}<s:Body>{body}</s:Body></s:Envelope>"
    try:
        reply = requests.post(
            url,
            data=envelope.encode(),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            timeout=6,
        )
    except requests.RequestException as exc:
        raise CameraError(f"Could not reach the camera: {type(exc).__name__}") from exc
    try:
        root = ET.fromstring(reply.content)
    except ET.ParseError as exc:
        raise CameraError(f"Unexpected reply from the camera (HTTP {reply.status_code})") from exc
    if reply.status_code != 200:
        code = [e.text for e in elements(root, "Value")]
        if any("NotAuthorized" in (c or "") for c in code):
            raise CameraError("The camera rejected the login. Check the camera account in .env.")
        reason = [e.text for e in elements(root, "Text")]
        raise CameraError(f"Camera error: {(reason or code or ['unknown'])[0]}")
    return root


class Onvif:
    """Looks up the service details once, and again if a request fails."""

    def __init__(self):
        self.lock = threading.Lock()
        self.session = None

    def _connect(self, settings):
        base = f"http://{settings['TAPO_HOST']}:2020"
        device_url = f"{base}/onvif/device_service"

        # The camera's clock can be wrong without internet access, and digest
        # logins are checked against it, so sign requests using its own time.
        time_reply = soap(device_url, "<tds:GetSystemDateAndTime/>")
        utc = elements(time_reply, "UTCDateTime")[0]
        camera_now = datetime.datetime(
            child_int(utc, "Year"), child_int(utc, "Month"), child_int(utc, "Day"),
            child_int(utc, "Hour"), child_int(utc, "Minute"), child_int(utc, "Second"),
            tzinfo=datetime.timezone.utc,
        )
        offset = camera_now - datetime.datetime.now(datetime.timezone.utc)

        caps = soap(
            device_url,
            "<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>",
            security_header(settings, offset),
        )

        def service_url(name):
            for element in elements(caps, name):
                for address in elements(element, "XAddr"):
                    return base + urllib.parse.urlparse(address.text).path
            raise CameraError(f"The camera does not offer the ONVIF {name} service.")

        profiles = soap(
            service_url("Media"), "<trt:GetProfiles/>", security_header(settings, offset)
        )
        tokens = [e.get("token") for e in elements(profiles, "Profiles") if e.get("token")]
        if not tokens:
            raise CameraError("The camera returned no ONVIF profiles.")
        return {
            "login": (settings["TAPO_HOST"], settings["TAPO_USER"], settings["TAPO_PASSWORD"]),
            "ptz_url": service_url("PTZ"),
            "token": tokens[0],
            "offset": offset,
        }

    def ptz(self, build_body):
        settings = require_settings()
        with self.lock:
            login = (settings["TAPO_HOST"], settings["TAPO_USER"], settings["TAPO_PASSWORD"])
            if self.session is None or self.session["login"] != login:
                self.session = self._connect(settings)
            session = self.session
            try:
                return soap(session["ptz_url"], build_body(escape(session["token"])), security_header(settings, session["offset"]))
            except CameraError:
                self.session = None
                raise


onvif = Onvif()
control_lock = threading.RLock()


def pan_tilt_move(pan, tilt, timeout=MOVE_FAILSAFE):
    onvif.ptz(lambda token: (
        f"<tptz:ContinuousMove><tptz:ProfileToken>{token}</tptz:ProfileToken>"
        f'<tptz:Velocity><tt:PanTilt x="{pan:.2f}" y="{tilt:.2f}"/></tptz:Velocity>'
        f"<tptz:Timeout>{timeout}</tptz:Timeout></tptz:ContinuousMove>"
    ))


def pan_tilt_stop():
    onvif.ptz(lambda token: (
        f"<tptz:Stop><tptz:ProfileToken>{token}</tptz:ProfileToken>"
        "<tptz:PanTilt>true</tptz:PanTilt><tptz:Zoom>false</tptz:Zoom></tptz:Stop>"
    ))


def pan_tilt_position():
    reply = onvif.ptz(lambda token: f"<tptz:GetStatus><tptz:ProfileToken>{token}</tptz:ProfileToken></tptz:GetStatus>")
    pan_tilt = elements(reply, "PanTilt")
    if not pan_tilt:
        raise CameraError("The camera did not report its position.")
    return float(pan_tilt[0].get("x")), float(pan_tilt[0].get("y"))


def pan_tilt_go_to(x, y):
    onvif.ptz(lambda token: (
        f"<tptz:AbsoluteMove><tptz:ProfileToken>{token}</tptz:ProfileToken>"
        f'<tptz:Position><tt:PanTilt x="{x}" y="{y}"/></tptz:Position></tptz:AbsoluteMove>'
    ))


@app.post("/api/move")
def move():
    data = request.get_json()
    try:
        pan = max(-1.0, min(1.0, float(data["pan"])))
        tilt = max(-1.0, min(1.0, float(data["tilt"])))
    except (KeyError, TypeError, ValueError):
        abort(400)
    # Manual control takes over from following.
    with control_lock:
        if follower:
            follower.set_target(None)
        pan_tilt_move(pan, tilt)
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    with control_lock:
        pan_tilt_stop()
    return jsonify(ok=True)


# --- Following, gestures and voice ---------------------------------------


follower = None
voice_agent = None
follower_lock = threading.Lock()


def open_follow_video():
    from tracker import FRAME_HEIGHT, FRAME_WIDTH

    output = [
        "-an", "-fps_mode", "passthrough", "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc, _ = start_ffmpeg(require_settings(), STREAM_PATHS["low"], output)
    return proc


def get_follower():
    global follower
    with follower_lock:
        if follower is None:
            try:
                from tracker import Follower
            except ImportError as exc:
                raise CameraError("Following needs the packages in requirements.txt.") from exc
            follower = Follower(
                open_follow_video, stop_ffmpeg, pan_tilt_move, pan_tilt_stop, camera_error=CameraError,
                control_lock=control_lock, position=pan_tilt_position, go_to=pan_tilt_go_to,
            )
        return follower


def open_mic():
    output = ["-vn", "-c:a", "copy", "-f", "alaw", "pipe:1"]
    proc, _ = start_ffmpeg(require_settings(), STREAM_PATHS["low"], output)
    return proc


def voice_is_on():
    return bool(voice_agent and voice_agent.get_status()["on"])


@app.get("/api/follow")
def follow_status():
    return jsonify(follower.get_status() if follower else {"enabled": False, "state": "off", "target": None, "scene": []})


@app.post("/api/follow")
def follow():
    target = request.get_json().get("target")
    require_settings()
    if target is None:
        if follower:
            # Voice uses the camera's view for answering, so keep watching for it.
            if voice_is_on():
                follower.set_target(None)
            else:
                follower.disable()
        return jsonify(ok=True)
    active = get_follower()
    from tracker import clean_target

    target = clean_target(target)
    if not target:
        abort(400)
    active.enable(target, search=request.get_json().get("search") is True)
    return jsonify(ok=True)


@app.post("/api/gesture")
def gesture():
    kind = request.get_json().get("kind")
    if kind not in ("yes", "no"):
        abort(400)
    require_settings()
    get_follower().gesture(kind)
    return jsonify(ok=True)


@app.get("/api/voice")
def voice_status():
    return jsonify(voice_agent.get_status() if voice_agent else {"on": False, "state": "off", "error": None, "lines": []})


@app.post("/api/voice")
def voice():
    global voice_agent
    on = request.get_json().get("on")
    if not isinstance(on, bool):
        abort(400)
    if not on:
        if voice_agent:
            voice_agent.stop()
        return jsonify(ok=True)
    settings = require_settings()
    if not settings.get("OPENAI_API_KEY"):
        raise CameraError(f"Add OPENAI_API_KEY to {ENV_FILE}")
    active = get_follower()
    with follower_lock:
        if voice_agent is None:
            from voice import VoiceAgent

            voice_agent = VoiceAgent(settings["OPENAI_API_KEY"], active, open_mic, stop_ffmpeg, get_speaker(settings))
    # Following starts only when asked to, by voice or on the page.
    try:
        voice_agent.start()
    except RuntimeError as exc:
        raise CameraError(str(exc)) from exc
    return jsonify(ok=True)


# --- Camera speaker and the toy rattle ------------------------------------
# Voice and the rattle share one speaker: go2rtc can only run once.

speaker = None
speaker_lock = threading.Lock()
rattle_lock = threading.Lock()
rattle_played = 0.0
SPEAKER_IDLE_SECONDS = 120  # Stays connected a while, so pressing again plays at once.
MAX_RECORDING_BYTES = 10 * 1024 * 1024


def get_speaker(settings):
    global speaker
    with speaker_lock:
        if speaker is None:
            from speaker import BINARY, Speaker

            if not (settings.get("TAPO_CLOUD_PASSWORD") and BINARY.exists()):
                return None
            speaker = Speaker(settings["TAPO_HOST"], settings["TAPO_CLOUD_PASSWORD"])
        return speaker


def stop_speaker_if_idle():
    with rattle_lock:
        if speaker and not voice_is_on() and time.monotonic() - rattle_played >= SPEAKER_IDLE_SECONDS - 1:
            speaker.stop()


@app.get("/api/rattle")
def rattle_status():
    from rattle import recorded

    return jsonify(recorded=recorded() is not None)


@app.post("/api/rattle")
def rattle():
    global rattle_played
    active = get_speaker(require_settings())
    if active is None:
        raise CameraError(f"The camera speaker needs TAPO_CLOUD_PASSWORD in {ENV_FILE} and bin/go2rtc.")
    from rattle import rattle_audio

    with rattle_lock:
        # Presses while one is still playing would just pile up.
        if time.monotonic() - rattle_played < 1.5:
            return jsonify(ok=True)
        fresh = active.proc is None or active.proc.poll() is not None
        try:
            active.start()
            if fresh:
                time.sleep(3)  # go2rtc needs a moment to reach the camera.
            active.play(rattle_audio())
            active.finish()
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise CameraError(f"The rattle did not play: {exc}") from exc
        rattle_played = time.monotonic()
    timer = threading.Timer(SPEAKER_IDLE_SECONDS, stop_speaker_if_idle)
    timer.daemon = True
    timer.start()
    return jsonify(ok=True)


@app.post("/api/rattle/sound")
def rattle_sound():
    """Saves a recording of the real toy, which the button then plays instead."""
    from rattle import SOUNDS

    upload = request.files.get("sound")
    if upload is None:
        abort(400)
    data = upload.stream.read(MAX_RECORDING_BYTES + 1)
    if len(data) > MAX_RECORDING_BYTES:
        raise CameraError("That recording is too long. A few seconds of shaking is plenty.")
    SOUNDS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "upload"
        source.write_bytes(data)
        target = Path(folder) / "rattle.wav"
        # Converting also checks it is audio. Keeps 10 s at most, minus silence at the start.
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
             "-af", "silenceremove=start_periods=1:start_threshold=-45dB", "-t", "10", str(target)],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0 or not target.exists() or target.stat().st_size < 2000:
            raise CameraError("Could not read that recording as audio.")
        for old in SOUNDS.glob("rattle.*"):
            old.unlink()
        target.replace(SOUNDS / "rattle.wav")
    return jsonify(ok=True)


@app.delete("/api/rattle/sound")
def rattle_sound_delete():
    from rattle import SOUNDS

    for old in SOUNDS.glob("rattle.*"):
        old.unlink()
    return jsonify(ok=True)


# --- Cat watch -------------------------------------------------------------
# Looks for the cat without ever moving the camera, and keeps the last sighting.

watcher = None
watcher_lock = threading.Lock()


def open_watch_video(clip_output):
    """One camera connection for both the detector's frames and the clip pieces.

    The camera accepts only 4 video connections at once (tested), and the page, the
    microphone and following each need one too.
    """
    from tracker import FRAME_HEIGHT, FRAME_WIDTH

    output = [
        "-map", "0:v", "-fps_mode", "passthrough", "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ] + clip_output
    # Linked: the clip pieces are files, so a killed server must not leave it running.
    proc, _ = start_ffmpeg(require_settings(), STREAM_PATHS["high"], output, linked=bool(clip_output))
    return proc


def get_watcher():
    global watcher
    with watcher_lock:
        if watcher is None:
            from catwatch import CatWatcher
            from clips import ClipRecorder

            watcher = CatWatcher(open_watch_video, stop_ffmpeg, pan_tilt_position, camera_error=CameraError, recorder=ClipRecorder())
        return watcher


@app.get("/api/watch")
def watch_status():
    return jsonify(get_watcher().get_status())


@app.post("/api/watch")
def watch():
    on = request.get_json().get("on")
    if not isinstance(on, bool):
        abort(400)
    if on:
        require_settings()
        get_watcher().enable()
    else:
        get_watcher().disable()
    return jsonify(ok=True)


@app.get("/api/watch/last.jpg")
def watch_last_image():
    from catwatch import LAST_IMAGE

    if not LAST_IMAGE.exists():
        abort(404)
    return Response(LAST_IMAGE.read_bytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/watch/goto")
def watch_goto():
    """Points the camera where it was when the cat was last seen. Only on request."""
    last = get_watcher().get_status()["last"]
    if not last or last.get("pan") is None:
        raise CameraError("No saved camera position for the last sighting.")
    with control_lock:
        if follower:
            follower.set_target(None)
        pan_tilt_go_to(last["pan"], last["tilt"])
    return jsonify(ok=True)


CLIP_NAME = re.compile(r"clip-\d{8}-\d{6}")


def clip_name(name):
    if not CLIP_NAME.fullmatch(name):
        abort(404)
    return name


@app.get("/api/clips")
def clips_list():
    from clips import list_clips

    return jsonify(clips=list_clips())


@app.get("/clips/<name>.mp4")
def clip_video(name):
    from clips import CLIPS

    return send_from_directory(CLIPS, f"{clip_name(name)}.mp4", mimetype="video/mp4", conditional=True)


@app.get("/clips/<name>.jpg")
def clip_picture(name):
    from clips import CLIPS

    return send_from_directory(CLIPS, f"{clip_name(name)}.jpg", mimetype="image/jpeg", max_age=86400)


@app.delete("/api/clips/<name>")
def clip_delete(name):
    from clips import delete_clip

    delete_clip(clip_name(name))
    return jsonify(ok=True)


# --- Camera settings (pytapo, direct to the camera) ----------------------


def tapo_client(settings):
    from pytapo import Tapo

    cloud_password = settings.get("TAPO_CLOUD_PASSWORD")
    if cloud_password:
        return Tapo(settings["TAPO_HOST"], "admin", cloud_password, cloud_password)
    return Tapo(settings["TAPO_HOST"], settings["TAPO_USER"], settings["TAPO_PASSWORD"])


@app.post("/api/setting")
def setting():
    data = request.get_json()
    name, on = data.get("name"), data.get("on")
    if name not in ("privacy", "led") or not isinstance(on, bool):
        abort(400)
    settings = require_settings()
    # Only runs when a button is pressed: repeated failed logins can lock the
    # camera's login for a while.
    try:
        client = tapo_client(settings)
        if name == "privacy":
            client.setPrivacyMode(on)
        else:
            client.setLEDEnabled(on)
    except Exception as exc:
        raise CameraError(f"Camera setting failed: {redact(str(exc), settings)}") from exc
    return jsonify(ok=True)


# --- Page and guards -----------------------------------------------------


@app.before_request
def only_this_pc():
    # Stops other websites open in the browser from driving the camera.
    if request.host not in ALLOWED_HOSTS:
        abort(403)
    origin = request.headers.get("Origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin not in (f"http://{request.host}", f"https://{request.host}"):
        abort(403)


@app.errorhandler(CameraError)
def camera_error(exc):
    return jsonify(error=str(exc)), 502


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    settings = load_settings()
    return jsonify(
        configured=all(settings.get(k) for k in ("TAPO_HOST", "TAPO_USER", "TAPO_PASSWORD")),
        host=settings.get("TAPO_HOST", ""),
        env_file=str(ENV_FILE),
        streams={q: {"watching": s.clients > 0, "error": s.error} for q, s in streams.items()},
    )


if __name__ == "__main__":
    # The page polls for status, which would otherwise fill the log.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # Carry on watching for the cat after a restart if it was on.
    from catwatch import CatWatcher

    if CatWatcher.was_on() and all(load_settings().get(k) for k in ("TAPO_HOST", "TAPO_USER", "TAPO_PASSWORD")):
        get_watcher().enable()
    app.run(host="127.0.0.1", port=PORT, threaded=True)
