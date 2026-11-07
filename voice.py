"""Talks with people through OpenAI's Realtime voice model, which can call camera tools."""
import audioop
import base64
import collections
import json
import queue
import threading
import time

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from tracker import clean_target

MODEL = "gpt-realtime-2.1-mini"
VOICE = "marin"
MIC_CHUNK_BYTES = 800  # 100 ms of 8 kHz A-law, the camera's microphone format.
ECHO_GUARD_SECONDS = 0.4  # Keep the microphone closed this long after speaking.
MIC_GAIN = 6  # The camera's microphone is very quiet from across a room.
MIC_PEAK_SECONDS = 3

INSTRUCTIONS = """You are the voice of a small pan-and-tilt camera in someone's home. You hear
them through its microphone and speak through its speaker. Keep replies short and natural, in the
language the person uses.

Use your tools to act. When asked to follow something, call follow. If you are asked to find or
look for something, call find. Both look around the room by themselves when needed, which takes up
to about a minute and a half: say you are looking, and you will get a system message when it ends.
Tell the person the result then, and do not call any tools while waiting for it. Only call
stop_following when the person clearly tells you to stop; it cancels any search. To answer a yes or no question you can nod or shake your head as well as speak.
Never describe your actions in words or write stage directions like *nods*: call the tool instead.
Call what_do_you_see before describing what is in view, and never invent things you cannot see.
You follow the person's face unless told otherwise. You can look for anything described in a few
words, such as cat, red backpack or coffee mug. You cannot tell specific people or pets apart."""

TOOLS = [
    {
        "type": "function",
        "name": "follow",
        "description": (
            "Keep the camera pointed at something. Use face or hand for the person; these wait until seen. "
            "Objects and animals are searched for around the room first if they are not in view."
        ),
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "description": "face, hand, or a few words such as cat or red backpack."}},
            "required": ["target"],
        },
    },
    {
        "type": "function",
        "name": "find",
        "description": "Move the camera around the room until the target is found, then follow it. Use face for find me.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "description": "face, hand, or a few words such as cat or red backpack."}},
            "required": ["target"],
        },
    },
    {
        "type": "function",
        "name": "stop_following",
        "description": "Stop moving and stay pointed where the camera is now.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "nod_yes",
        "description": "Nod the camera up and down, meaning yes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "shake_no",
        "description": "Turn the camera left and right, meaning no.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "what_do_you_see",
        "description": "List the things the camera recognises in view right now.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def session_config(speaker):
    audio = {
        "input": {
            "format": {"type": "audio/pcma"},
            "turn_detection": {"type": "semantic_vad"},
            "transcription": {"model": "gpt-realtime-whisper"},
        },
    }
    if speaker:
        audio["output"] = {"format": {"type": "audio/pcma"}, "voice": VOICE}
    return {
        "type": "realtime",
        "instructions": INSTRUCTIONS,
        "output_modalities": ["audio"] if speaker else ["text"],
        "audio": audio,
        "tools": TOOLS,
        "tool_choice": "auto",
    }


class VoiceAgent:
    def __init__(self, api_key, follower, open_mic, close_mic, speaker=None):
        self.api_key = api_key
        self.follower = follower
        self.follower.on_search_started = self.search_started
        self.follower.on_search_done = self.search_done
        self.socket = None
        self.responding = False
        self.pending_notice = None
        self.notices = queue.Queue()
        self.open_mic = open_mic
        self.close_mic = close_mic
        self.speaker = speaker
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.lines = collections.deque(maxlen=30)
        self.state = "off"
        self.error = None
        self.muted_until = 0.0
        self.speech_ends = 0.0
        self.mic_peaks = collections.deque(maxlen=100)
        self.chunks_sent = 0
        self.events = collections.Counter()

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            if self.speaker:
                self.speaker.start()
            # Anything queued while voice was off is out of date.
            self.notices = queue.Queue()
            self.pending_notice = None
            self.stop_event = threading.Event()
            self.error = None
            self.state = "connecting"
            self.thread = threading.Thread(target=self._run, args=(self.stop_event,), daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            thread = self.thread
            self.stop_event.set()
            if self.socket:
                self.socket.close()
            if thread:
                thread.join()
            if self.speaker:
                self.speaker.stop()
            self.state = "off"

    def get_status(self):
        cutoff = time.monotonic() - MIC_PEAK_SECONDS
        recent = [level for stamp, level in list(self.mic_peaks) if stamp >= cutoff]
        return {
            "on": self.state != "off",
            "state": self.state,
            "error": self.error,
            "lines": list(self.lines),
            "mic_level": max(recent, default=0),
            "chunks_sent": self.chunks_sent,
            "events": dict(self.events),
        }

    def search_started(self, target):
        """Tells the model the camera has started searching without being asked."""
        self.lines.append({"who": "tool", "text": f"lost {target}: searching the room"})
        self.queue_notice(f"You lost sight of {target}, and the camera is now searching the room for it. Tell the person briefly. You will be told when the search ends.")

    def search_done(self, target, found, error=None, returned=False):
        """Tells the model how a room search ended, so it can tell the person."""
        result = error or ("found" if found else "not found")
        self.lines.append({"who": "tool", "text": f"search for {target}: {result}" + (", returning to where it was last seen" if returned else "")})
        ending = "The camera is going back to where it last saw it." if returned else "The camera has stopped."
        text = (
            f"The search ended: {target} is in view and is now being followed."
            if found else f"The search ended: {target} was not found anywhere the camera looked. {ending}"
        )
        if error:
            text = f"The search for {target} could not finish: {error}"
        self.queue_notice(text)

    def queue_notice(self, text):
        self.notices.put((text, self.follower.generation))

    def send_notice(self, socket):
        if self.responding:
            return
        if self.pending_notice is None:
            try:
                self.pending_notice = self.notices.get_nowait()
            except queue.Empty:
                return
        text, generation = self.pending_notice
        # Any command since then makes this out of date.
        if generation != self.follower.generation:
            self.pending_notice = None
            return
        socket.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "system", "content": [{"type": "input_text", "text": text}]},
        }))
        self.create_response(socket)
        self.pending_notice = None

    def create_response(self, socket):
        self.responding = True
        socket.send(json.dumps({"type": "response.create"}))

    def call_tool(self, name, arguments):
        if name in ("follow", "find"):
            target = clean_target(arguments["target"])
            if not target:
                return {"error": "Describe the target in a few words."}
            # A person's own face or hand comes to the camera; a pet has to be looked for.
            # Hands only make sense once someone is already in front of the camera.
            search = target != "hand" and (name == "find" or target != "face")
            self.follower.enable(target, search=search)
            return {"following": target, "searching": search}
        if name == "stop_following":
            self.follower.set_target(None)
            return {"following": None}
        if name == "nod_yes":
            self.follower.gesture("yes")
            return {"done": True}
        if name == "shake_no":
            self.follower.gesture("no")
            return {"done": True}
        if name == "what_do_you_see":
            status = self.follower.get_status()
            if not status.get("scene_available"):
                return {"error": "The camera's current view is unavailable. Try again after video and detection recover."}
            return {"objects": [{"label": o["label"], "confidence": o["score"]} for o in status["scene"]]}
        return {"error": f"Unknown tool {name}"}

    def _send_mic(self, socket, stop_event, microphone, mic_lock):
        # Reconnects because the camera can drop us, or refuse a fourth connection.
        while not stop_event.is_set():
            proc = self.open_mic()
            with mic_lock:
                microphone[0] = proc
            got_audio = False
            try:
                while not stop_event.is_set():
                    data = proc.stdout.read(MIC_CHUNK_BYTES)
                    if not data:
                        break
                    if not got_audio:
                        got_audio = True
                        self.error = None
                    # A-law stores loudness on a 0-7 scale in bits 4-6 of each byte.
                    now = time.monotonic()
                    self.mic_peaks.append((now, max((byte ^ 0x55) >> 4 & 7 for byte in data)))
                    # The camera's microphone would otherwise hear its own speaker.
                    if now < self.muted_until:
                        continue
                    if stop_event.is_set():
                        break
                    louder = audioop.lin2alaw(audioop.mul(audioop.alaw2lin(data, 2), 2, MIC_GAIN), 2)
                    socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(louder).decode()}))
                    self.chunks_sent += 1
            except ConnectionClosed:
                return
            finally:
                with mic_lock:
                    if microphone[0] is proc:
                        self.close_mic(proc)
                        microphone[0] = None
            if not stop_event.is_set():
                self.error = "Lost the camera's microphone, reconnecting. Closing other camera pages may help."
                stop_event.wait(2)

    def handle_event(self, socket, event):
        kind = event.get("type")
        self.events[kind] += 1
        if kind == "response.created":
            self.responding = True
        elif kind == "input_audio_buffer.speech_started":
            self.state = "hearing"
        elif kind == "input_audio_buffer.speech_stopped":
            self.state = "thinking"
        elif kind == "response.output_audio.delta":
            audio = base64.b64decode(event["delta"])
            if self.speaker:
                # A-law at 8 kHz is one byte per sample.
                self.speech_ends = max(self.speech_ends, time.monotonic()) + len(audio) / 8000
                self.muted_until = self.speech_ends + ECHO_GUARD_SECONDS
                try:
                    self.speaker.play(audio)
                except RuntimeError as exc:
                    self.error = str(exc)
        elif kind == "response.output_audio.done":
            if self.speaker:
                self.speaker.finish()
        elif kind == "conversation.item.input_audio_transcription.completed":
            if event.get("transcript", "").strip():
                self.lines.append({"who": "heard", "text": event["transcript"].strip()})
        elif kind in ("response.output_audio_transcript.done", "response.output_text.done"):
            text = event.get("transcript") or event.get("text") or ""
            if text.strip():
                self.lines.append({"who": "camera", "text": text.strip()})
        elif kind == "error":
            self.error = event.get("error", {}).get("message", "Unknown error from OpenAI.")
        elif kind == "response.done":
            self.responding = False
            self.state = "listening"
            calls = [o for o in event["response"].get("output", []) if o.get("type") == "function_call"]
            for call in calls:
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                    self.lines.append({"who": "tool", "text": f"{call['name']}({', '.join(map(str, arguments.values()))})"})
                    result = self.call_tool(call["name"], arguments)
                except Exception as exc:
                    result = {"error": str(exc)}
                socket.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)},
                }))
            if calls:
                self.create_response(socket)

    def connect(self):
        return connect(
            f"wss://api.openai.com/v1/realtime?model={MODEL}",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
            open_timeout=15,
            close_timeout=2,
        )

    def _run(self, stop_event):
        while not stop_event.is_set():
            mic_stop = threading.Event()
            microphone, mic_lock = [None], threading.Lock()
            mic_thread = None
            try:
                with self.connect() as socket:
                    if stop_event.is_set():
                        break
                    config = session_config(self.speaker)
                    status = self.follower.get_status()
                    config["instructions"] += f"\nCurrent camera state: {status['state']}; target: {status['target']}."
                    socket.send(json.dumps({"type": "session.update", "session": config}))
                    self.socket = socket
                    self.responding = False
                    self.state = "listening"
                    mic_thread = threading.Thread(target=self._send_mic, args=(socket, mic_stop, microphone, mic_lock), daemon=True)
                    mic_thread.start()
                    while not stop_event.is_set():
                        self.send_notice(socket)
                        try:
                            message = socket.recv(timeout=0.5)
                        except TimeoutError:
                            continue
                        if not stop_event.is_set():
                            self.handle_event(socket, json.loads(message))
            except Exception as exc:
                if not stop_event.is_set():
                    self.error = f"Voice connection problem: {type(exc).__name__}. Reconnecting…"
                    self.state = "connecting"
            finally:
                self.socket = None
                self.responding = False
                mic_stop.set()
                with mic_lock:
                    if microphone[0]:
                        self.close_mic(microphone[0])
                        microphone[0] = None
                if mic_thread:
                    mic_thread.join()
            if not stop_event.is_set():
                stop_event.wait(3)
