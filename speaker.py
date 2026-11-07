"""Plays speech on the camera's speaker through go2rtc, which talks directly to the camera."""
import ctypes
import errno
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

import requests

BINARY = Path(__file__).parent / "bin" / "go2rtc"
API = "http://127.0.0.1:1984"


def die_with_parent():
    # Otherwise go2rtc outlives a killed server and keeps its port.
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)


class Speaker:
    def __init__(self, host, password):
        self.host = host
        self.password = password
        self.proc = None
        self.folder = None
        self.holding = None
        self.lock = threading.Lock()
        self.chunks = None
        self.stopped = threading.Event()

    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            self.stopped = threading.Event()
            self.folder = tempfile.mkdtemp(prefix="tapo-speaker-")
            config = os.path.join(self.folder, "go2rtc.yaml")
            secret = urllib.parse.quote(self.password, safe="")
            descriptor = os.open(config, os.O_WRONLY | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                handle.write(
                    'api:\n  listen: "127.0.0.1:1984"\nrtsp:\n  listen: ""\nwebrtc:\n  listen: ""\n'
                    'rtmp:\n  listen: ""\nsrtp:\n  listen: ""\nlog:\n  level: error\n'
                    'ffmpeg:\n  alaw: "-re -fflags nobuffer -probesize 32 -analyzeduration 0 -f alaw -ar 8000 -ac 1 -i {input}"\n'
                    f'streams:\n  camera: "tapo://{secret}@{self.host}?subtype=1"\n'
                )
            started = threading.Event()

            def supervise():
                # The death signal follows the thread that starts go2rtc, so this
                # must outlive the web request that turned voice on.
                self.proc = subprocess.Popen(
                    [str(BINARY), "-config", config],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    preexec_fn=die_with_parent,
                )
                started.set()
                self.proc.wait()

            threading.Thread(target=supervise, daemon=True).start()
            started.wait(timeout=5)
            ready = False
            for _ in range(50):
                try:
                    requests.get(f"{API}/api", timeout=1)
                    ready = True
                    break
                except requests.RequestException:
                    time.sleep(0.1)
            # go2rtc has read it now, so the password need not stay on disk.
            os.remove(config)
            if not ready or self.proc.poll() is not None:
                self.proc.terminate()
                self.proc = None
                raise RuntimeError("The camera speaker could not start: port 1984 may be in use.")
            self.holding = threading.Event()
            threading.Thread(target=self._hold_connection, args=(self.holding,), daemon=True).start()

    def _hold_connection(self, holding):
        # go2rtc only stays connected to the camera while something is watching.
        # Reconnecting takes seconds, which would delay the start of every reply.
        while not holding.is_set():
            try:
                with requests.get(f"{API}/api/stream.mp4?src=camera&video=h264", stream=True, timeout=(3, 15)) as reply:
                    for _ in reply.iter_content(65536):
                        if holding.is_set():
                            return
            except requests.RequestException:
                pass
            holding.wait(2)

    def stop(self):
        self.stopped.set()
        self.finish()
        if self.holding:
            self.holding.set()
        with self.lock:
            if self.proc:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
                self.proc = None
            if self.folder:
                shutil.rmtree(self.folder, ignore_errors=True)
                self.folder = None

    def play(self, audio):
        """Queues 8 kHz A-law audio, starting a new utterance if none is playing."""
        if self.stopped.is_set() or self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("The camera speaker stopped. Turn Voice off and on to restart it.")
        if self.chunks is None:
            self.chunks = queue.Queue()
            threading.Thread(target=self._stream, args=(self.chunks, self.folder, self.proc, self.stopped), daemon=True).start()
        self.chunks.put(audio)

    def finish(self):
        """Marks the end of an utterance; what is queued still plays."""
        if self.chunks is not None:
            self.chunks.put(None)
            self.chunks = None

    def _stream(self, chunks, folder, proc, stopped):
        fifo = os.path.join(folder, f"speech-{time.monotonic_ns()}.alaw")
        failed = threading.Event()
        descriptor = None
        try:
            os.mkfifo(fifo, 0o600)
            source = urllib.parse.quote(f"ffmpeg:{fifo}#input=alaw#audio=pcma", safe="")

            def play():
                try:
                    reply = requests.post(f"{API}/api/streams?dst=camera&src={source}", timeout=60)
                    reply.raise_for_status()
                except requests.RequestException:
                    failed.set()

            threading.Thread(target=play, daemon=True).start()
            deadline = time.monotonic() + 5
            while not stopped.is_set() and not failed.is_set() and proc.poll() is None:
                try:
                    descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError as exc:
                    if exc.errno != errno.ENXIO:
                        raise
                    if time.monotonic() >= deadline:
                        return
                    stopped.wait(0.05)
            if descriptor is None:
                return
            while not stopped.is_set() and not failed.is_set() and proc.poll() is None:
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                data = memoryview(chunk)
                deadline = time.monotonic() + 5
                while data and not stopped.is_set() and not failed.is_set() and proc.poll() is None:
                    try:
                        data = data[os.write(descriptor, data):]
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            return
                        stopped.wait(0.05)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.remove(fifo)
            except OSError:
                pass
