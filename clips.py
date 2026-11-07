"""Records a clip whenever the cat is in view, ending 30 s after she was last seen.

While the watch is on, its ffmpeg also copies the camera's high stream into 2 s pieces in memory
(no re-encoding, so it costs almost no CPU). A clip is those pieces, starting a few seconds
before she was spotted so it shows her arriving, compressed into a smaller MP4 once she has
gone. Pieces nobody needs are deleted after a few seconds.
"""
import datetime
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).parent
CLIPS = PROJECT / "clips"
BUFFER = Path("/dev/shm/tapo-cam-clips")

PIECE_SECONDS = 2  # Pieces start on a keyframe; the camera sends one about every 2 s.
BEFORE_SECONDS = 5  # Kept from before she was spotted.
AFTER_SECONDS = 30  # A clip ends once she has been out of sight this long.
MAX_CLIP_SECONDS = 15 * 60  # A longer visit carries on in a new clip, with no gap.
MAX_BYTES = 5 * 1024 ** 3  # Oldest clips are deleted beyond this,
MIN_FREE_BYTES = 2 * 1024 ** 3  # or when the disk gets this full.
# Saved clips are re-encoded: 1600x900 at this quality came out about 6 times smaller than
# the camera's 2304x1296 and looked as sharp. One at a time, on 3 threads at low priority,
# so cat detection keeps its speed; a minute of video takes about 7 s.
ENCODE = [
    "-vf", "scale=1600:900", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
    "-pix_fmt", "yuv420p", "-threads", "3", "-c:a", "copy",
]
encoding = threading.Lock()


def log(text):
    from catwatch import log as watch_log

    watch_log(text)


def clip_files(name):
    return [CLIPS / f"{name}{ext}" for ext in (".mp4", ".jpg", ".json")]


def list_clips():
    found = []
    for info in sorted(CLIPS.glob("clip-*.json"), reverse=True):
        try:
            data = json.loads(info.read_text())
        except (OSError, ValueError):
            continue
        if (CLIPS / f"{info.stem}.mp4").exists():
            found.append({"name": info.stem, **data})
    return found


def delete_clip(name):
    for path in clip_files(name):
        path.unlink(missing_ok=True)


def prune():
    """Deletes the oldest clips while over the size limit or short of disk space."""
    videos = sorted(CLIPS.glob("clip-*.mp4"))
    total = sum(v.stat().st_size for v in videos)
    while videos and (total > MAX_BYTES or shutil.disk_usage(CLIPS).free < MIN_FREE_BYTES):
        oldest = videos.pop(0)
        total -= oldest.stat().st_size
        delete_clip(oldest.stem)
        log(f"deleted old clip {oldest.stem} to make room")


def duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return round(float(out.strip()), 1)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


class Clip:
    def __init__(self, started, thumbnail, score, infrared):
        self.started = started  # When she was spotted, as a local datetime.
        self.name = f"clip-{started:%Y%m%d-%H%M%S}"
        self.pieces = []
        self.best = score
        self.infrared = infrared
        self.opened = time.monotonic()
        CLIPS.mkdir(exist_ok=True)
        if thumbnail:
            (CLIPS / f"{self.name}.jpg").write_bytes(thumbnail)


class ClipRecorder:
    """Keeps the pieces that the cat watcher's ffmpeg writes, and makes clips from them.

    The camera takes only 4 video connections at once, so the watcher's one connection
    feeds both the detector and these pieces: see output_args().
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = None
        self.thread = None
        self.clip = None
        self.last_seen = 0.0
        self.pending = None  # A sighting waiting for the next round to open a clip.
        self.folder = None  # Where the current video connection writes its pieces.
        self.runs = 0
        self.finishing = []
        CLIPS.mkdir(exist_ok=True)
        for partial in CLIPS.glob("*.part"):
            partial.unlink()

    # --- Control ------------------------------------------------------

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            # Also stops an ffmpeg left over from a server that was killed: with its
            # folder gone it cannot start its next piece, so it exits.
            shutil.rmtree(BUFFER, ignore_errors=True)
            self.folder = None
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._run, args=(self.stop_event,), daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            if self.stop_event:
                self.stop_event.set()
            thread = self.thread
        if thread:
            thread.join(timeout=60)

    def output_args(self):
        """ffmpeg output arguments that save the camera's own video and sound in pieces.

        Called for each new video connection, which gets a folder of its own.
        """
        with self.lock:
            self.runs += 1
            folder = BUFFER / f"{os.getpid()}-{self.runs}"
            folder.mkdir(parents=True, exist_ok=True)
            self.folder = folder
        return [
            "-map", "0:v", "-map", "0:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "32k",
            "-f", "segment", "-segment_time", str(PIECE_SECONDS), "-segment_format", "mpegts",
            "-reset_timestamps", "0", str(folder / "piece%06d.ts"),
        ]

    def cat_seen(self, thumbnail, score, infrared):
        """Called on every confirmed sighting. thumbnail() makes a JPEG, only if a clip opens."""
        with self.lock:
            self.last_seen = time.monotonic()
            if self.clip:
                self.clip.best = max(self.clip.best, score)
                self.clip.infrared = self.clip.infrared or infrared
            elif not self.pending:
                self.pending = (datetime.datetime.now().astimezone(), thumbnail(), score, infrared)

    def status(self):
        with self.lock:
            clip = self.clip
            return {
                "recording": clip.name if clip else None,
                "recording_since": clip.started.isoformat(timespec="seconds") if clip else None,
            }

    # --- Recording ----------------------------------------------------

    def _run(self, stop_event):
        current = None
        while True:
            stopping = stop_event.wait(0.5)
            folder = self.folder
            # A new connection means the old one is closed and its last piece complete.
            if current and (folder != current or stopping):
                self._close(current)
                current = None
            if stopping:
                break
            if folder and folder.exists():
                current = folder
                try:
                    self._tick(folder)
                except OSError as exc:
                    log(f"clip recording problem: {exc}")
        shutil.rmtree(BUFFER, ignore_errors=True)

    def _close(self, folder):
        with self.lock:
            clip, self.clip = self.clip, None
        if clip:
            self._claim(clip, folder, include_last=True)
            self._finish(self._set_aside(clip, folder))
        for thread in self.finishing:
            thread.join()
        self.finishing = []
        shutil.rmtree(folder, ignore_errors=True)

    def _tick(self, folder):
        pieces = sorted(folder.glob("piece*.ts"))
        now = time.monotonic()
        with self.lock:
            clip = self.clip
            if clip is None and self.pending:
                started, thumbnail, score, infrared = self.pending
                self.pending = None
                clip = self.clip = Clip(started, thumbnail, score, infrared)
                log(f"recording {clip.name}")
                # The last few seconds before she was spotted come along too.
                cutoff = time.time() - BEFORE_SECONDS - PIECE_SECONDS
                clip.pieces = [p for p in pieces[:-1] if p.stat().st_mtime >= cutoff]
                clip.pieces = clip.pieces or pieces[-2:-1]
            gone = now - self.last_seen > AFTER_SECONDS
        if clip is None:
            # Nobody needs pieces this old.
            cutoff = time.time() - BEFORE_SECONDS - 3 * PIECE_SECONDS
            for piece in pieces[:-1]:
                if piece.stat().st_mtime < cutoff:
                    piece.unlink(missing_ok=True)
            return

        self._claim(clip, folder)
        too_long = now - clip.opened > MAX_CLIP_SECONDS
        if gone or too_long:
            with self.lock:
                self.clip = None
                if too_long and not gone:
                    # Still here: carry straight on in a new clip.
                    self.pending = (datetime.datetime.now().astimezone(), None, clip.best, clip.infrared)
            thread = threading.Thread(target=self._finish, args=(self._set_aside(clip, folder),), daemon=True)
            thread.start()
            self.finishing = [t for t in self.finishing if t.is_alive()] + [thread]

    @staticmethod
    def _set_aside(clip, folder):
        """Moves an ended clip's pieces away from the clean-up of unneeded ones."""
        own = folder / clip.name
        own.mkdir(exist_ok=True)
        moved = []
        for piece in clip.pieces:
            try:
                moved.append(piece.replace(own / piece.name))
            except FileNotFoundError:
                pass
        clip.pieces = moved
        return clip

    @staticmethod
    def _claim(clip, folder, include_last=False):
        """Adds the pieces ffmpeg has finished writing since the clip's last one."""
        pieces = sorted(folder.glob("piece*.ts"))
        if not include_last:
            pieces = pieces[:-1]  # Still being written.
        newest = clip.pieces[-1].name if clip.pieces else ""
        clip.pieces += [p for p in pieces if p.name > newest and p not in clip.pieces]

    def _finish(self, clip):
        """Joins the clip's pieces into an MP4 in the clips folder."""
        pieces = [p for p in clip.pieces if p.exists()]
        video = CLIPS / f"{clip.name}.mp4"
        partial = CLIPS / f"{clip.name}.mp4.part"
        try:
            if not pieces:
                raise OSError("no video was recorded")
            with encoding:
                proc = subprocess.Popen(
                    ["nice", "-n", "10", "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "mpegts", "-i", "pipe:0", *ENCODE, "-movflags", "+faststart", "-f", "mp4", str(partial)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                for piece in pieces:
                    proc.stdin.write(piece.read_bytes())
                proc.stdin.close()
                errors = proc.stderr.read().decode("utf-8", "replace").strip()
                if proc.wait(timeout=1800) != 0:
                    raise OSError(errors or "ffmpeg failed")
            partial.replace(video)
            seconds = duration(video)
            info = {
                "started": clip.started.isoformat(timespec="seconds"),
                "seconds": seconds,
                "bytes": video.stat().st_size,
                "score": clip.best,
                "infrared": clip.infrared,
            }
            (CLIPS / f"{clip.name}.json").write_text(json.dumps(info, indent=1))
            log(f"saved {clip.name} ({seconds} s, {info['bytes'] / 1e6:.1f} MB)")
            if not (CLIPS / f"{clip.name}.jpg").exists():
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "1", "-i", str(video),
                     "-frames:v", "1", "-vf", "scale=640:-2", str(CLIPS / f"{clip.name}.jpg")],
                    capture_output=True, timeout=30,
                )
            prune()
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"could not save {clip.name}: {exc}")
            partial.unlink(missing_ok=True)
            if not video.exists():
                (CLIPS / f"{clip.name}.jpg").unlink(missing_ok=True)
        finally:
            for piece in clip.pieces:
                piece.unlink(missing_ok=True)
            if clip.pieces and clip.pieces[0].parent.name == clip.name:
                shutil.rmtree(clip.pieces[0].parent, ignore_errors=True)
