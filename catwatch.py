"""Watches for the cat without ever moving the camera, and keeps where it was last seen.

Runs YOLO26 (trained on COCO, whose classes include cat) on the CPU with ONNX Runtime.
The old Kepler GPUs stay at full power the whole time a program holds them, which kept
them above their 80 C rating and caused "illegal memory access" errors overnight.
Infrared frames get their contrast boosted first, which brings out the dark room.
Checks run back to back, as fast as the CPU allows. Smaller, faster models did not
recognise her at all: a cat across the room is only about 80 px wide.
"""
import datetime
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from tracker import FRAME_HEIGHT, FRAME_WIDTH

PROJECT = Path(__file__).parent
# Medium found her in real night frames better than large (62% against 27%), and faster.
MODEL = PROJECT / "models" / "yolo" / "yolo26m.onnx"
DATA = PROJECT / "sightings"
LAST_IMAGE = DATA / "last_cat.jpg"
LAST_INFO = DATA / "last_seen.json"
STATE_FILE = DATA / "watch.json"

CHECK_SECONDS = 0.0  # No pause: check again as soon as the last check is done.
THREADS = 6  # Of the 12, leaving room for video, following and the web server.
COCO_CAT = 15
# Whole frame: 960 found small dark cats as well as 1280 in tests, in half the time; 640 missed them.
IMAGE_SIZE = 960
MIN_SCORE = {False: 0.40, True: 0.40}  # Daylight, night vision.
NMS_OVERLAP = 0.5
CONFIRM = (2, 3)  # Seen in 2 of the last 3 checks.
GONE_SECONDS = 5.0  # Unseen this long ends a visit.
SAVE_SECONDS = 2.0  # While in view, refresh the last picture this often.
POSITION_SECONDS = 10.0  # And ask the camera where it points this often.
KEEP_VISITS = 300  # Pictures kept, one per visit.
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def is_infrared(rgb):
    """Night vision frames have identical colour channels."""
    sample = rgb[::16, ::16].astype(np.int16)
    return float(np.abs(sample[:, :, 0] - sample[:, :, 1]).mean() + np.abs(sample[:, :, 1] - sample[:, :, 2]).mean()) < 2.0


def enhance(rgb):
    return cv2.cvtColor(CLAHE.apply(np.ascontiguousarray(rgb[:, :, 0])), cv2.COLOR_GRAY2RGB)


class CatDetector:
    def __init__(self):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def _run(self, image, width, threshold):
        """Detects cats in an image, scaled to the given width. Boxes are fractions of the image."""
        height = image.shape[0]
        scaled = round(height * width / image.shape[1])
        padded = -(-scaled // 32) * 32
        top = (padded - scaled) // 2
        canvas = np.full((padded, width, 3), 114, np.uint8)
        canvas[top:top + scaled] = cv2.resize(image, (width, scaled), interpolation=cv2.INTER_AREA)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
        # One row per candidate box: centre x, centre y, width, height, then a score per class.
        out = self.session.run(None, {self.input: blob})[0][0]
        scores = out[4 + COCO_CAT]
        keep = scores >= threshold
        cx, cy, w, h, scores = out[0, keep], out[1, keep], out[2, keep], out[3, keep], scores[keep]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
        chosen = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, NMS_OVERLAP), int).flatten()
        found = []
        for i in chosen:
            left, upper, box_w, box_h = boxes[i]
            box = [left / width, (upper - top) / scaled, (left + box_w) / width, (upper + box_h - top) / scaled]
            found.append({"box": [min(max(float(v), 0.0), 1.0) for v in box], "score": round(float(scores[i]), 2)})
        return found

    def detect(self, rgb):
        """Checks the whole frame. Returns the cats, and whether it was night vision."""
        infrared = is_infrared(rgb)
        image = enhance(rgb) if infrared else rgb
        cats = self._run(image, IMAGE_SIZE, MIN_SCORE[infrared])
        cats = [{**c, "box": [round(v, 4) for v in c["box"]], "models": ["yolo26"]} for c in cats]
        return sorted(cats, key=lambda c: -c["score"]), infrared


def picture(rgb, cats, infrared):
    """The frame as a JPEG with the cats boxed; infrared ones brightened like the detector saw them."""
    if infrared:
        image = cv2.cvtColor(CLAHE.apply(rgb[:, :, 0].copy()), cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for cat in cats:
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(cat["box"], (FRAME_WIDTH, FRAME_HEIGHT) * 2))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(image, f"cat {round(cat['score'] * 100)}%", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return data.tobytes() if ok else None


def log(text):
    print(f"[cat watch {datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {text}", file=sys.stderr, flush=True)


def write_atomic(path, data):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_bytes(data)
    partial.replace(path)


def read_frames(proc, latest, ready, stop_event):
    size = FRAME_WIDTH * FRAME_HEIGHT * 3
    try:
        while not stop_event.is_set():
            data = proc.stdout.read(size)
            if len(data) < size:
                break
            latest[0] = (data, time.monotonic())
            ready.set()
    finally:
        ready.set()


class CatWatcher:
    def __init__(self, open_video, close_video, position, camera_error=OSError, recorder=None):
        self.open_video = open_video
        self.recorder = recorder  # Records clips of her; see clips.py.
        self.close_video = close_video
        self.position = position  # Only reads where the camera points; never moves it.
        self.camera_error = camera_error
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.status = {"state": "off"}
        self.cats = []
        self.cats_time = 0.0
        self.last = self._load_last()
        DATA.mkdir(exist_ok=True)

    # --- Saved state --------------------------------------------------

    @staticmethod
    def _load_last():
        try:
            return json.loads(LAST_INFO.read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def was_on():
        try:
            return json.loads(STATE_FILE.read_text()).get("enabled") is True
        except (OSError, ValueError):
            return False

    @staticmethod
    def _remember_on(on):
        write_atomic(STATE_FILE, json.dumps({"enabled": on}).encode())

    # --- Control ------------------------------------------------------

    def enable(self):
        with self.lock:
            self._remember_on(True)
            if self.thread and self.thread.is_alive():
                return
            self.stop_event = threading.Event()
            self.status = {"state": "starting"}
            if self.recorder:
                self.recorder.start()
            self.thread = threading.Thread(target=self._run, args=(self.stop_event,), daemon=True)
            self.thread.start()

    def disable(self):
        with self.lock:
            self._remember_on(False)
            self.stop_event.set()
            thread = self.thread
        if thread:
            thread.join(timeout=10)
        if self.recorder:
            self.recorder.stop()
        with self.lock:
            self.cats = []
            self.status = {"state": "off"}

    def get_status(self):
        running = bool(self.thread and self.thread.is_alive())
        fresh = running and time.monotonic() - self.cats_time < 2
        recording = self.recorder.status() if self.recorder and running else {}
        return {**self.status, **recording, "enabled": running, "cats": self.cats if fresh else [], "last": self.last}

    # --- Watching -----------------------------------------------------

    def _read_position(self):
        try:
            return self.position()
        except (OSError, self.camera_error):
            return None

    def _save(self, rgb, cats, infrared, visit_start, position, new_visit):
        now = datetime.datetime.now().astimezone()
        image = picture(rgb, cats, infrared)
        if image is None:
            return
        write_atomic(LAST_IMAGE, image)
        if new_visit:
            write_atomic(DATA / f"cat-{now:%Y%m%d-%H%M%S}.jpg", image)
            visits = sorted(DATA.glob("cat-*.jpg"))
            for old in visits[:-KEEP_VISITS]:
                old.unlink(missing_ok=True)
        best = cats[0]
        self.last = {
            "time": now.isoformat(timespec="seconds"),
            "visit_started": visit_start.isoformat(timespec="seconds"),
            "box": best["box"],
            "score": best["score"],
            "models": best["models"],
            "count": len(cats),
            "infrared": infrared,
            "pan": position[0] if position else None,
            "tilt": position[1] if position else None,
        }
        write_atomic(LAST_INFO, json.dumps(self.last, indent=1).encode())

    def _run(self, stop_event):
        # Meant to run all night, so any failure is retried rather than ending the watch.
        while not stop_event.is_set():
            try:
                detector = CatDetector()
                log(f"detector started on the CPU ({MODEL.name}, {IMAGE_SIZE} px)")
                while not stop_event.is_set():
                    self._watch(detector, stop_event)
                    if not stop_event.is_set():
                        self.status = {"state": "error", "error": "The video stopped. Reconnecting…"}
                        stop_event.wait(3)
            except Exception as exc:
                if not stop_event.is_set():
                    log(f"detector failed, restarting: {exc}")
                    self.status = {"state": "error", "error": f"Detector problem: {exc}. Restarting…"}
                    stop_event.wait(10)
        self.status = {"state": "off"}

    def _watch(self, detector, stop_event):
        latest, ready, reader_stop = [None], threading.Event(), threading.Event()
        proc = reader = None
        recent = []
        visit_start = None
        last_save = last_position_time = seen_at = 0.0
        position = None
        check_times = []
        try:
            # The same connection saves the camera's video for clips, if they are on.
            proc = self.open_video(self.recorder.output_args() if self.recorder else [])
            reader = threading.Thread(target=read_frames, args=(proc, latest, ready, reader_stop), daemon=True)
            reader.start()
            while not stop_event.is_set():
                if not ready.wait(timeout=5):
                    return
                ready.clear()
                frame, latest[0] = latest[0], None
                if frame is None:
                    if not reader.is_alive():
                        return
                    continue
                data, stamp = frame
                rgb = np.frombuffer(data, np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)
                started = time.perf_counter()
                next_check = started + CHECK_SECONDS
                cats, infrared = detector.detect(rgb)
                detect_ms = round((time.perf_counter() - started) * 1000)
                now = time.monotonic()
                self.cats, self.cats_time = cats, now
                check_times = (check_times + [now])[-20:]
                fps = (len(check_times) - 1) / (check_times[-1] - check_times[0]) if len(check_times) > 1 else 0.0

                recent = (recent + [bool(cats)])[-CONFIRM[1]:]
                if cats and sum(recent) >= CONFIRM[0]:
                    new_visit = visit_start is None
                    if new_visit:
                        visit_start = datetime.datetime.now().astimezone()
                        log(f"cat seen ({cats[0]['score']:.0%}{', night vision' if infrared else ''})")
                    seen_at = now
                    if self.recorder:
                        self.recorder.cat_seen(lambda: picture(rgb, cats, infrared), cats[0]["score"], infrared)
                    if new_visit or now - last_position_time >= POSITION_SECONDS:
                        position = self._read_position() or position
                        last_position_time = now
                    if new_visit or now - last_save >= SAVE_SECONDS:
                        self._save(rgb, cats, infrared, visit_start, position, new_visit)
                        last_save = now
                elif visit_start and now - seen_at > GONE_SECONDS:
                    visit_start = None

                self.status = {
                    "state": "cat" if visit_start else "watching",
                    "infrared": infrared,
                    "detect_ms": detect_ms,
                    "fps": round(fps, 1),
                }
                stop_event.wait(max(0.0, next_check - time.perf_counter()))
        except (OSError, self.camera_error) as exc:
            self.status = {"state": "error", "error": f"Camera connection problem: {exc}. Reconnecting…"}
        finally:
            reader_stop.set()
            if proc:
                self.close_video(proc)
            if reader:
                reader.join(timeout=5)
