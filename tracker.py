"""Points the camera at a face, a hand or anything named. All models run on this PC."""
import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJECT = Path(__file__).parent
MODELS = PROJECT / "models"
GPU_PYTHON = PROJECT / ".venv-gpu" / "bin" / "python"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_TARGET_LENGTH = 40

# What YOLOE looks for by default; anything else named gets added to this.
SCENE_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def clean_target(text):
    """Returns a named target in its usual form, or None if it is unusable."""
    target = " ".join(str(text).lower().split())
    target = {"hands": "hand", "faces": "face"}.get(target, target)
    return target if target and len(target) <= MAX_TARGET_LENGTH else None


# Errors are fractions of the frame, measured from its centre. Video reaches
# the detector about 0.4 s late, so the camera slows down as the target nears
# the centre, and only starts again once it has clearly left the stop zone.
STOP_BAND = 0.07
START_BAND = 0.12
SPEED_STEPS = ((0.30, 1.0), (0.18, 0.6), (0.0, 0.4))
TILT_LIMIT = 0.6  # Faster tilting overshoots and hunts up and down.
# Aim for where the target will be once the video and the camera catch up.
PREDICT_SECONDS = 0.5
VELOCITY_SMOOTHING = 0.5
VELOCITY_MAX_GAP = 0.5

CONFIRM_FRAMES = 3  # Sightings needed before moving, to ignore false detections.
LOST_FRAMES = 3  # Misses allowed before treating the target as gone.
# A target that leaves the frame gets chased the way it went, then searched for.
CHASE_SECONDS = 2.5
CHASE_PAN = 1.0
CHASE_TILT = 0.6
CHASE_EDGE = 0.2
MOVE_TIMEOUT = "PT2S"  # The camera stops by itself if this program dies mid-move.
REFRESH_SECONDS = 1.0
SEARCH_SPEED = 0.6
SEARCH_MOVE_SECONDS = 0.8
# After each search step: discard frames from while the camera moved, then look.
SEARCH_SKIP_SECONDS = 0.5
SEARCH_LOOK_SECONDS = 1.2
OBJECT_SCORE = 0.35
SCENE_SECONDS = 0.5
SCENE_SECONDS_FOLLOWING = 2.0  # Both share the GPU, so following gets most of it.
FACE_ON_PERSON = 0.12  # Where a face sits on a person's box, when too far to detect.

# (pan, tilt, seconds): out, through the start point, and back again.
GESTURES = {
    "yes": ((0, 0.6, 0.3), (0, -0.6, 0.6), (0, 0.6, 0.3)),
    "no": ((-0.6, 0, 0.3), (0.6, 0, 0.6), (-0.6, 0, 0.3)),
}


def search_steps():
    """Sweeps right, then up and left, then down and right, ending near the start."""
    right, left, up, down = (SEARCH_SPEED, 0), (-SEARCH_SPEED, 0), (0, SEARCH_SPEED), (0, -SEARCH_SPEED)
    return [right] * 6 + [up] + [left] * 12 + [down] * 2 + [right] * 12 + [up] + [left] * 6


class Search:
    def __init__(self):
        now = time.perf_counter()
        self.steps = search_steps()
        self.moving = False
        self.until = now + SEARCH_LOOK_SECONDS
        self.skip_until = now
        self.sightings = 0
        self.total = len(self.steps)

    def look_again(self):
        """Starts a fresh look, ignoring frames from before it."""
        self.moving = False
        self.skip_until = time.perf_counter() + SEARCH_SKIP_SECONDS
        self.until = self.skip_until + SEARCH_LOOK_SECONDS
        self.sightings = 0


def ahead(error, velocity):
    """The error expected soon. Zero if the target should have crossed the centre by then."""
    expected = error + velocity * PREDICT_SECONDS
    return expected if expected * error > 0 else 0.0


class Motion:
    """Tracks how fast a target moves across the frame, including the camera's own turning."""

    def __init__(self):
        self.x = self.y = None
        self.velocity_x = self.velocity_y = 0.0
        self.stamp = 0.0

    def update(self, x, y, stamp):
        gap = stamp - self.stamp
        if self.x is not None and 0 < gap < VELOCITY_MAX_GAP:
            self.velocity_x += VELOCITY_SMOOTHING * ((x - self.x) / gap - self.velocity_x)
            self.velocity_y += VELOCITY_SMOOTHING * ((y - self.y) / gap - self.velocity_y)
        else:
            self.velocity_x = self.velocity_y = 0.0
        self.x, self.y, self.stamp = x, y, stamp

    def predicted_error(self):
        return ahead(self.x - 0.5, self.velocity_x), ahead(self.y - 0.5, self.velocity_y)

    def chase_velocity(self):
        """A target right of centre needs a pan right; one below centre, a tilt down."""
        side = self.x - 0.5 + self.velocity_x * PREDICT_SECONDS
        height = self.y - 0.5 + self.velocity_y * PREDICT_SECONDS
        pan = CHASE_PAN if side > 0 else -CHASE_PAN
        tilt = -CHASE_TILT if height > 0 else CHASE_TILT
        return pan if abs(side) > CHASE_EDGE else 0.0, tilt if abs(height) > CHASE_EDGE else 0.0


def axis_speed(error, moving):
    size = abs(error)
    if size < STOP_BAND or (not moving and size < START_BAND):
        return 0.0
    speed = next(speed for floor, speed in SPEED_STEPS if size >= floor)
    return speed if error > 0 else -speed


def choose_velocity(error_x, error_y, pan, tilt):
    """A target right of centre needs a pan right; one below centre, a tilt down."""
    new_tilt = axis_speed(error_y, tilt != 0)
    return axis_speed(error_x, pan != 0), -max(-TILT_LIMIT, min(TILT_LIMIT, new_tilt))


def to_image(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def to_box(detection):
    box = detection.bounding_box
    left, top = box.origin_x / FRAME_WIDTH, box.origin_y / FRAME_HEIGHT
    return [left, top, left + box.width / FRAME_WIDTH, top + box.height / FRAME_HEIGHT]


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class Yoloe:
    """Talks to yoloe_worker.py, which runs YOLOE on the GPU."""

    def __init__(self, stop_event, device="cuda:0", scene=True):
        self.lock = threading.Lock()
        self.stop_event = stop_event
        self.proc = subprocess.Popen(
            [str(GPU_PYTHON), str(PROJECT / "yoloe_worker.py")],
            cwd=MODELS / "yolo",
            env={**os.environ, "OMP_NUM_THREADS": "4", "YOLO_DEVICE": device},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        os.set_blocking(self.proc.stdin.fileno(), False)
        os.set_blocking(self.proc.stdout.fileno(), False)
        try:
            self._request(None)
            self.target = None
            if scene:
                self._request({"cmd": "prompts", "model": "scene", "labels": list(SCENE_LABELS)})
        except Exception:
            self.close()
            raise

    def _request(self, message, payload=b""):
        # Uncached scene prompts can take over a minute on Kepler.
        deadline = time.monotonic() + 120
        acquired = False
        try:
            while not acquired:
                self._check_deadline(deadline)
                acquired = self.lock.acquire(timeout=0.1)
            data = memoryview(json.dumps(message).encode() + b"\n" + payload) if message is not None else memoryview(b"")
            while data:
                self._check_deadline(deadline)
                if select.select([], [self.proc.stdin], [], 0.1)[1]:
                    data = data[os.write(self.proc.stdin.fileno(), data):]
            line = bytearray()
            while b"\n" not in line:
                self._check_deadline(deadline)
                if select.select([self.proc.stdout], [], [], 0.1)[0]:
                    chunk = os.read(self.proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError("the GPU detector stopped")
                    line.extend(chunk)
            reply = json.loads(line)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply
        except Exception:
            self.close()
            raise
        finally:
            if acquired:
                self.lock.release()

    def _check_deadline(self, deadline):
        if self.stop_event.is_set():
            raise RuntimeError("the GPU detector was stopped")
        if time.monotonic() >= deadline:
            raise RuntimeError("the GPU detector timed out")

    def _detect(self, model, rgb):
        # Enhancing only touches infrared frames, where it helps every model.
        return self.detect_raw(model, rgb, enhance=True)["objects"]

    def detect_raw(self, model, rgb, score=OBJECT_SCORE, classes=None, enhance=False, size=None):
        """Returns the worker's whole reply: the objects, and whether the frame was infrared."""
        message = {"cmd": "detect", "model": model, "width": rgb.shape[1], "height": rgb.shape[0], "score": score, "classes": classes, "enhance": enhance, "size": size}
        return self._request(message, rgb.tobytes())

    def detect(self, rgb):
        """Returns everything recognised from the general list of labels."""
        return self._detect("scene", rgb)

    def find(self, rgb, label):
        """Returns matches for one named thing, looking for nothing else."""
        self.set_target(label)
        return self._detect("target", rgb)

    def set_target(self, label):
        if label != self.target:
            self._request({"cmd": "prompts", "model": "target", "labels": [label]})
            self.target = label

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)
        self.proc.stdin.close()
        self.proc.stdout.close()


class Detectors:
    def __init__(self, stop_event):
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        ))
        self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        ))
        try:
            self.objects = Yoloe(stop_event)
        except Exception:
            self.hand.close()
            self.face.close()
            raise

    def close(self):
        self.hand.close()
        self.face.close()
        self.objects.close()


class Follower:
    def __init__(self, open_video, close_video, move, stop, camera_error=OSError, control_lock=None, position=None, go_to=None):
        self.open_video = open_video
        self.close_video = close_video
        self.move = move
        self.stop_camera = stop
        self.position = position
        self.go_to = go_to
        self.last_seen_at = None
        self.lock = control_lock if control_lock is not None else threading.RLock()
        self.lifecycle_lock = threading.Lock()
        self.camera_error = camera_error
        self.generation = 0
        self.gesture_lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.target = None
        self.search = None
        self.search_moving = False
        self.on_search_started = None
        self.on_search_done = None
        self.last_position = None
        self.latest_rgb = None
        self.frame_time = 0.0
        self.scene_time = 0.0
        self.stream_generation = 0
        self.detector_error = None
        self.scene = []
        self.status = {"state": "off"}

    def enable(self, target, search=False):
        with self.lifecycle_lock:
            with self.lock:
                if self.is_running():
                    self._stop_if_active()
                self.search_moving = False
                self.generation += 1
                self.last_seen_at = None
                self.search = Search() if search else None
                self.target = target
                if self.thread and self.thread.is_alive():
                    return
                self.stop_event = threading.Event()
                self.detector_error = None
                self.status = {"state": "starting"}
                self.thread = threading.Thread(target=self._run, args=(self.stop_event,), daemon=True)
                self.thread.start()

    def set_target(self, target):
        with self.lock:
            self._stop_if_active()
            self.generation += 1
            self.last_seen_at = None
            self.search = None
            self.target = target
            self.search_moving = False

    def _remember_last_seen(self):
        """Saves where the camera is pointing, so it can come back after losing the target."""
        if not self.position:
            return
        try:
            self.last_seen_at = self.position()
        except (OSError, self.camera_error):
            self.last_seen_at = None

    def _return_to_last_seen(self):
        if not (self.go_to and self.last_seen_at):
            return False
        try:
            self.go_to(*self.last_seen_at)
        except (OSError, self.camera_error):
            return False
        # Stop and new commands must be able to cancel the return, like a search move.
        self.search_moving = True
        return True

    def _stop_if_active(self):
        # A needless Stop would jerk a camera being held in a direction by hand.
        if self.target is None and not self.search_moving and not self.paused.is_set():
            return
        try:
            self.stop_camera()
        except (OSError, self.camera_error):
            pass  # The move's own timeout stops it, and following recovers by itself.

    def disable(self):
        with self.lifecycle_lock:
            with self.lock:
                self.generation += 1
                self.search = None
                self.target = None
                thread = self.thread
                self.stop_event.set()
                try:
                    self.stop_camera()
                except (OSError, self.camera_error):
                    pass
                self.search_moving = False
            if thread:
                thread.join()
            with self.lock:
                self._clear_scene()
                self.status = {"state": "off"}

    def is_running(self):
        return bool(self.thread and self.thread.is_alive())

    def get_status(self):
        # No lock: it is held across camera commands, which would stall the page.
        fresh = self.is_running() and self.latest_rgb is not None and self.scene_time > 0 and time.monotonic() - self.frame_time < 5 and time.monotonic() - self.scene_time < 5
        return {**self.status, "enabled": self.is_running(), "target": self.target, "scene": self.scene if fresh else [], "scene_available": fresh}

    def gesture(self, kind):
        """Nods or shakes the camera, pausing any following. Any new command cancels it."""
        with self.gesture_lock:
            try:
                with self.lock:
                    self.generation += 1
                    generation = self.generation
                    self.paused.set()
                    self.stop_camera()
                    self.search_moving = False
                # The lock is free between movements so that Stop can take over.
                for pan, tilt, seconds in GESTURES[kind]:
                    with self.lock:
                        if self.generation != generation:
                            return
                        self.move(pan, tilt, MOVE_TIMEOUT)
                    time.sleep(seconds)
                with self.lock:
                    if self.generation != generation:
                        return
                    self.stop_camera()
                # Let video from during the gesture pass before following again.
                time.sleep(0.6)
            finally:
                with self.lock:
                    if self.search:
                        self.search.look_again()
                    self.paused.clear()

    def create_detectors(self, stop_event):
        return Detectors(stop_event)

    def detect(self, detectors, target, rgb, timestamp_ms):
        """Returns the target's position and box as fractions of the frame, or None."""
        if target == "hand":
            result = detectors.hand.detect_for_video(to_image(rgb), timestamp_ms)
            if not result.hand_landmarks:
                return None
            xs = [point.x for point in result.hand_landmarks[0]]
            ys = [point.y for point in result.hand_landmarks[0]]
            return {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys), "box": [min(xs), min(ys), max(xs), max(ys)]}

        if target == "face":
            faces = [to_box(d) for d in detectors.face.detect_for_video(to_image(rgb), timestamp_ms).detections]
            if faces:
                box = max(faces, key=box_area)
                x, y = centre(box)
                return {"x": x, "y": y, "box": box}
            # The face model only works up close, so aim for the face on a distant person.
            people = [o["box"] for o in detectors.objects.find(rgb, "person")]
            if not people:
                return None
            box = max(people, key=box_area)
            return {"x": centre(box)[0], "y": box[1] + (box[3] - box[1]) * FACE_ON_PERSON, "box": box}

        matches = [o["box"] for o in detectors.objects.find(rgb, target)]
        if not matches:
            return None
        # Stay on the same one when several are in view.
        if self.last_position:
            last_x, last_y = self.last_position
            box = min(matches, key=lambda b: (centre(b)[0] - last_x) ** 2 + (centre(b)[1] - last_y) ** 2)
        else:
            box = max(matches, key=box_area)
        x, y = centre(box)
        return {"x": x, "y": y, "box": box}

    def _finish_search(self, search, target, found, error=None):
        with self.lock:
            if self.search is not search or self.target != target:
                return
            self.search = None
            returned = False
            if not found:
                self.target = None
                returned = not error and self._return_to_last_seen()
            if self.on_search_done:
                self.on_search_done(target, found, error, returned)

    def _current_control(self, generation, stop_event):
        return self.generation == generation and not stop_event.is_set() and not self.paused.is_set()

    def _advance_search(self, search, detectors, target, rgb, generation, stop_event):
        """Runs one frame of the room search. Returns the target once it is confirmed."""
        with self.lock:
            if not self._current_control(generation, stop_event) or self.search is not search:
                return None
            now = time.perf_counter()
            if search.moving and now >= search.until:
                self.stop_camera()
                self.search_moving = False
                search.look_again()
            look = not search.moving and now >= search.skip_until
        if look:
            found = self.detect(detectors, target, rgb, int(now * 1000))
            with self.lock:
                if not self._current_control(generation, stop_event) or self.search is not search:
                    return None
                search.sightings = search.sightings + 1 if found else 0
                if search.sightings >= CONFIRM_FRAMES:
                    self._finish_search(search, target, True)
                    return found
                if time.perf_counter() >= search.until:
                    if not search.steps:
                        self._finish_search(search, target, False)
                        return None
                    pan, tilt = search.steps.pop(0)
                    self.move(pan, tilt, MOVE_TIMEOUT)
                    self.search_moving = True
                    search.moving = True
                    search.until = time.perf_counter() + SEARCH_MOVE_SECONDS
        with self.lock:
            if self._current_control(generation, stop_event) and self.search is search:
                self.status = {
                    "state": "scanning",
                    "box": None,
                    "moving": search.moving,
                    "progress": round(1 - len(search.steps) / search.total, 2),
                }
        return None

    def _read_frames(self, proc, latest, ready, stop_event):
        size = FRAME_WIDTH * FRAME_HEIGHT * 3
        try:
            while not stop_event.is_set():
                data = proc.stdout.read(size)
                if len(data) < size:
                    break
                # Keep only the newest frame so a slow detection never adds delay.
                latest[0] = (data, time.monotonic())
                ready.set()
        finally:
            ready.set()

    def _clear_scene(self):
        self.latest_rgb = None
        self.frame_time = self.scene_time = 0.0
        self.stream_generation += 1
        self.scene = []

    def _scene_loop(self, detectors, stop_event):
        while not stop_event.wait(SCENE_SECONDS_FOLLOWING if self.target else SCENE_SECONDS):
            with self.lock:
                rgb, stamp, generation = self.latest_rgb, self.frame_time, self.stream_generation
            if rgb is not None and time.monotonic() - stamp < 5:
                try:
                    objects = detectors.objects.detect(rgb)
                    with self.lock:
                        if generation == self.stream_generation and time.monotonic() - stamp < 5:
                            self.scene = objects
                            self.scene_time = stamp
                except Exception as exc:
                    with self.lock:
                        self.scene = []
                        self.scene_time = 0.0
                        if not stop_event.is_set():
                            self.detector_error = str(exc)
                            stop_event.set()

    def _run(self, stop_event):
        pan = tilt = 0.0
        last_command = 0.0
        active_generation = None
        detectors = scene = None
        error = None
        try:
            detectors = self.create_detectors(stop_event)
            scene = threading.Thread(target=self._scene_loop, args=(detectors, stop_event), daemon=True)
            scene.start()
            while not stop_event.is_set():
                latest, ready = [None], threading.Event()
                reader_stop = threading.Event()
                proc = reader = None
                try:
                    proc = self.open_video()
                    reader = threading.Thread(target=self._read_frames, args=(proc, latest, ready, reader_stop), daemon=True)
                    reader.start()
                    sightings = misses = 0
                    motion = Motion()
                    chase_until = 0.0
                    first_frame = True
                    while not stop_event.is_set():
                        if not ready.wait(timeout=5) or (latest[0] is None and not reader.is_alive()):
                            break
                        ready.clear()
                        frame, latest[0] = latest[0], None
                        if frame is None:
                            if not reader.is_alive():
                                break
                            continue

                        data, stamp = frame
                        rgb = np.frombuffer(data, np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)
                        with self.lock:
                            self.latest_rgb, self.frame_time = rgb, stamp
                            target, generation, search = self.target, self.generation, self.search
                            if generation != active_generation:
                                active_generation = generation
                                sightings = misses = 0
                                motion = Motion()
                                chase_until = 0.0
                                self.last_position = None
                                pan = tilt = 0.0
                            # Time spent reconnecting must not use up a search's look.
                            if first_frame and search:
                                search.look_again()
                            first_frame = False
                            if self.paused.is_set() or target is None:
                                self.status = {"state": "gesture" if self.paused.is_set() else "idle", "box": None}
                                continue

                        if search:
                            confirmed = self._advance_search(search, detectors, target, rgb, generation, stop_event)
                            if confirmed:
                                with self.lock:
                                    if self._current_control(generation, stop_event):
                                        # Keep the confirmation, so losing it again starts a chase.
                                        sightings, misses = CONFIRM_FRAMES, 0
                                        motion = Motion()
                                        motion.update(confirmed["x"], confirmed["y"], stamp)
                                        self.last_position = (confirmed["x"], confirmed["y"])
                            continue

                        started = time.perf_counter()
                        found = self.detect(detectors, target, rgb, int(started * 1000))
                        detect_ms = (time.perf_counter() - started) * 1000

                        with self.lock:
                            if not self._current_control(generation, stop_event):
                                continue
                            now = time.perf_counter()
                            new_pan, new_tilt = pan, tilt
                            if found:
                                # A target seen again mid-chase is the one being chased.
                                sightings = CONFIRM_FRAMES if chase_until else sightings + 1
                                chase_until = 0.0
                                misses = 0
                                self.last_position = (found["x"], found["y"])
                                motion.update(found["x"], found["y"], stamp)
                                if sightings >= CONFIRM_FRAMES:
                                    new_pan, new_tilt = choose_velocity(*motion.predicted_error(), pan, tilt)
                            else:
                                misses += 1
                                give_up = False
                                if misses == LOST_FRAMES and sightings >= CONFIRM_FRAMES:
                                    self._remember_last_seen()
                                    new_pan, new_tilt = motion.chase_velocity()
                                    chase_until = now + CHASE_SECONDS
                                if misses >= LOST_FRAMES:
                                    sightings = 0
                                    self.last_position = None
                                    if now >= chase_until:
                                        new_pan, new_tilt = 0.0, 0.0
                                        # People come back to the camera; pets have to be looked for.
                                        if chase_until and target not in ("face", "hand"):
                                            self.search = Search()
                                            if self.on_search_started:
                                                self.on_search_started(target)
                                        elif chase_until:
                                            give_up = True
                                        chase_until = 0.0

                            if (new_pan, new_tilt) != (pan, tilt) or ((pan or tilt) and now - last_command > REFRESH_SECONDS):
                                if new_pan or new_tilt:
                                    self.move(new_pan, new_tilt, MOVE_TIMEOUT)
                                else:
                                    self.stop_camera()
                                pan, tilt, last_command = new_pan, new_tilt, now

                            if not found and give_up:
                                self._return_to_last_seen()

                            self.status = {
                                "state": "chasing" if chase_until else "following" if sightings >= CONFIRM_FRAMES else "searching",
                                "box": found["box"] if found else None,
                                "moving": bool(pan or tilt),
                                "detect_ms": round(detect_ms, 1),
                            }
                except (OSError, self.camera_error) as exc:
                    self.status = {"state": "error", "error": f"Camera connection problem: {exc}. Reconnecting…"}
                finally:
                    reader_stop.set()
                    with self.lock:
                        self._clear_scene()
                    try:
                        if proc:
                            self.close_video(proc)
                    finally:
                        if reader:
                            reader.join()
                    with self.lock:
                        if active_generation == self.generation:
                            try:
                                if pan or tilt or self.search_moving:
                                    self.stop_camera()
                            except (OSError, self.camera_error):
                                pass
                            self.search_moving = False
                            if self.search:
                                self.search.moving = False
                        pan = tilt = 0.0
                if not stop_event.is_set():
                    self.status = {"state": "error", "error": "The video stopped. Reconnecting…"}
                    stop_event.wait(2)
        except Exception as exc:
            error = str(exc)
        finally:
            stop_event.set()
            if scene:
                scene.join()
            if detectors:
                try:
                    detectors.close()
                except Exception as exc:
                    error = error or str(exc)
            with self.lock:
                self._clear_scene()
                error = error or self.detector_error
                if error:
                    if self.search:
                        self._finish_search(self.search, self.target, False, error)
                    self.status = {"state": "error", "error": f"Following stopped: {error}"}
                else:
                    self.status = {"state": "off"}
