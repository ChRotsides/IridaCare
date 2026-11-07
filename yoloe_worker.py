"""Runs YOLOE on the Kepler GPU for tracker.py.

Lives in its own process because the Kepler torch needs NumPy 1 and the server's
packages need NumPy 2. Requests and replies are JSON lines on stdin and stdout; a
detect request is followed by the frame's raw RGB bytes.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

MODEL = "yoloe-26l-seg.pt"
# Trained on fixed COCO classes (cat is 15). Holds up better than YOLOE on infrared.
COCO_MODEL = "yolo26l.pt"
DEVICE = os.environ.get("YOLO_DEVICE", "cuda:0")
# Night vision gives a grey picture whose far side is nearly black; this brings it out.
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
IMAGE_SIZE = 640
# Text embeddings take about a second each to compute, so each is only done once.
# They differ between model sizes: another size's embeddings give wrong detections.
EMBEDDINGS = Path(MODEL).with_suffix(".embeddings.pt")

# Kepler cannot use cuDNN, and the Kepler environment's torchvision is CPU-only,
# so this one step of each detection runs on the CPU.
torch.backends.cudnn.enabled = False
cpu_nms = torchvision.ops.nms
torchvision.ops.nms = lambda boxes, scores, threshold: cpu_nms(boxes.cpu(), scores.cpu(), threshold).to(boxes.device)


def embed(model, labels, cache):
    missing = [label for label in labels if label not in cache]
    if missing:
        computed = model.get_text_pe(missing)
        for index, label in enumerate(missing):
            cache[label] = computed[:, index:index + 1].cpu()
        # Two workers can run at once (following and the cat watcher), so each writes its own file.
        partial = EMBEDDINGS.with_suffix(f".{os.getpid()}.tmp")
        torch.save(cache, partial)
        partial.replace(EMBEDDINGS)
    return torch.cat([cache[label] for label in labels], dim=1)


def is_infrared(frame):
    """Night vision frames have identical colour channels."""
    sample = frame[::16, ::16].astype(np.int16)
    return float(np.abs(sample[:, :, 0] - sample[:, :, 1]).mean() + np.abs(sample[:, :, 1] - sample[:, :, 2]).mean()) < 2.0


def enhance(frame):
    grey = CLAHE.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def main():
    # Libraries print to stdout, which would corrupt the replies.
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    requests = sys.stdin.buffer

    def reply(**message):
        replies.write(json.dumps(message) + "\n")

    # Similar labels compete, so a named target gets a model that looks for nothing else.
    # Loaded on first use, so a worker only holds the models it is asked for.
    files = {"scene": MODEL, "target": MODEL, "coco": COCO_MODEL}
    models = {}
    cache = torch.load(EMBEDDINGS) if EMBEDDINGS.exists() else {}
    reply(ready=True)
    while True:
        line = requests.readline()
        if not line:
            return
        message = json.loads(line)
        try:
            if message["model"] not in models:
                models[message["model"]] = YOLO(files[message["model"]])
            model = models[message["model"]]
            if message["cmd"] == "prompts":
                model.set_classes(message["labels"], embed(model, message["labels"], cache))
                # The predictor keeps the old labels otherwise.
                model.predictor = None
                reply(ok=True)
            else:
                width, height = message["width"], message["height"]
                data = requests.read(width * height * 3)
                # YOLO expects BGR.
                frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)[:, :, ::-1].copy()
                infrared = is_infrared(frame)
                if infrared and message.get("enhance"):
                    frame = enhance(frame)
                result = model.predict(
                    frame, device=DEVICE, imgsz=message.get("size") or IMAGE_SIZE, conf=message["score"], classes=message.get("classes"), verbose=False,
                )[0]
                reply(infrared=infrared, objects=[
                    {"label": result.names[int(c)], "score": round(float(s), 2), "box": [round(float(v), 4) for v in b]}
                    for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxyn)
                ])
        except Exception as exc:
            reply(error=f"{type(exc).__name__}: {exc}"[:300])


if __name__ == "__main__":
    main()
