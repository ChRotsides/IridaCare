"""A cat-toy rattle for the camera speaker: beads shaken inside a small plastic ball.

To use the real toy instead, record it and save the file as sounds/rattle.wav
(or .mp3, .m4a, .ogg); it is used in place of the made-up one.
"""
import subprocess
from pathlib import Path

import numpy as np

RATE = 8000  # The camera speaker takes 8 kHz A-law.
SOUNDS = Path(__file__).parent / "sounds"


def synthesize(seed=None):
    """Returns 16-bit samples: a few quick shakes, like a paw batting the toy."""
    rng = np.random.default_rng(seed)
    total = np.zeros(int(RATE * 2.2))
    at = int(RATE * 0.05)
    for burst in range(rng.integers(3, 5)):
        # Each shake throws the beads one way: a dense cluster of clicks.
        length = int(RATE * rng.uniform(0.09, 0.16))
        for _ in range(rng.integers(14, 26)):
            start = at + int(rng.beta(1.5, 2.5) * length)
            click_len = int(RATE * rng.uniform(0.002, 0.006))
            if start + click_len >= len(total):
                continue
            t = np.arange(click_len) / RATE
            # Hard plastic rings high; the speaker tops out just under 4 kHz.
            tone = sum(np.sin(2 * np.pi * rng.uniform(1800, 3900) * t + rng.uniform(0, 6.3)) for _ in range(3))
            click = (tone + rng.normal(0, 1.2, click_len)) * np.exp(-t * rng.uniform(900, 1600))
            total[start:start + click_len] += click * rng.uniform(0.4, 1.0)
        at += length + int(RATE * rng.uniform(0.05, 0.12))
        if burst == 1:
            at += int(RATE * rng.uniform(0.15, 0.3))  # A pause, as if the toy rolled.
    total = total[:at + int(RATE * 0.05)]
    # Thin out the low end so the small speaker spends its power where the rattle is.
    total = np.diff(total, prepend=0.0)
    # Clicks are short and spiky; saturating them makes the rattle louder without clipping.
    total = np.tanh(3.0 * total / max(np.abs(total).max(), 1e-9)) * 0.95
    return (total * 32767).astype(np.int16)


def recorded():
    for path in sorted(SOUNDS.glob("rattle.*")) if SOUNDS.exists() else []:
        if path.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"):
            return path
    return None


def to_alaw(source):
    """Converts a file path, or 16-bit samples at 8 kHz, to the camera's A-law."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if isinstance(source, Path):
        # A recording gets evened out, since phones record quietly.
        command += ["-i", str(source), "-af", "loudnorm=I=-12:TP=-1"]
        data = None
    else:
        command += ["-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0"]
        data = source.tobytes()
    command += ["-ar", str(RATE), "-ac", "1", "-f", "alaw", "pipe:1"]
    return subprocess.run(command, input=data, capture_output=True, check=True, timeout=20).stdout


def rattle_audio():
    """A fresh rattle each time, so it never sounds like the same recording."""
    path = recorded()
    return to_alaw(path) if path else to_alaw(synthesize())
