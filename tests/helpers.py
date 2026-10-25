"""Small, local media fixtures; tests never download footage or model weights."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSPECTION_SCRIPTS = PROJECT_ROOT / "cv_projects" / "01_video_inspection" / "scripts"
DETECTION_SCRIPTS = PROJECT_ROOT / "cv_projects" / "02_person_detection" / "scripts"
TRACKING_SCRIPTS = PROJECT_ROOT / "cv_projects" / "03_person_tracking" / "scripts"
TRAJECTORY_SCRIPTS = (
    PROJECT_ROOT / "cv_projects" / "04_trajectory_extraction" / "scripts"
)
sys.path.insert(0, str(INSPECTION_SCRIPTS))
sys.path.insert(0, str(DETECTION_SCRIPTS))
sys.path.insert(0, str(TRACKING_SCRIPTS))
sys.path.insert(0, str(TRAJECTORY_SCRIPTS))


def make_sequence(root: Path, *, name: str = "sequence", count: int = 5) -> Path:
    """Create a numbered MOT-style sequence with distinguishable frames."""
    source = root / name
    images = source / "img1"
    images.mkdir(parents=True)
    (source / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={name}\nimDir=img1\nframeRate=10\nseqLength={count}\n"
        "imWidth=64\nimHeight=48\nimExt=.jpg\n",
        encoding="utf-8",
    )
    for index in range(count):
        frame = np.full((48, 64, 3), index * 40 % 256, dtype=np.uint8)
        if not cv2.imwrite(str(images / f"{index + 1:06d}.jpg"), frame):
            raise OSError("Could not create test image")
    return source


def make_video(root: Path, *, name: str = "clip.avi", count: int = 5) -> Path:
    """Write a tiny MJPEG clip using the same colors as make_sequence."""
    path = root / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    try:
        if not writer.isOpened():
            raise OSError("MJPEG codec is unavailable")
        for index in range(count):
            writer.write(np.full((48, 64, 3), index * 40 % 256, dtype=np.uint8))
    finally:
        writer.release()
    return path
