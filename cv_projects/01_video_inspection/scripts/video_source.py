"""Dual-mode video reader: plain video files and MOT17-style frame-sequence directories.

A frame-sequence source is a directory containing a ``seqinfo.ini`` (MOT17 layout:
``imDir``/``imExt`` point at numbered JPEG frames). Anything else is treated as a
video file opened via OpenCV's ``VideoCapture``.
"""

from __future__ import annotations

import configparser
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    source_type: str  # "video_file" | "frame_sequence"
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    codec: str | None


class VideoSource:
    """Common interface over a video file or an image-sequence directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self._is_frame_sequence(self.path):
            self._impl = _FrameSequenceSource(self.path)
        else:
            self._impl = _VideoFileSource(self.path)

    @staticmethod
    def _is_frame_sequence(path: Path) -> bool:
        return path.is_dir() and (path / "seqinfo.ini").exists()

    @property
    def metadata(self) -> VideoMetadata:
        return self._impl.metadata

    def __len__(self) -> int:
        return self._impl.metadata.frame_count

    def read_frame(self, index: int) -> np.ndarray:
        return self._impl.read_frame(index)

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _VideoFileSource:
    def __init__(self, path: Path):
        self.path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = self._decode_fourcc(int(self._cap.get(cv2.CAP_PROP_FOURCC)))

        self.metadata = VideoMetadata(
            source_type="video_file",
            path=str(path),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=(frame_count / fps) if fps else 0.0,
            codec=codec,
        )

    @staticmethod
    def _decode_fourcc(value: int) -> str:
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))

    def read_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.metadata.frame_count:
            raise IndexError(f"Frame {index} out of range (0..{self.metadata.frame_count - 1})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise IOError(f"Failed to read frame {index} from {self.path}")
        return frame

    def close(self) -> None:
        self._cap.release()


class _FrameSequenceSource:
    def __init__(self, path: Path):
        self.path = path
        config = configparser.ConfigParser(interpolation=None)
        try:
            with (path / "seqinfo.ini").open(encoding="utf-8-sig") as handle:
                config.read_file(handle)
            seq = config["Sequence"]
            fps = float(seq["frameRate"])
            frame_count = int(seq["seqLength"])
            width = int(seq["imWidth"])
            height = int(seq["imHeight"])
        except (configparser.Error, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid seqinfo.ini in {path}: {exc}") from exc

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"frameRate must be finite and positive in {path}")
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"seqLength, imWidth and imHeight must be positive in {path}")
        if not math.isfinite(frame_count / fps):
            raise ValueError(f"Sequence duration must be finite in {path}")

        im_dir = (path / seq.get("imDir", "img1")).resolve()
        if not im_dir.is_relative_to(path.resolve()) or not im_dir.is_dir():
            raise ValueError(f"imDir must be an existing directory inside {path}")
        im_ext = seq.get("imExt", ".jpg")
        if not re.fullmatch(r"\.[A-Za-z0-9]+", im_ext):
            raise ValueError(f"imExt must be a literal file extension in {path}")
        frame_paths = [
            entry for entry in im_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() == im_ext.lower()
        ]
        if len(frame_paths) != frame_count:
            raise ValueError(
                f"seqLength declares {frame_count} frames but found {len(frame_paths)} in {im_dir}"
            )
        for frame_path in frame_paths:
            if not frame_path.stem.isascii() or not frame_path.stem.isdecimal():
                raise ValueError(f"Frame names must be numbered from 1: {frame_path}")
            if not frame_path.resolve().is_relative_to(im_dir):
                raise ValueError(f"Frame path must stay inside {im_dir}: {frame_path}")
        self.frame_paths = sorted(frame_paths, key=lambda entry: int(entry.stem))
        if any(int(entry.stem) != index for index, entry in enumerate(self.frame_paths, 1)):
            raise ValueError(f"Frames must be contiguous and numbered from 1 in {im_dir}")

        self.metadata = VideoMetadata(
            source_type="frame_sequence",
            path=str(path),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
            codec=None,
        )

    def read_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.metadata.frame_count:
            raise IndexError(f"Frame {index} out of range (0..{self.metadata.frame_count - 1})")
        frame = cv2.imread(str(self.frame_paths[index]))
        if frame is None:
            raise IOError(f"Failed to read frame file: {self.frame_paths[index]}")
        if frame.shape[:2] != (self.metadata.height, self.metadata.width):
            raise ValueError(
                f"Frame dimensions disagree with seqinfo.ini: {self.frame_paths[index]}"
            )
        return frame

    def close(self) -> None:
        pass
