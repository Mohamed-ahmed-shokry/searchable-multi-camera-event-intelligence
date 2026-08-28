"""Dual-mode video reader: plain video files and MOT17-style frame-sequence directories.

A frame-sequence source is a directory containing a ``seqinfo.ini`` (MOT17 layout:
``imDir``/``imExt`` point at numbered JPEG frames). Anything else is treated as a
video file opened via OpenCV's ``VideoCapture``.
"""

from __future__ import annotations

import configparser
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
        config = configparser.ConfigParser()
        config.read(path / "seqinfo.ini")
        seq = config["Sequence"]

        im_dir = path / seq.get("imDir", "img1")
        im_ext = seq.get("imExt", ".jpg")
        self.frame_paths = sorted(im_dir.glob(f"*{im_ext}"))

        declared_length = seq.getint("seqLength", fallback=len(self.frame_paths))
        frame_count = min(declared_length, len(self.frame_paths)) or len(self.frame_paths)
        fps = seq.getfloat("frameRate", fallback=0.0)

        self.metadata = VideoMetadata(
            source_type="frame_sequence",
            path=str(path),
            fps=fps,
            width=seq.getint("imWidth", fallback=0),
            height=seq.getint("imHeight", fallback=0),
            frame_count=frame_count,
            duration_seconds=(frame_count / fps) if fps else 0.0,
            codec=None,
        )

    def read_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.metadata.frame_count:
            raise IndexError(f"Frame {index} out of range (0..{self.metadata.frame_count - 1})")
        frame = cv2.imread(str(self.frame_paths[index]))
        if frame is None:
            raise IOError(f"Failed to read frame file: {self.frame_paths[index]}")
        return frame

    def close(self) -> None:
        pass
