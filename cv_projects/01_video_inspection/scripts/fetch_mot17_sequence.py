"""Remotely extract chosen MOT17 training sequences from motchallenge.net's
MOT17.zip using HTTP Range requests, without downloading the whole archive.

Usage:
    uv run python cv_projects/01_video_inspection/scripts/fetch_mot17_sequence.py <dest_dir> <SEQ> [<SEQ> ...]

    e.g. .../fetch_mot17_sequence.py data/mot17 MOT17-04-FRCNN MOT17-05-FRCNN MOT17-11-FRCNN

How it works:
- The archive's central directory is read over a few ranged requests.
- For every entry belonging to the requested sequences, one ranged GET fetches
  exactly that entry's compressed bytes; they are decompressed, CRC-verified,
  and written into ``<dest_dir>/`` under the entry's relative path.
- Files that already exist locally with matching sizes are CRC-checked locally
  and skipped when valid, so an interrupted run resumes cheaply.
- Kept here so data acquisition is reproducible; the ~5.9 GB archive itself is
  never stored on disk in full.
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE_URL = "https://motchallenge.net/data/MOT17.zip"
ARCHIVE_LENGTH = 5860214001
CHUNK = 1024 * 1024
WORKERS = 10


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file served with Range support."""

    def __init__(self, url: str, length: int):
        self._url = url
        self._len = length
        self._pos = 0
        self._start = -1
        self._buf = b""

    @property
    def name(self):
        return self._url

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset: int, whence: int = 0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._len + offset
        self._pos = min(max(self._pos, 0), self._len)
        return self._pos

    def tell(self):
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._len - self._pos
        size = min(size, self._len - self._pos)
        out = io.BytesIO()
        while size > 0:
            if self._start <= self._pos < self._start + len(self._buf):
                offset = self._pos - self._start
                take = min(size, len(self._buf) - offset)
                out.write(self._buf[offset : offset + take])
                self._pos += take
                size -= take
            else:
                self._fetch_chunk()
        return out.getvalue()

    def _fetch_chunk(self) -> None:
        self._start = self._pos
        end = min(self._pos + CHUNK, self._len) - 1
        request = urllib.request.Request(
            self._url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    self._buf = response.read()
                return
            except Exception:
                if attempt == 4:
                    raise
        if not self._buf:
            return


def open_archive() -> zipfile.ZipFile:
    remote = RemoteFile(ARCHIVE_URL, ARCHIVE_LENGTH)
    return zipfile.ZipFile(remote)


def _entry_data(info: zipfile.ZipInfo, archive: zipfile.ZipFile) -> int:
    """Byte offset of an entry's compressed payload inside the archive."""
    with archive.open(info):
        pass
    header_offset = info.header_offset
    remote = archive.fp  # type: ignore[attr-defined]
    remote.seek(header_offset)
    local_header = remote.read(30)
    name_length = int.from_bytes(local_header[26:28], "little")
    extra_length = int.from_bytes(local_header[28:30], "little")
    return header_offset + 30 + name_length + extra_length


def _range_bytes(start: int, end: int, attempts: int = 5) -> bytes:
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
    raise OSError("unreachable")


def _decompress(payload: bytes, method: int) -> bytes:
    if method == 0:
        return payload
    return zlib.decompressobj(-15).decompress(payload)


def _local_crc_matches(path: Path, expected_crc: str) -> bool:
    if not path.is_file():
        return False
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            crc = zlib.crc32(block, crc)
    return f"{crc & 0xFFFFFFFF:08x}" == expected_crc.lower()


def extract(archive: zipfile.ZipFile, dest: Path, seqs: list[str]) -> None:
    wanted = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and any(info.filename.startswith(f"MOT17/train/{seq}/") for seq in seqs)
    ]
    total = len(wanted)
    megabytes = sum(info.file_size for info in wanted) / 1e6
    print(
        f"to extract: {total} files, {megabytes:.0f} MB (CRC-checked skip on resume)",
        flush=True,
    )
    done = 0
    started = time.time()

    def worker(info: zipfile.ZipInfo) -> None:
        nonlocal done
        relative = Path(*info.filename.split("/")[2:])  # strip MOT17/train/
        target = dest / relative
        if _local_crc_matches(target, f"{info.CRC:08x}"):
            done += 1
            return
        data_offset = _entry_data(info, archive)
        payload = _range_bytes(data_offset, data_offset + info.compress_size - 1)
        if len(payload) != info.compress_size:
            raise OSError(f"short read for {info.filename}: {len(payload)} bytes")
        content = _decompress(payload, info.compress_type)
        if len(content) != info.file_size:
            raise OSError(f"size mismatch for {info.filename}")
        if info.CRC != zlib.crc32(content) & 0xFFFFFFFF:
            raise OSError(f"CRC mismatch for {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        done += 1
        if done % 200 == 0:
            rate = done / max(time.time() - started, 1e-6)
            print(f"  {done}/{total} ({rate:.1f} files/s)", flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        for result in [pool.submit(worker, info) for info in wanted]:
            result.result()
    print(f"done: {total} entries verified -> {dest}", flush=True)


def main(argv: list[str]) -> None:
    dest = Path(argv[1])
    seqs = argv[2:]
    dest.mkdir(parents=True, exist_ok=True)
    extract(open_archive(), dest, seqs)


if __name__ == "__main__":
    main(sys.argv)
