"""Remotely extract chosen MOT17 training sequences from motchallenge.net's
MOT17.zip using HTTP Range requests, without downloading the whole archive.

Usage:
    uv run python cv_projects/01_video_inspection/scripts/fetch_mot17_sequence.py <dest_dir> <SEQ> [<SEQ> ...]

    e.g. .../fetch_mot17_sequence.py data/mot17 MOT17-04-FRCNN MOT17-05-FRCNN MOT17-11-FRCNN

The archive's central directory is read over a few ranged requests, then only the
byte ranges of the requested sequences are fetched (with retries), extracted and
CRC-validated by ``zipfile``. Kept here so data acquisition is reproducible; the
archive is never stored on disk in full.
"""

import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://motchallenge.net/data/MOT17.zip"
URL_LEN = 5860214001
CHUNK = 1024 * 1024


class RemoteFile(io.RawIOBase):
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

    def seek(self, offset, whence=0):
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

    def read(self, size=-1):
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
                self._start = self._pos
                end = min(self._pos + CHUNK, self._len) - 1
                req = urllib.request.Request(
                    self._url, headers={"Range": f"bytes={self._pos}-{end}"}
                )
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            self._buf = resp.read()
                        break
                    except Exception:
                        if attempt == 4:
                            raise
                if not self._buf:
                    return out.getvalue()
        return out.getvalue()


def main(dest: Path, seqs: list[str]) -> None:
    remote = RemoteFile(URL, URL_LEN)
    archive = zipfile.ZipFile(remote)
    wanted = [
        info
        for info in archive.infolist()
        if any(info.filename.startswith(f"MOT17/train/{seq}/") for seq in seqs)
    ]
    sizes = {
        seq: sum(
            i.file_size for i in wanted if i.filename.startswith(f"MOT17/train/{seq}/")
        )
        for seq in seqs
    }
    print("to extract:", {s: f"{v / 1e6:.0f} MB" for s, v in sizes.items()}, flush=True)
    count = 0

    def worker(info):
        target = dest / info.filename
        if info.is_dir():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, target.open("wb") as dst:
            while True:
                block = src.read(CHUNK)
                if not block:
                    break
                dst.write(block)

    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(worker, info) for info in wanted]
        for future in futures:
            future.result()
            count += 1
            if count % 400 == 0:
                print(f"extracted {count}/{len(wanted)}", flush=True)
    print(f"done: {count} files -> {dest}")


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    sequences = sys.argv[2:]
    destination.mkdir(parents=True, exist_ok=True)
    main(destination, sequences)
