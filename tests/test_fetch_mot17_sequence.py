from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests import helpers  # noqa: F401 — adds scripts dir to sys.path

import fetch_mot17_sequence as fm  # isort:skip


def _build_in_memory_mot17_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MOT17/train/MOT17-11-FRCNN/seqinfo.ini", b"[Sequence]\nname=11\n")
        for idx in range(1, 16):
            zf.writestr(
                f"MOT17/train/MOT17-11-FRCNN/img1/{idx:06d}.jpg",
                f"frame-{idx:06d}-payload".encode() * 20,
            )
        zf.writestr("MOT17/train/MOT17-04-FRCNN/seqinfo.ini", b"[Sequence]\nname=04\n")
    return buf.getvalue()


class FetchMot17SequenceTests(unittest.TestCase):
    def test_extract_concurrent_workers_and_crc_resume(self):
        raw_zip = _build_in_memory_mot17_zip()
        calls: list[tuple[int, int]] = []

        def fake_range_fetcher(start: int, end: int) -> bytes:
            calls.append((start, end))
            return raw_zip[start : end + 1]

        with (
            tempfile.TemporaryDirectory() as tmp,
            zipfile.ZipFile(io.BytesIO(raw_zip)) as archive,
        ):
            dest = Path(tmp)
            fm.extract(
                archive,
                dest,
                ["MOT17-11-FRCNN"],
                range_fetcher=fake_range_fetcher,
            )

            seq_dir = dest / "MOT17-11-FRCNN"
            self.assertTrue((seq_dir / "seqinfo.ini").is_file())
            self.assertFalse((dest / "MOT17-04-FRCNN").exists())
            frames = sorted((seq_dir / "img1").glob("*.jpg"))
            self.assertEqual(len(frames), 15)
            self.assertEqual(len(calls), 16)

            # Second run should skip all 16 existing CRC-matching files with 0 range calls
            calls.clear()
            fm.extract(
                archive,
                dest,
                ["MOT17-11-FRCNN"],
                range_fetcher=fake_range_fetcher,
            )
            self.assertEqual(len(calls), 0)

    def test_fetch_entry_payload_rejects_bad_magic(self):
        raw_zip = _build_in_memory_mot17_zip()
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
            info = archive.infolist()[0]
            with self.assertRaises(zipfile.BadZipFile):
                fm._fetch_entry_payload(info, range_fetcher=lambda _s, _e: b"BAD!" * 20)


if __name__ == "__main__":
    unittest.main()
