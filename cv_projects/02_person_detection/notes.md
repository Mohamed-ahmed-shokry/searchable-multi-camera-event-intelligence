# Notes — Person Detection

Observations, model comparisons, failures, and decisions for Phase 2.

## GPU Verification

| Item | Value |
|------|-------|
| Device | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.44 GB (`torch.cuda.get_device_properties(0).total_memory`) |
| `torch.cuda.is_available()` | `True` |
| torch | 2.11.0+cu128 |
| CUDA runtime (wheel) | 12.8 |
| Driver-reported CUDA (nvidia-smi) | 13.4 |

Matches the locked hardware decision in `docs/project_phases.md`.

## Model Comparison

_(none yet — needs real footage from Phase 1)_

## Decisions

- Used the `cu128` PyTorch CUDA wheel index instead of the newer `cu130`:
  `cu130`'s `torchvision` wheel for `win_amd64`/`cp312` was corrupted on
  PyTorch's CDN (hash mismatch persisted across repeated attempts and a full
  `uv cache clean`). The RTX 3060's driver (CUDA 13.4) is backward compatible
  with `cu128`'s CUDA 12.8 runtime, so this has no practical downside.
- `pyproject.toml` pins `torch`/`torchvision` to the `pytorch-cu128` index via
  `[tool.uv.sources]` so `uv sync` always pulls the CUDA build, not PyPI's
  CPU-only default.

## Failures / Limitations

_(none yet — pending real footage)_
