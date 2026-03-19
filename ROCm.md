# ROCm Support

DFloat11 on AMD GPUs uses HIPRTC to compile the decode kernel at runtime inside PyTorch's process. No pre-compiled binaries needed.

## Requirements

- ROCm 7.2+ (Windows via pip, Linux via system install)
- PyTorch ROCm build (`torch.version.hip` must be set)
- Tested on RX 7800 XT (gfx1101)

## Windows Setup

### 1. Install ROCm SDK + PyTorch

```powershell
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
```

Then install the matching PyTorch ROCm build.

### 2. Install DFloat11

```bash
pip install .
```

### 3. Verify

```python
import torch
print(torch.version.hip)

from dfloat11 import DFloat11Model
model = DFloat11Model.from_pretrained("DFloat11/Qwen3-8B-DF11", device_map="auto")
```

## GPU Architecture

Architecture is auto-detected from PyTorch (`gcnArchName`). Override with:

```bash
set DFLOAT11_HIP_ARCH=gfx90a    # MI200
set DFLOAT11_HIP_ARCH=gfx942    # MI300
```

## Performance (RX 7800 XT, NLLB-1.3B)

99 sentences, eng→pol, beam=4. All outputs bit-identical to BF16.

| Batch | BF16 (ms/sent) | DF11 (ms/sent) | Overhead |
|-------|----------------|----------------|----------|
| 1     | 464            | 602            | +30%     |
| 4     | 174            | 214            | +23%     |
| 8     | 123            | 149            | +21%     |

Peak VRAM: BF16 4473 MB, DF11 3811 MB (-662 MB).

Decompression overhead is constant per forward pass and amortizes at larger batch sizes.

## Troubleshooting

**"Could not load hiprtc0702.dll"** — The HIPRTC DLL ships with `rocm_sdk_core`. Run `pip list | grep rocm` to check it's installed.

**Kernel produces zeros** — Version mismatch between HIPRTC and PyTorch's HIP runtime. If you have ROCm 6.4 installed system-wide alongside ROCm 7.2 pip packages, the wrong DLL may load. The pip package (`_rocm_sdk_core/bin/`) takes priority in the search order.

**"[WARNING] failed to run offload-arch"** — Harmless on Windows.

## Profiling

```bash
# Kernel trace
rocprofv3 --hip-trace python your_script.py

# Hardware counters
rocprofv3 --pmc "SQ_WAVES,SQ_INSTS_VALU" python your_script.py
```

Notes on the decode kernel and AMD architecture:
- The kernel uses block-level sync (`__syncthreads()`) only, no warp primitives — safe regardless of warpSize (32 or 64).
- LUT access is divergent per-thread (Huffman stream dependent). `__constant__` memory would serialize these reads. Current approach (global + L2 cache) is correct.
- `__ldg()` is patched out at compile time. AMD GPUs cache global reads in L1/L2 by default.
