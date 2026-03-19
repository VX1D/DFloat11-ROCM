# ROCm

DFloat11 on AMD GPUs. The decode kernel is built at runtime with HIPRTC, so there's no `.so` or `.dll` to ship.

Tested on an RX 7800 XT (gfx1101) under ROCm 7.2 on Windows. CDNA targets should work but I haven't actually run them.

## Requirements

- ROCm 7.2 or newer
- PyTorch built against HIP (`torch.version.hip` non-empty)
- A working HIPRTC

## Windows install

Grab the ROCm SDK wheels Radeon publishes for 7.2:

```powershell
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
```

Then install the matching PyTorch HIP build from pytorch.org and `pip install .` from this repo.

Smoke test:

```python
import torch
print(torch.version.hip)  # should not be None

from dfloat11 import DFloat11Model
m = DFloat11Model.from_pretrained("DFloat11/Qwen3-8B-DF11", device_map="auto")
```

## Arch detection

The backend reads `gcnArchName` off the current device. To force a target (cross-compile, or you trust me less than the runtime):

```
DFLOAT11_HIP_ARCH=gfx1101
```

I only ever set this while debugging.

## Numbers

RX 7800 XT, NLLB-1.3B, eng→pol, beam=4, 99 sentences. Outputs match BF16 exactly.

| batch | bf16 ms/sent | df11 ms/sent | overhead |
|-------|--------------|--------------|----------|
| 1     | 464          | 602          | 30%      |
| 4     | 174          | 214          | 23%      |
| 8     | 123          | 149          | 21%      |

Peak VRAM 4473 → 3811 MB. Decode is a fixed cost per forward pass, so the gap closes once batches get bigger.

## Things that bit me

- `Could not load hiprtc0702.dll`: the dll lives inside `rocm_sdk_core`'s site-packages. If `pip list | grep rocm` shows it installed, the search path is wrong, usually a leftover system ROCm install ahead of the wheel on PATH.
- Kernel returns all zeros: HIPRTC and PyTorch's HIP runtime out of sync. Same root cause as above, mixing 6.x and 7.2. Make sure `_rocm_sdk_core/bin/` resolves first.
- `[WARNING] failed to run offload-arch` on Windows is noise, ignore.

## Profiling

`rocprofv3 --hip-trace python whatever.py` for a kernel trace, `rocprofv3 --pmc "SQ_WAVES,SQ_INSTS_VALU" python whatever.py` for counters. RGP works too if you prefer GUIs.

## Notes on the kernel port

A few CUDA-isms had to go:

- Block-level `__syncthreads()` only. No warp shuffles, because warpSize differs between RDNA (32) and CDNA (64) and I didn't want two code paths.
- LUT lives in LDS. `__constant__` looked tempting but the access pattern is fully divergent (driven by the Huffman stream) and constant memory serialises divergent reads on AMD. LDS with a bank-friendly layout came out ~1.5x over the global-memory version.
- `__ldg()` is stripped during source rewrite. AMD caches global loads in L1/L2 anyway.
