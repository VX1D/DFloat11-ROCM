"""HIP/ROCm backend for DFloat11 decode kernel via HIPRTC."""
import ctypes
import hashlib
import math
import os
from sys import stderr

import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC_PATH = os.path.join(_PKG_DIR, "decode_hip.cu")
_CACHE_DIR = os.path.join(_PKG_DIR, ".cache")

_hip = None
_hiprtc = None
_module = None
_function = None


def _rocm_search_dirs():
    dirs = []
    try:
        import _rocm_sdk_core
        sdk_bin = os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "bin")
        if os.path.isdir(sdk_bin):
            dirs.append(sdk_bin)
    except ImportError:
        pass
    for var in ("HIP_PATH", "ROCM_PATH"):
        p = os.environ.get(var)
        if p:
            dirs.append(os.path.join(p, "bin") if not p.endswith("bin") else p)
    dirs.append(r"C:\Program Files\AMD\ROCm\6.4\bin")
    dirs.append("")
    return dirs


def _find_dll(names):
    search_dirs = _rocm_search_dirs()
    tried = []
    for name in names:
        for d in search_dirs:
            path = os.path.join(d, name) if d else name
            try:
                return ctypes.CDLL(path)
            except OSError:
                tried.append(path)
    raise RuntimeError(
        f"Could not load any of {names}.\nSearched: {tried}\n"
        "Ensure ROCm is installed (pip install rocm_sdk_core or set ROCM_PATH)."
    )


def _load_runtime():
    global _hip, _hiprtc
    if _hip is not None:
        return
    _hip = _find_dll(["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"])
    _hiprtc = _find_dll(["hiprtc0702.dll", "hiprtc0604.dll", "hiprtc.dll"])


class _HipError(RuntimeError):
    pass


def _check_hip(err, msg="HIP call failed"):
    if err != 0:
        raise _HipError(f"{msg}: hipError_t = {err}")


def _check_rtc(err, msg="HIPRTC call failed"):
    if err != 0:
        try:
            _hiprtc.hiprtcGetErrorString.restype = ctypes.c_char_p
            s = _hiprtc.hiprtcGetErrorString(err)
            msg = f"{msg}: {s.decode()}"
        except Exception:
            msg = f"{msg}: hiprtcResult = {err}"
        raise _HipError(msg)


def _cache_path(arch, src_hash):
    return os.path.join(_CACHE_DIR, f"decode_{arch}_{src_hash}.co")


def _compile_kernel():
    global _module, _function
    _load_runtime()

    with open(_KERNEL_SRC_PATH, "r") as f:
        src = f.read()

    arch = os.environ.get("DFLOAT11_HIP_ARCH")
    if not arch:
        arch = torch.cuda.get_device_properties(0).gcnArchName

    src_hash = hashlib.sha256(src.encode()).hexdigest()[:16]
    cached = _cache_path(arch, src_hash)

    if os.path.exists(cached):
        with open(cached, "rb") as f:
            code_bytes = f.read()
        print(f"DFloat11 HIP backend: loaded cached code object for {arch}", file=stderr)
    else:
        src_bytes = src.encode("utf-8")

        prog = ctypes.c_void_p()
        _check_rtc(
            _hiprtc.hiprtcCreateProgram(
                ctypes.byref(prog), src_bytes, b"decode_hip.cu", 0, None, None,
            ),
            "hiprtcCreateProgram",
        )

        opts = [f"--offload-arch={arch}".encode()]
        c_opts = (ctypes.c_char_p * len(opts))(*opts)
        ret = _hiprtc.hiprtcCompileProgram(prog, len(opts), c_opts)

        if ret != 0:
            log_size = ctypes.c_size_t()
            _hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            _hiprtc.hiprtcGetProgramLog(prog, log_buf)
            _hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
            raise _HipError(
                f"HIPRTC compile failed (hiprtcResult={ret}):\n"
                f"{log_buf.value.decode(errors='replace')}"
            )

        code_size = ctypes.c_size_t()
        _check_rtc(_hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size)), "hiprtcGetCodeSize")
        code_buf = ctypes.create_string_buffer(code_size.value)
        _check_rtc(_hiprtc.hiprtcGetCode(prog, code_buf), "hiprtcGetCode")
        _check_rtc(_hiprtc.hiprtcDestroyProgram(ctypes.byref(prog)), "hiprtcDestroyProgram")

        code_bytes = code_buf.raw
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cached, "wb") as f:
            f.write(code_bytes)

        print(f"DFloat11 HIP backend: compiled for {arch} (cached to {cached})", file=stderr)

    code_buf = ctypes.create_string_buffer(code_bytes)
    _module = ctypes.c_void_p()
    _check_hip(_hip.hipModuleLoadData(ctypes.byref(_module), code_buf), "hipModuleLoadData")

    _function = ctypes.c_void_p()
    _check_hip(
        _hip.hipModuleGetFunction(ctypes.byref(_function), _module, b"decode"),
        "hipModuleGetFunction('decode')",
    )


def _ensure_compiled():
    global _function
    if _function is None:
        _compile_kernel()
    return _function


def launch_decode(
    luts, encoded_exponent, sign_mantissa, output_positions, gaps, output,
    n_luts, n_bytes, n_elements,
    threads_per_block=512, bytes_per_thread=8, shared_mem_size=0,
):
    _ensure_compiled()
    blocks = int(math.ceil(n_bytes / (threads_per_block * bytes_per_thread)))

    # Add LUT size to shared memory (kernel stores LUT in LDS before accumulators/write_buffer)
    lut_bytes = n_luts * 256
    shared_mem_size += lut_bytes

    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    arg_luts = ctypes.c_void_p(luts.data_ptr())
    arg_codes = ctypes.c_void_p(encoded_exponent.data_ptr())
    arg_sm = ctypes.c_void_p(sign_mantissa.data_ptr())
    arg_pos = ctypes.c_void_p(output_positions.data_ptr())
    arg_gaps = ctypes.c_void_p(gaps.data_ptr())
    arg_out = ctypes.c_void_p(output.data_ptr())
    arg_nluts = ctypes.c_int(n_luts)
    arg_nbytes = ctypes.c_int(n_bytes)
    arg_nelem = ctypes.c_int(n_elements)

    args = [arg_luts, arg_codes, arg_sm, arg_pos, arg_gaps, arg_out,
            arg_nluts, arg_nbytes, arg_nelem]
    kernel_params = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    _check_hip(
        _hip.hipModuleLaunchKernel(
            _function,
            blocks, 1, 1,
            threads_per_block, 1, 1,
            shared_mem_size,
            ctypes.c_void_p(stream_ptr),
            kernel_params,
            None,
        ),
        "hipModuleLaunchKernel",
    )
