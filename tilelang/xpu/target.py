"""Intel XPU (Arc) target detection and normalization.

The Phase 1 backend uses TVM's ``opencl`` target kind. Battlemage G21
(Arc Pro B60) hardware facts used as defaults:

- max threads (work-items) per work-group: 1024
- SLM (shared local memory) per Xe-core: 128 KiB
- subgroup (warp) sizes: 16/32 (SIMD16 is the native DPAS width)

Detection is intentionally read-only: it never creates a GPU context, so it
cannot perturb the xe driver.
"""

from __future__ import annotations

import os

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

# Battlemage (Xe2) defaults; conservative enough for Alchemist as well.
_XPU_TARGET_DEFAULTS = {
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 131072,  # 128 KiB SLM per Xe-core
    "thread_warp_size": 32,
    "max_function_args": 128,
    "max_num_threads": 1024,
}


def check_xpu_availability() -> bool:
    """Return True if an Intel GPU with an OpenCL-capable stack is present.

    Purely passive: checks for DRI render nodes and the Intel OpenCL ICD
    registration. Does not initialize any GPU context.
    """
    has_render_node = False
    try:
        has_render_node = any(
            name.startswith("renderD") for name in os.listdir("/dev/dri")
        )
    except OSError:
        pass
    if not has_render_node:
        return False
    # Intel NEO registers an ICD vendor file; presence means the compute
    # stack is installed. (Driver remains untouched either way.)
    for icd_dir in ("/etc/OpenCL/vendors", "/usr/local/etc/OpenCL/vendors"):
        try:
            if any("intel" in name.lower() for name in os.listdir(icd_dir)):
                return True
        except OSError:
            continue
    # Fall back to libOpenCL being loadable; render node alone may suffice
    # for Level-Zero-based stacks.
    return os.path.exists("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1")


def _xpu_target_config() -> dict[str, object]:
    return {"kind": "opencl", "keys": ["opencl", "gpu", "xpu"], **_XPU_TARGET_DEFAULTS}


def _detect_xpu_target() -> Target | str | None:
    if check_xpu_availability():
        return Target(_xpu_target_config())
    return None


def normalize_xpu_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "opencl" and "xpu" in target.keys:
            return target
        if target.kind.name == "opencl":
            return target
        return None
    if isinstance(target, dict):
        if target.get("kind") not in ("xpu", "opencl"):
            return None
        merged = {**_xpu_target_config(), **{k: v for k, v in target.items() if k != "kind"}}
        return Target(merged)
    if not isinstance(target, str):
        return None
    name = target.strip()
    if name not in ("xpu", "opencl"):
        return None
    return Target(_xpu_target_config())


register_target_detector("xpu", _detect_xpu_target, override=True)
register_target_normalizer("xpu", normalize_xpu_target, override=True)
