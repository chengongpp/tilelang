"""Device codegen entry points for the Intel XPU backend.

Phase 1 delegates to TVM's OpenCL source codegen, registered as
``target.build.opencl`` when TVM is built with ``USE_OPENCL=ON``.
"""

from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

build_xpu = global_func_device_codegen("target.build.opencl")
build_xpu_without_compile = global_func_device_codegen("target.build.opencl")
