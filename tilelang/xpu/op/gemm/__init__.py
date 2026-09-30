"""XPU GEMM op registrations.

- ``xpu.fma``: generic scalar-FMA implementation shared with CUDA
  (target-agnostic), used as the fallback.
- ``xpu.dpas``: XMX systolic path via OpenCL subgroup matrix builtins.
"""

from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.cuda.op.gemm.gemm_fma import GemmFMA

from .gemm_dpas import GEMM_INST_XPU_DPAS, GemmDPAS

GEMM_INST_XPU_FMA = "xpu.fma"


def _match_xpu(target) -> bool:
    return target.kind.name == "opencl"


register_gemm_impl("xpu.fma", GEMM_INST_XPU_FMA, _match_xpu, GemmFMA)
register_gemm_impl("xpu.dpas", GEMM_INST_XPU_DPAS, _match_xpu, GemmDPAS)
