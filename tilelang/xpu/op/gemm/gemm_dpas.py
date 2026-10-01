"""Intel XPU XMX (DPAS) GEMM via OpenCL subgroup matrix builtins.

Emits ``intel_sub_group_{bf16,f16}_matrix_mad_k16`` calls (OpenCL extension
``cl_intel_subgroup_matrix_multiply_accumulate``):

- subgroup size 16 (pinned via ``intel_reqd_sub_group_size`` emitted by
  CodeGenOpenCL when these builtins are present);
- micro-tile M=8, N=16, K=16 (``..._mad_k16``);
- per-lane operands: ``a`` = short8 with lane holding ``A[m][k=lane]``,
  ``b`` = int8 with component ``j`` packing ``B[2j][n=lane]`` (low half) and
  ``B[2j+1][n=lane]`` (high half), ``acc`` = float8 with lane holding
  ``C[m][n=lane]``.

Vectors are staged through small private arrays (scalar stores with constant
indices after unrolling, then one vector load via ``T.ramp``), which the IGC
compiler promotes to registers.

Supported:
- shared path: A row-major (M, K) in SLM, B transposed (N, K) in SLM;
- global-direct path: A (M, K) and B (K, N) straight from global memory via
  ``cl_intel_subgroup_2d_block_io`` reads (a-operand via plain 16b reads,
  b-operand via the VNNI "transform" read), no SLM staging and no barriers.
  Triggers when both operands are global-scope buffers covering full K.
C fp32 fragment. Everything else falls back to the scalar FMA path.
"""

from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.utils.language import is_fragment, is_full_region
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx

GEMM_INST_XPU_DPAS = "xpu.dpas"

SUBGROUP = 16
MICRO_M = 8
MICRO_N = 16  # == SUBGROUP
MICRO_K = 16

_DPAS_FN = {
    "bfloat16": "intel_sub_group_bf16_bf16_matrix_mad_k16",
    "float16": "intel_sub_group_f16_f16_matrix_mad_k16",
}


class GemmDPAS(GemmBase):
    """XMX systolic GEMM using subgroup matrix multiply-accumulate builtins."""

    def _partition(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M, self.N, thread_nums, target, GEMM_INST_XPU_DPAS
        )
        return m_warp, n_warp

    def _c_fragment(self, C_buf, m_warp: int, n_warp: int):
        """Fragment layout matching the per-lane DPAS accumulator mapping."""
        M, N = int(C_buf.shape[-2]), int(C_buf.shape[-1])
        wtM, wtN = M // m_warp, N // n_warp
        repM, repN = wtM // MICRO_M, wtN // MICRO_N

        def forward(i, j):
            warp_m = i // wtM
            warp_n = j // wtN
            wi = (i % wtM) // MICRO_M
            m = i % MICRO_M
            ni = (j % wtN) // MICRO_N
            lane = j % MICRO_N
            thread = (warp_m * n_warp + warp_n) * SUBGROUP + lane
            idx = (wi * repN + ni) * MICRO_M + m
            return thread, idx

        return T.Fragment(list(C_buf.shape), forward_fn=forward)

    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self._partition(target, thread_nums)
        layouts = {}
        if is_fragment(self.C):
            layouts[self.C] = self._c_fragment(self.C, m_warp, n_warp)
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        trans_A, trans_B = self.trans_A, self.trans_B
        M, N, K = self.M, self.N, self.K
        A_region, B_region, C_region = self.ARegion, self.BRegion, self.CRegion
        A_buf, B_buf, C_buf = A_region.buffer, B_region.buffer, C_region.buffer

        assert is_full_region(C_region), "Fragment output C must be a full region"
        a_dtype = self.a_dtype
        accum_dtype = self.accum_dtype
        assert a_dtype == self.b_dtype, "DPAS path requires matching A/B dtypes"
        dpas_fn = _DPAS_FN[str(a_dtype)]

        clear_accum = self.clear_accum
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self._partition(target, thread_nums)
        wtM, wtN = M // m_warp, N // n_warp
        repM, repN = wtM // MICRO_M, wtN // MICRO_N

        # A: (M, K) row-major shared; B: (N, K) transposed shared.
        a0, a1 = A_region.region[-2].min, A_region.region[-1].min
        b0, b1 = B_region.region[-2].min, B_region.region[-1].min
        c0, c1 = C_region.region[-2].min, C_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]
        B_other = [r.min for r in B_region.region[:-2]]
        C_other = [r.min for r in C_region.region[:-2]]

        # ---- global-direct path: 2D block reads straight from DRAM --------
        # A (M, K) row-major + B (K, N) row-major (trans_B=False), full K in
        # one T.gemm call, both global scope: read XMX operands directly with
        # cl_intel_subgroup_2d_block_io (a: plain 16b read; b: VNNI transform
        # read).  No SLM staging, no barriers; ~54 TF vs ~26 TF for the SLM
        # path on B60 at 4096^3.
        def _global_direct_ok() -> bool:
            if trans_A or trans_B:
                return False
            if str(a_dtype) not in ("bfloat16", "float16"):
                return False
            try:
                if int(K) % MICRO_K != 0:
                    return False
            except Exception:
                return False
            scope_a = A_buf.scope() or "global"
            scope_b = B_buf.scope() or "global"
            if not (scope_a.startswith("global") and scope_b.startswith("global")):
                return False
            if len(A_region.region) != 2 or len(B_region.region) != 2:
                return False
            # K dim fully covered by the call (whole-K tile).
            try:
                if int(A_region.region[-1].extent) != int(K):
                    return False
                if int(B_region.region[-2].extent) != int(K):
                    return False
            except Exception:
                return False
            return True

        if _global_direct_ok():
            a_pitch = A_buf.shape[-1] * 2  # bytes per row
            a_h = A_buf.shape[-2]
            b_pitch = B_buf.shape[-1] * 2
            b_h = B_buf.shape[-2]

            @T.prim_func
            def _gemm_dpas_global() -> None:
                tx = thread_var - thread_bounds.min
                warp_id = tx // SUBGROUP
                lane = tx % SUBGROUP
                warp_m = warp_id // n_warp
                warp_n = warp_id % n_warp
                a_all = T.alloc_local((repM * MICRO_M,), T.int16)
                b_all = T.alloc_local((repN * MICRO_K // 2,), T.int32)
                acc_all = T.alloc_local((repM * repN * MICRO_M,), accum_dtype)
                for wi in T.serial(repM):
                    for ni in T.serial(repN):
                        for m in T.unroll(MICRO_M):
                            if clear_accum:
                                acc_all[(wi * repN + ni) * MICRO_M + m] = T.float32(0)
                            else:
                                acc_all[(wi * repN + ni) * MICRO_M + m] = C_buf[
                                    tuple(C_other)
                                    + (c0 + warp_m * wtM + wi * MICRO_M + m, c1 + warp_n * wtN + ni * MICRO_N + lane)
                                ]
                for ki in T.serial(K // MICRO_K):
                    # A operands via plain 16b block reads: lane holds
                    # A[m0..m0+7][k=lane] -- exactly the DPAS a operand.
                    for wi in T.unroll(repM):
                        y = a0 + warp_m * wtM + wi * MICRO_M
                        x = a1 + ki * MICRO_K
                        a_all[T.ramp(wi * MICRO_M, 1, MICRO_M)] = tirx.call_extern(
                            T.int16 + "x8", "tl_xpu_rd_a16",
                            A_buf.data, a_pitch, a_h, a_pitch, x, y,
                        )
                    # B operands via VNNI transform reads on B (K, N): lane
                    # holds pairs (B[2j][n], B[2j+1][n]) -- the b operand.
                    for ni in T.unroll(repN):
                        x = b1 + warp_n * wtN + ni * MICRO_N
                        y = b0 + ki * MICRO_K
                        b_all[T.ramp(ni * (MICRO_K // 2), 1, MICRO_K // 2)] = tirx.call_extern(
                            T.int32 + "x8", "tl_xpu_rd_b16",
                            B_buf.data, b_pitch, b_h, b_pitch, x, y,
                        )
                    for wi in T.unroll(repM):
                        for ni in T.unroll(repN):
                            a8 = a_all[T.ramp(wi * MICRO_M, 1, MICRO_M)]
                            b8 = b_all[T.ramp(ni * (MICRO_K // 2), 1, MICRO_K // 2)]
                            aoff = (wi * repN + ni) * MICRO_M
                            acc8 = acc_all[T.ramp(aoff, 1, MICRO_M)]
                            acc_all[T.ramp(aoff, 1, MICRO_M)] = tirx.call_extern(
                                accum_dtype + "x8", dpas_fn, a8, b8, acc8
                            )
                for wi in T.serial(repM):
                    for ni in T.serial(repN):
                        for m in T.unroll(MICRO_M):
                            C_buf[
                                tuple(C_other)
                                + (c0 + warp_m * wtM + wi * MICRO_M + m, c1 + warp_n * wtN + ni * MICRO_N + lane)
                            ] = acc_all[(wi * repN + ni) * MICRO_M + m]

            return _Simplify(_gemm_dpas_global, inline_let=True)

        # ---- shared (SLM) path -------------------------------------------------
        @T.prim_func
        def _gemm_dpas() -> None:
            tx = thread_var - thread_bounds.min
            warp_id = tx // SUBGROUP
            lane = tx % SUBGROUP
            warp_m = warp_id // n_warp
            warp_n = warp_id % n_warp
            # Register blocking: per-lane A/B operand staging plus the full
            # accumulator tile, all in private memory (promoted to registers
            # by IGC after unrolling).
            a_all = T.alloc_local((repM * MICRO_M,), T.int16)
            b_all = T.alloc_local((repN * MICRO_K // 2,), T.int32)
            acc_all = T.alloc_local((repM * repN * MICRO_M,), accum_dtype)
            # Initialize the accumulator from C (or zero on clear).
            for wi in T.serial(repM):
                for ni in T.serial(repN):
                    for m in T.unroll(MICRO_M):
                        if clear_accum:
                            acc_all[(wi * repN + ni) * MICRO_M + m] = T.float32(0)
                        else:
                            acc_all[(wi * repN + ni) * MICRO_M + m] = C_buf[
                                tuple(C_other)
                                + (c0 + warp_m * wtM + wi * MICRO_M + m, c1 + warp_n * wtN + ni * MICRO_N + lane)
                            ]
            for ki in T.serial(K // MICRO_K):
                # Load A operands: repM short8 vectors.
                if trans_A:
                    # A stored (K, M): lane reads 8 consecutive bf16 from row
                    # ki*16+lane -> one 16B vload, exactly the a operand.
                    for wi in T.unroll(repM):
                        row0 = warp_m * wtM + wi * MICRO_M
                        avec = A_buf[
                            tuple(A_other)
                            + (a0 + ki * MICRO_K + lane, T.ramp(a1 + row0, 1, MICRO_M))
                        ]
                        a_all[T.ramp(wi * MICRO_M, 1, MICRO_M)] = T.reinterpret(
                            T.int16 + "x8", avec
                        )
                else:
                    # A stored (M, K): 8 scalar SLM loads per operand.
                    for wi in T.unroll(repM):
                        row0 = warp_m * wtM + wi * MICRO_M
                        for m in T.unroll(MICRO_M):
                            a_all[wi * MICRO_M + m] = T.reinterpret(
                                T.int16,
                                A_buf[tuple(A_other) + (a0 + row0 + m, a1 + ki * MICRO_K + lane)],
                            )
                # Load B operands: repN int8 vectors. B stored (N, K): the 16
                # consecutive bf16 along k already match the VNNI pair packing.
                # Load via an int32 alias so the codegen emits fast uint4 loads
                # (a 16-wide ushort vload is slow on IGC).
                B32 = tirx.decl_buffer(
                    list(B_buf.shape[:-1]) + [B_buf.shape[-1] // 2],
                    "int32",
                    data=B_buf.data,
                )
                for ni in T.unroll(repN):
                    col0 = warp_n * wtN + ni * MICRO_N
                    b_all[T.ramp(ni * (MICRO_K // 2), 1, MICRO_K // 2)] = B32[
                        tuple(B_other)
                        + (b0 + col0 + lane, T.ramp((b1 + ki * MICRO_K) // 2, 1, MICRO_K // 2))
                    ]
                # repM x repN DPAS calls, accumulating in registers.
                for wi in T.unroll(repM):
                    for ni in T.unroll(repN):
                        a8 = a_all[T.ramp(wi * MICRO_M, 1, MICRO_M)]
                        b8 = b_all[T.ramp(ni * (MICRO_K // 2), 1, MICRO_K // 2)]
                        aoff = (wi * repN + ni) * MICRO_M
                        acc8 = acc_all[T.ramp(aoff, 1, MICRO_M)]
                        acc_all[T.ramp(aoff, 1, MICRO_M)] = tirx.call_extern(
                            accum_dtype + "x8", dpas_fn, a8, b8, acc8
                        )
            # Epilogue: write the accumulator tile back to the C fragment.
            for wi in T.serial(repM):
                for ni in T.serial(repN):
                    for m in T.unroll(MICRO_M):
                        C_buf[
                            tuple(C_other)
                            + (c0 + warp_m * wtM + wi * MICRO_M + m, c1 + warp_n * wtN + ni * MICRO_N + lane)
                        ] = acc_all[(wi * repN + ni) * MICRO_M + m]

        return _Simplify(_gemm_dpas, inline_let=True)
