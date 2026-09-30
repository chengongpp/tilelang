/*!
 * \file tl/xpu/op/gemm.cc
 * \brief Intel XPU implementation for tl.gemm instruction selection.
 *
 * Selects between:
 *  - "xpu.dpas": XMX systolic path via the OpenCL extension builtins
 *    intel_sub_group_{bf16,f16}_matrix_mad_k16 (subgroup size 16, micro-tile
 *    8x16x16), implemented by tilelang/xpu/op/gemm/gemm_dpas.py.
 *  - "xpu.fma": generic scalar-FMA fallback (target-agnostic implementation
 *    shared with the CUDA backend).
 */

#include "op/gemm.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tirx;

namespace xpu {
namespace {

constexpr const char *kXPUFMA = "xpu.fma";
constexpr const char *kXPUDPAS = "xpu.dpas";
constexpr int kSubgroupSize = 16;
constexpr int kDPASMicroM = 8;
constexpr int kDPASMicroN = 16;
constexpr int kDPASMicroK = 16;

bool DTypeIsBF16orFP16(DataType dt) {
  return (dt.is_bfloat16() || dt.is_float16()) && dt.bits() == 16 &&
         dt.lanes() == 1;
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)target;
    // XMX path requires: bf16/fp16 inputs, fp32 accumulator, trans_B layout
    // (B stored as (N, K)), no trans_A, block dims divisible by the DPAS
    // micro-tile, and a thread count divisible by the subgroup size.
    bool dtype_ok = DTypeIsBF16orFP16(op.a_->dtype) &&
                    DTypeIsBF16orFP16(op.b_->dtype) && op.c_->dtype.is_float() &&
                    op.c_->dtype.bits() == 32;
    bool layout_ok = op.transB_;
    int64_t m = op.m_;
    int64_t n = op.n_;
    int64_t k = op.k_;
    bool shape_ok = (m % kDPASMicroM == 0) && (n % kDPASMicroN == 0) &&
                    (k % kDPASMicroK == 0) && (block_size % kSubgroupSize == 0);
    if (dtype_ok && layout_ok && shape_ok) {
      return kXPUDPAS;
    }
    return kXPUFMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)target;
    if (gemm_inst == kXPUDPAS) {
      int num_sg = block_size / kSubgroupSize;
      int m_warp = num_sg, n_warp = 1;
      if (policy.IsFullRow()) {
        // m_warp = num_sg, n_warp = 1 (default above)
      } else if (policy.IsFullCol()) {
        m_warp = 1;
        n_warp = num_sg;
      } else {
        // Pick the legal partition minimizing per-lane register bytes:
        // acc: repM*repN*8 fp32, a: repM*8 i16, b: repN*8 i32.
        int best = -1;
        for (int nw = 1; nw <= num_sg; nw <<= 1) {
          if (num_sg % nw != 0) continue;
          int mw = num_sg / nw;
          if (N % (kDPASMicroN * nw) != 0 || M % (kDPASMicroM * mw) != 0)
            continue;
          int repM = M / (kDPASMicroM * mw);
          int repN = N / (kDPASMicroN * nw);
          int bytes = repM * repN * 8 * 4 + repM * 8 * 2 + repN * 8 * 4;
          if (best < 0 || bytes < best) {
            best = bytes;
            m_warp = mw;
            n_warp = nw;
          }
        }
      }
      policy.m_warp = m_warp;
      policy.n_warp = n_warp;
      return {m_warp, n_warp};
    }
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    // The DPAS path reads shared memory with explicit per-lane indices and
    // does not rely on a pre-existing swizzled layout contract.
    (void)gemm_inst;
    return true;
  }
};

} // namespace xpu

namespace {

bool MatchXPUGemmTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "opencl";
}

bool RegisterXPUGemm() {
  RegisterGemmImpl(GemmImpl{
      "xpu.Gemm",
      MatchXPUGemmTarget,
      xpu::Gemm::SelectInst,
      xpu::Gemm::ComputeWarpPartition,
      xpu::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool xpu_gemm_registered = RegisterXPUGemm();

} // namespace

} // namespace tl
} // namespace tvm
