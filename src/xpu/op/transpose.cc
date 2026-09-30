/*!
 * \file tl/xpu/op/transpose.cc
 * \brief XPU implementation for tl.transpose lowering.
 */

#include "op/transpose.h"
#include <tvm/ir/cast.h>

#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <dlpack/dlpack.h>
#include <vector>

namespace tvm {
namespace tl {

namespace xpu {

struct Transpose {
  static Stmt Lower(const TransposeNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    bool is_cpu_target = lower_args.target->GetTargetDeviceType() == kDLCPU;
    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

    if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
      return VectorizeLoop(fused_loop, lower_args.layout_map);
    }

    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(
        par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
        lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
        /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard());
  }
};

} // namespace xpu

namespace {

bool MatchXPUTransposeTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "opencl";
}

bool RegisterXPUTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "xpu.Transpose",
      MatchXPUTransposeTarget,
      xpu::Transpose::Lower,
  });
  return true;
}

const bool xpu_transpose_registered = RegisterXPUTranspose();

} // namespace

} // namespace tl
} // namespace tvm
