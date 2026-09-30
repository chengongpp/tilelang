/*!
 * \file tl/xpu/op/copy.cc
 * \brief Intel XPU (OpenCL/SYCL) implementation for tl.copy lowering.
 */

#include "op/copy.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace xpu {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace xpu

namespace {

bool MatchXPUCopyTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "opencl";
}

bool RegisterXPUCopy() {
  RegisterCopyImpl(CopyImpl{
      "xpu.Copy",
      MatchXPUCopyTarget,
      100,
      xpu::Copy::InferLayout,
      xpu::Copy::Lower,
  });
  return true;
}

const bool xpu_copy_registered = RegisterXPUCopy();

} // namespace

} // namespace tl
} // namespace tvm
