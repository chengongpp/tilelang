/*!
 * \file tl/xpu/op/finalize_reducer.cc
 * \brief Intel XPU implementation for tl.finalize_reducer lowering.
 *
 * Unlike CUDA/ROCm (which call the tl::AllReduce C++ template emitted into
 * generated source), the OpenCL/SYCL source model cannot rely on C++
 * templates at kernel-compile time. This lowering therefore emits the
 * cross-thread combine as plain TIR: a shared-memory tree reduction with
 * explicit `tvm_storage_sync("shared")` barriers (OpenCL
 * `barrier(CLK_LOCAL_MEM_FENCE)`).
 *
 * Constraints (MVP):
 *  - plan steps must have scale == 1 and thread offset == 0;
 *  - one shared slot per thread, reused across output elements;
 *  - barriers are always executed uniformly (never under divergent `if`),
 *    which is required by OpenCL barrier semantics.
 */

#include "backend/common/op/reduce.h"
#include "op/reducer.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace xpu {
namespace {

/*! \brief Emit one barrier that every work-item in the work-group reaches. */
Stmt XPUWorkgroupBarrier() {
  return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                       {StringImm("shared")}));
}

} // namespace

struct FinalizeReducer {
  static Stmt Lower(const FinalizeReducerOpNode &op,
                    const LowerArgs &lower_args, arith::Analyzer *) {
    auto buffer = lower_args.buffer_remap[op.reducer];
    auto opt_layout = lower_args.layout_map.Get(op.reducer);
    ICHECK(opt_layout);
    ICHECK(opt_layout->as<Fragment>());
    auto layout = opt_layout->as<Fragment>().value();

    Array<PrimExpr> indices_0;
    indices_0.reserve(layout->OutputDim());
    for (int i = 0; i < layout->OutputDim(); ++i) {
      indices_0.push_back(Var("__finred_" + std::to_string(i)));
    }

    // Collective steps (same contract as the shared lowerer).
    std::vector<std::pair<int, int>> steps;
    if (op.explicit_plan) {
      for (size_t i = 0; i + 1 < op.plan_steps.size(); i += 2) {
        steps.emplace_back(static_cast<int>(op.plan_steps[i]->value),
                           static_cast<int>(op.plan_steps[i + 1]->value));
      }
    } else {
      const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
      ICHECK(p_extent);
      int extent = *p_extent;
      ICHECK(extent == 1 ||
             extent == *as_const_int(lower_args.thread_bounds->extent))
          << "Illegal finalize_reducer: extent=" << extent
          << "; T.thread_bounds=" << lower_args.thread_bounds;
      if (extent > 1) {
        steps.emplace_back(extent, 1);
      }
    }

    const auto *p_block = as_const_int(lower_args.thread_bounds->extent);
    ICHECK(p_block) << "xpu finalize_reducer requires constant block size";
    const int block_threads = static_cast<int>(*p_block);
    const PrimExpr tid =
        lower_args.thread_index - lower_args.thread_bounds->min;

    Array<Stmt> step_stmts;
    for (const auto &[reducing_threads, scale] : steps) {
      ICHECK_EQ(scale, 1)
          << "xpu finalize_reducer: scale != 1 not supported yet";
      ICHECK(reducing_threads >= 1 && reducing_threads <= block_threads);

      // One shared slot per thread; reused element-wise because the
      // per-element steps are separated by barriers.
      Buffer ws = decl_buffer({IntImm(DataType::Int(32), block_threads)},
                              buffer->dtype, "xpu_finred_ws", "shared");

      // ws[tid] = buffer[indices_0]  (only replica owners run this, since the
      // enclosing kParallel loops are partitioned by the fragment layout;
      // non-participant slots are never read below).
      Array<Stmt> seq;
      seq.push_back(BufferStore(ws, BufferLoad(buffer, indices_0), {tid}));
      seq.push_back(XPUWorkgroupBarrier());

      // Pairwise tree reduction; works for non power-of-two thread counts.
      // if (tid % (2*off) == 0 && tid + off < R) ws[tid] += ws[tid+off]
      for (int off = 1; off < reducing_threads; off <<= 1) {
        PrimExpr cond = (truncmod(tid, 2 * off) == 0) &&
                        (tid + off < reducing_threads);
        Stmt combine = BufferStore(
            ws,
            ReducerV2Combine(op.op, BufferLoad(ws, {tid}),
                             BufferLoad(ws, {tid + off})),
            {tid});
        seq.push_back(IfThenElse(cond, combine));
        seq.push_back(XPUWorkgroupBarrier());
      }

      // Broadcast: every owner reloads the combined value from slot 0.
      seq.push_back(BufferStore(buffer, BufferLoad(ws, {IntImm(tid->dtype, 0)}),
                                indices_0));

      Stmt body = SeqStmt(seq);
      for (int i = layout->OutputDim() - 1; i >= 0; i--) {
        body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
                   ForKind::kParallel, body);
      }
      step_stmts.push_back(SeqStmt({AllocBuffer(ws), body}));
    }

    // Optional logical seed (same semantics as the shared lowerer).
    if (op.seed.defined()) {
      Stmt body =
          BufferStore(buffer,
                      ReducerV2Combine(op.op, BufferLoad(buffer, indices_0),
                                       op.seed.value()),
                      indices_0);
      for (int i = layout->OutputDim() - 1; i >= 0; i--) {
        body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
                   ForKind::kParallel, body);
      }
      step_stmts.push_back(body);
    }

    if (step_stmts.empty()) {
      return Evaluate(0);
    }
    return step_stmts.size() == 1 ? step_stmts[0] : SeqStmt(step_stmts);
  }
};

} // namespace xpu

namespace {

bool MatchXPUFinalizeReducerTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "opencl";
}

bool RegisterXPUFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "xpu.FinalizeReducer",
      MatchXPUFinalizeReducerTarget,
      xpu::FinalizeReducer::Lower,
  });
  return true;
}

const bool xpu_finalize_reducer_registered = RegisterXPUFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
