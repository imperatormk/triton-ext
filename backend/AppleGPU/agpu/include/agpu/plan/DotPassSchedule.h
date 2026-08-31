// DotPassSchedule - which phases one pass of a dot runs.
//
// A single-shot dot declares the accumulator fragments, multiplies into them,
// and drains them (store to the pool, barrier, read back). A dot whose
// accumulators outlive the pass (a Fused plan, run once per K iteration) only
// multiplies; the enclosing loop declares and drains them.
#ifndef AGPU_DOT_PASS_SCHEDULE_H
#define AGPU_DOT_PASS_SCHEDULE_H

#include "agpu/plan/DotPlan.h"

namespace agpu {

struct DotPassSchedule {
  // What `drainC` withholds is emitted once by `emitFusedLoop`.
  bool declareAccums = true;
  bool drainC = true;

  // The drain also fences the operand pool. A pass that does not drain leaves
  // its reads unfenced, so iteration N+1's B scatter would race iteration N's
  // MMA.
  bool barrierBeforeStage() const { return !drainC; }

  static DotPassSchedule of(const Plan &p) {
    if (p.accumulatorsOutlivePass())
      return {false, false};
    return {};
  }
};

} // namespace agpu

#endif // AGPU_DOT_PASS_SCHEDULE_H
