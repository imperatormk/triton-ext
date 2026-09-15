// Staging a tensor into a threadgroup buffer, and the coordinate check
// every staged tile needs.
#include "AgpuEmitter.h"
#include "AgpuShape.h"

#include "agpu/emit/EmitPoison.h"

namespace mlir::triton::applegpu::bridge {

namespace am = agpu::msl;

am::SmallVec<am::Str, 8> AgpuEmitter::stagedNamesOf(agpu::ValueId v,
                                                    int64_t regs) {
  am::SmallVec<am::Str, 8> names;
  for (int64_t r = 0; r < regs; ++r) {
    const am::Str *n = body_.sym.regAt(v, (std::size_t)r);
    names.push_back(n ? *n : am::Str{});
  }
  return names;
}

agpu::Decision AgpuEmitter::tileCoordsResolvable(RankedTensorType ty,
                                                 std::string_view where) {
  // `rangeOf` indexes `cs.dims`; fewer output dims than rank reads past the
  // end.
  if ((int)coordSourceOf(ty).dims.size() != ty.getRank())
    return declined(where, "the layout has no coordinates");
  for (int64_t r = 0; r < registerCount(ty); ++r)
    if (!registerCoordAt(ty, (int)r))
      return declined(where, "the layout has no coordinates");
  return agpu::Decision::emitted();
}

agpu::Decision
AgpuEmitter::planTileActions(agpu::ValueId v, RankedTensorType ty,
                             const std::vector<agpu::CoordWindow> &windows,
                             const agpu::TileView &dst, unsigned elemBits,
                             am::SmallVec<agpu::StageAction, 8> &actions,
                             std::string_view where) {
  if (const agpu::Decision d = tileCoordsResolvable(ty, where); !d.ok())
    return d;

  const agpu::CoordSource cs = coordSourceOf(ty);
  const int64_t regs = registerCount(ty);

  for (int64_t r = 0; r < regs; ++r) {
    std::vector<agpu::CoordRange> ranges;
    for (int d = 0; d < ty.getRank(); ++d)
      ranges.push_back(cs.rangeOf((int)r, d, ty.getShape()[d]));

    // Not `ranges[d].lo`, which is the reachable set's start and so is shared
    // by every register along a lane-varying dim.
    const std::optional<agpu::TileView::Coord> at = registerCoordAt(ty, (int)r);
    if (!at)
      return declined(where, "the layout has no coordinates");

    agpu::TileView::Coord coord;
    for (std::size_t d = 0; d < windows.size() && d < at->size(); ++d)
      coord.push_back((*at)[d] - windows[d].lo);

    if (const std::optional<agpu::StageAction> a =
            agpu::planStage((int)r, ranges, windows, coord))
      actions.push_back(*a);
  }

  agpu::planStageRuns(actions, cs.dims, dst, elemBits);
  return agpu::Decision::emitted();
}

agpu::Decision
AgpuEmitter::stageWholeTensor(agpu::ValueId v, RankedTensorType ty,
                              const am::Str &buffer, const agpu::TileView &dst,
                              const agpu::ElemType &elem,
                              std::string_view where, std::string_view what) {
  const int64_t regs = registerCount(ty);
  // Registers stage under the names they were declared with, so a register
  // declared wider than the buffer's element would need narrowing first.
  if (const agpu::ElemType *held = elemOf(v); held && *held != elem)
    return declined(where, std::string(what) +
                               " register is not held in the buffer's type");
  am::SmallVec<agpu::StageAction, 8> actions;
  const am::SmallVec<am::Str, 8> names = stagedNamesOf(v, regs);
  if (const agpu::Decision d = planTileActions(v, ty, wholeWindowsOf(ty), dst,
                                               elem.bits, actions, where);
      !d.ok())
    return d;
  int64_t covered = 0;
  for (const agpu::StageAction &a : actions)
    covered += a.width;
  if (covered < regs)
    return declined(where, std::string(what) + " register never lands");
  for (int64_t r = 0; r < regs; ++r)
    if (names[(std::size_t)r].empty())
      return declined(where, std::string(what) + " register has no name");

  agpu::emitStage(agpu_.context(), *cur_, dst, buffer, actions, names,
                  coordSourceOf(ty), elem);
  return agpu::Decision::emitted();
}

} // namespace mlir::triton::applegpu::bridge
