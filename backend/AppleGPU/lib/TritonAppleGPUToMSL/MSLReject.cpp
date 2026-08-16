// MSLReject.cpp - MSL_LOG_REJECT diagnostics for the fast-path classifier
// gates.
//
// Diagnostics only: nothing here participates in codegen. MSL_LOG_REJECT=1
// prints one MSL-REJECT line per gate rejection and, at process teardown, an
// MSL-REJECT-SITE summary keyed on the rejecting op's own identity.
//
// Two attribution traps this exists to avoid. The enclosing tt.FuncOp name is
// an inductor FUSION-GROUP name, so a GEMM template rejecting inside a fused
// reduction reads as "the reduction rejected"; the op's own source location and
// type are what actually identify it. And autotune recompiles one kernel under
// N configs, so N identical lines are one site, not N - hence the config tag on
// every line and the distinct-site count in the summary.

#include "MSLEmitter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::triton::applegpu {

bool mslLogReject() {
  static const bool on = getenv("MSL_LOG_REJECT") != nullptr;
  return on;
}

namespace {

StringRef enclosingFuncName(Operation *op) {
  if (auto f = op->getParentOfType<tt::FuncOp>())
    return f.getName();
  return "<unknown>";
}

std::string typeBrief(Type t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << t;
  return s;
}

// NameLoc/FusedLoc wrappers are peeled so one site reads identically across
// configs, which is what makes the summary's dedup exact.
std::string siteLoc(Operation *op) {
  Location loc = op->getLoc();
  while (true) {
    if (auto n = dyn_cast<NameLoc>(loc)) {
      loc = n.getChildLoc();
      continue;
    }
    if (auto f = dyn_cast<FusedLoc>(loc)) {
      if (!f.getLocations().empty()) {
        loc = f.getLocations().front();
        continue;
      }
    }
    break;
  }
  if (auto fl = dyn_cast<FileLineColLoc>(loc)) {
    llvm::StringRef file = fl.getFilename().getValue();
    size_t slash = file.rfind('/');
    if (slash != llvm::StringRef::npos)
      file = file.drop_front(slash + 1);
    return (file + ":" + llvm::Twine(fl.getLine()) + ":" +
            llvm::Twine(fl.getColumn()))
        .str();
  }
  return "<noloc>";
}

std::string configBrief(Operation *op) {
  auto mod = op->getParentOfType<ModuleOp>();
  if (!mod)
    return "cfg?";
  auto attr = [&](llvm::StringRef n) -> long {
    if (auto i = mod->getAttrOfType<IntegerAttr>(n))
      return i.getInt();
    return -1;
  };
  return ("warps=" + llvm::Twine(attr("ttg.num-warps")) +
          ",ctas=" + llvm::Twine(attr("ttg.num-ctas")) +
          ",tpw=" + llvm::Twine(attr("ttg.threads-per-warp")))
      .str();
}

// For convert_layout the src->dst pair is the only thing that separates a
// genuine cross-thread move from a scalar shuffle.
std::string opBrief(Operation *op) {
  if (auto cl = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    auto s = cast<RankedTensorType>(cl.getSrc().getType());
    auto d = cast<RankedTensorType>(cl.getResult().getType());
    return typeBrief(s) + " -> " + typeBrief(d);
  }
  if (op->getNumResults() == 0)
    return "void";
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::interleave(
      op->getResultTypes(), os, [&](Type t) { os << typeBrief(t); }, ",");
  return s;
}

struct RejectKey {
  std::string gate, reason, func, loc, opName, detail;
  bool operator<(const RejectKey &o) const {
    return std::tie(gate, reason, func, loc, opName, detail) <
           std::tie(o.gate, o.reason, o.func, o.loc, o.opName, o.detail);
  }
};

struct RejectTally {
  std::map<RejectKey, std::set<std::string>> sites;
  ~RejectTally() {
    if (sites.empty())
      return;
    llvm::errs() << "MSL-REJECT-SUMMARY\tdistinct sites: " << sites.size()
                 << '\n';
    for (auto &[k, cfgs] : sites)
      llvm::errs() << "MSL-REJECT-SITE\t" << k.gate << '\t' << k.reason << '\t'
                   << k.func << '\t' << k.loc << '\t' << k.opName << '\t'
                   << k.detail << "\tconfigs=" << cfgs.size() << '\n';
  }
};

RejectTally &tally() {
  static RejectTally t;
  return t;
}
} // namespace

void mslReject(Operation *op, StringRef gate, StringRef reason) {
  if (!mslLogReject())
    return;
  RejectKey k{gate.str(),
              reason.str(),
              enclosingFuncName(op).str(),
              siteLoc(op),
              op->getName().getStringRef().str(),
              opBrief(op)};
  std::string cfg = configBrief(op);
  tally().sites[k].insert(cfg);
  llvm::errs() << "MSL-REJECT\t" << k.gate << '\t' << k.reason << '\t' << k.func
               << '\t' << k.loc << '\t' << k.opName << '\t' << k.detail << '\t'
               << cfg << '\n';
}

} // namespace mlir::triton::applegpu
