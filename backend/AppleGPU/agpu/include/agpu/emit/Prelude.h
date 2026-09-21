// Prelude.h - the helper definitions a kernel needs before it can link.
#ifndef AGPU_PRELUDE_H
#define AGPU_PRELUDE_H

#include "agpu/core/EnumBitset.h"
#include "agpu/msl/Context.h"
#include "agpu/msl/Printer.h"
#include "agpu/plan/AssertPlan.h"
#include "agpu/plan/AtomicPlan.h"
#include "agpu/plan/Elementwise.h"
#include "agpu/plan/MathFn.h"
#include "agpu/plan/PrintPlan.h"
#include "agpu/plan/TypeConvert.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace agpu {

// Emission order. A helper must precede any helper that calls it.
enum class Helper {
#define HELPER(Name) Name,
#include "agpu/emit/Helpers.def"

  Count,
};

namespace hn = msl::builtin::helper;

inline const char *helperName(Helper h) {
  switch (h) {
#define HELPER(Name)                                                           \
  case Helper::Name:                                                           \
    return hn::Name;
#include "agpu/emit/Helpers.def"
  case Helper::Count:
    break;
  }
  return "";
}

// One row per fp8 encoding: which helper packs it and which unpacks it.
struct Fp8Helpers {
  FloatKind kind;
  Helper pack;
  Helper unpack;
};

inline constexpr Fp8Helpers kFp8Helpers[] = {
    {FloatKind::E4M3, Helper::Fp8PackE4M3, Helper::Fp8UnpackE4M3},
    {FloatKind::E5M2, Helper::Fp8PackE5M2, Helper::Fp8UnpackE5M2},
    {FloatKind::E4B8, Helper::Fp8PackE4B8, Helper::Fp8UnpackE4B8},
    {FloatKind::E5B16, Helper::Fp8PackE5B16, Helper::Fp8UnpackE5B16},
};

inline bool convertHelper(const ConvertPlan &p, Helper &out) {
  const bool brain = p.to.floatKind == FloatKind::Brain;
  switch (p.kind) {
  case ConvertKind::NarrowRtz:
    out = brain ? Helper::RtzBfloat : Helper::RtzHalf;
    return true;
  case ConvertKind::NarrowRtne:
    out = brain ? Helper::RtneBfloat : Helper::RtneHalf;
    return true;
  case ConvertKind::Fp8Pack:
  case ConvertKind::Fp8Unpack:
    for (const Fp8Helpers &h : kFp8Helpers)
      if (h.kind == p.fp8) {
        out = p.kind == ConvertKind::Fp8Pack ? h.pack : h.unpack;
        return true;
      }
    return false;
  default:
    return false;
  }
}

// The helper that narrows to a 16-bit float and returns its bits, for the
// plans that have one.
inline std::optional<Helper> narrowIntHelperFor(const ConvertPlan &p) {
  if (p.kind != ConvertKind::NarrowRtne || p.to.kind != ElemType::Kind::Float ||
      p.to.bits != 16)
    return std::nullopt;
  return p.to.floatKind == FloatKind::Brain ? Helper::RtneIntBfloat
                                            : Helper::RtneIntHalf;
}

// Which prelude helper a math function lowers to. False for everything Metal
// spells itself.
inline bool mathHelper(MathFn fn, Helper &out) {
  switch (fn) {
  case MathFn::Erf:
    out = Helper::Erf;
    return true;
  case MathFn::Cbrt:
    out = Helper::Cbrt;
    return true;
  default:
    return false;
  }
}

class HelperSet {
public:
  void add(Helper h) { bits_.add(h); }
  bool has(Helper h) const { return bits_.has(h); }
  bool any() const { return bits_.any(); }

  void require(const AtomicPlan &p) {
    switch (p.strategy) {
    case AtomicStrategy::FloatCas:
      add(Helper::AtomicRmwF32);
      return;
    case AtomicStrategy::Packed16:
      add(Helper::AtomicRmwPacked16);
      add(Helper::RtneIntHalf);
      add(Helper::RtneIntBfloat);
      return;
    default:
      return;
    }
  }

  void require(MathFn fn) {
    Helper h;
    if (mathHelper(fn, h))
      add(h);
  }

  void require(MathFn3 fn, ElemType elem) {
    if (fn != MathFn3::Fma || elem.kind != ElemType::Kind::Float ||
        elem.bits != 32)
      return;
    add(Helper::Fma);
    add(Helper::SoftFma);
  }

  void require(const ConvertPlan &p) {
    Helper h;
    if (!convertHelper(p, h))
      return;
    add(h);
    if (const std::optional<Helper> ih = narrowIntHelperFor(p))
      add(*ih);
    if (p.narrows != ConvertKind::None)
      require(p.narrowing());
  }

private:
  EnumBitset<Helper, std::uint64_t> bits_;
};

// The `op` ladder both CAS helpers carry, generated from the enum that
// `AtomicPlan.h` selects with. `cur` names the caller's current value.
inline msl::Expr *emuRmwLadderExpr(msl::Context &c, const char *cur) {
  msl::Expr *const value = c.var(cur);
  msl::Expr *const v = c.var("v");
  msl::Expr *const op = c.var("op");

  const auto isOp = [&](EmuRmw which) {
    return c.binary(msl::BinOp::Eq, op, c.lit(emuRmwCode(which)));
  };
  const auto callOn = [&](const char *fn) { return c.call(fn, {value, v}); };

  return c.ternary(isOp(EmuRmw::Add), c.add(value, v),
                   c.ternary(isOp(EmuRmw::Max), callOn(msl::builtin::math::Max),
                             c.ternary(isOp(EmuRmw::Min),
                                       callOn(msl::builtin::math::Min), v)));
}

inline msl::Function *helperFn(msl::Context &c, const char *name,
                               msl::Type ret) {
  msl::Function *fn = c.function();
  fn->isInline = true;
  fn->name = name;
  fn->returnType = std::move(ret);
  return fn;
}

// Trailing newline included: `printPrelude` joins bodies without one.
inline std::string renderHelper(const msl::Function *fn) {
  std::ostringstream os;
  msl::Printer(os).printStmt(fn);
  std::string out = os.str();
  while (!out.empty() && out.back() == '\n')
    out.pop_back();
  return out;
}

// The fp8 packers share one algorithm and differ only in field widths. `satHi`
// is the saturated magnitude and `nanHi` the all-exponent pattern; e4m3fn
// reserves its top slot for NaN, so the two differ there and coincide for e5m2.
inline msl::Function *fp8Packer(msl::Context &c, const char *name, int mantBits,
                                int bias, int satExp, int subnormalShift,
                                int subnormalFloor, int64_t nanHi,
                                int64_t satHi, int64_t quiet,
                                msl::Expr *extraSatCond, bool biasForm) {
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type u8 = msl::Type::scalar(msl::Scalar::U8);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  msl::Function *fn = helperFn(c, name, u8);
  fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

  msl::Expr *const u = c.var("u");
  msl::Expr *const sgn = c.var("sgn");
  msl::Expr *const mant = c.var("mant");
  msl::Expr *const ex = c.var("ex");
  msl::Expr *const m = c.var("m");
  msl::Expr *const rem = c.var("rem");
  msl::Expr *const sh = c.var("sh");
  msl::Expr *const halfWay = c.var("half_");
  const int tailShift = 23 - mantBits;

  const auto shr = [&](msl::Expr *e, msl::Expr *by) {
    return c.binary(msl::BinOp::Shr, e, by);
  };
  const auto bitAnd = [&](msl::Expr *e, msl::Expr *k) {
    return c.binary(msl::BinOp::And, e, k);
  };
  const auto bitOr = [&](msl::Expr *a, msl::Expr *b) {
    return c.binary(msl::BinOp::Or, a, b);
  };
  const auto roundUp = [&](msl::Expr *target, msl::Expr *mid) {
    return c.ifStmt(
        c.binary(msl::BinOp::LOr, c.binary(msl::BinOp::Gt, rem, mid),
                 c.binary(msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, rem, mid),
                          bitAnd(m, c.lit(1, u32)))),
        {c.assignOp(msl::BinOp::Add, target, c.lit(1))});
  };

  fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
  fn->body.push_back(
      c.declStmt(u32, "sgn", bitAnd(shr(u, c.lit(24)), c.litHex(0x80))));
  fn->body.push_back(c.declStmt(
      i32, "e32", c.cast(i32, bitAnd(shr(u, c.lit(23)), c.litHex(0xff)))));
  fn->body.push_back(c.declStmt(u32, "mant", bitAnd(u, c.litHex(0x7fffff))));
  msl::Expr *nan = bitOr(sgn, c.litHex(nanHi));
  if (biasForm)
    nan = c.ternary(mant, c.litHex(0x80), nan);
  else if (quiet)
    nan = bitOr(nan, c.ternary(mant, c.litHex(quiet), c.lit(0, u32)));
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Eq, c.var("e32"), c.litHex(0xff)),
               {c.returnStmt(c.cast(u8, nan))}));
  fn->body.push_back(c.declStmt(
      i32, "ex",
      c.add(c.binary(msl::BinOp::Sub, c.var("e32"), c.lit(127)), c.lit(bias))));

  msl::Expr *satCond = c.binary(msl::BinOp::Ge, ex, c.lit(satExp));
  if (extraSatCond)
    satCond = c.binary(msl::BinOp::LOr, satCond, extraSatCond);
  fn->body.push_back(c.ifStmt(
      satCond, {c.returnStmt(c.cast(u8, bitOr(sgn, c.litHex(satHi))))}));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Le, ex, c.lit(0)),
      {c.ifStmt(c.binary(msl::BinOp::Lt, ex, c.lit(subnormalFloor)),
                {c.returnStmt(c.cast(u8, biasForm ? c.lit(0, u32) : sgn))}),
       c.declStmt(u32, "fm", bitOr(mant, c.litHex(0x800000))),
       c.declStmt(i32, "sh",
                  c.binary(msl::BinOp::Sub, c.lit(subnormalShift), ex)),
       c.declStmt(u32, "m", shr(c.var("fm"), sh)),
       c.declStmt(u32, "rem",
                  bitAnd(c.var("fm"),
                         c.binary(msl::BinOp::Sub,
                                  c.binary(msl::BinOp::Shl, c.lit(1, u32), sh),
                                  c.lit(1, u32)))),
       c.declStmt(u32, "half_",
                  c.binary(msl::BinOp::Shl, c.lit(1, u32),
                           c.binary(msl::BinOp::Sub, sh, c.lit(1)))),
       roundUp(m, halfWay),
       c.returnStmt(c.cast(u8, biasForm
                                   ? c.ternary(m, bitOr(sgn, m), c.lit(0, u32))
                                   : bitOr(sgn, m)))}));

  fn->body.push_back(c.declStmt(u32, "m", shr(mant, c.lit(tailShift))));
  fn->body.push_back(c.declStmt(
      u32, "rem", bitAnd(mant, c.litHex((int64_t(1) << tailShift) - 1))));
  fn->body.push_back(
      c.declStmt(u32, "bits",
                 bitOr(bitOr(sgn, c.binary(msl::BinOp::Shl, c.cast(u32, ex),
                                           c.lit(mantBits))),
                       m)));
  fn->body.push_back(
      roundUp(c.var("bits"), c.litHex(int64_t(1) << (tailShift - 1))));
  fn->body.push_back(c.returnStmt(c.cast(u8, c.var("bits"))));
  return fn;
}

// The fp8 unpackers mirror the packers. `infArm` is null for the FNUZ
// encodings, which have no infinity and instead map 0x80 to NaN up front.
inline msl::Function *fp8Unpacker(msl::Context &c, const char *name,
                                  int mantBits, int bias, bool fnuz,
                                  msl::Expr *infCond, msl::Expr *infValue) {
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
  msl::Function *fn = helperFn(c, name, f32);
  fn->params.push_back({msl::Type::scalar(msl::Scalar::U8), "b", {}});

  msl::Expr *const b = c.cast(u32, c.var("b"));
  msl::Expr *const sgn = c.var("sgn");
  msl::Expr *const e = c.var("e");
  msl::Expr *const m = c.var("m");
  const int expBits = 7 - mantBits;
  const int64_t mantMask = (int64_t(1) << mantBits) - 1;
  const int mantShift = 23 - mantBits;

  const auto bitOr = [&](msl::Expr *x, msl::Expr *y) {
    return c.binary(msl::BinOp::Or, x, y);
  };
  const auto shl = [&](msl::Expr *x, int64_t by) {
    return c.binary(msl::BinOp::Shl, x, c.lit(by));
  };
  const auto compose = [&](msl::Expr *exp) {
    return c.bitcast(f32, bitOr(bitOr(sgn, shl(exp, 23)), shl(m, mantShift)));
  };

  if (fnuz)
    fn->body.push_back(
        c.ifStmt(c.binary(msl::BinOp::Eq, c.var("b"), c.litHex(0x80)),
                 {c.returnStmt(c.bitcast(f32, c.litHex(0x7fc00000)))}));

  fn->body.push_back(c.declStmt(
      u32, "sgn", shl(c.binary(msl::BinOp::And, b, c.litHex(0x80)), 24)));
  fn->body.push_back(c.declStmt(
      u32, "e",
      c.binary(msl::BinOp::And, c.binary(msl::BinOp::Shr, b, c.lit(mantBits)),
               c.litHex((int64_t(1) << expBits) - 1))));
  fn->body.push_back(
      c.declStmt(u32, "m", c.binary(msl::BinOp::And, b, c.litHex(mantMask))));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Eq, e, c.lit(0, u32)),
      {c.ifStmt(c.binary(msl::BinOp::Eq, m, c.lit(0, u32)),
                {c.returnStmt(c.bitcast(f32, sgn))}),
       c.declStmt(i32, "sh", c.lit(0)),
       c.whileStmt(c.binary(msl::BinOp::Eq,
                            c.binary(msl::BinOp::And, m,
                                     c.litHex(int64_t(1) << mantBits)),
                            c.lit(0, u32)),
                   {c.assignOp(msl::BinOp::Shl, m, c.lit(1)),
                    c.assignOp(msl::BinOp::Add, c.var("sh"), c.lit(1))}),
       c.assignOp(msl::BinOp::And, m, c.litHex(mantMask)),
       c.declStmt(u32, "e32",
                  c.cast(u32, c.binary(msl::BinOp::Sub, c.lit(127 - (bias - 1)),
                                       c.var("sh")))),
       c.returnStmt(compose(c.var("e32")))}));

  if (infCond)
    fn->body.push_back(
        c.ifStmt(infCond, {c.returnStmt(c.bitcast(f32, infValue))}));

  fn->body.push_back(c.declStmt(
      u32, "e32",
      c.binary(msl::BinOp::Sub, c.add(e, c.lit(127, u32)), c.lit(bias, u32))));
  fn->body.push_back(c.returnStmt(compose(c.var("e32"))));
  return fn;
}

// The print and assert buffers share a shape: bump a head counter, drop the
// record if the buffer is full, then store one word per field.
inline msl::Function *recordAppender(msl::Context &c, const char *name,
                                     int64_t headWord, int64_t capacity,
                                     int64_t headerWords, int64_t recordWords,
                                     const std::vector<msl::Str> &fields,
                                     const std::vector<int64_t> &offsets) {
  namespace at = msl::builtin::atomic;
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type atomicPtr =
      msl::Type::named(at::Uint).pointerTo(msl::AddrSpace::Device);
  msl::Function *fn = helperFn(c, name, msl::Type::named("void"));
  fn->params.push_back({atomicPtr, "buf", {}});
  for (const msl::Str &f : fields)
    fn->params.push_back({u32, f, {}});

  msl::Expr *const buf = c.var("buf");
  msl::Expr *const relaxed = c.var(msl::builtin::order::Relaxed);

  fn->body.push_back(
      c.declStmt(u32, "slot",
                 c.call(at::FetchAdd, {c.add(buf, c.lit(headWord)),
                                       c.lit(1, u32), relaxed})));
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Ge, c.var("slot"), c.lit(capacity, u32)),
               {c.returnStmt()}));
  fn->body.push_back(
      c.declStmt(atomicPtr, "rec",
                 c.add(c.add(buf, c.lit(headerWords)),
                       c.mul(c.var("slot"), c.lit(recordWords)))));
  for (std::size_t i = 0; i < fields.size(); ++i)
    fn->body.push_back(
        c.exprStmt(c.call(at::Store, {c.add(c.var("rec"), c.lit(offsets[i])),
                                      c.var(fields[i]), relaxed})));
  return fn;
}

// An f64 argument, narrowed to f32. Metal has no double, so the value never
// exists as a float: it arrives as its two words and the result is built from
// the bit pattern, rounded nearest-even.
inline msl::Function *narrowF64(msl::Context &c) {
  const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  const msl::Type u64 = msl::Type::scalar(msl::Scalar::U64);

  msl::Function *fn = helperFn(c, hn::NarrowF64, f32);
  fn->params.push_back({u32, "hi", {}});
  fn->params.push_back({u32, "lo", {}});

  msl::Expr *const hi = c.var("hi");
  msl::Expr *const E = c.var("E");
  msl::Expr *const q = c.var("q");
  msl::Expr *const sig = c.var("sig");
  const auto u64lit = [&](uint64_t v) {
    return c.cast(u64, c.lit((int64_t)v));
  };
  const auto shlOne = [&](msl::Expr *by) {
    return c.binary(msl::BinOp::Shl, u64lit(1), by);
  };
  const auto signBit = [&] {
    return c.binary(msl::BinOp::And, hi, c.litHex(0x80000000));
  };

  fn->body.push_back(c.declStmt(
      u32, "ex",
      c.binary(msl::BinOp::And, c.binary(msl::BinOp::Shr, hi, c.lit(20)),
               c.litHex(0x7ff))));
  fn->body.push_back(
      c.declStmt(u64, "mant",
                 c.binary(msl::BinOp::Or,
                          c.binary(msl::BinOp::Shl,
                                   c.cast(u64, c.binary(msl::BinOp::And, hi,
                                                        c.litHex(0xfffff))),
                                   c.lit(32)),
                          c.cast(u64, c.var("lo")))));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Eq, c.var("ex"), c.litHex(0x7ff)),
      {c.returnStmt(c.bitcast(
          f32,
          c.binary(msl::BinOp::Or, signBit(),
                   c.ternary(c.binary(msl::BinOp::Ne, c.var("mant"), u64lit(0)),
                             c.litHex(0x7fc00000), c.litHex(0x7f800000)))))}));
  // A subnormal f64 is far below the smallest f32, so it joins zero here.
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(0, u32)),
               {c.returnStmt(c.bitcast(f32, signBit()))}));

  fn->body.push_back(c.declStmt(
      i32, "E",
      c.binary(msl::BinOp::Sub, c.cast(i32, c.var("ex")), c.lit(1023))));
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Gt, E, c.lit(127)),
               {c.returnStmt(c.bitcast(f32, c.binary(msl::BinOp::Or, signBit(),
                                                     c.litHex(0x7f800000))))}));

  fn->body.push_back(c.declStmt(
      u64, "sig", c.binary(msl::BinOp::Or, shlOne(c.lit(52)), c.var("mant"))));
  // 29 = 52 - 23: the mantissa keeps 24 bits, and widens below the normal
  // range where the quantum is fixed at 2^-149.
  fn->body.push_back(c.declStmt(
      i32, "shift",
      c.ternary(c.binary(msl::BinOp::Ge, E, c.lit(-126)), c.lit(29),
                c.binary(msl::BinOp::Add, c.lit(29),
                         c.binary(msl::BinOp::Sub, c.lit(-126), E)))));
  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Gt, c.var("shift"), c.lit(54)),
               {c.returnStmt(c.bitcast(f32, signBit()))}));

  fn->body.push_back(
      c.declStmt(u64, "q", c.binary(msl::BinOp::Shr, sig, c.var("shift"))));
  fn->body.push_back(c.declStmt(
      u64, "rem",
      c.binary(msl::BinOp::And, sig,
               c.binary(msl::BinOp::Sub, shlOne(c.var("shift")), u64lit(1)))));
  fn->body.push_back(c.declStmt(
      u64, "hb", shlOne(c.binary(msl::BinOp::Sub, c.var("shift"), c.lit(1)))));
  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::LOr,
               c.binary(msl::BinOp::Gt, c.var("rem"), c.var("hb")),
               c.binary(msl::BinOp::LAnd,
                        c.binary(msl::BinOp::Eq, c.var("rem"), c.var("hb")),
                        c.binary(msl::BinOp::And, q, u64lit(1)))),
      {c.assign(q, c.binary(msl::BinOp::Add, q, u64lit(1)))}));

  msl::Block normal;
  normal.push_back(c.ifStmt(
      c.binary(msl::BinOp::Ge, q, shlOne(c.lit(24))),
      {c.assign(q, c.binary(msl::BinOp::Shr, q, c.lit(1))),
       c.assign(E, c.binary(msl::BinOp::Add, E, c.lit(1))),
       c.ifStmt(
           c.binary(msl::BinOp::Gt, E, c.lit(127)),
           {c.returnStmt(c.bitcast(f32, c.binary(msl::BinOp::Or, signBit(),
                                                 c.litHex(0x7f800000))))})}));
  normal.push_back(c.returnStmt(c.bitcast(
      f32, c.binary(msl::BinOp::Or,
                    c.binary(msl::BinOp::Or, signBit(),
                             c.binary(msl::BinOp::Shl,
                                      c.cast(u32, c.binary(msl::BinOp::Add, E,
                                                           c.lit(127))),
                                      c.lit(23))),
                    c.binary(msl::BinOp::And, c.cast(u32, q),
                             c.litHex(0x7fffff))))));

  fn->body.push_back(c.ifElse(
      c.binary(msl::BinOp::Eq, c.var("shift"), c.lit(29)), std::move(normal),
      {c.returnStmt(c.bitcast(
          f32, c.binary(msl::BinOp::Or, signBit(), c.cast(u32, q))))}));
  return fn;
}

// Rounds sign * sig * 2^exp to f32, nearest-even. `sig` is nonzero and below
// 2^63; bit 0 of it carries the sticky bit of everything already shifted out.
inline msl::Function *softFmaPacker(msl::Context &c) {
  const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  const msl::Type u64 = msl::Type::scalar(msl::Scalar::U64);

  msl::Function *fn = helperFn(c, hn::SoftFmaPack, f32);
  fn->params.push_back({u32, "sign", {}});
  fn->params.push_back({i32, "exp", {}});
  fn->params.push_back({u64, "sig", {}});

  msl::Expr *const sign = c.var("sign");
  msl::Expr *const exp = c.var("exp");
  msl::Expr *const sig = c.var("sig");
  msl::Expr *const E = c.var("E");
  msl::Expr *const shift = c.var("shift");
  msl::Expr *const q = c.var("q");

  const auto u64lit = [&](uint64_t v) {
    return c.cast(u64, c.lit((int64_t)v));
  };
  const auto one64 = [&] { return u64lit(1); };
  const auto shlOne = [&](msl::Expr *by) {
    return c.binary(msl::BinOp::Shl, one64(), by);
  };
  const auto signBit = [&] {
    return c.binary(msl::BinOp::Shl, sign, c.lit(31));
  };

  fn->body.push_back(c.declStmt(
      u32, "lz", c.cast(u32, c.call(msl::builtin::math::Clz, {sig}))));
  fn->body.push_back(
      c.assign(sig, c.binary(msl::BinOp::Shl, sig, c.var("lz"))));
  fn->body.push_back(
      c.assign(exp, c.binary(msl::BinOp::Sub, exp, c.cast(i32, c.var("lz")))));
  fn->body.push_back(
      c.declStmt(i32, "E", c.binary(msl::BinOp::Add, exp, c.lit(63))));

  // 40 = 63 - 23: the leading bit sits at 63 and the mantissa keeps 24 bits.
  // Below the normal range the quantum is fixed at 2^-149, so the shift widens
  // by however far the exponent falls short.
  fn->body.push_back(c.declStmt(
      i32, "shift",
      c.ternary(c.binary(msl::BinOp::Ge, E, c.lit(-126)), c.lit(40),
                c.binary(msl::BinOp::Add, c.lit(40),
                         c.binary(msl::BinOp::Sub, c.lit(-126), E)))));
  fn->body.push_back(c.ifStmt(c.binary(msl::BinOp::Gt, shift, c.lit(64)),
                              {c.returnStmt(c.bitcast(f32, signBit()))}));

  fn->body.push_back(c.declStmt(u64, "q"));
  fn->body.push_back(c.declStmt(u64, "rem"));
  fn->body.push_back(c.declStmt(u64, "hb"));
  fn->body.push_back(c.ifElse(
      c.binary(msl::BinOp::Eq, shift, c.lit(64)),
      {c.assign(q, u64lit(0)), c.assign(c.var("rem"), sig),
       c.assign(c.var("hb"), shlOne(c.lit(63)))},
      {c.assign(q, c.binary(msl::BinOp::Shr, sig, shift)),
       c.assign(c.var("rem"),
                c.binary(msl::BinOp::And, sig,
                         c.binary(msl::BinOp::Sub, shlOne(shift), one64()))),
       c.assign(c.var("hb"),
                shlOne(c.binary(msl::BinOp::Sub, shift, c.lit(1))))}));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::LOr,
               c.binary(msl::BinOp::Gt, c.var("rem"), c.var("hb")),
               c.binary(msl::BinOp::LAnd,
                        c.binary(msl::BinOp::Eq, c.var("rem"), c.var("hb")),
                        c.binary(msl::BinOp::And, q, one64()))),
      {c.assign(q, c.binary(msl::BinOp::Add, q, one64()))}));

  msl::Block normal;
  normal.push_back(
      c.ifStmt(c.binary(msl::BinOp::Ge, q, shlOne(c.lit(24))),
               {c.assign(q, c.binary(msl::BinOp::Shr, q, c.lit(1))),
                c.assign(E, c.binary(msl::BinOp::Add, E, c.lit(1)))}));
  normal.push_back(
      c.declStmt(i32, "be", c.binary(msl::BinOp::Add, E, c.lit(127))));
  normal.push_back(
      c.ifStmt(c.binary(msl::BinOp::Ge, c.var("be"), c.lit(255)),
               {c.returnStmt(c.bitcast(f32, c.binary(msl::BinOp::Or, signBit(),
                                                     c.litHex(0x7f800000))))}));
  normal.push_back(c.returnStmt(c.bitcast(
      f32, c.binary(msl::BinOp::Or,
                    c.binary(msl::BinOp::Or, signBit(),
                             c.binary(msl::BinOp::Shl, c.cast(u32, c.var("be")),
                                      c.lit(23))),
                    c.binary(msl::BinOp::And, c.cast(u32, q),
                             c.litHex(0x7fffff))))));

  // A subnormal target already has the quantum built in, so the rounded
  // integer is the whole field and a carry lands in the exponent by itself.
  fn->body.push_back(
      c.ifElse(c.binary(msl::BinOp::Eq, shift, c.lit(40)), std::move(normal),
               {c.returnStmt(c.bitcast(f32, c.binary(msl::BinOp::Or, signBit(),
                                                     c.cast(u32, q))))}));
  return fn;
}

// fma with every step in integer arithmetic. The ALU flushes subnormals in
// and out, so a operand or result in that range has to avoid it entirely:
// the 24x24 product is exact in 64 bits, the addend is aligned into the same
// frame, and the single rounding happens on the integer significand.
inline msl::Function *softFma(msl::Context &c) {
  const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
  const msl::Type u64 = msl::Type::scalar(msl::Scalar::U64);
  const msl::Type b = msl::Type::scalar(msl::Scalar::Bool);

  msl::Function *fn = helperFn(c, hn::SoftFma, f32);
  fn->params.push_back({f32, "a", {}});
  fn->params.push_back({f32, "b", {}});
  fn->params.push_back({f32, "c", {}});

  const auto u64lit = [&](uint64_t v) {
    return c.cast(u64, c.lit((int64_t)v));
  };
  const auto bitsOf = [&](const char *v) { return c.bitcast(u32, c.var(v)); };
  const auto magOf = [&](const char *v) {
    return c.binary(msl::BinOp::And, bitsOf(v), c.litHex(0x7fffffff));
  };
  const auto expField = [&](const char *v) {
    return c.binary(msl::BinOp::And,
                    c.binary(msl::BinOp::Shr, bitsOf(v), c.lit(23)),
                    c.litHex(0xff));
  };
  const auto isNan = [&](const char *v) {
    return c.binary(msl::BinOp::Gt, magOf(v), c.litHex(0x7f800000));
  };
  const auto isZero = [&](const char *v) {
    return c.binary(msl::BinOp::Eq, magOf(v), c.lit(0, u32));
  };
  const auto quietNan = [&] { return c.bitcast(f32, c.litHex(0x7fc00000)); };

  // sign, exponent and integer significand of each operand.
  for (const char *v : {"a", "b", "c"}) {
    const std::string s = std::string("s") + v;
    const std::string e = std::string("e") + v;
    const std::string m = std::string("m") + v;
    fn->body.push_back(
        c.declStmt(u32, s, c.binary(msl::BinOp::Shr, bitsOf(v), c.lit(31))));
    fn->body.push_back(c.declStmt(u32, e + "f", expField(v)));
    fn->body.push_back(
        c.declStmt(u32, m + "f",
                   c.binary(msl::BinOp::And, bitsOf(v), c.litHex(0x7fffff))));
    fn->body.push_back(c.declStmt(
        i32, e,
        c.ternary(c.binary(msl::BinOp::Eq, c.var(e + "f"), c.lit(0, u32)),
                  c.lit(-149),
                  c.binary(msl::BinOp::Sub, c.cast(i32, c.var(e + "f")),
                           c.lit(150)))));
    fn->body.push_back(c.declStmt(
        u32, m,
        c.ternary(
            c.binary(msl::BinOp::Eq, c.var(e + "f"), c.lit(0, u32)),
            c.var(m + "f"),
            c.binary(msl::BinOp::Or, c.var(m + "f"), c.litHex(0x800000)))));
  }

  // Inf and NaN never reach the hardware fma: it would read a subnormal
  // operand as zero and turn inf*subnormal into a NaN.
  msl::Block spec;
  spec.push_back(c.ifStmt(
      c.binary(msl::BinOp::LOr,
               c.binary(msl::BinOp::LOr, isNan("a"), isNan("b")), isNan("c")),
      {c.returnStmt(quietNan())}));
  spec.push_back(c.declStmt(
      u32, "ps", c.binary(msl::BinOp::Xor, c.var("sa"), c.var("sb"))));
  spec.push_back(c.declStmt(
      b, "ainf", c.binary(msl::BinOp::Eq, c.var("eaf"), c.litHex(0xff))));
  spec.push_back(c.declStmt(
      b, "binf", c.binary(msl::BinOp::Eq, c.var("ebf"), c.litHex(0xff))));
  spec.push_back(c.declStmt(
      b, "cinf", c.binary(msl::BinOp::Eq, c.var("ecf"), c.litHex(0xff))));
  spec.push_back(
      c.ifStmt(c.binary(msl::BinOp::LOr,
                        c.binary(msl::BinOp::LAnd, c.var("ainf"), isZero("b")),
                        c.binary(msl::BinOp::LAnd, c.var("binf"), isZero("a"))),
               {c.returnStmt(quietNan())}));
  spec.push_back(c.ifStmt(
      c.binary(msl::BinOp::LOr, c.var("ainf"), c.var("binf")),
      {c.ifStmt(c.binary(msl::BinOp::LAnd, c.var("cinf"),
                         c.binary(msl::BinOp::Ne, c.var("sc"), c.var("ps"))),
                {c.returnStmt(quietNan())}),
       c.returnStmt(c.bitcast(
           f32, c.binary(msl::BinOp::Or,
                         c.binary(msl::BinOp::Shl, c.var("ps"), c.lit(31)),
                         c.litHex(0x7f800000))))}));
  spec.push_back(c.returnStmt(c.var("c")));

  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::LOr,
               c.binary(msl::BinOp::LOr,
                        c.binary(msl::BinOp::Eq, c.var("eaf"), c.litHex(0xff)),
                        c.binary(msl::BinOp::Eq, c.var("ebf"), c.litHex(0xff))),
               c.binary(msl::BinOp::Eq, c.var("ecf"), c.litHex(0xff))),
      std::move(spec)));

  fn->body.push_back(c.declStmt(
      u32, "ps", c.binary(msl::BinOp::Xor, c.var("sa"), c.var("sb"))));
  fn->body.push_back(
      c.declStmt(u64, "pm",
                 c.binary(msl::BinOp::Mul, c.cast(u64, c.var("ma")),
                          c.cast(u64, c.var("mb")))));
  fn->body.push_back(c.declStmt(
      i32, "pe", c.binary(msl::BinOp::Add, c.var("ea"), c.var("eb"))));
  fn->body.push_back(c.declStmt(u64, "cm", c.cast(u64, c.var("mc"))));

  fn->body.push_back(
      c.ifStmt(c.binary(msl::BinOp::Eq, c.var("pm"), u64lit(0)),
               {c.ifStmt(c.binary(msl::BinOp::Eq, c.var("cm"), u64lit(0)),
                         {c.returnStmt(c.bitcast(
                             f32, c.binary(msl::BinOp::Shl,
                                           c.binary(msl::BinOp::And,
                                                    c.var("ps"), c.var("sc")),
                                           c.lit(31))))}),
                c.returnStmt(c.var("c"))}));
  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Eq, c.var("cm"), u64lit(0)),
      {c.returnStmt(
          c.call(hn::SoftFmaPack, {c.var("ps"), c.var("pe"), c.var("pm")}))}));

  // Both terms move into one frame. Bit 0 is left free so everything shifted
  // out of it can be OR'd back as a sticky bit for the rounding decision.
  fn->body.push_back(c.declStmt(
      i32, "E0",
      c.binary(msl::BinOp::Sub,
               c.call(msl::builtin::math::Min, {c.var("pe"), c.var("ec")}),
               c.lit(1))));
  fn->body.push_back(c.declStmt(
      i32, "spm", c.binary(msl::BinOp::Sub, c.var("pe"), c.var("E0"))));
  fn->body.push_back(c.declStmt(
      i32, "scm", c.binary(msl::BinOp::Sub, c.var("ec"), c.var("E0"))));
  fn->body.push_back(
      c.declStmt(i32, "over",
                 c.call(msl::builtin::math::Max,
                        {c.binary(msl::BinOp::Sub, c.var("spm"), c.lit(14)),
                         c.binary(msl::BinOp::Sub, c.var("scm"), c.lit(38))})));
  fn->body.push_back(c.ifStmt(
      c.binary(msl::BinOp::Gt, c.var("over"), c.lit(0)),
      {c.assign(c.var("E0"),
                c.binary(msl::BinOp::Add, c.var("E0"), c.var("over"))),
       c.assign(c.var("spm"),
                c.binary(msl::BinOp::Sub, c.var("spm"), c.var("over"))),
       c.assign(c.var("scm"),
                c.binary(msl::BinOp::Sub, c.var("scm"), c.var("over")))}));

  fn->body.push_back(c.declStmt(b, "sticky", c.litBool(false)));
  const auto alignInto = [&](const char *dst, const char *src, const char *by) {
    msl::Block down;
    down.push_back(c.declStmt(i32, std::string(by) + "n",
                              c.unary(msl::UnOp::Neg, c.var(by))));
    msl::Block wide;
    wide.push_back(
        c.assign(c.var("sticky"),
                 c.binary(msl::BinOp::LOr, c.var("sticky"),
                          c.binary(msl::BinOp::Ne, c.var(src), u64lit(0)))));
    wide.push_back(c.assign(c.var(dst), u64lit(0)));
    msl::Block part;
    part.push_back(c.assign(
        c.var("sticky"),
        c.binary(
            msl::BinOp::LOr, c.var("sticky"),
            c.binary(msl::BinOp::Ne,
                     c.binary(msl::BinOp::And, c.var(src),
                              c.binary(msl::BinOp::Sub,
                                       c.binary(msl::BinOp::Shl, u64lit(1),
                                                c.var(std::string(by) + "n")),
                                       u64lit(1))),
                     u64lit(0)))));
    part.push_back(
        c.assign(c.var(dst), c.binary(msl::BinOp::Shr, c.var(src),
                                      c.var(std::string(by) + "n"))));
    down.push_back(c.ifElse(
        c.binary(msl::BinOp::Ge, c.var(std::string(by) + "n"), c.lit(64)),
        std::move(wide), std::move(part)));
    return c.ifElse(c.binary(msl::BinOp::Ge, c.var(by), c.lit(0)),
                    {c.assign(c.var(dst), c.binary(msl::BinOp::Shl, c.var(src),
                                                   c.var(by)))},
                    std::move(down));
  };
  fn->body.push_back(c.declStmt(u64, "P"));
  fn->body.push_back(c.declStmt(u64, "Q"));
  fn->body.push_back(alignInto("P", "pm", "spm"));
  fn->body.push_back(alignInto("Q", "cm", "scm"));

  fn->body.push_back(c.declStmt(u64, "sig"));
  fn->body.push_back(c.declStmt(u32, "sign"));
  const auto carrySticky = [&](const char *v) {
    return c.ifStmt(
        c.var("sticky"),
        {c.assign(c.var(v), c.binary(msl::BinOp::Sub, c.var(v), u64lit(1))),
         c.assign(c.var(v), c.binary(msl::BinOp::Or, c.var(v), u64lit(1)))});
  };
  msl::Block same;
  same.push_back(c.assign(c.var("sig"),
                          c.binary(msl::BinOp::Add, c.var("P"), c.var("Q"))));
  same.push_back(c.assign(c.var("sign"), c.var("ps")));
  same.push_back(
      c.ifStmt(c.var("sticky"),
               {c.assign(c.var("sig"),
                         c.binary(msl::BinOp::Or, c.var("sig"), u64lit(1)))}));
  msl::Block diff;
  diff.push_back(
      c.ifElse(c.binary(msl::BinOp::Ge, c.var("P"), c.var("Q")),
               {c.assign(c.var("sig"),
                         c.binary(msl::BinOp::Sub, c.var("P"), c.var("Q"))),
                c.assign(c.var("sign"), c.var("ps")), carrySticky("sig")},
               {c.assign(c.var("sig"),
                         c.binary(msl::BinOp::Sub, c.var("Q"), c.var("P"))),
                c.assign(c.var("sign"), c.var("sc")), carrySticky("sig")}));
  fn->body.push_back(
      c.ifElse(c.binary(msl::BinOp::Eq, c.var("ps"), c.var("sc")),
               std::move(same), std::move(diff)));

  fn->body.push_back(c.ifStmt(c.binary(msl::BinOp::Eq, c.var("sig"), u64lit(0)),
                              {c.returnStmt(c.bitcast(f32, c.lit(0, u32)))}));
  fn->body.push_back(c.returnStmt(
      c.call(hn::SoftFmaPack, {c.var("sign"), c.var("E0"), c.var("sig")})));
  return fn;
}

// The hardware fma, except where a subnormal could reach it. Checking the
// result alone is not enough: a subnormal operand is read as zero, so a
// vanished product leaves a normal-looking answer that no result test catches.
inline msl::Function *guardedFma(msl::Context &c) {
  const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
  const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
  const msl::Type b = msl::Type::scalar(msl::Scalar::Bool);

  msl::Function *fn = helperFn(c, hn::Fma, f32);
  fn->params.push_back({f32, "a", {}});
  fn->params.push_back({f32, "b", {}});
  fn->params.push_back({f32, "c", {}});

  const auto expOf = [&](const char *v) {
    return c.binary(msl::BinOp::And, c.bitcast(u32, c.var(v)),
                    c.litHex(0x7f800000));
  };
  const auto flushed = [&](const char *v) {
    return c.binary(msl::BinOp::Eq, expOf(v), c.lit(0, u32));
  };

  fn->body.push_back(c.declStmt(
      f32, "r",
      c.call(msl::builtin::math::Fma, {c.var("a"), c.var("b"), c.var("c")})));
  fn->body.push_back(c.declStmt(
      b, "bad",
      c.binary(msl::BinOp::LOr,
               c.binary(msl::BinOp::LOr,
                        c.binary(msl::BinOp::LOr, flushed("a"), flushed("b")),
                        flushed("c")),
               flushed("r"))));
  fn->body.push_back(c.returnStmt(c.ternary(
      c.var("bad"), c.call(hn::SoftFma, {c.var("a"), c.var("b"), c.var("c")}),
      c.var("r"))));
  return fn;
}

inline std::string helperSource(Helper h) {
  switch (h) {
  case Helper::AtomicRmwF32: {
    namespace at = msl::builtin::atomic;
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::AtomicRmwF32, f32);
    fn->params.push_back(
        {msl::Type::named(at::Uint).pointerTo(msl::AddrSpace::Device),
         "p",
         {}});
    fn->params.push_back({f32, "v", {}});
    fn->params.push_back({msl::Type::scalar(msl::Scalar::I32), "op", {}});

    msl::Expr *const relaxed = c.var(msl::builtin::order::Relaxed);
    msl::Expr *const expected = c.var("expected");

    fn->body.push_back(
        c.declStmt(u32, "expected", c.call(at::Load, {c.var("p"), relaxed})));
    fn->body.push_back(c.declStmt(f32, "old"));
    fn->body.push_back(c.whileStmt(
        c.litBool(true),
        {c.assign(c.var("old"), c.bitcast(f32, expected)),
         c.declStmt(f32, "next", emuRmwLadderExpr(c, "old")),
         c.ifStmt(c.call(at::CompareExchangeWeak,
                         {c.var("p"), c.addrOf(expected),
                          c.bitcast(u32, c.var("next")), relaxed, relaxed}),
                  {c.breakStmt()})}));
    fn->body.push_back(c.returnStmt(c.var("old")));
    return renderHelper(fn);
  }

  // Narrows via `__agpu_rtne_int_*`. Under fast-math a `(half)` cast folds
  // away against the next iteration's re-widening `float()`, which
  // accumulates at f32 precision into a 16-bit slot.
  case Helper::AtomicRmwPacked16: {
    namespace at = msl::builtin::atomic;
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    const msl::Type tTy = msl::Type::named("T");

    msl::Function *decl = helperFn(c, hn::Narrow16, u16);
    decl->templateParams.push_back("T");
    decl->isPrototype = true;
    decl->params.push_back({f32, "v", {}});

    const auto narrowFor = [&](const char *elem, const char *rtne) {
      msl::Function *fn = helperFn(c, hn::Narrow16, u16);
      fn->isSpecialization = true;
      fn->templateArgs.push_back(elem);
      fn->params.push_back({f32, "v", {}});
      fn->body.push_back(c.returnStmt(c.call(rtne, {c.var("v")})));
      return fn;
    };

    msl::Function *fn = helperFn(c, hn::AtomicRmwPacked16, tTy);
    fn->templateParams.push_back("T");
    fn->params.push_back(
        {msl::Type::named(at::Uint).pointerTo(msl::AddrSpace::Device),
         "word",
         {}});
    fn->params.push_back({msl::Type::scalar(msl::Scalar::Bool), "high", {}});
    fn->params.push_back({f32, "v", {}});
    fn->params.push_back({msl::Type::scalar(msl::Scalar::I32), "op", {}});

    msl::Expr *const expected = c.var("expected");
    msl::Expr *const high = c.var("high");
    msl::Expr *const nb = c.var("nb");
    msl::Expr *const relaxed = c.var(msl::builtin::order::Relaxed);
    const auto shr = [&](msl::Expr *e, int64_t by) {
      return c.binary(msl::BinOp::Shr, e, c.lit(by));
    };
    const auto bitAnd = [&](msl::Expr *e, int64_t m) {
      return c.binary(msl::BinOp::And, e, c.litHex(m));
    };

    fn->body.push_back(c.declStmt(u32, "expected",
                                  c.call(at::Load, {c.var("word"), relaxed})));
    fn->body.push_back(c.declStmt(tTy, "old"));
    fn->body.push_back(c.whileStmt(
        c.litBool(true),
        {c.declStmt(u16, "bits",
                    c.ternary(high, c.construct(u16, shr(expected, 16)),
                              c.construct(u16, bitAnd(expected, 0xffff)))),
         c.assign(c.var("old"), c.bitcast(tTy, c.var("bits"))),
         c.declStmt(f32, "cur", c.construct(f32, c.var("old"))),
         c.declStmt(f32, "next", emuRmwLadderExpr(c, "cur")),
         c.declStmt(u16, "nb", c.call(hn::Narrow16, {"T"}, {c.var("next")})),
         c.declStmt(
             u32, "merged",
             c.ternary(high,
                       c.binary(msl::BinOp::Or, bitAnd(expected, 0x0000ffff),
                                c.binary(msl::BinOp::Shl, c.construct(u32, nb),
                                         c.lit(16))),
                       c.binary(msl::BinOp::Or, bitAnd(expected, 0xffff0000),
                                c.construct(u32, nb)))),
         c.ifStmt(c.call(at::CompareExchangeWeak,
                         {c.var("word"), c.addrOf(expected), c.var("merged"),
                          relaxed, relaxed}),
                  {c.breakStmt()})}));
    fn->body.push_back(c.returnStmt(c.var("old")));

    return renderHelper(decl) + "\n" +
           renderHelper(narrowFor("half", hn::RtneIntHalf)) + "\n" +
           renderHelper(narrowFor("bfloat", hn::RtneIntBfloat)) + "\n\n" +
           renderHelper(fn);
  }

  // Metal has no cbrt. `pow(|x|, 1/3)` needs the sign taken out and restored,
  // plus one Newton step for the last few ulp. The zero guard avoids 0/0 in
  // that step and preserves the sign of -0.0.
  case Helper::Cbrt: {
    msl::Context c;
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::Cbrt, f32);
    fn->params.push_back({f32, "x", {}});

    msl::Expr *const x = c.var("x");
    msl::Expr *const a = c.var("a");
    msl::Expr *const y = c.var("y");
    msl::Expr *const third =
        c.binary(msl::BinOp::Div, c.litF(1.0), c.litF(3.0));

    fn->body.push_back(
        c.declStmt(f32, "a", c.call(msl::builtin::math::Abs, {x})));
    fn->body.push_back(
        c.ifStmt(c.binary(msl::BinOp::Eq, a, c.litF(0.0)), {c.returnStmt(x)}));
    fn->body.push_back(
        c.declStmt(f32, "y",
                   c.call(spell(msl::builtin::accuracy::Pow, Accuracy::Exact),
                          {a, third})));
    fn->body.push_back(c.assign(
        y, c.binary(msl::BinOp::Sub, y,
                    c.mul(c.binary(msl::BinOp::Sub, y,
                                   c.binary(msl::BinOp::Div, a, c.mul(y, y))),
                          third))));
    fn->body.push_back(
        c.returnStmt(c.ternary(c.binary(msl::BinOp::Lt, x, c.litF(0.0)),
                               c.unary(msl::UnOp::Neg, y), y)));
    return renderHelper(fn);
  }

  // Metal has no erf. Abramowitz-Stegun 7.1.26.
  case Helper::Erf: {
    msl::Context c;
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::Erf, f32);
    fn->params.push_back({f32, "x", {}});

    msl::Expr *const x = c.var("x");
    msl::Expr *const a = c.var("a");
    msl::Expr *const t = c.var("t");

    // Horner, innermost coefficient first.
    msl::Expr *poly = c.litF(1.061405429);
    for (const double k :
         {-1.453152027, 1.421413741, -0.284496736, 0.254829592}) {
      poly = c.mul(poly, t);
      poly = c.binary(k < 0 ? msl::BinOp::Sub : msl::BinOp::Add, poly,
                      c.litF(k < 0 ? -k : k));
    }

    fn->body.push_back(
        c.declStmt(f32, "s", c.call(msl::builtin::math::Sign, {x})));
    fn->body.push_back(
        c.declStmt(f32, "a", c.call(msl::builtin::math::Abs, {x})));
    fn->body.push_back(
        c.declStmt(f32, "t",
                   c.binary(msl::BinOp::Div, c.litF(1.0),
                            c.add(c.litF(1.0), c.mul(c.litF(0.3275911), a)))));
    fn->body.push_back(c.declStmt(
        f32, "y",
        c.binary(
            msl::BinOp::Sub, c.litF(1.0),
            c.mul(c.mul(poly, t),
                  c.call(spell(msl::builtin::accuracy::Exp, Accuracy::Tolerant),
                         {c.mul(c.unary(msl::UnOp::Neg, a), a)})))));
    fn->body.push_back(c.returnStmt(c.mul(c.var("s"), c.var("y"))));
    return renderHelper(fn);
  }

  // Round-to-nearest-even, spelled out because MSL's `(half)` cast does not
  // guarantee NaN/Inf/subnormal handling under fast-math. Returns bits rather
  // than `half` so the CAS loops' result is not a fold candidate.
  case Helper::RtneIntHalf: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
    msl::Function *fn = helperFn(c, hn::RtneIntHalf, u16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    msl::Expr *const u = c.var("u");
    msl::Expr *const sgn = c.var("sgn");
    msl::Expr *const mant = c.var("mant");
    msl::Expr *const ex = c.var("ex");
    msl::Expr *const m = c.var("m");
    msl::Expr *const rem = c.var("rem");
    const auto shr = [&](msl::Expr *e, msl::Expr *by) {
      return c.binary(msl::BinOp::Shr, e, by);
    };
    const auto bitAnd = [&](msl::Expr *e, msl::Expr *k) {
      return c.binary(msl::BinOp::And, e, k);
    };
    const auto bitOr = [&](msl::Expr *a, msl::Expr *b) {
      return c.binary(msl::BinOp::Or, a, b);
    };
    const auto odd = [&] { return bitAnd(m, c.lit(1, u32)); };

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(
        c.declStmt(u32, "sgn", bitAnd(shr(u, c.lit(16)), c.litHex(0x8000))));
    fn->body.push_back(c.declStmt(
        i32, "e32", c.cast(i32, bitAnd(shr(u, c.lit(23)), c.litHex(0xff)))));
    fn->body.push_back(c.declStmt(u32, "mant", bitAnd(u, c.litHex(0x7fffff))));
    fn->body.push_back(c.ifStmt(
        c.binary(msl::BinOp::Eq, c.var("e32"), c.litHex(0xff)),
        {c.returnStmt(c.cast(
            u16, bitOr(bitOr(sgn, c.litHex(0x7c00)),
                       c.ternary(mant, c.litHex(0x200), c.lit(0, u32)))))}));
    fn->body.push_back(c.declStmt(
        i32, "ex",
        c.add(c.binary(msl::BinOp::Sub, c.var("e32"), c.lit(127)), c.lit(15))));
    fn->body.push_back(
        c.ifStmt(c.binary(msl::BinOp::Ge, ex, c.lit(31)),
                 {c.returnStmt(c.cast(u16, bitOr(sgn, c.litHex(0x7c00))))}));

    msl::Expr *const sh = c.var("sh");
    msl::Expr *const fm = c.var("fm");
    msl::Expr *const halfWay = c.var("half_");
    fn->body.push_back(c.ifStmt(
        c.binary(msl::BinOp::Le, ex, c.lit(0)),
        {c.ifStmt(c.binary(msl::BinOp::Lt, ex, c.lit(-10)),
                  {c.returnStmt(c.cast(u16, sgn))}),
         c.declStmt(u32, "fm", bitOr(mant, c.litHex(0x800000))),
         c.declStmt(i32, "sh", c.binary(msl::BinOp::Sub, c.lit(14), ex)),
         c.declStmt(u32, "m", shr(fm, sh)),
         c.declStmt(
             u32, "rem",
             bitAnd(fm, c.binary(msl::BinOp::Sub,
                                 c.binary(msl::BinOp::Shl, c.lit(1, u32), sh),
                                 c.lit(1, u32)))),
         c.declStmt(u32, "half_",
                    c.binary(msl::BinOp::Shl, c.lit(1, u32),
                             c.binary(msl::BinOp::Sub, sh, c.lit(1)))),
         c.ifStmt(
             c.binary(msl::BinOp::LOr, c.binary(msl::BinOp::Gt, rem, halfWay),
                      c.binary(msl::BinOp::LAnd,
                               c.binary(msl::BinOp::Eq, rem, halfWay), odd())),
             {c.assignOp(msl::BinOp::Add, m, c.lit(1))}),
         c.returnStmt(c.cast(u16, bitOr(sgn, m)))}));

    fn->body.push_back(c.declStmt(u32, "m", shr(mant, c.lit(13))));
    fn->body.push_back(c.declStmt(u32, "rem", bitAnd(mant, c.litHex(0x1fff))));
    fn->body.push_back(c.declStmt(
        u32, "bits",
        bitOr(bitOr(sgn, c.binary(msl::BinOp::Shl, c.cast(u32, ex), c.lit(10))),
              m)));
    fn->body.push_back(c.ifStmt(
        c.binary(
            msl::BinOp::LOr, c.binary(msl::BinOp::Gt, rem, c.litHex(0x1000)),
            c.binary(msl::BinOp::LAnd,
                     c.binary(msl::BinOp::Eq, rem, c.litHex(0x1000)), odd())),
        {c.assignOp(msl::BinOp::Add, c.var("bits"), c.lit(1))}));
    fn->body.push_back(c.returnStmt(c.cast(u16, c.var("bits"))));
    return renderHelper(fn);
  }

  case Helper::RtneHalf: {
    msl::Context c;
    msl::Function *fn =
        helperFn(c, hn::RtneHalf, msl::Type::scalar(msl::Scalar::F16));
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});
    fn->body.push_back(
        c.returnStmt(c.bitcast(msl::Type::scalar(msl::Scalar::F16),
                               c.call(hn::RtneIntHalf, {c.var("v")}))));
    return renderHelper(fn);
  }

  // bfloat is f32's top 16 bits. The NaN arm is separate because the round
  // increment can carry a quiet NaN's payload into an infinity.
  case Helper::RtneIntBfloat: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    msl::Function *fn = helperFn(c, hn::RtneIntBfloat, u16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    msl::Expr *const u = c.var("u");
    const auto shr = [&](msl::Expr *e, int64_t by) {
      return c.binary(msl::BinOp::Shr, e, c.lit(by));
    };
    const auto andHex = [&](msl::Expr *e, int64_t m) {
      return c.binary(msl::BinOp::And, e, c.litHex(m));
    };

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(c.declStmt(
        u32, "lsb", c.binary(msl::BinOp::And, shr(u, 16), c.lit(1, u32))));
    fn->body.push_back(c.declStmt(
        u32, "sum", c.add(c.add(u, c.litHex(0x7fff)), c.var("lsb"))));
    fn->body.push_back(c.declStmt(u32, "rounded", shr(c.var("sum"), 16)));
    fn->body.push_back(
        c.declStmt(u32, "special",
                   c.binary(msl::BinOp::Or, andHex(shr(u, 16), 0xff80),
                            c.ternary(andHex(u, 0x7fffff), c.litHex(0x40),
                                      c.lit(0, u32)))));
    fn->body.push_back(c.returnStmt(
        c.cast(u16, c.ternary(c.binary(msl::BinOp::Eq, andHex(shr(u, 23), 0xff),
                                       c.litHex(0xff)),
                              c.var("special"), c.var("rounded")))));
    return renderHelper(fn);
  }

  case Helper::RtneBfloat: {
    msl::Context c;
    msl::Function *fn =
        helperFn(c, hn::RtneBfloat, msl::Type::scalar(msl::Scalar::BF16));
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});
    fn->body.push_back(
        c.returnStmt(c.bitcast(msl::Type::scalar(msl::Scalar::BF16),
                               c.call(hn::RtneIntBfloat, {c.var("v")}))));
    return renderHelper(fn);
  }

  // Round-toward-zero. MSL's cast is always nearest-even, so this truncates
  // the mantissa directly. Overflow saturates to 0x7bff (largest finite
  // half): RTZ cannot turn a finite input into inf (0x7c00).
  case Helper::RtzHalf: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type u16 = msl::Type::scalar(msl::Scalar::U16);
    const msl::Type i32 = msl::Type::scalar(msl::Scalar::I32);
    const msl::Type f16 = msl::Type::scalar(msl::Scalar::F16);
    msl::Function *fn = helperFn(c, hn::RtzHalf, f16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    msl::Expr *const u = c.var("u");
    msl::Expr *const sgn = c.var("sgn");
    msl::Expr *const ex = c.var("ex");
    msl::Expr *const mant = c.var("mant");
    msl::Expr *const bits = c.var("bits");
    const auto shr = [&](msl::Expr *e, int64_t by) {
      return c.binary(msl::BinOp::Shr, e, c.lit(by));
    };
    const auto orAll = [&](std::vector<msl::Expr *> parts) {
      return c.chain(msl::BinOp::Or, parts);
    };
    const auto setBits = [&](msl::Expr *e) {
      return c.assign(bits, c.cast(u16, e));
    };
    msl::Expr *const exponent =
        c.binary(msl::BinOp::And, shr(u, 23), c.litHex(0xff));

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(c.declStmt(u16, "bits"));
    fn->body.push_back(c.declStmt(
        u32, "sgn", c.binary(msl::BinOp::And, shr(u, 16), c.litHex(0x8000))));
    fn->body.push_back(c.declStmt(
        i32, "ex",
        c.binary(msl::BinOp::Sub, c.cast(i32, exponent), c.lit(112))));
    fn->body.push_back(c.declStmt(
        u32, "mant", c.binary(msl::BinOp::And, u, c.litHex(0x7fffff))));

    msl::Block subnormal{c.ifElse(
        c.binary(msl::BinOp::Lt, ex, c.lit(-10)), {setBits(sgn)},
        {c.declStmt(u32, "m",
                    c.binary(msl::BinOp::Shr,
                             c.binary(msl::BinOp::Or, mant, c.litHex(0x800000)),
                             c.binary(msl::BinOp::Sub, c.lit(14), ex))),
         setBits(c.binary(msl::BinOp::Or, sgn, c.var("m")))})};

    fn->body.push_back(c.ifElse(
        c.binary(msl::BinOp::Eq, exponent, c.litHex(0xff)),
        {setBits(orAll({sgn, c.litHex(0x7c00),
                        c.ternary(mant, c.litHex(0x200), c.lit(0, u32))}))},
        {c.ifElse(
            c.binary(msl::BinOp::Ge, ex, c.lit(31)),
            {setBits(c.binary(msl::BinOp::Or, sgn, c.litHex(0x7bff)))},
            {c.ifElse(
                c.binary(msl::BinOp::Le, ex, c.lit(0)), subnormal,
                {setBits(orAll(
                    {sgn, c.binary(msl::BinOp::Shl, c.cast(u32, ex), c.lit(10)),
                     shr(mant, 13)}))})})}));
    fn->body.push_back(c.returnStmt(c.bitcast(f16, bits)));
    return renderHelper(fn);
  }

  // bfloat is f32's top 16 bits, so RTZ is a plain truncation with no
  // exponent rebiasing.
  case Helper::RtzBfloat: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    const msl::Type bf16 = msl::Type::scalar(msl::Scalar::BF16);
    msl::Function *fn = helperFn(c, hn::RtzBfloat, bf16);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::F32), "v", {}});

    fn->body.push_back(c.declStmt(u32, "u", c.bitcast(u32, c.var("v"))));
    fn->body.push_back(c.returnStmt(c.bitcast(
        bf16, c.cast(msl::Type::scalar(msl::Scalar::U16),
                     c.binary(msl::BinOp::And,
                              c.binary(msl::BinOp::Shr, c.var("u"), c.lit(16)),
                              c.litHex(0xffff))))));
    return renderHelper(fn);
  }

  // fp8 has no MSL type and travels as a byte. Rounds to nearest-even and
  // handles subnormals. e4m3fn's top slot is NaN, so saturation lands on
  // the half-way mark below 448, since 0x7f is unavailable as a finite max.
  case Helper::Fp8PackE4M3: {
    msl::Context c;
    msl::Expr *const nearMax = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(15)),
        c.binary(msl::BinOp::Gt, c.var("mant"), c.litHex(0x600000)));
    return renderHelper(fp8Packer(c, hn::Fp8PackE4M3, /*mantBits=*/3,
                                  /*bias=*/7, /*satExp=*/16,
                                  /*subnormalShift=*/21, /*subnormalFloor=*/-6,
                                  /*nanHi=*/0x7f, /*satHi=*/0x7e,
                                  /*quiet=*/0, nearMax, /*biasForm=*/false));
  }

  case Helper::Fp8UnpackE4M3: {
    msl::Context c;
    const msl::Type u32 = msl::Type::scalar(msl::Scalar::U32);
    msl::Expr *const isNan = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("e"), c.litHex(0xf)),
        c.binary(msl::BinOp::Eq, c.var("m"), c.litHex(0x7)));
    msl::Expr *const nanBits =
        c.binary(msl::BinOp::Or,
                 c.binary(msl::BinOp::Or, c.var("sgn"), c.litHex(0x7f800000)),
                 c.litHex(0x400000));
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE4M3, /*mantBits=*/3,
                                    /*bias=*/7, /*fnuz=*/false, isNan,
                                    nanBits));
  }

  case Helper::Fp8PackE5M2: {
    msl::Context c;
    return renderHelper(fp8Packer(c, hn::Fp8PackE5M2, /*mantBits=*/2,
                                  /*bias=*/15, /*satExp=*/31,
                                  /*subnormalShift=*/22, /*subnormalFloor=*/-2,
                                  /*nanHi=*/0x7c, /*satHi=*/0x7c,
                                  /*quiet=*/0x2, nullptr, /*biasForm=*/false));
  }

  case Helper::Fp8UnpackE5M2: {
    msl::Context c;
    msl::Expr *const isInf =
        c.binary(msl::BinOp::Eq, c.var("e"), c.litHex(0x1f));
    msl::Expr *const infBits =
        c.binary(msl::BinOp::Or,
                 c.binary(msl::BinOp::Or, c.var("sgn"), c.litHex(0x7f800000)),
                 c.binary(msl::BinOp::Shl, c.var("m"), c.lit(21)));
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE5M2, /*mantBits=*/2,
                                    /*bias=*/15, /*fnuz=*/false, isInf,
                                    infBits));
  }

  // FNUZ variants e4b8/e5b16: no inf/NaN (0x7f is the finite max), 0x80 is
  // the NaN encoding so signed zero must pack to +0 and the bias is one
  // larger than the encoding they otherwise mirror (8 vs 7, 16 vs 15).
  case Helper::Fp8PackE4B8: {
    msl::Context c;
    msl::Expr *const nearMax = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(15)),
        c.binary(msl::BinOp::Gt, c.var("mant"), c.litHex(0x700000)));
    return renderHelper(fp8Packer(c, hn::Fp8PackE4B8, /*mantBits=*/3,
                                  /*bias=*/8, /*satExp=*/16,
                                  /*subnormalShift=*/21, /*subnormalFloor=*/-6,
                                  /*nanHi=*/0x7f, /*satHi=*/0x7f,
                                  /*quiet=*/0, nearMax, /*biasForm=*/true));
  }

  case Helper::Fp8UnpackE4B8: {
    msl::Context c;
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE4B8, /*mantBits=*/3,
                                    /*bias=*/8, /*fnuz=*/true, nullptr,
                                    nullptr));
  }

  case Helper::Fp8PackE5B16: {
    msl::Context c;
    msl::Expr *const nearMax = c.binary(
        msl::BinOp::LAnd, c.binary(msl::BinOp::Eq, c.var("ex"), c.lit(31)),
        c.binary(msl::BinOp::Gt, c.var("mant"), c.litHex(0x600000)));
    return renderHelper(fp8Packer(c, hn::Fp8PackE5B16, /*mantBits=*/2,
                                  /*bias=*/16, /*satExp=*/32,
                                  /*subnormalShift=*/22, /*subnormalFloor=*/-2,
                                  /*nanHi=*/0x7f, /*satHi=*/0x7f,
                                  /*quiet=*/0, nearMax, /*biasForm=*/true));
  }

  case Helper::Fp8UnpackE5B16: {
    msl::Context c;
    return renderHelper(fp8Unpacker(c, hn::Fp8UnpackE5B16, /*mantBits=*/2,
                                    /*bias=*/16, /*fnuz=*/true, nullptr,
                                    nullptr));
  }

  // e2m1: one sign bit, two exponent bits, one mantissa bit, bias 1. All
  // sixteen values fit in a table and there is no inf/NaN.
  case Helper::Fp4Unpack: {
    msl::Context c;
    const msl::Type f32 = msl::Type::scalar(msl::Scalar::F32);
    msl::Function *fn = helperFn(c, hn::Fp4Unpack, f32);
    fn->params.push_back({msl::Type::scalar(msl::Scalar::U8), "nib", {}});

    msl::SmallVec<msl::Expr *, 4> values;
    for (const double v : {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0})
      values.push_back(c.litF(v));
    for (const double v : {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0})
      values.push_back(c.unary(msl::UnOp::Neg, c.litF(v)));

    fn->body.push_back(
        c.arrayDecl(f32.withQual(msl::Type::Const), "kV", std::move(values)));
    fn->body.push_back(c.returnStmt(c.subscript(
        c.var("kV"),
        c.binary(msl::BinOp::And,
                 c.cast(msl::Type::scalar(msl::Scalar::U32), c.var("nib")),
                 c.litHex(0xf)))));
    return renderHelper(fn);
  }

  // Offsets come from `PrintField`, kept in sync with `PrintPlan`. The head
  // bump happens before the bounds test so the host can count lost records.
  case Helper::PrintAppend: {
    msl::Context c;
    return renderHelper(recordAppender(
        c, hn::PrintAppend, printHeaderWord(PrintHeader::Head),
        kPrintBufferRecords, kPrintHeaderWords, kPrintRecordWords,
        {"site", "pid", "tid", "index", "type", "value", "operand"},
        {printFieldWord(PrintField::Site), printFieldWord(PrintField::Pid),
         printFieldWord(PrintField::Tid), printFieldWord(PrintField::Index),
         printFieldWord(PrintField::Type), printFieldWord(PrintField::Value),
         printFieldWord(PrintField::Operand)}));
  }

  // Offsets come from `AssertField`, kept in sync with `assertLayoutText`.
  // No barrier: only failing threads reach here and a barrier in divergent
  // control flow is undefined in Metal.
  case Helper::AssertRecord: {
    msl::Context c;
    return renderHelper(recordAppender(
        c, hn::AssertRecord, assertHeaderWord(AssertHeader::Head),
        kAssertBufferRecords, kAssertHeaderWords, kAssertRecordWords,
        {"site", "pid", "tid"},
        {assertFieldWord(AssertField::Site), assertFieldWord(AssertField::Pid),
         assertFieldWord(AssertField::Tid)}));
  }
  case Helper::NarrowF64: {
    msl::Context c;
    return renderHelper(narrowF64(c));
  }
  case Helper::SoftFma: {
    msl::Context c;
    return renderHelper(softFmaPacker(c)) + "\n" + renderHelper(softFma(c));
  }
  case Helper::Fma: {
    msl::Context c;
    return renderHelper(guardedFma(c));
  }
  case Helper::Count:
    break;
  }
  return {};
}

} // namespace agpu

#endif // AGPU_PRELUDE_H
