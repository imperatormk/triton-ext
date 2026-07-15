// EmitMSLRawLeaves.cpp - design-sanctioned Raw escape-hatch leaves.
//
// The fp32->half/bfloat narrowing (rtne/rtz) and the emulated fp32/packed-fp16
// CAS loops. These are the only builders that write MSL text imperatively
// through `os`/`ind()`/`indent`/`fresh()` (the Raw escape hatch); every
// mask/shift/round-bias is IEEE-exact.

#include "MSLEmitter.h"

using namespace mlir;

namespace mlir::triton::applegpu {

// IEEE round-to-nearest-even narrowing of an f32 `v` to half/bfloat, with
// correct NaN/Inf/overflow/subnormal handling (used by fp_to_fp rtne).
std::string MSLEmitter::emitRoundedHalfValueFull(const std::string &sc,
                                                 const std::string &v) {
  std::string f = fresh(), u = fresh(), h = fresh(), bits = fresh();
  os << ind() << "float " << f << " = (float)(" << v << ");\n";
  os << ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
  os << ind() << "ushort " << bits << ";\n";
  std::string sgn = fresh(), e32 = fresh(), mant = fresh();
  os << ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
  os << ind() << "int " << e32 << " = (int)((" << u << " >> 23) & 0xffu);\n";
  os << ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
  if (sc == "bfloat") {
    std::string r = fresh();
    os << ind() << "if (" << e32 << " == 0xff) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(((" << u << " >> 16) & 0xffffu) | ("
       << mant << " ? 0x40u : 0u));\n";
    --indent;
    os << ind() << "} else {\n";
    ++indent;
    os << ind() << "uint " << r << " = (" << u << " >> 16) & 1u;\n";
    os << ind() << "uint __t = (" << u << " + 0x7fffu + " << r << ");\n";
    os << ind() << bits << " = (ushort)((__t >> 16) & 0xffffu);\n";
    --indent;
    os << ind() << "}\n";
  } else {
    std::string ex = fresh();
    os << ind() << "int " << ex << " = " << e32 << " - 112;\n";
    os << ind() << "if (" << e32 << " == 0xff) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u | ("
       << mant << " ? 0x200u : 0u));\n";
    --indent;
    os << ind() << "} else if (" << ex << " >= 31) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u);\n";
    --indent;
    os << ind() << "} else if (" << ex << " <= 0) {\n";
    ++indent;
    os << ind() << "if (" << ex << " < -10) { " << bits << " = (ushort)"
       << sgn << "; }\n";
    os << ind() << "else {\n";
    ++indent;
    os << ind() << "uint __fm = " << mant << " | 0x800000u;\n";
    os << ind() << "int __sh = 14 - " << ex << ";\n";
    os << ind() << "uint __m = __fm >> __sh;\n";
    os << ind() << "uint __rem = __fm & ((1u << __sh) - 1u);\n";
    os << ind() << "uint __half = 1u << (__sh - 1);\n";
    os << ind() << "if (__rem > __half || (__rem == __half && (__m & 1u))) "
       << "__m += 1;\n";
    os << ind() << bits << " = (ushort)(" << sgn << " | __m);\n";
    --indent;
    os << ind() << "}\n";
    --indent;
    os << ind() << "} else {\n";
    ++indent;
    std::string m = fresh(), rem = fresh();
    os << ind() << "uint " << m << " = " << mant << " >> 13;\n";
    os << ind() << "uint " << rem << " = " << mant << " & 0x1fffu;\n";
    os << ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
       << " << 10) | " << m << ");\n";
    os << ind() << "if (" << rem << " > 0x1000u || (" << rem
       << " == 0x1000u && (" << m << " & 1u))) " << bits << " += 1;\n";
    --indent;
    os << ind() << "}\n";
  }
  os << ind() << sc << " " << h << " = as_type<" << sc << ">(" << bits
     << ");\n";
  return h;
}

// Round-toward-zero (truncating) narrowing of an f32 `v` to half/bfloat.
// RTZ drops the low mantissa bits with no rounding increment.
std::string MSLEmitter::emitTruncatedFloatValue(const std::string &sc,
                                                const std::string &v) {
  std::string f = fresh(), u = fresh(), h = fresh(), bits = fresh();
  os << ind() << "float " << f << " = (float)(" << v << ");\n";
  os << ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
  os << ind() << "ushort " << bits << ";\n";
  if (sc == "bfloat") {
    os << ind() << bits << " = (ushort)((" << u << " >> 16) & 0xffffu);\n";
  } else {
    std::string sgn = fresh(), ex = fresh(), mant = fresh();
    os << ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
    os << ind() << "int " << ex << " = (int)((" << u
       << " >> 23) & 0xffu) - 112;\n";
    os << ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
    os << ind() << "if (((" << u << " >> 23) & 0xffu) == 0xffu) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u | ("
       << mant << " ? 0x200u : 0u));\n";
    --indent;
    os << ind() << "} else if (" << ex << " >= 31) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(" << sgn << " | 0x7bffu);\n";
    --indent;
    os << ind() << "} else if (" << ex << " <= 0) {\n";
    ++indent;
    os << ind() << "if (" << ex << " < -10) { " << bits << " = (ushort)"
       << sgn << "; }\n";
    os << ind() << "else { uint __m = (" << mant
       << " | 0x800000u) >> (14 - " << ex << "); " << bits
       << " = (ushort)(" << sgn << " | __m); }\n";
    --indent;
    os << ind() << "} else {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
       << " << 10) | (" << mant << " >> 13));\n";
    --indent;
    os << ind() << "}\n";
  }
  os << ind() << sc << " " << h << " = as_type<" << sc << ">(" << bits
     << ");\n";
  return h;
}

// Integer RTNE narrowing of `v` to `sc`. Metal fast-math elides a plain
// (half)/(bfloat) cast whose only consumer re-widens to float (the CAS add),
// dropping the round; the integer path forces one the optimizer cannot cancel.
std::string MSLEmitter::emitRoundedHalfValue(const std::string &sc,
                                             const std::string &v) {
  std::string f = fresh(), u = fresh(), h = fresh();
  os << ind() << "float " << f << " = (float)(" << v << ");\n";
  os << ind() << "uint " << u << " = as_type<uint>(" << f << ");\n";
  std::string bits = fresh();
  os << ind() << "ushort " << bits << ";\n";
  if (sc == "bfloat") {
    std::string r = fresh();
    os << ind() << "uint " << r << " = (" << u << " >> 16) & 1u;\n";
    os << ind() << bits << " = (ushort)(((" << u << " + 0x7fffu + " << r
       << ") >> 16) & 0xffffu);\n";
  } else {
    std::string sgn = fresh(), ex = fresh(), mant = fresh(), m = fresh(),
                rem = fresh();
    os << ind() << "uint " << sgn << " = (" << u << " >> 16) & 0x8000u;\n";
    os << ind() << "int " << ex << " = (int)((" << u
       << " >> 23) & 0xffu) - 112;\n";
    os << ind() << "uint " << mant << " = " << u << " & 0x7fffffu;\n";
    os << ind() << "if (" << ex << " <= 0) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)" << sgn << ";\n";
    --indent;
    os << ind() << "} else if (" << ex << " >= 31) {\n";
    ++indent;
    os << ind() << bits << " = (ushort)(" << sgn << " | 0x7c00u);\n";
    --indent;
    os << ind() << "} else {\n";
    ++indent;
    os << ind() << "uint " << m << " = " << mant << " >> 13;\n";
    os << ind() << "uint " << rem << " = " << mant << " & 0x1fffu;\n";
    os << ind() << bits << " = (ushort)(" << sgn << " | ((uint)" << ex
       << " << 10) | " << m << ");\n";
    os << ind() << "if (" << rem << " > 0x1000u || (" << rem
       << " == 0x1000u && (" << m << " & 1u))) " << bits << " += 1;\n";
    --indent;
    os << ind() << "}\n";
  }
  os << ind() << sc << " " << h << " = as_type<" << sc << ">(" << bits
     << ");\n";
  return h;
}

// For a 16-bit-float element pointer p, emit statements binding a uint* to the
// containing aligned 32-bit word and a bool selecting the high half-word.
// Returns {wordPtr, isHigh} identifiers.
std::pair<std::string, std::string>
MSLEmitter::emitPacked16Base(const std::string &p, const std::string &sc) {
  std::string bytePtr = fresh(), wordAddr = fresh(), isHigh = fresh(),
              wordPtr = fresh();
  os << ind() << "device uchar *" << bytePtr
     << " = (device uchar *)(" << p << ");\n";
  os << ind() << "size_t " << wordAddr << " = (size_t)" << bytePtr
     << " & ~(size_t)3;\n";
  os << ind() << "bool " << isHigh << " = ((size_t)" << bytePtr
     << " & 2u) != 0u;\n";
  os << ind() << "device atomic_uint *" << wordPtr
     << " = (device atomic_uint *)" << wordAddr << ";\n";
  (void)sc;
  return {wordPtr, isHigh};
}

// Extract the selected 16-bit-float lane from a loaded 32-bit word into a
// float, apply `newFloatExpr` (which may reference `curId`), repack into the
// word, and CAS-loop until it lands. Binds `id` to the pre-op value.
void MSLEmitter::emitPacked16CASLoop(const std::string &wordPtr,
                                     const std::string &isHigh,
                                     const std::string &sc,
                                     const std::string &curId,
                                     const std::string &newHalfExpr,
                                     const std::string &id) {
  std::string word = fresh(), lane = fresh(), newLane = fresh(),
              newWord = fresh();
  os << ind() << "uint " << word
     << " = atomic_load_explicit(" << wordPtr << ", memory_order_relaxed);\n";
  os << ind() << "while (true) {\n";
  ++indent;
  os << ind() << "ushort " << lane << " = (ushort)((" << isHigh << ") ? ("
     << word << " >> 16) : (" << word << " & 0xffffu));\n";
  os << ind() << id << " = as_type<" << sc << ">(" << lane << ");\n";
  os << ind() << sc << " " << curId << " = as_type<" << sc << ">(" << lane
     << ");\n";
  os << ind() << sc << " " << newLane << " = " << newHalfExpr << ";\n";
  os << ind() << "uint " << newWord << " = (" << isHigh
     << ") ? ((" << word << " & 0x0000ffffu) | ((uint)as_type<ushort>("
     << newLane << ") << 16)) : ((" << word
     << " & 0xffff0000u) | (uint)as_type<ushort>(" << newLane << "));\n";
  os << ind() << "if (atomic_compare_exchange_weak_explicit(" << wordPtr
     << ", &" << word << ", " << newWord
     << ", memory_order_relaxed, memory_order_relaxed)) break;\n";
  --indent;
  os << ind() << "}\n";
}

// f32 atomic RMW via a 32-bit CAS loop (Metal's atomic_float only supports
// fetch_add; max/min/xchg need emulation). `newFloatExpr` references `curId`.
void MSLEmitter::emitFloat32CASLoop(const std::string &p,
                                    const std::string &curId,
                                    const std::string &newFloatExpr,
                                    const std::string &id) {
  std::string wordPtr = fresh(), word = fresh(), newWord = fresh();
  os << ind() << "device atomic_uint *" << wordPtr
     << " = (device atomic_uint *)(" << p << ");\n";
  os << ind() << "uint " << word
     << " = atomic_load_explicit(" << wordPtr << ", memory_order_relaxed);\n";
  os << ind() << "while (true) {\n";
  ++indent;
  os << ind() << id << " = as_type<float>(" << word << ");\n";
  os << ind() << "float " << curId << " = as_type<float>(" << word << ");\n";
  os << ind() << "uint " << newWord << " = as_type<uint>((float)("
     << newFloatExpr << "));\n";
  os << ind() << "if (atomic_compare_exchange_weak_explicit(" << wordPtr
     << ", &" << word << ", " << newWord
     << ", memory_order_relaxed, memory_order_relaxed)) break;\n";
  --indent;
  os << ind() << "}\n";
}

} // namespace mlir::triton::applegpu
