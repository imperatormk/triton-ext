// MSLAstRoundTrip.cpp - standalone printer round-trip self-test.
//
// Excluded from the normal dylib build (guarded by MSL_AST_SELFTEST so it emits
// no symbols into libapplegpu_backend). Compile + run standalone (LLVMROOT is
// the prebuilt Triton LLVM; adjust the hash for your machine):
//
//   LLVMROOT=~/.triton/llvm/llvm-62b7cf96-macos-arm64-2
//   clang++ -std=c++17 -DMSL_AST_SELFTEST -I$LLVMROOT/include \
//     MSLAstRoundTrip.cpp MSLPrinter.cpp \
//     $LLVMROOT/lib/libLLVMSupport.a $LLVMROOT/lib/libLLVMDemangle.a \
//     -lz -lcurses -o /tmp/msl_selftest && /tmp/msl_selftest
//
// Exits nonzero and prints an expected-vs-got diff on mismatch.
#ifdef MSL_AST_SELFTEST

#include "MSLPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir::triton::applegpu::msl;

int main() {
  MSLContext ctx;

  // Kernel body -------------------------------------------------------------
  Block body;

  // for (int v0 = 0; v0 < 4; v0 += 1) {
  //     float v1 = x[v0];
  //     acc = acc + v1;
  //     threadgroup_barrier(mem_threadgroup);
  //     threadgroup_barrier(mem_threadgroup | mem_device);   <- collapses
  // }
  {
    Block forBody;
    forBody.push_back(ctx.declStmt(
        ctx.scalar(Scalar::F32), "v1",
        ctx.subscript(ctx.var("x"), ctx.var("v0"))));
    forBody.push_back(ctx.assignStmt(
        ctx.var("acc"), ctx.binary(BinOp::Add, ctx.var("acc"), ctx.var("v1"))));
    forBody.push_back(ctx.barrier(/*device=*/false));
    forBody.push_back(ctx.barrier(/*device=*/true));

    Stmt *initDecl =
        ctx.declStmt(ctx.scalar(Scalar::I32), "v0", ctx.lit("0"));
    Expr *cond = ctx.binary(BinOp::Lt, ctx.var("v0"), ctx.lit("4"));
    Stmt *step = ctx.addAssignStmt(ctx.var("v0"), ctx.lit("1"));
    body.push_back(ctx.forScope(initDecl, cond, step, std::move(forBody)));
  }

  // empty-for: proves an empty body prints `{\n    }\n`, not a dangling header.
  {
    Stmt *initDecl =
        ctx.declStmt(ctx.scalar(Scalar::I32), "e", ctx.lit("0"));
    Expr *cond = ctx.binary(BinOp::Lt, ctx.var("e"), ctx.lit("2"));
    Stmt *step =
        ctx.assignStmt(ctx.var("e"),
                       ctx.binary(BinOp::Add, ctx.var("e"), ctx.lit("1")));
    body.push_back(ctx.forScope(initDecl, cond, step, Block{}));
  }

  // if (acc > 0.0f) { acc = as_type<uint>(acc); } else { acc = 0.0f; }
  {
    Block thenB, elseB;
    thenB.push_back(ctx.assignStmt(
        ctx.var("acc"),
        ctx.cast(Cast::Style::AsType, ctx.scalar(Scalar::U32),
                 ctx.var("acc"))));
    elseB.push_back(ctx.assignStmt(ctx.var("acc"), ctx.lit("0.0f")));
    body.push_back(ctx.ifElseScope(
        ctx.binary(BinOp::Gt, ctx.var("acc"), ctx.lit("0.0f")),
        std::move(thenB), std::move(elseB)));
  }

  // simdgroup_multiply_accumulate(a, b, c, c);
  body.push_back(ctx.exprStmt(ctx.call(
      "simdgroup_multiply_accumulate",
      {ctx.var("a"), ctx.var("b"), ctx.var("c"), ctx.var("c")})));

  // Wide-IV trip-count loop: iv declared INSIDE the body (dodges the i65 fold).
  {
    Block tcBody;
    tcBody.push_back(ctx.declStmt(
        ctx.scalar(Scalar::I64), "iv",
        ctx.binary(BinOp::Add, ctx.lit("0"),
                   ctx.binary(BinOp::Mul, ctx.var("tc"), ctx.lit("1")))));
    tcBody.push_back(ctx.ifScope(
        ctx.unary(UnOp::LNot,
                  ctx.paren(ctx.binary(BinOp::Lt, ctx.var("iv"),
                                       ctx.lit("100")))),
        [&] {
          Block b;
          b.push_back(ctx.breakStmt());
          return b;
        }()));
    body.push_back(
        ctx.tripCountForScope(ctx.scalar(Scalar::I64), "tc", std::move(tcBody)));
  }

  // Kernel signature: device float* x [[buffer(0)]], uint3 tid [[...]]
  KernelFn *kernel = ctx.kernelFn(
      ctx.maxThreadsAttr(128), "triton_demo",
      {ctx.param(ctx.ptr(ctx.scalar(Scalar::F32), AddrSpace::Device), "x",
                 ctx.bufferAttr(0)),
       ctx.param(ctx.vector(Scalar::U32, 3), "tid", ctx.threadPosAttr())},
      std::move(body));

  std::string got;
  {
    llvm::raw_string_ostream os(got);
    MSLPrinter p(os);
    p.printPreamble();
    p.print(kernel);
    os.flush();
  }

  static const char *kExpected =
      "#include <metal_stdlib>\n"
      "#include <metal_simdgroup_matrix>\n"
      "using namespace metal;\n"
      "\n"
      "static inline float tt_erf(float x){\n"
      "  float t = 1.0f/(1.0f+0.5f*metal::fabs(x));\n"
      "  float y = t*metal::exp(-x*x-1.26551223f+t*(1.00002368f+t*(0.37409196f"
      "+t*(0.09678418f+t*(-0.18628806f+t*(0.27886807f+t*(-1.13520398f"
      "+t*(1.48851587f+t*(-0.82215223f+t*0.17087277f)))))))));\n"
      "  float r = 1.0f - y;\n"
      "  return x >= 0.0f ? r : -r;\n"
      "}\n"
      "\n"
      "[[max_total_threads_per_threadgroup(128)]]\n"
      "kernel void triton_demo(\n"
      "    device float* x [[buffer(0)]],\n"
      "    uint3 tid [[thread_position_in_threadgroup]]\n"
      ") {\n"
      "    for (int v0 = 0; v0 < 4; v0 += 1) {\n"
      "        float v1 = x[v0];\n"
      "        acc = acc + v1;\n"
      "        threadgroup_barrier(mem_flags::mem_threadgroup | "
      "mem_flags::mem_device);\n"
      "    }\n"
      "    for (int e = 0; e < 2; e = e + 1) {\n"
      "    }\n"
      "    if (acc > 0.0f) {\n"
      "        acc = as_type<uint>(acc);\n"
      "    } else {\n"
      "        acc = 0.0f;\n"
      "    }\n"
      "    simdgroup_multiply_accumulate(a, b, c, c);\n"
      "    for (long tc = 0; ; tc += 1) {\n"
      "        long iv = 0 + tc * 1;\n"
      "        if (!(iv < 100)) {\n"
      "            break;\n"
      "        }\n"
      "    }\n"
      "}\n";

  if (got != kExpected) {
    llvm::errs() << "=== MISMATCH ===\n--- expected ---\n"
                 << kExpected << "\n--- got ---\n"
                 << got << "\n";
    return 1;
  }
  llvm::outs() << got;
  llvm::outs() << "=== MSL AST round-trip OK ===\n";
  return 0;
}

#endif // MSL_AST_SELFTEST
