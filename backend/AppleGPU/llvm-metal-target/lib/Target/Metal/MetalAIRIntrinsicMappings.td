//===- MetalAIRIntrinsicMappings.td ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mapping of LLVM intrinsic names to their Apple AIR equivalents. The records
// declared here are consumed by -gen-searchable-tables to produce a binary-
// searchable table used by the MetalLLVMToAIRIntrinsics pass.
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

class IntrinsicRename<string llvm, string air> {
  string LLVMName = llvm;
  string AIRName = air;
}

def AIRIntrinsicRenames : GenericTable {
  let FilterClass = "IntrinsicRename";
  let Fields = ["LLVMName", "AIRName"];
}

// Keep in sync with Apple's air::kIntrinsicRenames table.

def : IntrinsicRename<"llvm.maxnum.f32", "air.fmax.f32">;
def : IntrinsicRename<"llvm.minnum.f32", "air.fmin.f32">;
def : IntrinsicRename<"llvm.maxnum.f16", "air.fmax.f16">;
def : IntrinsicRename<"llvm.minnum.f16", "air.fmin.f16">;

// sin/cos use the *precise* AIR variants. tan is implemented in the libdevice
// stub as sin(x)/cos(x); near the poles (x ~ pi/2, tan ~ 1e3) the low-precision
// fast_sin/fast_cos plus the division amplify the relative error past test
// tolerances (abs err 0.027 vs eager precise::tan err 2e-4). Precise sin/cos is
// only marginally slower and strictly more accurate, so no other kernel
// regresses (same rationale as exp2/log2/pow below).
def : IntrinsicRename<"llvm.sin.f32", "air.sin.f32">;
def : IntrinsicRename<"llvm.cos.f32", "air.cos.f32">;
def : IntrinsicRename<"llvm.sin.f16", "air.sin.f16">;
def : IntrinsicRename<"llvm.cos.f16", "air.cos.f16">;

def : IntrinsicRename<"llvm.exp.f32",  "air.fast_exp.f32">;
def : IntrinsicRename<"llvm.log.f32",  "air.fast_log.f32">;
// exp2/log2 use the *precise* AIR variants: they are the building blocks of
// pow (ax**y == exp2(y*log2(ax))) where a low-precision fast_log2 is amplified
// by the exponent (e.g. x**1000) far beyond test tolerances. Precise is only
// marginally slower and strictly more accurate, so no other kernel regresses.
def : IntrinsicRename<"llvm.exp2.f32", "air.exp2.f32">;
def : IntrinsicRename<"llvm.log2.f32", "air.log2.f32">;
// Native AIR pow (Metal precise::pow); used directly by the libdevice pow stub
// to avoid the lossy exp2/log2 decomposition entirely.
def : IntrinsicRename<"llvm.pow.f32", "air.pow.f32">;
def : IntrinsicRename<"llvm.exp.f16",  "air.fast_exp.f16">;
def : IntrinsicRename<"llvm.log.f16",  "air.fast_log.f16">;
def : IntrinsicRename<"llvm.exp2.f16", "air.fast_exp2.f16">;
def : IntrinsicRename<"llvm.log2.f16", "air.fast_log2.f16">;

def : IntrinsicRename<"llvm.sqrt.f32",  "air.fast_sqrt.f32">;
def : IntrinsicRename<"llvm.fabs.f32",  "air.fabs.f32">;
def : IntrinsicRename<"llvm.floor.f32", "air.fast_floor.f32">;
def : IntrinsicRename<"llvm.ceil.f32",  "air.fast_ceil.f32">;
def : IntrinsicRename<"llvm.sqrt.f16",  "air.fast_sqrt.f16">;
def : IntrinsicRename<"llvm.fabs.f16",  "air.fabs.f16">;
def : IntrinsicRename<"llvm.floor.f16", "air.fast_floor.f16">;
def : IntrinsicRename<"llvm.ceil.f16",  "air.fast_ceil.f16">;

def : IntrinsicRename<"llvm.fma.f32", "air.fma.f32">;
def : IntrinsicRename<"llvm.fma.f16", "air.fma.f16">;

def : IntrinsicRename<"llvm.rint.f32", "air.fast_rint.f32">;
def : IntrinsicRename<"llvm.rint.f16", "air.fast_rint.f16">;
