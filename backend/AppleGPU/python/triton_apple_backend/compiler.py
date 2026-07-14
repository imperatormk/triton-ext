"""Apple MPS Triton backend.

Compiles Triton kernels through TTIR → TTGIR → MSL → metallib, then dispatches
via MTLComputeCommandEncoder.
"""

from dataclasses import dataclass
import hashlib
import os
import re
import tempfile
import threading

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes


# Libdevice patching: see _LibdevicePatchFinder in __init__.py
_plugin = getattr(passes, 'plugin', None)

# EmitMSL hands its output path to the C++ pass through the process-global
# TRITON_MSL_OUT env var. Inductor autotuning precompiles choices on a thread
# pool, so two concurrent make_msl calls would race that global and one would
# read the other's path, leaving its own metallib empty ("no kernel void").
# Serialize the env-var critical section.
_msl_out_lock = threading.Lock()


def _pmaybe_enable_debug(pm):
    if os.environ.get('TRITON_MPS_DEBUG'):
        pm.enable_debug()


@dataclass(frozen=True)
class MPSOptions:
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    arch: str = "apple_m"
    backend_name: str = "mps"
    warp_size: int = 32

    # simdgroup tile — fixed by hardware
    simdgroup_m: int = 8
    simdgroup_n: int = 8
    simdgroup_k: int = 8

    # Standard Triton options (accepted but largely unused on MPS)
    debug: bool = False
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    instrumentation_mode: str = "none"
    sanitize_overflow: bool = False
    allowed_dot_input_precisions: tuple = ("ieee", )
    default_dot_input_precision: str = "ieee"
    supported_fp8_dtypes: tuple = ()
    extern_libs: tuple = ()

    def hash(self):
        return hashlib.md5(str(self.__dict__).encode()).hexdigest()


class MPSBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "mps"

    def __init__(self, target: GPUTarget):
        super().__init__(target)
        self.target = target
        self.binary_ext = "metallib"
        if not _plugin:
            raise RuntimeError(
                "Apple GPU plugin not loaded. Set TRITON_PASS_PLUGIN_PATH to "
                "the TritonAppleGPUBackend dylib built from triton-ext.")

    def parse_options(self, opts) -> MPSOptions:
        args = {
            k: opts[k]
            for k in MPSOptions.__dataclass_fields__ if k in opts
        }
        nw = args.get("num_warps", MPSOptions.num_warps)
        assert nw > 0 and (nw &
                           (nw - 1)) == 0, "num_warps must be a power of 2"
        return MPSOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):

        def min_dot_size(lhs_type, rhs_type):
            # Apple simdgroup tile is 8×8; minimum dot operand = (1, 1, 8)
            return (1, 1, 8)

        return {"min_dot_size": min_dot_size}

    def get_module_map(self):
        return {}

    def get_target_name(self, options) -> str:
        return f"mps:{options.arch}"

    def load_dialects(self, ctx):
        # Plugin dialect is registered automatically via TRITON_PASS_PLUGIN_PATH
        ir.load_dialects(ctx)

    def hash(self):
        return "mps-v0.1"

    # ── Stage 1: Triton IR optimization (shared) ───────────────────────────
    def make_ttir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    # ── Stage 2: GPU tiling — THE make-or-break ────────────────────────────
    def make_ttgir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)

        # Convert generic TritonIR → TritonGPU IR (shared pass)
        passes.ttir.add_convert_to_ttgpuir(
            pm,
            f"mps:{options.arch}",
            options.num_warps,
            32,  # warp_size = 32 (simdgroup size)
            options.num_ctas)

        # Shared layout optimization passes
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        # Apple plugin passes (loaded via TRITON_PASS_PLUGIN_PATH)
        _plugin.add_simplify_gather(pm)
        _plugin.add_accelerate_matmul(pm)

        # Clean up redundant layout conversions introduced by the rewrite
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, True)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        # Re-lay MMA epilogue stores so the #mma -> #blocked convert is a
        # within-simdgroup shuffle (no threadgroup round-trip). Must run after
        # the final remove_layout_conversions or it gets reverted.
        _plugin.add_store_shuffle_layout(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        # Fuse nested loops marked with tt.flatten (tl.range(flatten=True))
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)

        # The MSL path has no async copy (staging lowers to a synchronous
        # threadgroup copy), so it stays single-buffered — the software
        # pipeliner would only add staging, barriers and slot rotation with no
        # overlap benefit. No pipelining pass runs here.

        pm.run(mod, 'make_ttgir')
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0
        return mod

    # ── TTGIR → MSL source → metallib ─────────────────────────────────────
    def make_msl(self, mod, metadata, options):
        with tempfile.NamedTemporaryFile(suffix='.metal', delete=False) as f:
            msl_path = f.name
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)
        with _msl_out_lock:
            old = os.environ.get('TRITON_MSL_OUT')
            os.environ['TRITON_MSL_OUT'] = msl_path
            try:
                _plugin.add_emit_msl(pm)
                pm.run(mod, 'make_msl')
            finally:
                if old is None:
                    os.environ.pop('TRITON_MSL_OUT', None)
                else:
                    os.environ['TRITON_MSL_OUT'] = old
        with open(msl_path, 'r') as f:
            msl = f.read()
        os.unlink(msl_path)
        if os.environ.get('TRITON_MPS_DEBUG'):
            print("=== emitted MSL ===")
            print(msl)
        metadata["_msl_src"] = msl
        dump = os.environ.get('TRITON_MSL_DUMP')
        if dump:
            with open(dump, 'w') as df:
                df.write(msl)
        m = re.search(r'kernel void (\w+)\(', msl)
        if not m:
            raise RuntimeError("no 'kernel void' entry found in emitted MSL")
        metadata["name"] = m.group(1)
        metadata["shared"] = 0
        return mod

    def make_msl_metallib(self, mod, metadata, options):
        msl = metadata.pop("_msl_src")
        # Safe math globally. Metal fast-math assumes no NaN/Inf and reassociates
        # FP, silently miscompiling any kernel that produces or consumes Inf/NaN
        # or relies on RTNE. Which kernels see Inf/NaN is a runtime property, so
        # scoping is unsound; safe math is proven zero-cost on the GEMM path.
        from triton_apple_backend import metal_utils
        return metal_utils.compile_source(msl, 'safe')

    # ── Stage 3: LLVM IR with simdgroup intrinsics ─────────────────────────
    # ── Gluon frontend: AST is lowered directly to TTGIR, so we skip make_ttir
    # and only run the dialect-generic Gluon passes that resolve explicit/auto
    # layouts before rejoining the shared TTGIR → MSL → metallib path.
    def gluon_to_ttgir(self, src, metadata, options):
        mod = src
        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)

        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def add_stages(self, stages, options, language):
        if language == Language.GLUON:
            stages["ttgir"] = lambda src, meta: self.gluon_to_ttgir(
                src, meta, options)
        else:
            stages["ttir"] = lambda src, meta: self.make_ttir(
                src, meta, options)
            stages["ttgir"] = lambda src, meta: self.make_ttgir(
                src, meta, options)
        stages["msl"] = lambda src, meta: self.make_msl(src, meta, options)
        stages["metallib"] = lambda src, meta: self.make_msl_metallib(
            src, meta, options)
