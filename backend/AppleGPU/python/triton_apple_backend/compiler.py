"""Apple MPS Triton backend.

Compiles Triton kernels through TTIR → TTGIR → LLVM IR → metallib,
then dispatches via MTLComputeCommandEncoder.
"""

from dataclasses import dataclass
import hashlib
import os
import re
import subprocess
import tempfile

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm


def _host_macos_major() -> int:
    """Target macOS major version for AIR emission.

    Triton JITs on the machine it runs on, so the host macOS version IS the
    target. Apple changed the simdgroup-MMA intrinsic signature + several AIR
    version fields across OS versions (verified via `xcrun metal
    -mmacosx-version-min=N`): the discriminator is the macOS major. macOS 26
    (the 2025 renumber) maps to the "16" era for AIR purposes.

    Override with TRITON_MPS_TARGET_OS_MAJOR (e.g. for cross-compiling).
    """
    env = os.environ.get("TRITON_MPS_TARGET_OS_MAJOR")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        import platform
        major = int(platform.mac_ver()[0].split(".")[0])
        # Apple renumbered macOS 16 -> 26; both are the same AIR era ("16").
        major = 16 if major >= 16 else major
    except Exception:
        major = 16  # default: current shipping target
    # Export so the C++ DotOp lowering pass (reads TRITON_MPS_TARGET_OS_MAJOR)
    # selects the matching simdgroup intrinsic signature.
    os.environ.setdefault("TRITON_MPS_TARGET_OS_MAJOR", str(major))
    return major


def _air_triple(os_major: int) -> str:
    """Canonical AIR triple for a target macOS major. subarch _vNN = major+12;
    macOS 16-era is written as macosx26 (Apple renumber)."""
    sub = os_major + 12
    triple_os = 26 if os_major >= 16 else os_major
    return f"air64_v{sub}-apple-macosx{triple_os}.0.0"


# Libdevice patching: see _LibdevicePatchFinder in __init__.py
_plugin = getattr(passes, 'plugin', None)


def _pmaybe_enable_debug(pm):
    if os.environ.get('TRITON_MPS_DEBUG'):
        pm.enable_debug()


def _find_llc():
    """Locate the metal-llc binary shipped by the AppleGPU backend.

    `metal-llc` is produced by `backend/AppleGPU/llvm-metal-target/`. Two
    layouts are supported:
      - Nested (default `make build`):
          <triton-ext>/build/backend/AppleGPU/llvm-metal-target/bin/metal-llc
      - Standalone (`cmake -S llvm-metal-target -B llvm-metal-target/build`):
          <triton-ext>/backend/AppleGPU/llvm-metal-target/build/bin/metal-llc
    Override with METAL_LLC_PATH for ad-hoc dev.
    """
    if os.environ.get('METAL_LLC_PATH'):
        return os.environ['METAL_LLC_PATH']
    here = os.path.dirname(os.path.abspath(__file__))
    apple_gpu = os.path.abspath(os.path.join(here, '..', '..'))
    triton_ext = os.path.abspath(os.path.join(apple_gpu, '..', '..'))
    candidates = [
        os.path.join(triton_ext, 'build', 'bin', 'metal-llc'),
        os.path.join(apple_gpu, 'llvm-metal-target', 'build', 'bin',
                     'metal-llc'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# Sized scalar types that may appear as the element type of an `addrspace(3)`
# global. Vectors are handled by multiplying through.
_LLVM_SCALAR_BYTES = {
    'i1': 1,
    'i8': 1,
    'i16': 2,
    'i32': 4,
    'i64': 8,
    'half': 2,
    'bfloat': 2,
    'float': 4,
    'double': 8,
}


def _llvm_type_size(ty: str) -> int:
    """Total bytes for an LLVM IR type literal as it appears in a global decl."""
    ty = ty.strip()
    # Vector: <N x T>
    m = re.fullmatch(r'<\s*(\d+)\s*x\s*(.+?)\s*>', ty)
    if m:
        return int(m.group(1)) * _llvm_type_size(m.group(2))
    # Array: [N x T]
    m = re.fullmatch(r'\[\s*(\d+)\s*x\s*(.+?)\s*\]', ty)
    if m:
        return int(m.group(1)) * _llvm_type_size(m.group(2))
    # Sized scalar
    if ty in _LLVM_SCALAR_BYTES:
        return _LLVM_SCALAR_BYTES[ty]
    # Generic iN
    m = re.fullmatch(r'i(\d+)', ty)
    if m:
        return (int(m.group(1)) + 7) // 8
    raise ValueError(f"unsupported LLVM type for tg-memory sizing: {ty!r}")


def _tg_memory_bytes(llvm_ir: str) -> int:
    """Sum bytes for `addrspace(3)` globals in the IR, with alignment padding.

    Matches the C bridge's metalir_tg_memory_bytes: walks each addrspace(3)
    global in declaration order, pads the running total up to that global's
    align, then adds its allocation size.
    """
    total = 0
    # `@name = ... addrspace(3) global <type> ..., align N`
    pat = re.compile(
        r'^@[\w$.]+\s*=\s*(?:[^@\n]*?\s)?addrspace\(3\)\s+(?:[\w]+\s+)?global\s+'
        r'(.+?)(?:,\s*align\s+(\d+))?\s*$', re.MULTILINE)
    for m in pat.finditer(llvm_ir):
        # The captured initializer-or-type group may start with the type
        # followed by the initializer; take the leading well-formed type.
        head = m.group(1).strip()
        tm = re.match(r'(<[^>]+>|\[[^\]]+\]|[\w]+)', head)
        if not tm:
            raise ValueError(f"unrecognized addrspace(3) type head: {head!r}")
        ty = tm.group(1)
        align = int(m.group(2)) if m.group(2) else 1
        if total % align:
            total += align - (total % align)
        total += _llvm_type_size(ty)
    return total


def _load_metalir():
    """Return a compile function backed by the out-of-tree `metal-llc`."""
    llc = _find_llc()
    if not llc:
        raise RuntimeError(
            "metal-llc not found. Build the out-of-tree Metal target:\n"
            "  cd <triton-ext>/llvm-metal-target && \\\n"
            "    cmake -B build -G Ninja && cmake --build build")

    def compile_ir(llvm_ir: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix='.metallib',
                                         delete=False) as out_f:
            out_path = out_f.name
        try:
            if os.environ.get('TRITON_MPS_DEBUG'):
                print(
                    f"[mps] llc: {llc} -mtriple={_air_triple(_host_macos_major())} -filetype=obj (os_major={_host_macos_major()})"
                )
            proc = subprocess.run([
                llc, '-mtriple=' + _air_triple(_host_macos_major()),
                '-filetype=obj', '-o', out_path, '-'
            ],
                                  input=llvm_ir.encode(),
                                  capture_output=True,
                                  check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"llc failed: {proc.stderr.decode(errors='replace')}")
            with open(out_path, 'rb') as f:
                return f.read()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    compile_ir.tg_memory_bytes = _tg_memory_bytes
    return compile_ir


_metalir_compile = None


def _get_metalir_compile():
    global _metalir_compile
    if _metalir_compile is None:
        _metalir_compile = _load_metalir()
    return _metalir_compile


@dataclass(frozen=True)
class MPSOptions:
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    arch: str = "apple_m"
    backend_name: str = "mps"

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

        # Fuse nested loops marked with tt.flatten (tl.range(flatten=True))
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)

        # Software pipeliner: multi-buffer loads across loop iterations.
        # Generates ttg.async_copy_global_to_local + async_commit + async_wait
        # which we lower to air.simdgroup_async_copy_2d (hardware DMA).
        if options.num_stages > 1:
            passes.ttgpuir.add_assign_latencies(pm, options.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, options.num_stages, False)

        pm.run(mod, 'make_ttgir')
        # Preliminary shared memory estimate (reduction scratchpad only).
        # Overwritten in make_llir with the real total after convert_layout
        # adds __tg_cvt_* threadgroup globals.
        metadata["shared"] = mod.get_int_attr("ttg.shared") or 0
        return mod

    # ── Stage 3: LLVM IR with simdgroup intrinsics ─────────────────────────
    def make_llir(self, mod, metadata, options):
        # Resolve the target macOS major *before* running the lowering passes.
        # The C++ DotOp→AIR pass reads TRITON_MPS_TARGET_OS_MAJOR via getenv to
        # pick the simdgroup-matrix intrinsic signature (canonical for macOS<=15
        # vs the 3-vector form for macOS>=16). `_host_macos_major()` exports the
        # env var as a side effect; it must run here, since make_metallib (which
        # also calls it, for the llc triple) runs only AFTER this pass — too late
        # to influence the signature. Without this, macOS 14 was handed the
        # macOS-16 3-vector MMA intrinsic and the driver crashed PSO creation
        # ("Compiler encountered an internal error").
        _host_macos_major()

        pm = ir.pass_manager(mod.context)
        _pmaybe_enable_debug(pm)

        # Standard TritonGPU → LLVM lowering prerequisites
        passes.convert.add_scf_to_cf(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        passes.convert.add_index_to_llvmir(pm)

        # Apple plugin passes
        _plugin.add_to_llvmir(pm)
        _plugin.add_lower_gpu_to_air(pm)
        _plugin.add_reconcile_unrealized_casts(pm)

        # Shared cleanup
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, 'make_llir')

        # Convert MLIR LLVM dialect → LLVM module
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        llvm_ir = str(llvm_mod)
        if os.environ.get('TRITON_MPS_DEBUG'):
            open('/tmp/raw_pre.ll', 'w').write(llvm_ir)

        # Recompute shared memory: Triton's ttg.shared only counts the
        # reduction scratchpad (global_smem). The Apple GPU convert_layout
        # lowering adds __tg_cvt_* threadgroup globals whose sizes depend
        # on the tile configuration. Compute the real total from the LLVM IR
        # so the autotuner can reject configs that exceed the 32 KB limit.
        metadata["shared"] = _get_metalir_compile().tg_memory_bytes(llvm_ir)
        metadata["_llvm_ir"] = llvm_ir

        return llvm_mod

    # ── Stage 4: LLVM IR → metallib ───────────────────────────────────────
    def make_metallib(self, llvm_mod, metadata, options):
        llvm_ir = metadata.pop("_llvm_ir", None) or str(llvm_mod)

        # Extract kernel name (first defined void function = kernel entry)
        for line in llvm_ir.splitlines():
            if line.startswith('define void @'):
                metadata["name"] = line[len('define void @'):].split('(')[0]
                break

        debug = os.environ.get('TRITON_MPS_DEBUG')

        if debug:
            kname = metadata["name"]
            open(f'/tmp/dot_kernel_{kname}.ll', 'w').write(llvm_ir)

        # MetalIR C++ pipeline: LLVM IR → AIR transforms → v1 bitcode → metallib
        result = _get_metalir_compile()(llvm_ir)
        if debug:
            open(f'/tmp/dot_kernel_{kname}.metallib', 'wb').write(result)
        return result

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttgir"] = lambda src, meta: self.make_ttgir(src, meta, options)
        stages["llir"] = lambda src, meta: self.make_llir(src, meta, options)
        stages["metallib"] = lambda src, meta: self.make_metallib(
            src, meta, options)
