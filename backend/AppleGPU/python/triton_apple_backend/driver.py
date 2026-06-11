"""
Apple GPU Triton backend driver.

Dispatch pipeline:
  metallib bytes (from compiler.py make_metallib stage)
    → metal_utils.load_metallib(bytes)   [prebuilt extension, links against libtorch]
    → MetalLibrary.get_function(name)    → MetalKernel (PSO)
    → kernel(*tensors, threads=, group_size=)  [zero-copy via getMTLBufferStorage]
"""

import os as _os
import re as _re
import struct as _struct
import torch
from triton.backends.driver import DriverBase, decompose_descriptor, expand_signature
from triton.runtime.errors import OutOfResources
from triton.tools.tensor_descriptor import TensorDescriptor


def _load_metal_utils():
    """Load the prebuilt metal_utils extension (compiled at pip install time)."""
    from triton_apple_backend import metal_utils
    return metal_utils


# Use torch's native MPS shader API (torch.mps.load_metallib) instead of the
# bundled metal_utils.m launcher. The native path dispatches on torch's own MPS
# command stream, so kernels are timed by the same events inductor's autotuner
# records (fixing the "End event N was not recorded" benchmarker failure) and we
# shed the custom ObjC++ launcher. Set TRITON_MPS_NATIVE_DISPATCH=0 to fall back.
_USE_NATIVE_DISPATCH = _os.environ.get("TRITON_MPS_NATIVE_DISPATCH",
                                       "0") == "1"


def _native_load_metallib_available():
    return hasattr(torch, "mps") and hasattr(torch.mps, "load_metallib")


def _materialize_offline_error(metallib_bytes, name=None):
    """Replay a metallib through the offline Metal toolchain to surface the real
    lowering error behind an opaque in-process PSO failure.

    The Metal PSO compiler runs in the out-of-process MTLCompilerService, so the
    genuine LLVM/AIR backend error never reaches the NSError we catch (we only
    see "Failed to materializeAll" / "XPC_ERROR_CONNECTION_INTERRUPTED"). Feeding
    the SAME metallib to the offline tools reproduces the failure with the real
    diagnostic:
      - `metal-objdump -d` runs the bitcode VERIFIER and prints the precise error
        (e.g. "Explicit gep type does not match pointee type ..."). This is the
        most actionable signal, so it runs first.
      - `xcrun metallib` is the blunter cross-check ("Unexpected bitcode file").
    Returns a combined diagnostic str (always including the saved metallib path
    for manual inspection) or None if no tool is available. Best-effort: any
    failure here is swallowed so it never masks the original error.

    The failing metallib is saved to a STABLE, named location when
    METAL_PSO_FAIL_DIR is set (the test harness points it at the persisted dump
    base), so the exact failing config survives the run for offline triage even
    when inductor's per-worker tempdir is reaped. The filename embeds the kernel
    name; a `.txt` sidecar records the verifier error. Falls back to a tempfile.
    """
    import subprocess as _sp
    import tempfile as _tf

    def _run(argv):
        try:
            p = _sp.run(argv, capture_output=True, check=False, timeout=60)
            err = (p.stderr or b'').decode(errors='replace').strip()
            out = (p.stdout or b'').decode(errors='replace').strip()
            return p.returncode, (err or out)
        except (OSError, _sp.SubprocessError, ValueError):
            return None, None

    mlib = None
    fail_dir = _os.environ.get('METAL_PSO_FAIL_DIR')
    if fail_dir:
        try:
            import hashlib as _hl
            _os.makedirs(fail_dir, exist_ok=True)
            h = _hl.sha1(bytes(metallib_bytes)).hexdigest()[:12]
            stem = f"{name or 'kernel'}-{h}"
            mlib = _os.path.join(fail_dir, stem + '.metallib')
            with open(mlib, 'wb') as f:
                f.write(metallib_bytes)
        except (OSError, ValueError):
            mlib = None
    if mlib is None:
        try:
            with _tf.NamedTemporaryFile(suffix='.metallib', delete=False) as f:
                f.write(metallib_bytes)
                mlib = f.name
        except (OSError, ValueError):
            return None

    parts = [f"(metallib saved for inspection: {mlib})"]

    # metal-objdump: precise verifier diagnostic.
    rc, txt = _run(['xcrun', '-sdk', 'macosx', 'metal-objdump', '-d', mlib])
    if txt:
        parts.append(f"metal-objdump -d:\n{txt}")

    # xcrun metallib: blunt cross-check.
    rc2, txt2 = _run(
        ['xcrun', '-sdk', 'macosx', 'metallib', mlib, '-o', mlib + '.out'])
    try:
        _os.unlink(mlib + '.out')
    except OSError:
        pass
    if txt2:
        parts.append(f"xcrun metallib:\n{txt2}")

    if rc is None and rc2 is None:
        return None  # no offline toolchain available
    diagnostic = "\n".join(parts)
    # Persist a sidecar so the failing config is greppable after the run.
    if fail_dir and mlib and mlib.startswith(fail_dir):
        try:
            with open(mlib + '.txt', 'w') as f:
                f.write(f"kernel: {name}\n{diagnostic}\n")
        except OSError:
            pass
    return diagnostic


class _NativeKernel:
    """Adapts a native _mps_MetalKernel to the attribute the launcher reads.

    The launcher reads `max_total_threads_per_threadgroup`; the native kernel
    exposes `max_threads_per_threadgroup`. The call signature
    (`fn(*args, threads=, group_size=)`) is identical, so __call__ forwards.
    """

    def __init__(self, fn):
        self._fn = fn
        self.max_total_threads_per_threadgroup = getattr(
            fn, "max_threads_per_threadgroup", 1024)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "u1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "fp64": "double",
    }[ty]


# Scalar type → (struct.pack format char, byte size, alignment)
_SCALAR_PACK_INFO = {
    "i1": ("b", 1, 1),  # i1 stored as 1 byte
    "u1": ("B", 1, 1),  # u1 (unsigned boolean)
    "i8": ("b", 1, 1),
    "u8": ("B", 1, 1),
    "i16": ("h", 2, 2),
    "u16": ("H", 2, 2),
    "i32": ("i", 4, 4),
    "i64": ("q", 8, 8),
    "u32": ("I", 4, 4),
    "u64": ("Q", 8, 8),
    "fp16": ("e", 2, 2),
    # bf16 is 2 bytes but is NOT IEEE fp16: _pack_scalars handles it explicitly
    # (truncate the f32 bit pattern to the high 16 bits). The pack char below is
    # only used for its size/alignment (2), never for struct.pack of bf16.
    "bf16": ("e", 2, 2),
    "fp32": ("f", 4, 4),
    "fp64": ("d", 8, 8),
}


def _is_pointer_type(ty):
    """Check if a type string represents a pointer (tensor) argument."""
    return isinstance(ty, str) and ty.startswith('*')


def _compute_scalar_layout(scalar_types):
    """Compute packed buffer layout for scalar parameters.

    Returns (total_size, field_offsets) where field_offsets[i] is the byte
    offset for scalar i in the packed buffer.
    """
    offsets = []
    current = 0
    for ty in scalar_types:
        info = _SCALAR_PACK_INFO.get(ty)
        if info is None:
            raise ValueError(f"Unknown scalar type for packing: {ty}")
        _, size, align = info
        padding = (align - (current % align)) % align
        current += padding
        offsets.append(current)
        current += size
    return current, offsets


def _pack_scalars(scalar_types, scalar_values, total_size, offsets):
    """Pack scalar values into a bytes buffer with natural alignment."""
    buf = bytearray(total_size)
    for ty, val, offset in zip(scalar_types, scalar_values, offsets):
        fmt, size, _ = _SCALAR_PACK_INFO[ty]
        if ty in ("i1", "u1"):
            val = 1 if val else 0
        elif ty == "bf16":
            # bf16: convert float → bf16 bits using IEEE754 layout
            bits = _struct.unpack('<I', _struct.pack('<f', float(val)))[0]
            bf16_bits = bits >> 16
            _struct.pack_into("H", buf, offset, bf16_bits)
            continue
        _struct.pack_into(fmt, buf, offset, val)
    return bytes(buf)


class MPSUtils:
    """
    Metal GPU utils — JIT-compiles metal_utils.m (links against libtorch)
    for zero-copy MPS tensor dispatch. Works with any PyTorch 2.0+.
    """

    def __init__(self):
        self._native = _USE_NATIVE_DISPATCH and _native_load_metallib_available(
        )
        self._metal = None if self._native else _load_metal_utils()

    def load_binary(self, name, metallib_bytes, shared_mem, device):
        """
        Returns (module, function, n_regs, n_spills, n_max_threads).
        """
        try:
            if self._native:
                module = torch.mps.load_metallib(bytes(metallib_bytes))
                function = _NativeKernel(getattr(module, name))
            else:
                module = self._metal.load_metallib(bytes(metallib_bytes))
                function = module.get_function(name)
            # Report the PSO's real maxTotalThreadsPerThreadgroup as n_max_threads.
            # Triton (compiler.py) rejects configs where num_warps*warp_size
            # exceeds this via OutOfResources, letting the autotuner drop them.
            # This MUST be accurate: a kernel using cross-warp threadgroup memory
            # (e.g. a multi-warp scan/reduction) is only correct when ALL its
            # warps launch. If register pressure caps the PSO below the required
            # thread count, silently launching fewer threads leaves some warps'
            # threadgroup-memory slots unwritten -> uninitialized reads -> racy,
            # nondeterministic results. Reporting the true max forces such a
            # config to be discarded instead of producing wrong answers.
            max_threads = getattr(function,
                                  'max_total_threads_per_threadgroup', 1024)
            return module, function, 0, 0, max_threads
        except RuntimeError as e:
            msg = str(e)
            m = _re.search(
                r'Threadgroup (?:memory )?size \((\d+)\) exceeds the maximum .+ \((\d+)\)',
                msg)
            if m:
                raise OutOfResources(int(m.group(1)), int(m.group(2)),
                                     "Metal PSO") from e
            # Over-budget register/stack footprint: skip the config (like an OOM)
            # instead of hard-failing the compile. See test_large_block_sizes.
            if 'exceeds available stack space' in msg:
                raise OutOfResources(0, 0, "Metal PSO stack space") from e
            # Opaque PSO-compile errors: the real LLVM/AIR lowering diagnostic
            # dies in the out-of-process MTLCompilerService and never reaches
            # this NSError, leaving only blunt strings like "Failed to
            # materializeAll" or "XPC_ERROR_CONNECTION_INTERRUPTED". Replay the
            # exact metallib through the offline Metal toolchain (metal-objdump,
            # which prints the precise verifier error, plus xcrun metallib),
            # and rethrow with the real backend error inlined.
            _opaque = ('materializeAll', 'XPC_ERROR_CONNECTION_INTERRUPTED',
                       'XPC_CONNECTION_INTERRUPTED', 'PSO creation failed',
                       'Unexpected bitcode')
            if any(s in msg for s in _opaque):
                detail = _materialize_offline_error(bytes(metallib_bytes), name)
                if detail:
                    raise RuntimeError(f"{msg}\n\n"
                                       f"offline Metal toolchain diagnostic:\n"
                                       f"{detail}") from e
            raise

    def unload_module(self, module):
        del module

    def get_device_properties(self, device):
        return {
            "warpSize":
            32,
            "max_shared_mem":
            32768,
            "multiprocessorCount":
            getattr(torch._C, '_mps_get_core_count', lambda: 10)(),
        }

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        pass  # single MPS device

    def get_current_stream(self, device):
        return 0  # MPS manages its own stream internally


class MPSLauncher:
    """
    Called by Triton's JIT runtime to dispatch a compiled kernel.
    `function` = _mps_MetalKernel returned by load_binary.
    """

    def __init__(self, src, metadata):
        self.signature = dict(src.signature)
        self.constants = getattr(src, "constants", {})

        # Constexpr args appear in Python *args but NOT in the compiled IR.
        # We strip them before passing to _mps_MetalKernel so Metal buffer slots
        # match IR arg positions exactly. self.constexpr_py_slots is the set of
        # Python *args indices that are constexpr (to be stripped at launch).
        self.constexpr_py_slots = frozenset(
            i for i, (k, ty) in enumerate(self.signature.items())
            if ty == 'constexpr')

        # Expand tensor descriptor types into flat scalar types.
        # MPS has no hardware TMA, so tensordesc_meta is always None —
        # descriptors are decomposed to (ptr, *shape, *strides, padding, tf32, *shape, *strides).
        non_constexpr_sig = [
            ty for ty in self.signature.values() if ty != 'constexpr'
        ]
        expanded = expand_signature(non_constexpr_sig, None, None)

        # Classify each expanded arg as pointer or scalar.
        # All scalars are packed into ONE device buffer,
        # so the IR param order is: [pointers..., packed_scalar_buf, system_values].
        # We need to separate pointers from scalars at launch time.
        #
        # Python tuple kernel args are flattened recursively: an empty tuple
        # contributes nothing; nested tuples expand to their leaf elements.
        # Leaf entries with type 'constexpr' (inlined constants inside tuples)
        # are skipped — they have no GPU arg slot.
        #
        # self._flat_arg_keep[i] says whether flat_arg[i] (after full tuple
        # flattening in __call__) should be forwarded to the GPU.  We need this
        # because constexpr values inside tuples ARE present in flat_args but
        # must NOT be passed to the kernel.
        self.ptr_indices = []  # indices into the KEPT slice of flat_args
        self.scalar_indices = []  # indices into the KEPT slice of flat_args
        self.scalar_types = []  # type strings for scalars (for packing)

        def _flatten_types_with_mask(types):
            """Recursively flatten tuple types; return (all_types, keep_mask).

            all_types: leaf type for every position (including constexpr)
            keep_mask: True where the leaf is a real GPU arg (not constexpr)
            """
            all_tys, keep = [], []
            for ty in types:
                if isinstance(ty, tuple):
                    sub_tys, sub_keep = _flatten_types_with_mask(ty)
                    all_tys.extend(sub_tys)
                    keep.extend(sub_keep)
                else:
                    all_tys.append(ty)
                    keep.append(ty != 'constexpr')
            return all_tys, keep

        all_types, keep_mask = _flatten_types_with_mask(expanded)
        self._flat_arg_keep = keep_mask  # used in __call__ to filter flat_args

        kept_slot = 0
        for ty, keep in zip(all_types, keep_mask):
            if not keep:
                continue
            if _is_pointer_type(ty):
                self.ptr_indices.append(kept_slot)
            else:
                self.scalar_indices.append(kept_slot)
                self.scalar_types.append(ty)
            kept_slot += 1

        # Pre-compute packed buffer layout
        if self.scalar_types:
            self.total_size, self.field_offsets = _compute_scalar_layout(
                self.scalar_types)
        else:
            self.total_size = 0
            self.field_offsets = []

        warp_size = 32
        self._requested_threads = getattr(metadata, "num_warps", 4) * warp_size
        self.ly = 1
        self.lz = 1

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *args):

        # The kernel requires exactly num_warps*warp_size threads; with fewer,
        # cross-warp threadgroup-memory cooperation breaks (unwritten smem slots
        # -> uninitialized reads -> nondeterministic results). load_binary
        # reports the PSO's real maxTotalThreadsPerThreadgroup so triton drops
        # any config that needs more threads than the PSO supports. If we ever
        # reach dispatch with a deficit, fail loudly instead of silently capping
        # and returning wrong answers.
        max_threads = getattr(function, 'max_total_threads_per_threadgroup',
                              1024)
        if self._requested_threads > max_threads:
            raise RuntimeError(
                f"kernel needs {self._requested_threads} threads/threadgroup "
                f"but PSO supports only {max_threads}; this config should have "
                f"been rejected at load_binary (OutOfResources)")
        self.lx = self._requested_threads

        if launch_enter_hook:
            launch_enter_hook(launch_metadata)

        # Strip constexpr args and decompose TensorDescriptors.
        # Python tuple args are flattened recursively so that the positional
        # index structure matches _flat_arg_keep built in __init__.
        # After full flattening, positions with keep=False (constexpr values
        # inside tuples) are dropped before indexing with ptr_indices /
        # scalar_indices.
        from triton.runtime.jit import TensorWrapper

        def _flatten_arg(a, out):
            """Recursively flatten an arg value, expanding tuples to leaves."""
            if isinstance(a, TensorWrapper):
                out.append(a.base)
            elif isinstance(a, torch.Tensor):
                # Pass tensors (including views) through unchanged. Do NOT unwrap
                # to `a._base`: for a view like `base[4:20]`, `._base` is the
                # offset-0 base tensor, which drops the storage_offset and makes
                # the kernel read from the wrong location. The launcher applies
                # storage_offset() itself via setBuffer:offset:.
                out.append(a)
            elif isinstance(a, TensorDescriptor):
                out.extend(decompose_descriptor(a))
            elif isinstance(a, tuple):
                for elem in a:
                    _flatten_arg(elem, out)
            else:
                out.append(a)

        all_flat_args = []
        for i, a in enumerate(args):
            if i in self.constexpr_py_slots:
                continue
            _flatten_arg(a, all_flat_args)

        # Apply keep mask: drop constexpr-inside-tuple positions.
        # The flattened runtime args must line up 1:1 with the keep mask built
        # from the signature in __init__; a mismatch means zip() would silently
        # truncate and pass the wrong buffers. Fail loudly instead.
        if len(all_flat_args) != len(self._flat_arg_keep):
            raise RuntimeError(
                f"flat arg count {len(all_flat_args)} does not match signature "
                f"keep-mask length {len(self._flat_arg_keep)}; kernel call "
                f"signature is out of sync with the compiled IR")
        flat_args = [
            v for v, keep in zip(all_flat_args, self._flat_arg_keep) if keep
        ]

        # Separate pointer args from scalar args.
        # IR param order after Pass 5b: [ptr0, ptr1, ..., packed_scalar_buf]
        ptr_args = [flat_args[i] for i in self.ptr_indices]
        scalar_values = [flat_args[i] for i in self.scalar_indices]

        if scalar_values:
            # Pack all scalars into a small MPS tensor (acts as device buffer)
            packed_bytes = _pack_scalars(self.scalar_types, scalar_values,
                                         self.total_size, self.field_offsets)
            scalar_buf = torch.frombuffer(bytearray(packed_bytes),
                                          dtype=torch.uint8).to('mps')
            reordered_args = tuple(ptr_args) + (scalar_buf, )
        else:
            reordered_args = tuple(ptr_args)

        if _os.environ.get('TRITON_MPS_DEBUG'):
            _threads = [gridX * self.lx, gridY * self.ly, gridZ * self.lz]
            _gs = [self.lx, self.ly, self.lz]
            print(
                f'[MPS] threads={_threads} group_size={_gs} grid=({gridX},{gridY},{gridZ})'
            )
            print(f'[MPS] reordered_args={reordered_args}')
            if scalar_values:
                print(
                    f'[MPS] scalar_types={self.scalar_types} scalar_values={scalar_values}'
                )
                print(
                    f'[MPS] packed_bytes={packed_bytes.hex()} total_size={self.total_size}'
                )
        function(
            *reordered_args,
            threads=[gridX * self.lx, gridY * self.ly, gridZ * self.lz],
            group_size=[self.lx, self.ly, self.lz],
        )

        if launch_exit_hook:
            launch_exit_hook(launch_metadata)


class MPSDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = MPSUtils()
        self.launcher_cls = MPSLauncher

    @staticmethod
    def is_active():
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_device_interface(self):
        return torch.mps

    def get_current_target(self):
        from triton.backends.compiler import GPUTarget
        return GPUTarget("mps", "apple_m", 32)

    def get_active_torch_device(self):
        return torch.device("mps", 0)

    def get_current_device(self):
        return 0

    def get_current_stream(self, device):
        return 0  # MPS manages its own stream internally

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        return torch.empty(256 * 1024 * 1024 // 4,
                           dtype=torch.int32,
                           device='mps')

    def clear_cache(self, cache):
        cache.zero_()
