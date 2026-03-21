"""Metal-compatible stubs for CUDA libdevice functions.

Metal has no libdevice — these are Triton JIT composites that provide
the same API using tl.* and tl.math.* primitives.
"""
import triton
import triton.language as tl

# ── Direct mappings (tl.* → libdevice) ──────────────────────────────────

DIRECT = {
    'exp': tl.exp, 'exp2': tl.exp2, 'log': tl.log, 'log2': tl.log2,
    'sin': tl.sin, 'cos': tl.cos, 'sqrt': tl.sqrt, 'abs': tl.abs,
    'fabs': tl.abs,
}

# tl.math.* that exist on this triton version
for _name in ['floor', 'ceil', 'rsqrt', 'erf', 'exp2', 'log2', 'div_rn']:
    _fn = getattr(tl.math, _name, None)
    if _fn is not None:
        DIRECT[_name] = _fn


# ── Composite stubs ─────────────────────────────────────────────────────

@triton.jit
def _log1p(x):
    return tl.log(1.0 + x)

@triton.jit
def _cbrt(x):
    return tl.exp(tl.log(x) / 3.0)

@triton.jit
def _pow(x, y):
    return tl.exp(y * tl.log(x))

@triton.jit
def _tan(x):
    return tl.sin(x) / tl.cos(x)

@triton.jit
def _tanh(x):
    return (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))

@triton.jit
def _acos(x):
    return 1.5707963 - x * (1.0 + x * x * (-0.1666666 + x * x * 0.075))

@triton.jit
def _atan(x):
    return x / (1.0 + 0.28125 * x * x)

@triton.jit
def _atan2(y, x):
    return tl.where(x > 0, _atan(y / x),
           tl.where(x < 0, _atan(y / x) + tl.where(y >= 0, 3.14159265, -3.14159265),
           tl.where(y > 0, 1.5707963, tl.where(y < 0, -1.5707963, 0.0))))

@triton.jit
def _fmod(x, y):
    return x - tl.math.floor(x / y) * y

@triton.jit
def _rint(x):
    return tl.math.floor(x + 0.5)

@triton.jit
def _trunc(x):
    return tl.where(x >= 0, tl.math.floor(x), tl.math.ceil(x))

@triton.jit
def _isinf(x):
    return (x == float('inf')) | (x == float('-inf'))

@triton.jit
def _isnan(x):
    return x != x

@triton.jit
def _finitef(x):
    return (x == x) & (x != float('inf')) & (x != float('-inf'))

@triton.jit
def _div_rz(x, y):
    return _trunc(x / y)

@triton.jit
def _fast_gelu(x):
    return 0.5 * x * (1.0 + _tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

COMPOSITES = {
    'log1p': _log1p, 'cbrt': _cbrt, 'pow': _pow, 'tan': _tan,
    'tanh': _tanh, 'acos': _acos, 'atan': _atan, 'atan2': _atan2,
    'fmod': _fmod, 'rint': _rint, 'trunc': _trunc,
    'isinf': _isinf, 'isnan': _isnan,
    'finitef': _finitef, 'isfinited': _finitef,
    'div_rz': _div_rz,
    'fast_tanh': _tanh, 'fast_erf': DIRECT.get('erf', _tanh),
    'fast_gelu': _fast_gelu,
}

ALL_STUBS = {**DIRECT, **COMPOSITES}
