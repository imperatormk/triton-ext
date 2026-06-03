"""Metal-compatible stubs for CUDA libdevice functions.

Metal has no libdevice — these are Triton JIT composites that provide
the same API using tl.* and tl.math.* primitives.
"""
import triton
import triton.language as tl

# ── Direct mappings (tl.* → libdevice) ──────────────────────────────────

DIRECT = {
    'exp': tl.exp,
    'exp2': tl.exp2,
    'log': tl.log,
    'log2': tl.log2,
    'sin': tl.sin,
    'cos': tl.cos,
    'sqrt': tl.sqrt,
    'abs': tl.abs,
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
def _round_to_int(y):
    # Round-half-away-from-zero to the nearest integral float value.
    return tl.where(y >= 0.0, tl.floor(y + 0.5), tl.ceil(y - 0.5))


@triton.jit
def _pow_mag(ax, y):
    # |ax|**y computed with a compensated product to retain precision for
    # large exponents (Metal has no f64, so a plain exp(y*log(ax)) in f32 is
    # too lossy for e.g. x**1000). Work in base 2: ax**y = 2**(y*log2(ax)).
    # Compute t = y*log2(ax) as a (hi, lo) double-single pair so the rounding
    # error of the product is carried into the exponent, then evaluate
    # 2**t = 2**ki * 2**frac, keeping the exp2 argument small.
    lg = tl.log2(ax)
    hi = y * lg
    e = tl.math.fma(y, lg, -hi)  # exact rounding error of the product
    t = hi + e
    ki = _round_to_int(t)
    tf = (hi - ki) + e
    return tl.math.exp2(tf) * tl.math.exp2(ki)


@triton.jit
def _pow(x, y):
    # exp(y*log(x)) is only valid for x > 0. For x < 0 with an integer
    # exponent, pow is real-valued: |x|**y, negated when y is odd. For x < 0
    # with a non-integer exponent the result is NaN (IEEE / metal::pow). x == 0
    # is handled below via exp2(y*log2(0)) -> 0 / inf as expected.
    ax = tl.abs(x)
    mag = _pow_mag(ax, y)
    # Sign for negative base: -1 when y is an odd integer, +1 when even, NaN
    # when y is not an integer.
    y_rounded = _round_to_int(y)
    is_int = y_rounded == y
    # (-1)**y for integer y: odd -> -1, even -> +1.
    odd = (y_rounded * 0.5 - tl.floor(y_rounded * 0.5)) != 0.0
    neg_sign = tl.where(odd, -1.0, 1.0)
    nan = float("nan")
    neg_result = tl.where(is_int, neg_sign * mag, nan)
    return tl.where(x < 0.0, neg_result, mag)


@triton.jit
def _tan(x):
    return tl.sin(x) / tl.cos(x)


@triton.jit
def _tanh(x):
    return (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))


@triton.jit
def _atan(x):
    # Range-reduce to |x| <= 1 via atan(x) = pi/2 - atan(1/x), then a
    # minimax polynomial in z = x^2. Accurate to < 1e-6 over all reals.
    ax = tl.abs(x)
    inv = ax > 1.0
    z = tl.where(inv, 1.0 / ax, ax)
    z2 = z * z
    p = 0.0028662257
    p = -0.0161657367 + p * z2
    p = 0.0429096138 + p * z2
    p = -0.0752896400 + p * z2
    p = 0.1065626393 + p * z2
    p = -0.1420889944 + p * z2
    p = 0.1999355085 + p * z2
    p = -0.3333314528 + p * z2
    r = z + z * z2 * p
    r = tl.where(inv, 1.5707963267948966 - r, r)
    return tl.where(x < 0.0, -r, r)


@triton.jit
def _atan2(y, x):
    pi = 3.141592653589793
    halfpi = 1.5707963267948966
    return tl.where(
        x > 0, _atan(y / x),
        tl.where(x < 0,
                 _atan(y / x) + tl.where(y >= 0, pi, -pi),
                 tl.where(y > 0, halfpi, tl.where(y < 0, -halfpi, 0.0))))


@triton.jit
def _asin(x):
    # asin(x) = atan2(x, sqrt(1 - x^2)); sqrt arg clamped to >= 0 for |x|~1.
    return _atan2(x, tl.sqrt(tl.maximum(1.0 - x * x, 0.0)))


@triton.jit
def _acos(x):
    # acos(x) = atan2(sqrt(1 - x^2), x), full [-1, 1] domain, < 1e-6 accurate.
    return _atan2(tl.sqrt(tl.maximum(1.0 - x * x, 0.0)), x)


@triton.jit
def _fmod(x, y):
    # C fmod: remainder with quotient truncated toward zero (not floored, which
    # would be wrong for negative operands).
    q = x / y
    t = tl.where(q >= 0.0, tl.math.floor(q), tl.math.ceil(q))
    return x - t * y


@triton.jit
def _rint(x):
    # Round half to even (IEEE nearbyint / rint default mode).
    f = tl.math.floor(x)
    d = x - f
    # exact .5 -> round to even; otherwise round half up.
    up = tl.where(d == 0.5, (f - 2.0 * tl.math.floor(f * 0.5)) == 1.0, d > 0.5)
    return tl.where(up, f + 1.0, f)


@triton.jit
def _erfc(x):
    return 1.0 - tl.math.erf(x)


@triton.jit
def _expm1(x):
    return tl.exp(x) - 1.0


@triton.jit
def _erfcx(x):
    # Scaled complementary error function: exp(x^2) * erfc(x).
    return tl.exp(x * x) * (1.0 - tl.math.erf(x))


@triton.jit
def _lgamma_pos(x):
    # Lanczos approximation (g=5) for x > 0. Returns log|Gamma(x)|.
    c0 = 1.000000000190015
    c1 = 76.18009172947146
    c2 = -86.50532032941677
    c3 = 24.01409824083091
    c4 = -1.231739572450155
    c5 = 0.1208650973866179e-2
    c6 = -0.5395239384953e-5
    xm1 = x - 1.0
    ser = (c0 + c1 / (xm1 + 1.0) + c2 / (xm1 + 2.0) + c3 / (xm1 + 3.0) + c4 /
           (xm1 + 4.0) + c5 / (xm1 + 5.0) + c6 / (xm1 + 6.0))
    tmp = xm1 + 5.5
    return (xm1 + 0.5) * tl.log(tmp) - tmp + tl.log(2.5066282746310005 * ser)


@triton.jit
def _lgamma(x):
    # Reflection for x <= 0: lgamma(x) = log(pi/|sin(pi*x)|) - lgamma(1-x).
    pi = 3.141592653589793
    pos = _lgamma_pos(x)
    refl = tl.log(pi / tl.abs(tl.sin(pi * x))) - _lgamma_pos(1.0 - x)
    return tl.where(x > 0.0, pos, refl)


@triton.jit
def _erfinv(x):
    # Rational approximation (Giles 2010), single-precision accurate over
    # the open interval (-1, 1).
    w = -tl.log((1.0 - x) * (1.0 + x))
    w1 = w - 2.5
    p1 = 2.81022636e-08
    p1 = 3.43273939e-07 + p1 * w1
    p1 = -3.5233877e-06 + p1 * w1
    p1 = -4.39150654e-06 + p1 * w1
    p1 = 0.00021858087 + p1 * w1
    p1 = -0.00125372503 + p1 * w1
    p1 = -0.00417768164 + p1 * w1
    p1 = 0.246640727 + p1 * w1
    p1 = 1.50140941 + p1 * w1
    ws = tl.sqrt(w) - 3.0
    p2 = -0.000200214257
    p2 = 0.000100950558 + p2 * ws
    p2 = 0.00134934322 + p2 * ws
    p2 = -0.00367342844 + p2 * ws
    p2 = 0.00573950773 + p2 * ws
    p2 = -0.0076224613 + p2 * ws
    p2 = 0.00943887047 + p2 * ws
    p2 = 1.00167406 + p2 * ws
    p2 = 2.83297682 + p2 * ws
    p = tl.where(w < 5.0, p1, p2)
    return p * x


@triton.jit
def _trunc(x):
    return tl.where(x >= 0, tl.math.floor(x), tl.math.ceil(x))


@triton.jit
def _signbit(x):
    # Nonzero iff the sign bit is set. 1/(-0.0) = -inf < 0 catches -0.0 too.
    return ((x < 0.0) | (1.0 / x == float('-inf'))).to(tl.int32)


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
    'log1p': _log1p,
    'cbrt': _cbrt,
    'pow': _pow,
    'tan': _tan,
    'tanh': _tanh,
    'acos': _acos,
    'asin': _asin,
    'atan': _atan,
    'atan2': _atan2,
    'fmod': _fmod,
    'rint': _rint,
    'nearbyint': _rint,
    'llrint': _rint,
    'lrint': _rint,
    'erfc': _erfc,
    'erfcx': _erfcx,
    'expm1': _expm1,
    'lgamma': _lgamma,
    'erfinv': _erfinv,
    'trunc': _trunc,
    'signbit': _signbit,
    'isinf': _isinf,
    'isnan': _isnan,
    'finitef': _finitef,
    'isfinited': _finitef,
    'div_rz': _div_rz,
    'fast_tanh': _tanh,
    'fast_erf': DIRECT.get('erf', _tanh),
    'fast_gelu': _fast_gelu,
}

ALL_STUBS = {**DIRECT, **COMPOSITES}
