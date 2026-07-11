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
    # |ax|**y via compensated product (Metal has no f64; plain f32
    # exp(y*log(ax)) is too lossy for large exponents). Base 2:
    # ax**y = 2**(y*log2(ax)), with t = y*log2(ax) as a (hi, lo) double-single
    # pair, then 2**t = 2**ki * 2**frac to keep the exp2 argument small.
    lg = tl.log2(ax)
    hi = y * lg
    e = tl.math.fma(y, lg, -hi)  # exact rounding error of the product
    t = hi + e
    ki = _round_to_int(t)
    tf = (hi - ki) + e
    return tl.math.exp2(tf) * tl.math.exp2(ki)


@triton.jit
def _pow(x, y):
    # exp(y*log(x)) is only valid for x > 0. x < 0: |x|**y, negated for odd
    # integer y, NaN for non-integer y (IEEE / metal::pow). x == 0 handled via
    # exp2(y*log2(0)) -> 0 / inf.
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


@triton.jit
def _cyl_bessel_i0(x):
    # Modified Bessel function I0, Abramowitz & Stegun 9.8.1 / 9.8.2.
    # Even; work in |x|. Split at 3.75: polynomial in (x/3.75)^2 for small,
    # asymptotic exp(ax)/sqrt(ax) * poly(3.75/ax) for large. |err| < ~1.6e-7.
    ax = tl.abs(x)
    small = ax < 3.75
    # --- small branch: t = (x/3.75)^2 ---
    ts = (ax / 3.75) * (ax / 3.75)
    ps = 0.0045813
    ps = 0.0360768 + ps * ts
    ps = 0.2659732 + ps * ts
    ps = 1.2067492 + ps * ts
    ps = 3.0899424 + ps * ts
    ps = 3.5156229 + ps * ts
    ps = 1.0 + ps * ts
    # --- large branch: t = 3.75/|x| ---
    # guard the division so the unused branch never sees ax==0 (NaN poisons
    # tl.where even on the not-taken side).
    axl = tl.where(small, 3.75, ax)
    tl_ = 3.75 / axl
    pl = 0.00392377
    pl = -0.01647633 + pl * tl_
    pl = 0.02635537 + pl * tl_
    pl = -0.02057706 + pl * tl_
    pl = 0.00916281 + pl * tl_
    pl = -0.00157565 + pl * tl_
    pl = 0.00225319 + pl * tl_
    pl = 0.01328592 + pl * tl_
    pl = 0.39894228 + pl * tl_
    large = (tl.exp(axl) / tl.sqrt(axl)) * pl
    return tl.where(small, ps, large)


@triton.jit
def _cyl_bessel_i1(x):
    # Modified Bessel function I1, Abramowitz & Stegun 9.8.3 / 9.8.4.
    # I1 is odd: compute for |x| then restore the sign. Same 3.75 split.
    ax = tl.abs(x)
    small = ax < 3.75
    # --- small branch: t = (x/3.75)^2, result is ax * poly(t) ---
    ts = (ax / 3.75) * (ax / 3.75)
    ps = 0.00032411
    ps = 0.00301532 + ps * ts
    ps = 0.02658733 + ps * ts
    ps = 0.15084934 + ps * ts
    ps = 0.51498869 + ps * ts
    ps = 0.87890594 + ps * ts
    ps = 0.5 + ps * ts
    small_val = ax * ps
    # --- large branch: t = 3.75/|x|, result is exp(ax)/sqrt(ax) * poly(t) ---
    axl = tl.where(small, 3.75, ax)
    tl_ = 3.75 / axl
    pl = -0.00420059
    pl = 0.01787654 + pl * tl_
    pl = -0.02895312 + pl * tl_
    pl = 0.02282967 + pl * tl_
    pl = -0.01031555 + pl * tl_
    pl = 0.00163801 + pl * tl_
    pl = -0.00362018 + pl * tl_
    pl = -0.03988024 + pl * tl_
    pl = 0.39894228 + pl * tl_
    large_val = (tl.exp(axl) / tl.sqrt(axl)) * pl
    mag = tl.where(small, small_val, large_val)
    # restore odd sign: I1(-x) = -I1(x)
    return tl.where(x < 0.0, -mag, mag)


@triton.jit
def _hypot(x, y):
    return tl.sqrt(x * x + y * y)


@triton.jit
def _copysign(x, y):
    ax = tl.abs(x)
    neg = (y < 0.0) | (1.0 / y == float('-inf'))
    return tl.where(neg, -ax, ax)


@triton.jit
def _j0(x):
    # Bessel J0, Abramowitz & Stegun 9.4.1 / 9.4.3 (Numerical Recipes form).
    # Even function. Small |x|<8: rational polynomial. Large: amplitude/phase
    # asymptotic. Single-precision accurate.
    ax = tl.abs(x)
    small = ax < 8.0
    # --- small branch: rational in y = x^2 ---
    y = x * x
    p1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y *
                                                    (-11214424.18 + y *
                                                     (77392.33017 + y *
                                                      (-184.9052456)))))
    q1 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y *
                                                  (59272.64853 + y *
                                                   (267.8532712 + y))))
    small_val = p1 / q1
    # --- large branch: xx = |x| - 0.785398164 ---
    axl = tl.where(small, 8.0, ax)
    z = 8.0 / axl
    y2 = z * z
    xx = axl - 0.785398164
    pa = 1.0 + y2 * (-0.1098628627e-2 + y2 *
                     (0.2734510407e-4 + y2 *
                      (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
    pb = -0.1562499995e-1 + y2 * (0.1430488765e-3 + y2 *
                                  (-0.6911147651e-5 + y2 *
                                   (0.7621095161e-6 + y2 * (-0.934935152e-7))))
    large_val = tl.sqrt(
        0.636619772 / axl) * (tl.cos(xx) * pa - z * tl.sin(xx) * pb)
    return tl.where(small, small_val, large_val)


@triton.jit
def _j1(x):
    # Bessel J1, A&S 9.4.4 / 9.4.6. Odd function.
    ax = tl.abs(x)
    small = ax < 8.0
    y = x * x
    p1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y *
                                                        (-2972611.439 + y *
                                                         (15704.48260 + y *
                                                          (-30.16036606))))))
    q1 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y *
                                                   (99447.43394 + y *
                                                    (376.9991397 + y))))
    small_val = p1 / q1
    axl = tl.where(small, 8.0, ax)
    z = 8.0 / axl
    y2 = z * z
    xx = axl - 2.356194491
    pa = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 + y2 *
                                         (0.2457520174e-5 + y2 *
                                          (-0.240337019e-6))))
    pb = 0.04687499995 + y2 * (-0.2002690873e-3 + y2 *
                               (0.8449199096e-5 + y2 *
                                (-0.88228987e-6 + y2 * 0.105787412e-6)))
    mag = tl.sqrt(0.636619772 / axl) * (tl.cos(xx) * pa - z * tl.sin(xx) * pb)
    large_val = tl.where(x < 0.0, -mag, mag)
    return tl.where(small, small_val, large_val)


@triton.jit
def _y0(x):
    # Bessel Y0 (second kind), A&S 9.4.2 / 9.4.3. Defined for x > 0; singular
    # at 0 (-> -inf). Small x<8: rational + (2/pi) * J0(x) * ln(x). Large:
    # amplitude/phase asymptotic.
    small = x < 8.0
    y = x * x
    p1 = -2957821389.0 + y * (7062834065.0 + y *
                              (-512359803.6 + y *
                               (10879881.29 + y *
                                (-86327.92757 + y * 228.4622733))))
    q1 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y *
                                                 (47447.26470 + y *
                                                  (226.1030244 + y))))
    # ln(x) guarded so the unused (large) branch never evaluates ln of a big x
    # in a way that poisons the where; x>0 by domain so abs is just safety.
    xs = tl.where(small, x, 1.0)
    small_val = (p1 / q1) + 0.636619772 * _j0(x) * tl.log(xs)
    xl = tl.where(small, 8.0, x)
    z = 8.0 / xl
    y2 = z * z
    xx = xl - 0.785398164
    pa = 1.0 + y2 * (-0.1098628627e-2 + y2 *
                     (0.2734510407e-4 + y2 *
                      (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
    pb = -0.1562499995e-1 + y2 * (0.1430488765e-3 + y2 *
                                  (-0.6911147651e-5 + y2 *
                                   (0.7621095161e-6 + y2 * (-0.934935152e-7))))
    large_val = tl.sqrt(
        0.636619772 / xl) * (tl.sin(xx) * pa + z * tl.cos(xx) * pb)
    return tl.where(small, small_val, large_val)


@triton.jit
def _y1(x):
    # Bessel Y1 (second kind), A&S 9.4.5 / 9.4.6. x > 0.
    small = x < 8.0
    y = x * x
    p1 = x * (-0.4900604943e13 + y *
              (0.1275274390e13 + y *
               (-0.5153438139e11 + y *
                (0.7349264551e9 + y *
                 (-0.4237922726e7 + y * 0.8511937935e4)))))
    q1 = 0.2499580570e14 + y * (0.4244419664e12 + y *
                                (0.3733650367e10 + y *
                                 (0.2245904002e8 + y *
                                  (0.1020426050e6 + y *
                                   (0.3549632885e3 + y)))))
    xs = tl.where(small, x, 1.0)
    small_val = (p1 / q1) + 0.636619772 * (_j1(x) * tl.log(xs) - 1.0 / xs)
    xl = tl.where(small, 8.0, x)
    z = 8.0 / xl
    y2 = z * z
    xx = xl - 2.356194491
    pa = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 + y2 *
                                         (0.2457520174e-5 + y2 *
                                          (-0.240337019e-6))))
    pb = 0.04687499995 + y2 * (-0.2002690873e-3 + y2 *
                               (0.8449199096e-5 + y2 *
                                (-0.88228987e-6 + y2 * 0.105787412e-6)))
    large_val = tl.sqrt(
        0.636619772 / xl) * (tl.sin(xx) * pa + z * tl.cos(xx) * pb)
    return tl.where(small, small_val, large_val)


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
    'hypot': _hypot,
    'copysign': _copysign,
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
    'cyl_bessel_i0': _cyl_bessel_i0,
    'cyl_bessel_i1': _cyl_bessel_i1,
    'j0': _j0,
    'j1': _j1,
    'y0': _y0,
    'y1': _y1,
}

ALL_STUBS = {**DIRECT, **COMPOSITES}
