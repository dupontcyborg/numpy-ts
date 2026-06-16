//! Shared vectorized transcendental cores (f64, ≈1 ulp) + complex drivers.
//!
//! Single source of truth for the f64 polynomial cores reused across the
//! exp / exp2 / sin / cos / log kernels and their complex variants:
//!   sinv_f64 / cosv_f64   — Cephes sin/cos (Cody-Waite reduction + quadrant)
//!   expv_f64 / sinhcosh   — Cephes exp + hyperbolic pair
//!   logv_f64              — Cephes natural log (raw; domain handling is per-op)
//!   atanv_f64 / atan2v    — Cephes atan / atan2
//! plus cdrive_c128 / cdrive_c64 — interleaved-complex map drivers.
//!
//! All FMAs lower to f64x2.relaxed_madd on +relaxed_simd builds. The f32 cores
//! (expv_f32, logv_f32, exp2v_*) live in their op files since they are not
//! shared — complex routes through these f64 cores then narrows.

const math = @import("std").math;
const simd = @import("simd.zig");

// ---------------------------------------------------------------------------
// sin / cos (Cephes, 3-part Cody-Waite reduction)
// ---------------------------------------------------------------------------

const TWO_OVER_PI: f64 = 0.6366197723675813430755; // 2/π
const PIO2_1: f64 = 1.57079632673412561417e+00;
const PIO2_2: f64 = 6.07710050650619224932e-11;
const PIO2_3: f64 = 2.02226624879595063154e-21;

/// sin polynomial on r ∈ [-π/4, π/4]: r + r·z·S(z), z = r².
inline fn sinPoly(r: simd.V2f64, z: simd.V2f64) simd.V2f64 {
    var s: simd.V2f64 = @splat(1.58962301576546568060e-10);
    s = simd.mulAdd_f64x2(s, z, @splat(-2.50507477628578072866e-8));
    s = simd.mulAdd_f64x2(s, z, @splat(2.75573136213857245213e-6));
    s = simd.mulAdd_f64x2(s, z, @splat(-1.98412698295895385996e-4));
    s = simd.mulAdd_f64x2(s, z, @splat(8.33333333332211858878e-3));
    s = simd.mulAdd_f64x2(s, z, @splat(-1.66666666666666307295e-1));
    return simd.mulAdd_f64x2(r * z, s, r);
}

/// cos polynomial on r ∈ [-π/4, π/4]: 1 - 0.5·z + z²·C(z), z = r².
inline fn cosPoly(z: simd.V2f64) simd.V2f64 {
    var c: simd.V2f64 = @splat(-1.13585365213876817300e-11);
    c = simd.mulAdd_f64x2(c, z, @splat(2.08757008419747316778e-9));
    c = simd.mulAdd_f64x2(c, z, @splat(-2.75573141792967388112e-7));
    c = simd.mulAdd_f64x2(c, z, @splat(2.48015872888517045348e-5));
    c = simd.mulAdd_f64x2(c, z, @splat(-1.38888888888730564116e-3));
    c = simd.mulAdd_f64x2(c, z, @splat(4.16666666666665929218e-2));
    const base = simd.nmulAdd_f64x2(@splat(0.5), z, @splat(1.0)); // 1 - 0.5·z
    return simd.mulAdd_f64x2(z * z, c, base);
}

/// Reduce x to (q, r): x ≈ q·(π/2) + r, q integer, r ∈ [-π/4, π/4].
inline fn reduce(x: simd.V2f64, q_out: *simd.V2i64) simd.V2f64 {
    const fq = @floor(simd.mulAdd_f64x2(x, @as(simd.V2f64, @splat(TWO_OVER_PI)), @as(simd.V2f64, @splat(0.5))));
    q_out.* = @intFromFloat(fq);
    var r = simd.nmulAdd_f64x2(fq, @splat(PIO2_1), x);
    r = simd.nmulAdd_f64x2(fq, @splat(PIO2_2), r);
    r = simd.nmulAdd_f64x2(fq, @splat(PIO2_3), r);
    return r;
}

/// Vectorized sin for a 2-wide f64 lane.
pub inline fn sinv_f64(x: simd.V2f64) simd.V2f64 {
    var q: simd.V2i64 = undefined;
    const r = reduce(x, &q);
    const z = r * r;
    const sp = sinPoly(r, z);
    const cp = cosPoly(z);

    const swap = (q & @as(simd.V2i64, @splat(1))) != @as(simd.V2i64, @splat(0));
    const base = @select(f64, swap, cp, sp);
    const neg = (q & @as(simd.V2i64, @splat(2))) != @as(simd.V2i64, @splat(0));
    return @select(f64, neg, -base, base);
}

/// Vectorized cos for a 2-wide f64 lane.
pub inline fn cosv_f64(x: simd.V2f64) simd.V2f64 {
    var q: simd.V2i64 = undefined;
    const r = reduce(x, &q);
    const z = r * r;
    const sp = sinPoly(r, z);
    const cp = cosPoly(z);

    const swap = (q & @as(simd.V2i64, @splat(1))) != @as(simd.V2i64, @splat(0));
    const base = @select(f64, swap, sp, cp);
    const neg = ((q +% @as(simd.V2i64, @splat(1))) & @as(simd.V2i64, @splat(2))) != @as(simd.V2i64, @splat(0));
    return @select(f64, neg, -base, base);
}

// ---------------------------------------------------------------------------
// exp (Cephes, degree-13 Taylor) + sinh/cosh
// ---------------------------------------------------------------------------

const EXP_LOG2E: f64 = 1.4426950408889634073599;
const EXP_C1: f64 = 6.93145751953125e-1; // ln2 high
const EXP_C2: f64 = 1.42860682030941723212e-6; // ln2 low
const EXP_MAXLOG: f64 = 7.09782712893383996843e2;
const EXP_MINLOG: f64 = -7.08396418532264106224e2;

/// Vectorized exp for a 2-wide f64 lane.
pub inline fn expv_f64(x_in: simd.V2f64) simd.V2f64 {
    const maxv: simd.V2f64 = @splat(EXP_MAXLOG);
    const minv: simd.V2f64 = @splat(EXP_MINLOG);
    const x = simd.max_f64x2(simd.min_f64x2(x_in, maxv), minv);

    const n = @floor(simd.mulAdd_f64x2(x, @as(simd.V2f64, @splat(EXP_LOG2E)), @as(simd.V2f64, @splat(0.5))));
    var r = simd.nmulAdd_f64x2(n, @splat(EXP_C1), x);
    r = simd.nmulAdd_f64x2(n, @splat(EXP_C2), r);

    var expr: simd.V2f64 = @splat(1.6059043836821613e-10);
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.08767569878681e-9));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.5052108385441720e-8));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.7557319223985893e-7));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.7557319223985888e-6));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.4801587301587302e-5));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.9841269841269841e-4));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.3888888888888889e-3));
    expr = simd.mulAdd_f64x2(expr, r, @splat(8.3333333333333332e-3));
    expr = simd.mulAdd_f64x2(expr, r, @splat(4.1666666666666664e-2));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.6666666666666666e-1));
    expr = simd.mulAdd_f64x2(expr, r, @splat(5.0e-1));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.0));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.0));

    const ni: simd.V2i64 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V2i64, @splat(1023))) << @as(simd.V2i64, @splat(52));
    const pow2n: simd.V2f64 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f64, x_in > maxv, @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x_in < minv, @as(simd.V2f64, @splat(0.0)), result);
    return result;
}

/// cosh/sinh pair for a 2-wide f64 lane (via exp(±x)).
pub inline fn sinhcosh_f64(x: simd.V2f64, sinh_out: *simd.V2f64, cosh_out: *simd.V2f64) void {
    const ex = expv_f64(x);
    const enx = expv_f64(-x);
    const half: simd.V2f64 = @splat(0.5);
    cosh_out.* = (ex + enx) * half;
    sinh_out.* = (ex - enx) * half;
}

// ---------------------------------------------------------------------------
// log (Cephes, raw natural log on positive reals; domain handling is per-op)
// ---------------------------------------------------------------------------

const LOG_SQRTH: f64 = 0.70710678118654752440;
const LOG_LN2_HI: f64 = 0.693359375;
const LOG_LN2_LO: f64 = -2.121944400546905827679e-4;

pub inline fn logv_f64(x: simd.V2f64) simd.V2f64 {
    const bits: simd.V2u64 = @bitCast(x);
    const raw_e: simd.V2u64 = (bits >> @as(simd.V2u64, @splat(52))) & @as(simd.V2u64, @splat(0x7ff));
    const ei: simd.V2i64 = @as(simd.V2i64, @bitCast(raw_e)) -% @as(simd.V2i64, @splat(1022));
    const mant_bits = (bits & @as(simd.V2u64, @splat(0x000fffffffffffff))) |
        @as(simd.V2u64, @splat(0x3fe0000000000000));
    var m: simd.V2f64 = @bitCast(mant_bits);
    var e: simd.V2f64 = @floatFromInt(ei);

    const lo = m < @as(simd.V2f64, @splat(LOG_SQRTH));
    e = @select(f64, lo, e - @as(simd.V2f64, @splat(1.0)), e);
    m = @select(f64, lo, m + m - @as(simd.V2f64, @splat(1.0)), m - @as(simd.V2f64, @splat(1.0)));

    const z = m * m;
    var p: simd.V2f64 = @splat(1.01875663804580931796e-4);
    p = simd.mulAdd_f64x2(p, m, @splat(4.97494994976747001425e-1));
    p = simd.mulAdd_f64x2(p, m, @splat(4.70579119878881725854e0));
    p = simd.mulAdd_f64x2(p, m, @splat(1.44989225341610930846e1));
    p = simd.mulAdd_f64x2(p, m, @splat(1.79368678507819816313e1));
    p = simd.mulAdd_f64x2(p, m, @splat(7.70838733755885391666e0));
    var q: simd.V2f64 = m + @as(simd.V2f64, @splat(1.12873587189167450590e1));
    q = simd.mulAdd_f64x2(q, m, @splat(4.52279145837532221105e1));
    q = simd.mulAdd_f64x2(q, m, @splat(8.29875266912776603211e1));
    q = simd.mulAdd_f64x2(q, m, @splat(7.11544750618563894466e1));
    q = simd.mulAdd_f64x2(q, m, @splat(2.31251620126765340583e1));

    var y = m * (z * (p / q));
    y = simd.mulAdd_f64x2(e, @splat(LOG_LN2_LO), y);
    y = simd.nmulAdd_f64x2(@splat(0.5), z, y);
    const result = m + y;
    return simd.mulAdd_f64x2(e, @splat(LOG_LN2_HI), result);
}

// ---------------------------------------------------------------------------
// atan / atan2 (Cephes double)
// ---------------------------------------------------------------------------

const ATAN_T3P8: f64 = 2.41421356237309504880;
const ATAN_TP8: f64 = 0.41421356237309504880;
const ATAN_PIO2: f64 = 1.5707963267948966;
const ATAN_PIO4: f64 = 0.7853981633974483;
const ATAN_PI: f64 = 3.141592653589793;
const ATAN_MOREBITS: f64 = 6.123233995736765886130e-17;

/// atan for non-negative lane values.
inline fn atanPos(ax: simd.V2f64) simd.V2f64 {
    const one: simd.V2f64 = @splat(1.0);
    const gt3 = ax > @as(simd.V2f64, @splat(ATAN_T3P8));
    const gtp8 = (ax > @as(simd.V2f64, @splat(ATAN_TP8))) & (ax <= @as(simd.V2f64, @splat(ATAN_T3P8)));

    const xa = -(one / ax);
    const xb = (ax - one) / (ax + one);
    var xr = ax;
    xr = @select(f64, gtp8, xb, xr);
    xr = @select(f64, gt3, xa, xr);

    var yoff: simd.V2f64 = @splat(0.0);
    yoff = @select(f64, gtp8, @as(simd.V2f64, @splat(ATAN_PIO4)), yoff);
    yoff = @select(f64, gt3, @as(simd.V2f64, @splat(ATAN_PIO2)), yoff);

    const z = xr * xr;
    var p: simd.V2f64 = @splat(-8.750608600031904122785e-1);
    p = simd.mulAdd_f64x2(p, z, @splat(-1.615753718733365076637e1));
    p = simd.mulAdd_f64x2(p, z, @splat(-7.500855792314704667340e1));
    p = simd.mulAdd_f64x2(p, z, @splat(-1.228866684490136173410e2));
    p = simd.mulAdd_f64x2(p, z, @splat(-6.485021904942025371773e1));
    var q: simd.V2f64 = z + @as(simd.V2f64, @splat(2.485846490142306297962e1));
    q = simd.mulAdd_f64x2(q, z, @splat(1.650270098316988542046e2));
    q = simd.mulAdd_f64x2(q, z, @splat(4.328810604912902668951e2));
    q = simd.mulAdd_f64x2(q, z, @splat(4.853903996359136964868e2));
    q = simd.mulAdd_f64x2(q, z, @splat(1.945506571482613964425e2));

    var zz = simd.mulAdd_f64x2(xr * z, p / q, xr);
    var corr: simd.V2f64 = @splat(0.0);
    corr = @select(f64, gtp8, @as(simd.V2f64, @splat(0.5 * ATAN_MOREBITS)), corr);
    corr = @select(f64, gt3, @as(simd.V2f64, @splat(ATAN_MOREBITS)), corr);
    zz = zz + corr;
    return yoff + zz;
}

/// atan over all reals (odd): atan(x) = sign(x)·atanPos(|x|).
pub inline fn atanv_f64(x: simd.V2f64) simd.V2f64 {
    const r = atanPos(@abs(x));
    return @select(f64, x < @as(simd.V2f64, @splat(0.0)), -r, r);
}

/// atan2(y, x) with full quadrant handling (NumPy-faithful for finite inputs).
pub inline fn atan2v_f64(y: simd.V2f64, x: simd.V2f64) simd.V2f64 {
    const zero: simd.V2f64 = @splat(0.0);
    var base = atanv_f64(y / x);
    const corr = @select(f64, y < zero, @as(simd.V2f64, @splat(-ATAN_PI)), @as(simd.V2f64, @splat(ATAN_PI)));
    base = @select(f64, x < zero, base + corr, base);
    base = @select(f64, (x == zero) & (y == zero), zero, base);
    return base;
}

/// expm1(x) = exp(x) − 1, accurate near 0. Same range reduction as expv, but the
/// polynomial computes expm1(r) = exp(r)−1 (Taylor without the leading 1) and the
/// result is pow2n·expm1(r) + (pow2n−1) — the (pow2n−1) is exact for integer n and
/// 0 near x=0, so no cancellation.
pub inline fn expm1v_f64(x_in: simd.V2f64) simd.V2f64 {
    const maxv: simd.V2f64 = @splat(EXP_MAXLOG);
    const minv: simd.V2f64 = @splat(EXP_MINLOG);
    const x = simd.max_f64x2(simd.min_f64x2(x_in, maxv), minv);
    const n = @floor(simd.mulAdd_f64x2(x, @as(simd.V2f64, @splat(EXP_LOG2E)), @as(simd.V2f64, @splat(0.5))));
    var r = simd.nmulAdd_f64x2(n, @splat(EXP_C1), x);
    r = simd.nmulAdd_f64x2(n, @splat(EXP_C2), r);

    var p: simd.V2f64 = @splat(1.6059043836821613e-10); // 1/13!
    p = simd.mulAdd_f64x2(p, r, @splat(2.08767569878681e-9));
    p = simd.mulAdd_f64x2(p, r, @splat(2.5052108385441720e-8));
    p = simd.mulAdd_f64x2(p, r, @splat(2.7557319223985893e-7));
    p = simd.mulAdd_f64x2(p, r, @splat(2.7557319223985888e-6));
    p = simd.mulAdd_f64x2(p, r, @splat(2.4801587301587302e-5));
    p = simd.mulAdd_f64x2(p, r, @splat(1.9841269841269841e-4));
    p = simd.mulAdd_f64x2(p, r, @splat(1.3888888888888889e-3));
    p = simd.mulAdd_f64x2(p, r, @splat(8.3333333333333332e-3));
    p = simd.mulAdd_f64x2(p, r, @splat(4.1666666666666664e-2));
    p = simd.mulAdd_f64x2(p, r, @splat(1.6666666666666666e-1));
    p = simd.mulAdd_f64x2(p, r, @splat(5.0e-1));
    p = simd.mulAdd_f64x2(p, r, @splat(1.0));
    const expm1r = p * r; // exp(r) − 1

    const ni: simd.V2i64 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V2i64, @splat(1023))) << @as(simd.V2i64, @splat(52));
    const pow2n: simd.V2f64 = @bitCast(biased);
    var result = simd.mulAdd_f64x2(pow2n, expm1r, pow2n - @as(simd.V2f64, @splat(1.0)));
    result = @select(f64, x_in > maxv, @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x_in < minv, @as(simd.V2f64, @splat(-1.0)), result);
    return result;
}

/// log1p(x) = log(1+x), accurate near 0 via Kahan's correction:
/// log1p(x) = log(u)·x/(u−1) with u = fl(1+x) (and = x when u rounds to 1).
/// x<−1 → NaN, x=−1 → −inf.
pub inline fn log1pv_f64(x: simd.V2f64) simd.V2f64 {
    const one: simd.V2f64 = @splat(1.0);
    const zero: simd.V2f64 = @splat(0.0);
    const u = one + x;
    const d = u - one;
    const ratio = @select(f64, d == zero, one, x / d);
    var r = logv_f64(u) * ratio;
    r = @select(f64, d == zero, x, r); // u rounded to 1 → log1p ≈ x
    r = @select(f64, x < -one, @as(simd.V2f64, @splat(math.nan(f64))), r);
    r = @select(f64, x == -one, @as(simd.V2f64, @splat(-math.inf(f64))), r);
    return @select(f64, x != x, x, r);
}

// softplus(d) = log(1 + e^d) for d ≤ 0, used by logaddexp as logaddexp(a,b) =
// max + softplus(min − max). Identity: log(1+e^d) = d/2 + ln2 + logcosh(d/2),
// and logcosh is even, so for |d| ≤ 2 it is a pure polynomial in u = (d/2)²
// (degree-13 minimax of logcosh(√u) on u∈[0,1], ~4e-14 rel) — NO exp, NO log,
// NO divide. Per-vector early-out: lanes spread past |d|>2 fall back to the
// general log1p∘exp path (accurate everywhere). The common clustered case (and
// logaddexp(x,x) → d=0) takes the transcendental-free branch.
const SP_LN2: f64 = 0.6931471805599453;
const SP_UMAX: f64 = 1.0; // (D/2)² with D = 2
const LC1: f64 = 0.50000000000014;
const LC2: f64 = -0.08333333333623849;
const LC3: f64 = 0.022222222229196967;
const LC4: f64 = -0.0067460313311044386;
const LC5: f64 = 0.0021869428489678443;
const LC6: f64 = -0.0007385615011211132;
const LC7: f64 = 0.0002564045805104835;
const LC8: f64 = -9.048920967545006e-05;
const LC9: f64 = 3.1787275678495595e-05;
const LC10: f64 = -1.0558913149522725e-05;
const LC11: f64 = 2.9944680866524453e-06;
const LC12: f64 = -6.103316525926454e-07;
const LC13: f64 = 6.370339031094745e-08;

pub inline fn softplusv_f64(d: simd.V2f64) simd.V2f64 {
    const half: simd.V2f64 = @splat(0.5);
    const w = d * half;
    const u = w * w; // (d/2)² ≥ 0

    // Spread lanes (|d| > 2) → general, accurate path for the whole vector.
    if (@reduce(.Or, u > @as(simd.V2f64, @splat(SP_UMAX)))) {
        return log1pv_f64(expv_f64(d));
    }

    // logcosh(d/2) = P(u) via Horner (LC0 ≡ 0, so the chain ends at u·(LC1+…)).
    var p: simd.V2f64 = @splat(LC13);
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC12)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC11)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC10)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC9)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC8)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC7)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC6)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC5)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC4)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC3)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC2)));
    p = simd.mulAdd_f64x2(p, u, @as(simd.V2f64, @splat(LC1)));
    p = p * u; // ×u (LC0 = 0)

    return w + @as(simd.V2f64, @splat(SP_LN2)) + p;
}

// --- inverse hyperbolic (compose the log core; domain-guarded) ---

/// asinh(x) = sign(x)·log(|x| + sqrt(x²+1)). Defined for all reals (compute on
/// |x| to avoid cancellation for large negative x).
pub inline fn asinhv_f64(x: simd.V2f64) simd.V2f64 {
    const one: simd.V2f64 = @splat(1.0);
    const ax = @abs(x);
    var r = logv_f64(ax + @sqrt(ax * ax + one));
    r = @select(f64, x < @as(simd.V2f64, @splat(0.0)), -r, r);
    return @select(f64, x != x, x, r); // propagate NaN
}

/// acosh(x) = log(x + sqrt(x²−1)). x < 1 → NaN; x = 1 → 0.
pub inline fn acoshv_f64(x: simd.V2f64) simd.V2f64 {
    const one: simd.V2f64 = @splat(1.0);
    const r = logv_f64(x + @sqrt(x * x - one));
    // x<1 (incl. NaN, since NaN<1 is false → handle via the <1 guard returning
    // NaN only for ordered <1; NaN input falls through r which is NaN-ish) →
    // force NaN for x<1, and propagate NaN inputs explicitly.
    var out = @select(f64, x < one, @as(simd.V2f64, @splat(math.nan(f64))), r);
    out = @select(f64, x != x, x, out);
    return out;
}

/// atanh(x) = ½·log((1+x)/(1−x)). |x|>1 → NaN, x=±1 → ±inf.
pub inline fn atanhv_f64(x: simd.V2f64) simd.V2f64 {
    const one: simd.V2f64 = @splat(1.0);
    var r = logv_f64((one + x) / (one - x)) * @as(simd.V2f64, @splat(0.5));
    r = @select(f64, @abs(x) > one, @as(simd.V2f64, @splat(math.nan(f64))), r);
    r = @select(f64, x == one, @as(simd.V2f64, @splat(math.inf(f64))), r);
    r = @select(f64, x == -one, @as(simd.V2f64, @splat(-math.inf(f64))), r);
    return @select(f64, x != x, x, r); // propagate NaN
}

// ---------------------------------------------------------------------------
// Complex drivers: map a per-element op over interleaved [re, im, ...] storage.
// Two complex elements per step (deinterleave → op → reinterleave).
// `op` is comptime anytype (the op fns use the inline calling convention).
// ---------------------------------------------------------------------------

const V2 = simd.V2f64;

/// complex128 driver.
pub inline fn cdrive_c128(a: [*]const f64, out: [*]f64, N: u32, comptime op: anytype) void {
    var k: u32 = 0;
    while (k + 2 <= N) : (k += 2) {
        const idx = k * 2;
        const a0 = simd.load2_f64(a, idx); // re0, im0
        const a1 = simd.load2_f64(a, idx + 2); // re1, im1
        const re = @shuffle(f64, a0, a1, [2]i32{ 0, -1 });
        const im = @shuffle(f64, a0, a1, [2]i32{ 1, -2 });
        var ore: V2 = undefined;
        var oim: V2 = undefined;
        op(re, im, &ore, &oim);
        simd.store2_f64(out, idx, @shuffle(f64, ore, oim, [2]i32{ 0, -1 }));
        simd.store2_f64(out, idx + 2, @shuffle(f64, ore, oim, [2]i32{ 1, -2 }));
    }
    if (k < N) {
        const idx = k * 2;
        const re: V2 = @splat(a[idx]);
        const im: V2 = @splat(a[idx + 1]);
        var ore: V2 = undefined;
        var oim: V2 = undefined;
        op(re, im, &ore, &oim);
        out[idx] = ore[0];
        out[idx + 1] = oim[0];
    }
}

/// complex64 driver: widen f32→f64, compute, narrow back to f32.
pub inline fn cdrive_c64(a: [*]const f32, out: [*]f32, N: u32, comptime op: anytype) void {
    var k: u32 = 0;
    while (k + 2 <= N) : (k += 2) {
        const idx = k * 2;
        const re: V2 = .{ a[idx], a[idx + 2] };
        const im: V2 = .{ a[idx + 1], a[idx + 3] };
        var ore: V2 = undefined;
        var oim: V2 = undefined;
        op(re, im, &ore, &oim);
        out[idx] = @floatCast(ore[0]);
        out[idx + 1] = @floatCast(oim[0]);
        out[idx + 2] = @floatCast(ore[1]);
        out[idx + 3] = @floatCast(oim[1]);
    }
    if (k < N) {
        const idx = k * 2;
        const re: V2 = @splat(a[idx]);
        const im: V2 = @splat(a[idx + 1]);
        var ore: V2 = undefined;
        var oim: V2 = undefined;
        op(re, im, &ore, &oim);
        out[idx] = @floatCast(ore[0]);
        out[idx + 1] = @floatCast(oim[0]);
    }
}

// --- Tests ---

test "sinv_f64 / cosv_f64 match std over a wide range" {
    const std = @import("std");
    const testing = std.testing;
    const xs = [_]f64{ 0.0, 0.5, 1.0, 1.5708, 3.14159, -2.7, 12.34, 100.0, 1000.5, 9999.0 };
    var i: usize = 0;
    while (i + 2 <= xs.len) : (i += 2) {
        const v: simd.V2f64 = .{ xs[i], xs[i + 1] };
        const s = sinv_f64(v);
        const c = cosv_f64(v);
        try testing.expectApproxEqAbs(s[0], @sin(xs[i]), 1e-12);
        try testing.expectApproxEqAbs(s[1], @sin(xs[i + 1]), 1e-12);
        try testing.expectApproxEqAbs(c[0], @cos(xs[i]), 1e-12);
        try testing.expectApproxEqAbs(c[1], @cos(xs[i + 1]), 1e-12);
    }
}

test "expv_f64 / logv_f64 round-trip" {
    const testing = @import("std").testing;
    const v: simd.V2f64 = .{ 2.0, 7.5 };
    try testing.expectApproxEqRel(expv_f64(v)[0], @exp(2.0), 1e-12);
    try testing.expectApproxEqRel(logv_f64(v)[1], @log(7.5), 1e-12);
}

test "atan2v matches std.math.atan2 across quadrants" {
    const std = @import("std");
    const testing = std.testing;
    const ys = [_]f64{ 1.0, -1.0, 3.0, -2.5, 0.0, 5.0 };
    const xs = [_]f64{ 1.0, -1.0, -4.0, 2.0, -1.0, 0.0 };
    var i: usize = 0;
    while (i + 2 <= ys.len) : (i += 2) {
        const yv: simd.V2f64 = .{ ys[i], ys[i + 1] };
        const xv: simd.V2f64 = .{ xs[i], xs[i + 1] };
        const r = atan2v_f64(yv, xv);
        try testing.expectApproxEqAbs(r[0], std.math.atan2(ys[i], xs[i]), 1e-12);
        try testing.expectApproxEqAbs(r[1], std.math.atan2(ys[i + 1], xs[i + 1]), 1e-12);
    }
}
