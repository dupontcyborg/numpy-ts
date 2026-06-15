//! Shared vectorized sin/cos cores (Cephes, double precision ≈1 ulp).
//!
//! sin/cos have no CPU opcode — they're software polynomials. We reduce
//! x = q·(π/2) + r with q an integer and r ∈ [-π/4, π/4], pick the sin or cos
//! polynomial by quadrant, and fix the sign. The reduction uses 3-part
//! Cody-Waite (π/2 split into hi/mid/lo) so it stays accurate for the large
//! arguments the benchmarks feed in (arange up to ~1e4). All the heavy lifting
//! (floor, the polynomials, the quadrant selects) is f64x2 SIMD; only the
//! float→int quadrant extraction touches scalar lanes. The fused multiply-adds
//! lower to f64x2.relaxed_madd (1 op) on +relaxed_simd builds.

const simd = @import("simd.zig");

const TWO_OVER_PI: f64 = 0.6366197723675813430755; // 2/π
// 3-part π/2 for Cody-Waite range reduction.
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
    return simd.mulAdd_f64x2(r * z, s, r); // r·z·S + r
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
    return simd.mulAdd_f64x2(z * z, c, base); // z²·C + base
}

/// Reduce x to (q, r): x ≈ q·(π/2) + r, q integer, r ∈ [-π/4, π/4].
inline fn reduce(x: simd.V2f64, q_out: *simd.V2i64) simd.V2f64 {
    const fq = @floor(simd.mulAdd_f64x2(x, @as(simd.V2f64, @splat(TWO_OVER_PI)), @as(simd.V2f64, @splat(0.5))));
    q_out.* = @intFromFloat(fq);
    var r = simd.nmulAdd_f64x2(fq, @splat(PIO2_1), x); // x - fq·PIO2_1
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
