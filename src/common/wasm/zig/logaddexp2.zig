//! WASM element-wise logaddexp2: log2(2^a + 2^b).
//!
//! Numerically stable form: max(a,b) + log2(1 + 2^(−|a−b|)). Composed from the
//! shared SIMD cores: 2^(mn−mx) = expv((mn−mx)·ln2), then log2(1+y) =
//! log1pv(y)·log2(e). Float-only fast path (f64 2-wide, f32 via the f64 core);
//! integer/complex inputs keep the JS fallback (the wrapper returns null).

const simd = @import("simd.zig");
const t = @import("transcend.zig");

const LN2: f64 = 0.6931471805599453;
const LOG2E: f64 = 1.4426950408889634;

/// logaddexp2 of a 2-wide f64 lane.
inline fn lae2_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const mx = simd.max_f64x2(a, b);
    const mn = simd.min_f64x2(a, b);
    const y = t.expv_f64((mn - mx) * @as(simd.V2f64, @splat(LN2))); // 2^(mn−mx) ∈ (0,1]
    return mx + t.log1pv_f64(y) * @as(simd.V2f64, @splat(LOG2E));
}

export fn logaddexp2_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, lae2_f64(simd.load2_f64(a, i), simd.load2_f64(b, i)));
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        const bv: simd.V2f64 = .{ b[i], b[i] };
        out[i] = lae2_f64(av, bv)[0];
    }
}

export fn logaddexp2_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const bv: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, lae2_f64(simd.load2_f64(a, i), bv));
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        out[i] = lae2_f64(av, bv)[0];
    }
}

export fn logaddexp2_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ a[i], a[i + 1] };
        const bv: simd.V2f64 = .{ b[i], b[i + 1] };
        const r = lae2_f64(av, bv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        const bv: simd.V2f64 = .{ b[i], b[i] };
        out[i] = @floatCast(lae2_f64(av, bv)[0]);
    }
}

export fn logaddexp2_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: f64 = scalar;
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ a[i], a[i + 1] };
        const bv: simd.V2f64 = .{ s, s };
        const r = lae2_f64(av, bv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        const bv: simd.V2f64 = .{ s, s };
        out[i] = @floatCast(lae2_f64(av, bv)[0]);
    }
}

// --- Tests ---

test "logaddexp2_f64 matches reference" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f64{ 0.0, 1.0, 3.0, -2.0, 5.0 };
    const b = [_]f64{ 0.0, 2.0, 3.0, 4.0, -5.0 };
    var out: [5]f64 = undefined;
    logaddexp2_f64(&a, &b, &out, 5);
    for (0..5) |i| {
        const mx = @max(a[i], b[i]);
        const mn = @min(a[i], b[i]);
        const ref = mx + std.math.log1p(std.math.pow(f64, 2.0, mn - mx)) * LOG2E;
        try testing.expectApproxEqRel(out[i], ref, 1e-12);
    }
    // logaddexp2(x, x) = x + 1
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-12);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-12);
}

test "logaddexp2_scalar_f64 / f32" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f64{ 0.0, 3.0 };
    var out: [2]f64 = undefined;
    logaddexp2_scalar_f64(&a, &out, 2, 0.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-12); // logaddexp2(0,0)=1
    const a32 = [_]f32{ 1.0, 2.0 };
    const b32 = [_]f32{ 1.0, 2.0 };
    var o32: [2]f32 = undefined;
    logaddexp2_f32(&a32, &b32, &o32, 2);
    try testing.expectApproxEqAbs(o32[0], 2.0, 1e-4); // logaddexp2(1,1)=2
    try testing.expectApproxEqAbs(o32[1], 3.0, 1e-4); // logaddexp2(2,2)=3
}
