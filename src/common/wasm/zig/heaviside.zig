//! WASM element-wise heaviside step function kernels for numeric types.
//!
//! Scalar: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2 : 1
//! Binary: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2[i] : 1
//!
//! Branchless SIMD: start at 1 (x1 > 0), then @select x1==0 → x2 and x1<0 → 0.
//! f64 runs 2-wide, f32 4-wide. NaN falls through to 1 (matching the prior scalar
//! kernels exactly — both comparisons are false for NaN).

const simd = @import("simd.zig");

inline fn step2(v: simd.V2f64, x2: simd.V2f64) simd.V2f64 {
    const zero: simd.V2f64 = @splat(0.0);
    var r: simd.V2f64 = @splat(1.0);
    r = @select(f64, v == zero, x2, r);
    r = @select(f64, v < zero, zero, r);
    return r;
}

inline fn step4(v: simd.V4f32, x2: simd.V4f32) simd.V4f32 {
    const zero: simd.V4f32 = @splat(0.0);
    var r: simd.V4f32 = @splat(1.0);
    r = @select(f32, v == zero, x2, r);
    r = @select(f32, v < zero, zero, r);
    return r;
}

export fn heaviside_scalar_f64(x1: [*]const f64, out: [*]f64, N: u32, x2: f64) void {
    const x2v: simd.V2f64 = @splat(x2);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, step2(simd.load2_f64(x1, i), x2v));
    }
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2 else 1.0;
    }
}

export fn heaviside_scalar_f32(x1: [*]const f32, out: [*]f32, N: u32, x2: f32) void {
    const x2v: simd.V4f32 = @splat(x2);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, step4(simd.load4_f32(x1, i), x2v));
    }
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2 else 1.0;
    }
}

export fn heaviside_f64(x1: [*]const f64, x2: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, step2(simd.load2_f64(x1, i), simd.load2_f64(x2, i)));
    }
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2[i] else 1.0;
    }
}

export fn heaviside_f32(x1: [*]const f32, x2: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, step4(simd.load4_f32(x1, i), simd.load4_f32(x2, i)));
    }
    while (i < N) : (i += 1) {
        const v = x1[i];
        out[i] = if (v < 0.0) 0.0 else if (v == 0.0) x2[i] else 1.0;
    }
}

// --- Tests ---

test "heaviside_scalar_f64 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ -2, -1, 0, 1, 2 };
    var out: [5]f64 = undefined;
    heaviside_scalar_f64(&x1, &out, 5, 0.5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-10);
}

test "heaviside_f64 binary" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ -1, 0, 0, 1 };
    const x2 = [_]f64{ 99, 0.5, 0.7, 99 };
    var out: [4]f64 = undefined;
    heaviside_f64(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.7, 1e-10);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-10);
}

test "heaviside_scalar_f32 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ -2, -1, 0, 1, 2 };
    var out: [5]f32 = undefined;
    heaviside_scalar_f32(&x1, &out, 5, 0.5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-5);
}

test "heaviside_f32 binary" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ -1, 0, 0, 1 };
    const x2 = [_]f32{ 99, 0.5, 0.7, 99 };
    var out: [4]f32 = undefined;
    heaviside_f32(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.7, 1e-5);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-5);
}

test "heaviside tail (odd length) + NaN→1" {
    const std = @import("std");
    const testing = std.testing;
    const x1 = [_]f64{ -1, 0, 1, 2, 3, std.math.nan(f64), -5 };
    var out: [7]f64 = undefined;
    heaviside_scalar_f64(&x1, &out, 7, 0.5);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 1.0, 1e-10); // NaN → 1 (parity with scalar)
    try testing.expectApproxEqAbs(out[6], 0.0, 1e-10);
}
