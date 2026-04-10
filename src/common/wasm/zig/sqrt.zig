//! WASM element-wise square root kernels for all numeric types.
//!
//! Unary: out[i] = sqrt(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Float types output same type; integer types output f64 (NumPy promotion).
//! Integer kernels convert to f64 in WASM using SIMD, avoiding JS conversion.

const simd = @import("simd.zig");

/// Element-wise sqrt for f64 using 2-wide SIMD: out[i] = sqrt(a[i]).
export fn sqrt_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, @sqrt(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = @sqrt(a[i]);
    }
}

/// Element-wise sqrt for f32 using 4-wide SIMD: out[i] = sqrt(a[i]).
export fn sqrt_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, @sqrt(simd.load4_f32(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = @sqrt(a[i]);
    }
}

/// Element-wise sqrt for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn sqrt_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @sqrt(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise sqrt for i32 → f64 output.
/// Loads i32x4, converts to 2×f64x2 via element widening, sqrt, store.
export fn sqrt_i32(a: [*]const i32, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]) };
        simd.store2_f64(out, i, @sqrt(v));
    }
    while (i < N) : (i += 1) {
        out[i] = @sqrt(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise sqrt for i16 → f32 output.
export fn sqrt_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v: simd.V4f32 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]), @floatFromInt(a[i + 2]), @floatFromInt(a[i + 3]) };
        simd.store4_f32(out, i, @sqrt(v));
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(@sqrt(@as(f64, @floatFromInt(a[i]))));
    }
}

/// Element-wise sqrt for i8 → f32 output.
export fn sqrt_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v: simd.V4f32 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]), @floatFromInt(a[i + 2]), @floatFromInt(a[i + 3]) };
        simd.store4_f32(out, i, @sqrt(v));
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(@sqrt(@as(f64, @floatFromInt(a[i]))));
    }
}

// --- Tests ---

test "sqrt_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, 1, 4, 9, 16 };
    var out: [5]f64 = undefined;
    sqrt_f64(&a, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 4.0, 1e-10);
}

test "sqrt_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0, 1, 4, 9, 16, 25, 36 };
    var out: [7]f32 = undefined;
    sqrt_f32(&a, &out, 7);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[6], 6.0, 1e-5);
}

test "sqrt_f64 SIMD boundary N=1" {
    const testing = @import("std").testing;
    const a = [_]f64{25.0};
    var out: [1]f64 = undefined;
    sqrt_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
}

test "sqrt_f64 SIMD boundary N=3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    var out: [3]f64 = undefined;
    sqrt_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.41421356, 1e-6);
    try testing.expectApproxEqAbs(out[2], 1.73205080, 1e-6);
}

test "sqrt_f32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    var a: [7]f32 = undefined;
    for (0..7) |i| {
        const v: f32 = @floatFromInt((i + 1) * (i + 1));
        a[i] = v;
    }
    var out: [7]f32 = undefined;
    sqrt_f32(&a, &out, 7);
    for (0..7) |i| {
        const expected: f32 = @floatFromInt(i + 1);
        try testing.expectApproxEqAbs(out[i], expected, 1e-5);
    }
}

test "sqrt_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 4, 9, 16, 25, 36 };
    var out: [7]f64 = undefined;
    sqrt_i32(&a, &out, 7);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[6], 6.0, 1e-10);
}

test "sqrt_i32 SIMD boundary N=3" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 4, 9 };
    var out: [3]f64 = undefined;
    sqrt_i32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
}

test "sqrt_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, 4, 9, 16 };
    var out: [5]f32 = undefined;
    sqrt_i8_f32(&a, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 4.0, 1e-5);
}

test "sqrt_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 1, 4, 100 };
    var out: [4]f64 = undefined;
    sqrt_i64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 10.0, 1e-10);
}

test "sqrt_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{4};
    var out: [1]f32 = undefined;
    sqrt_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5);
}
