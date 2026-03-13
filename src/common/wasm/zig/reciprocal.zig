//! WASM element-wise reciprocal kernels for float types.
//!
//! Unary: out[i] = 1.0 / a[i]
//! Only float types (integer reciprocal promotes to float64 in NumPy).

const simd = @import("simd.zig");

/// Element-wise reciprocal for f64 using 2-wide SIMD: out[i] = 1.0 / a[i].
export fn reciprocal_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const one: simd.V2f64 = @splat(1.0);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, one / simd.load2_f64(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = 1.0 / a[i];
    }
}

/// Element-wise reciprocal for f32 using 4-wide SIMD: out[i] = 1.0 / a[i].
export fn reciprocal_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const one: simd.V4f32 = @splat(1.0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, one / simd.load4_f32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = 1.0 / a[i];
    }
}

/// i64-to-f64 reciprocal: out[i] = 1.0 / f64(a[i]).
export fn reciprocal_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = 1.0 / @as(f64, @floatFromInt(a[i]));
    }
}

/// i32-to-f64 reciprocal: out[i] = 1.0 / f64(a[i]).
export fn reciprocal_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = 1.0 / @as(f64, @floatFromInt(a[i]));
    }
}

/// i16-to-f64 reciprocal: out[i] = 1.0 / f64(a[i]).
export fn reciprocal_i16_f64(a: [*]const i16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = 1.0 / @as(f64, @floatFromInt(a[i]));
    }
}

/// i8-to-f64 reciprocal: out[i] = 1.0 / f64(a[i]).
export fn reciprocal_i8_f64(a: [*]const i8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = 1.0 / @as(f64, @floatFromInt(a[i]));
    }
}

// --- Tests ---

test "reciprocal_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 2, 4, 5, 10 };
    var out: [4]f64 = undefined;
    reciprocal_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.25, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.2, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.1, 1e-10);
}

test "reciprocal_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 2, 4, 5, 10, 8 };
    var out: [5]f32 = undefined;
    reciprocal_f32(&a, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.25, 1e-5);
    try testing.expectApproxEqAbs(out[4], 0.125, 1e-5);
}
