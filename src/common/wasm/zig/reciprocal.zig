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

/// Integer reciprocal for i64: out[i] = @divTrunc(1, a[i]). Keeps integer dtype.
/// Division by zero returns 0 (matching NumPy behavior for most platforms).
export fn reciprocal_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) @divTrunc(@as(i64, 1), a[i]) else 0;
    }
}

/// Integer reciprocal for u64: out[i] = 1 / a[i]. Keeps integer dtype.
export fn reciprocal_u64(a: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 / a[i] else 0;
    }
}

/// Integer reciprocal for i32: out[i] = @divTrunc(1, a[i]). Keeps integer dtype.
export fn reciprocal_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) @divTrunc(@as(i32, 1), a[i]) else 0;
    }
}

/// Integer reciprocal for u32: out[i] = 1 / a[i]. Keeps integer dtype.
export fn reciprocal_u32(a: [*]const u32, out: [*]u32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 / a[i] else 0;
    }
}

/// Integer reciprocal for i16: out[i] = @divTrunc(1, a[i]). Keeps integer dtype.
export fn reciprocal_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) @divTrunc(@as(i16, 1), a[i]) else 0;
    }
}

/// Integer reciprocal for u16: out[i] = 1 / a[i]. Keeps integer dtype.
export fn reciprocal_u16(a: [*]const u16, out: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 / a[i] else 0;
    }
}

/// Integer reciprocal for i8: out[i] = @divTrunc(1, a[i]). Keeps integer dtype.
export fn reciprocal_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) @divTrunc(@as(i8, 1), a[i]) else 0;
    }
}

/// Integer reciprocal for u8: out[i] = 1 / a[i]. Keeps integer dtype.
export fn reciprocal_u8(a: [*]const u8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 / a[i] else 0;
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

test "reciprocal_f64 SIMD boundary N=1" {
    const testing = @import("std").testing;
    const a = [_]f64{8.0};
    var out: [1]f64 = undefined;
    reciprocal_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.125, 1e-10);
}

test "reciprocal_f64 SIMD boundary N=3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, -2.0, 0.5 };
    var out: [3]f64 = undefined;
    reciprocal_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -0.5, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}

test "reciprocal_f64 inf for zero" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    const a = [_]f64{ 0.0, -0.0 };
    var out: [2]f64 = undefined;
    reciprocal_f64(&a, &out, 2);
    try testing.expect(math.isInf(out[0]) and out[0] > 0);
    try testing.expect(math.isInf(out[1]) and out[1] < 0);
}

test "reciprocal_f32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 4, 5, 8, 10, 20 };
    var out: [7]f32 = undefined;
    reciprocal_f32(&a, &out, 7);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.25, 1e-5);
}

test "reciprocal_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 4, -5 };
    var out: [4]i32 = undefined;
    reciprocal_i32(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0); // trunc(1/2) = 0
    try testing.expectEqual(out[2], 0); // trunc(1/4) = 0
    try testing.expectEqual(out[3], 0); // trunc(1/-5) = 0
}

test "reciprocal_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, -1, 0 };
    var out: [4]i8 = undefined;
    reciprocal_i8(&a, &out, 4);
    try testing.expectEqual(out[0], 1); // 1/1 = 1
    try testing.expectEqual(out[1], 0); // trunc(1/2) = 0
    try testing.expectEqual(out[2], -1); // 1/-1 = -1
    try testing.expectEqual(out[3], 0); // div by zero = 0
}

test "reciprocal_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 1, 2, 4, 0 };
    var out: [4]u8 = undefined;
    reciprocal_u8(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 0); // div by zero = 0
}

test "reciprocal_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, -1, 2 };
    var out: [3]i64 = undefined;
    reciprocal_i64(&a, &out, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], -1);
    try testing.expectEqual(out[2], 0);
}

test "reciprocal_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 4, 0 };
    var out: [3]i16 = undefined;
    reciprocal_i16(&a, &out, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
}

test "reciprocal_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 1, 2, 100, 0 };
    var out: [4]u64 = undefined;
    reciprocal_u64(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0); // 1/2 truncates to 0
    try testing.expectEqual(out[2], 0); // 1/100 truncates to 0
    try testing.expectEqual(out[3], 0); // div by zero → 0
}

test "reciprocal_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 1, 5, 0 };
    var out: [3]u32 = undefined;
    reciprocal_u32(&a, &out, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0); // div by zero → 0
}

test "reciprocal_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 1, 7, 0 };
    var out: [3]u16 = undefined;
    reciprocal_u16(&a, &out, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0); // div by zero → 0
}
