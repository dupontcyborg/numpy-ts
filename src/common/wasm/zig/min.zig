//! WASM element-wise minimum kernels for all numeric types.
//!
//! Binary: out[i] = min(a[i], b[i])  (propagates NaN, like np.minimum)
//! Scalar: out[i] = min(a[i], scalar)
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise minimum for f64 using 2-wide SIMD: out[i] = min(a[i], b[i]).
export fn min_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.min_f64x2(simd.load2_f64(a, i), simd.load2_f64(b, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for f64 using 2-wide SIMD.
export fn min_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.min_f64x2(simd.load2_f64(a, i), s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for f32 using 4-wide SIMD: out[i] = min(a[i], b[i]).
export fn min_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.min_f32x4(simd.load4_f32(a, i), simd.load4_f32(b, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for f32 using 4-wide SIMD.
export fn min_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.min_f32x4(simd.load4_f32(a, i), s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i64, scalar (LLVM scalarizes i64x2 @select).
export fn min_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i64, scalar.
export fn min_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u64, scalar (LLVM scalarizes i64x2 @select).
export fn min_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u64, scalar.
export fn min_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, scalar: u64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i32 using 4-wide SIMD with compare+select.
export fn min_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_i32(a, i);
        const vb = simd.load4_i32(b, i);
        simd.store4_i32(out, i, @select(i32, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i32 using 4-wide SIMD.
export fn min_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const s: simd.V4i32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_i32(a, i);
        simd.store4_i32(out, i, @select(i32, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u32 using 4-wide SIMD with unsigned compare+select.
export fn min_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_u32(a, i);
        const vb = simd.load4_u32(b, i);
        simd.store4_u32(out, i, @select(u32, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u32 using 4-wide SIMD.
export fn min_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, scalar: u32) void {
    const s: simd.V4u32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_u32(a, i);
        simd.store4_u32(out, i, @select(u32, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i16 using 8-wide SIMD with compare+select.
export fn min_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_i16(a, i);
        const vb = simd.load8_i16(b, i);
        simd.store8_i16(out, i, @select(i16, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i16 using 8-wide SIMD.
export fn min_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    const s: simd.V8i16 = @splat(scalar);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_i16(a, i);
        simd.store8_i16(out, i, @select(i16, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u16 using 8-wide SIMD with unsigned compare+select.
export fn min_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_u16(a, i);
        const vb = simd.load8_u16(b, i);
        simd.store8_u16(out, i, @select(u16, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u16 using 8-wide SIMD.
export fn min_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, scalar: u16) void {
    const s: simd.V8u16 = @splat(scalar);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_u16(a, i);
        simd.store8_u16(out, i, @select(u16, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i8 using 16-wide SIMD with compare+select.
export fn min_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_i8(a, i);
        const vb = simd.load16_i8(b, i);
        simd.store16_i8(out, i, @select(i8, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i8 using 16-wide SIMD.
export fn min_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    const s: simd.V16i8 = @splat(scalar);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_i8(a, i);
        simd.store16_i8(out, i, @select(i8, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u8 using 16-wide SIMD with unsigned compare+select.
export fn min_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_u8(a, i);
        const vb = simd.load16_u8(b, i);
        simd.store16_u8(out, i, @select(u8, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u8 using 16-wide SIMD.
export fn min_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, scalar: u8) void {
    const s: simd.V16u8 = @splat(scalar);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_u8(a, i);
        simd.store16_u8(out, i, @select(u8, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

// --- Tests ---

test "min_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 5, 3 };
    const b = [_]f64{ 2, 4, 6 };
    var out: [3]f64 = undefined;
    min_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
}

test "min_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 5, 3, -2, 7, 0, 8, -1, 9, 10, -5, 4, 6, -3, 2, 11, 12 };
    const b = [_]i8{ 2, 4, 6, -1, 3, 1, 7, 0, 8, 5, -4, 3, 7, -2, 1, 10, 13 };
    var out: [17]i8 = undefined;
    min_i8(&a, &b, &out, 17);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 4);
    try testing.expectEqual(out[3], -2);
}

test "min_u8 unsigned values above 127" {
    const testing = @import("std").testing;
    const a = [_]u8{ 200, 100, 255, 0, 128, 1, 254, 50, 130, 140, 150, 160, 170, 180, 190, 210, 220 };
    const b = [_]u8{ 100, 200, 128, 1, 255, 0, 50, 254, 140, 130, 160, 150, 180, 170, 210, 190, 230 };
    var out: [17]u8 = undefined;
    min_u8(&a, &b, &out, 17);
    try testing.expectEqual(out[0], 100); // 200 vs 100 → 100
    try testing.expectEqual(out[1], 100); // 100 vs 200 → 100
    try testing.expectEqual(out[2], 128); // 255 vs 128 → 128
    try testing.expectEqual(out[4], 128); // 128 vs 255 → 128
}

test "min_f64 SIMD boundary N=3 (V2f64 remainder=1)" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.5, 2.5, 0.0 };
    const b = [_]f64{ 1.0, -3.0, 0.0 };
    var out: [3]f64 = undefined;
    min_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], -1.5, 1e-10);
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
}

test "min_f64 negative values" {
    const testing = @import("std").testing;
    const a = [_]f64{ -10.0, -1.0, -100.0 };
    const b = [_]f64{ -5.0, -2.0, -50.0 };
    var out: [3]f64 = undefined;
    min_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], -10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -100.0, 1e-10);
}

test "min_f64 edge float values Inf -Inf -0.0" {
    const testing = @import("std").testing;
    const inf = @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    const neg_inf = @as(f64, @bitCast(@as(u64, 0xFFF0000000000000)));
    const neg_zero = @as(f64, @bitCast(@as(u64, 0x8000000000000000)));
    const a = [_]f64{ inf, neg_inf, neg_zero };
    const b = [_]f64{ 1.0, 1.0, 0.0 };
    var out: [3]f64 = undefined;
    min_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], neg_inf, 1e-10);
    // min(-0.0, 0.0) should be -0.0 or 0.0 (both approx equal)
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
}

test "min_scalar_f64 SIMD boundary N=3" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.0, 5.0, 2.0 };
    var out: [3]f64 = undefined;
    min_scalar_f64(&a, &out, 3, 3.0);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}

test "min_f32 SIMD boundary N=7 (V4f32 remainder=3)" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0 };
    const b = [_]f32{ -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0 };
    var out: [7]f32 = undefined;
    min_f32(&a, &b, &out, 7);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-6);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-6);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-6);
    try testing.expectApproxEqAbs(out[3], -4.0, 1e-6);
    try testing.expectApproxEqAbs(out[4], -5.0, 1e-6);
    try testing.expectApproxEqAbs(out[5], -6.0, 1e-6);
    try testing.expectApproxEqAbs(out[6], -7.0, 1e-6);
}

test "min_scalar_f32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    const a = [_]f32{ -10.0, 0.0, 5.0, -1.0, 3.0, 2.0, -7.0 };
    var out: [7]f32 = undefined;
    min_scalar_f32(&a, &out, 7, 1.0);
    try testing.expectApproxEqAbs(out[0], -10.0, 1e-6);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-6);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-6);
    try testing.expectApproxEqAbs(out[3], -1.0, 1e-6);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-6);
    try testing.expectApproxEqAbs(out[5], 1.0, 1e-6);
    try testing.expectApproxEqAbs(out[6], -7.0, 1e-6);
}

test "min_f32 edge float values Inf -Inf" {
    const testing = @import("std").testing;
    const inf = @as(f32, @bitCast(@as(u32, 0x7F800000)));
    const neg_inf = @as(f32, @bitCast(@as(u32, 0xFF800000)));
    const a = [_]f32{ inf, neg_inf, 0.0, -0.0 };
    const b = [_]f32{ 1.0, 1.0, -0.0, 0.0 };
    var out: [4]f32 = undefined;
    min_f32(&a, &b, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-6);
    try testing.expectApproxEqAbs(out[1], neg_inf, 1e-6);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-6);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-6);
}

test "min_i32 SIMD boundary N=7 (V4i32 remainder=3)" {
    const testing = @import("std").testing;
    const a = [_]i32{ 10, -20, 30, -40, 50, -60, 70 };
    const b = [_]i32{ -10, 20, -30, 40, -50, 60, -70 };
    var out: [7]i32 = undefined;
    min_i32(&a, &b, &out, 7);
    try testing.expectEqual(out[0], -10);
    try testing.expectEqual(out[1], -20);
    try testing.expectEqual(out[2], -30);
    try testing.expectEqual(out[3], -40);
    try testing.expectEqual(out[4], -50);
    try testing.expectEqual(out[5], -60);
    try testing.expectEqual(out[6], -70);
}

test "min_i32 negative values" {
    const testing = @import("std").testing;
    const a = [_]i32{ -100, -1, -2147483647 };
    const b = [_]i32{ -50, -200, -1 };
    var out: [3]i32 = undefined;
    min_i32(&a, &b, &out, 3);
    try testing.expectEqual(out[0], -100);
    try testing.expectEqual(out[1], -200);
    try testing.expectEqual(out[2], -2147483647);
}

test "min_scalar_i32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    const a = [_]i32{ -5, 0, 3, -10, 7, 1, -3 };
    var out: [7]i32 = undefined;
    min_scalar_i32(&a, &out, 7, 2);
    try testing.expectEqual(out[0], -5);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], -10);
    try testing.expectEqual(out[4], 2);
    try testing.expectEqual(out[5], 1);
    try testing.expectEqual(out[6], -3);
}

test "min_i16 SIMD boundary N=9 (V8i16 remainder=1)" {
    const testing = @import("std").testing;
    const a = [_]i16{ 100, -200, 300, -400, 500, -600, 700, -800, 900 };
    const b = [_]i16{ -100, 200, -300, 400, -500, 600, -700, 800, -900 };
    var out: [9]i16 = undefined;
    min_i16(&a, &b, &out, 9);
    try testing.expectEqual(out[0], -100);
    try testing.expectEqual(out[1], -200);
    try testing.expectEqual(out[2], -300);
    try testing.expectEqual(out[3], -400);
    try testing.expectEqual(out[4], -500);
    try testing.expectEqual(out[5], -600);
    try testing.expectEqual(out[6], -700);
    try testing.expectEqual(out[7], -800);
    try testing.expectEqual(out[8], -900);
}

test "min_i16 negative values" {
    const testing = @import("std").testing;
    const a = [_]i16{ -32767, -1, -100 };
    const b = [_]i16{ -1, -32767, -50 };
    var out: [3]i16 = undefined;
    min_i16(&a, &b, &out, 3);
    try testing.expectEqual(out[0], -32767);
    try testing.expectEqual(out[1], -32767);
    try testing.expectEqual(out[2], -100);
}

test "min_scalar_i16 SIMD boundary N=9" {
    const testing = @import("std").testing;
    const a = [_]i16{ -10, 0, 5, -1, 3, 2, -7, 8, -20 };
    var out: [9]i16 = undefined;
    min_scalar_i16(&a, &out, 9, 0);
    try testing.expectEqual(out[0], -10);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], -1);
    try testing.expectEqual(out[4], 0);
    try testing.expectEqual(out[5], 0);
    try testing.expectEqual(out[6], -7);
    try testing.expectEqual(out[7], 0);
    try testing.expectEqual(out[8], -20);
}

test "min_i8 SIMD boundary N=17 (V16i8 remainder=1)" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, -20, 30, -40, 50, -60, 70, -80, 9, -10, 11, -12, 13, -14, 15, -16, 17 };
    const b = [_]i8{ -10, 20, -30, 40, -50, 60, -70, 80, -9, 10, -11, 12, -13, 14, -15, 16, -17 };
    var out: [17]i8 = undefined;
    min_i8(&a, &b, &out, 17);
    try testing.expectEqual(out[0], -10);
    try testing.expectEqual(out[1], -20);
    try testing.expectEqual(out[2], -30);
    try testing.expectEqual(out[3], -40);
    try testing.expectEqual(out[4], -50);
    try testing.expectEqual(out[5], -60);
    try testing.expectEqual(out[6], -70);
    try testing.expectEqual(out[7], -80);
    try testing.expectEqual(out[8], -9);
    try testing.expectEqual(out[9], -10);
    try testing.expectEqual(out[10], -11);
    try testing.expectEqual(out[11], -12);
    try testing.expectEqual(out[12], -13);
    try testing.expectEqual(out[13], -14);
    try testing.expectEqual(out[14], -15);
    try testing.expectEqual(out[15], -16);
    try testing.expectEqual(out[16], -17);
}

test "min_i8 negative values" {
    const testing = @import("std").testing;
    const a = [_]i8{ -127, -1, -50 };
    const b = [_]i8{ -1, -127, -25 };
    var out: [3]i8 = undefined;
    min_i8(&a, &b, &out, 3);
    try testing.expectEqual(out[0], -127);
    try testing.expectEqual(out[1], -127);
    try testing.expectEqual(out[2], -50);
}

test "min_scalar_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    const a = [_]i8{ -5, 0, 3, -10, 7, 1, -3, 8, -2, 4, -6, 9, -1, 2, -4, 6, -8 };
    var out: [17]i8 = undefined;
    min_scalar_i8(&a, &out, 17, 0);
    try testing.expectEqual(out[0], -5);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], -10);
    try testing.expectEqual(out[4], 0);
    try testing.expectEqual(out[5], 0);
    try testing.expectEqual(out[6], -3);
    try testing.expectEqual(out[7], 0);
    try testing.expectEqual(out[8], -2);
    try testing.expectEqual(out[9], 0);
    try testing.expectEqual(out[10], -6);
    try testing.expectEqual(out[11], 0);
    try testing.expectEqual(out[12], -1);
    try testing.expectEqual(out[13], 0);
    try testing.expectEqual(out[14], -4);
    try testing.expectEqual(out[15], 0);
    try testing.expectEqual(out[16], -8);
}

test "min_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 0, 100, 18446744073709551615 };
    const b = [_]u64{ 1, 50, 0 };
    var out: [3]u64 = undefined;
    min_u64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 50);
    try testing.expectEqual(out[2], 0);
}

test "min_scalar_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 0, 10, 100 };
    var out: [3]u64 = undefined;
    min_scalar_u64(&a, &out, 3, 50);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 10);
    try testing.expectEqual(out[2], 50);
}

test "min_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0, 100, 4294967295, 50, 200 };
    const b = [_]u32{ 1, 50, 0, 100, 150 };
    var out: [5]u32 = undefined;
    min_u32(&a, &b, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 50);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 50);
    try testing.expectEqual(out[4], 150);
}

test "min_scalar_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0, 10, 100, 5, 200 };
    var out: [5]u32 = undefined;
    min_scalar_u32(&a, &out, 5, 50);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 10);
    try testing.expectEqual(out[2], 50);
    try testing.expectEqual(out[3], 5);
    try testing.expectEqual(out[4], 50);
}

test "min_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0, 100, 65535, 50, 200, 300, 400, 500, 600 };
    const b = [_]u16{ 1, 50, 0, 100, 150, 350, 350, 600, 500 };
    var out: [9]u16 = undefined;
    min_u16(&a, &b, &out, 9);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 50);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 50);
    try testing.expectEqual(out[4], 150);
    try testing.expectEqual(out[5], 300);
    try testing.expectEqual(out[6], 350);
    try testing.expectEqual(out[7], 500);
    try testing.expectEqual(out[8], 500);
}

test "min_scalar_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0, 10, 100, 5, 200, 300, 400, 500, 600 };
    var out: [9]u16 = undefined;
    min_scalar_u16(&a, &out, 9, 150);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 10);
    try testing.expectEqual(out[2], 100);
    try testing.expectEqual(out[3], 5);
    try testing.expectEqual(out[4], 150);
    try testing.expectEqual(out[5], 150);
    try testing.expectEqual(out[6], 150);
    try testing.expectEqual(out[7], 150);
    try testing.expectEqual(out[8], 150);
}

test "min_scalar_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 0, 10, 200, 5, 255, 100, 50, 150, 30, 40, 60, 70, 80, 90, 110, 120, 250 };
    var out: [17]u8 = undefined;
    min_scalar_u8(&a, &out, 17, 100);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 10);
    try testing.expectEqual(out[2], 100);
    try testing.expectEqual(out[3], 5);
    try testing.expectEqual(out[4], 100);
    try testing.expectEqual(out[5], 100);
    try testing.expectEqual(out[6], 50);
    try testing.expectEqual(out[7], 100);
    try testing.expectEqual(out[16], 100);
}

test "min_f64 single element N=1" {
    const testing = @import("std").testing;
    const a = [_]f64{3.14};
    const b = [_]f64{2.71};
    var out: [1]f64 = undefined;
    min_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 2.71, 1e-10);
}

test "min_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ -9223372036854775807, 0, 100 };
    const b = [_]i64{ 0, -1, 50 };
    var out: [3]i64 = undefined;
    min_i64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], -9223372036854775807);
    try testing.expectEqual(out[1], -1);
    try testing.expectEqual(out[2], 50);
}

test "min_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ -100, 0, 100 };
    var out: [3]i64 = undefined;
    min_scalar_i64(&a, &out, 3, 0);
    try testing.expectEqual(out[0], -100);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
}
