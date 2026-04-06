//! WASM element-wise multiplication kernels for all numeric types.
//!
//! Binary: out[i] = a[i] * b[i]
//! Scalar: out[i] = a[i] * scalar
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise multiply for f64 using 2-wide SIMD: out[i] = a[i] * b[i].
export fn mul_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) * simd.load2_f64(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

/// Element-wise multiply scalar for f64 using 2-wide SIMD: out[i] = a[i] * scalar.
export fn mul_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) * s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * scalar;
    }
}

/// Element-wise multiply for f32 using 4-wide SIMD: out[i] = a[i] * b[i].
export fn mul_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) * simd.load4_f32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

/// Element-wise multiply scalar for f32 using 4-wide SIMD: out[i] = a[i] * scalar.
export fn mul_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) * s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * scalar;
    }
}

/// Element-wise complex multiply for complex128 using shuffle-based SIMD.
/// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
/// N = number of complex elements (each = 2 f64s).
export fn mul_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    var k: u32 = 0;
    while (k + 2 <= N) : (k += 2) {
        const idx = k * 2;
        const a0 = simd.load2_f64(a, idx);
        const a1 = simd.load2_f64(a, idx + 2);
        const b0 = simd.load2_f64(b, idx);
        const b1 = simd.load2_f64(b, idx + 2);
        const a_re = @shuffle(f64, a0, a1, [2]i32{ 0, -1 });
        const a_im = @shuffle(f64, a0, a1, [2]i32{ 1, -2 });
        const b_re = @shuffle(f64, b0, b1, [2]i32{ 0, -1 });
        const b_im = @shuffle(f64, b0, b1, [2]i32{ 1, -2 });
        const re = a_re * b_re - a_im * b_im;
        const im = a_re * b_im + a_im * b_re;
        const lo = @shuffle(f64, re, im, [2]i32{ 0, -1 });
        const hi = @shuffle(f64, re, im, [2]i32{ 1, -2 });
        simd.store2_f64(out, idx, lo);
        simd.store2_f64(out, idx + 2, hi);
    }
    while (k < N) : (k += 1) {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        out[idx] = a_re * b_re - a_im * b_im;
        out[idx + 1] = a_re * b_im + a_im * b_re;
    }
}

/// Multiply complex128 by real scalar.
export fn mul_scalar_c128(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_f64 = N * 2;
    const n_simd = n_f64 & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) * s);
    }
    while (i < n_f64) : (i += 1) {
        out[i] = a[i] * scalar;
    }
}

/// Element-wise complex multiply for complex64 using shuffle-based SIMD.
/// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
/// N = number of complex elements (each = 2 f32s).
export fn mul_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    var k: u32 = 0;
    while (k + 4 <= N) : (k += 4) {
        const idx = k * 2;
        const a0 = simd.load4_f32(a, idx);
        const a1 = simd.load4_f32(a, idx + 4);
        const b0 = simd.load4_f32(b, idx);
        const b1 = simd.load4_f32(b, idx + 4);
        const a_re = @shuffle(f32, a0, a1, [4]i32{ 0, 2, -1, -3 });
        const a_im = @shuffle(f32, a0, a1, [4]i32{ 1, 3, -2, -4 });
        const b_re = @shuffle(f32, b0, b1, [4]i32{ 0, 2, -1, -3 });
        const b_im = @shuffle(f32, b0, b1, [4]i32{ 1, 3, -2, -4 });
        const re = a_re * b_re - a_im * b_im;
        const im = a_re * b_im + a_im * b_re;
        const lo = @shuffle(f32, re, im, [4]i32{ 0, -1, 1, -2 });
        const hi = @shuffle(f32, re, im, [4]i32{ 2, -3, 3, -4 });
        simd.store4_f32(out, idx, lo);
        simd.store4_f32(out, idx + 4, hi);
    }
    while (k < N) : (k += 1) {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        out[idx] = a_re * b_re - a_im * b_im;
        out[idx + 1] = a_re * b_im + a_im * b_re;
    }
}

/// Multiply complex64 by real scalar.
export fn mul_scalar_c64(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_f32 = N * 2;
    const n_simd = n_f32 & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) * s);
    }
    while (i < n_f32) : (i += 1) {
        out[i] = a[i] * scalar;
    }
}

/// Element-wise multiply for i64, scalar loop (no i64x2.mul in WASM SIMD).
/// Handles both signed (i64) and unsigned (u64).
export fn mul_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i] *% b[i];
    }
}

/// Element-wise multiply scalar for i64, scalar loop (no i64x2.mul in WASM SIMD).
/// Handles both signed (i64) and unsigned (u64).
export fn mul_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i] *% scalar;
    }
}

/// Element-wise multiply for i32 using 4-wide SIMD with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn mul_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) *% simd.load4_i32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% b[i];
    }
}

/// Element-wise multiply scalar for i32 using 4-wide SIMD with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn mul_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const s: simd.V4i32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) *% s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% scalar;
    }
}

/// Element-wise multiply for i16 using 8-wide SIMD with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn mul_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) *% simd.load8_i16(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% b[i];
    }
}

/// Element-wise multiply scalar for i16 using 8-wide SIMD with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn mul_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    const s: simd.V8i16 = @splat(scalar);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) *% s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% scalar;
    }
}

/// Element-wise multiply for i8 using 16-wide SIMD with wrapping arithmetic.
/// Uses widened i16 multiply since WASM SIMD has no i8x16.mul.
/// Handles both signed (i8) and unsigned (u8).
export fn mul_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.mul_i8x16(simd.load16_i8(a, i), simd.load16_i8(b, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% b[i];
    }
}

/// Element-wise multiply scalar for i8 using 16-wide SIMD with wrapping arithmetic.
/// Uses widened i16 multiply since WASM SIMD has no i8x16.mul.
/// Handles both signed (i8) and unsigned (u8).
export fn mul_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    const s: simd.V16i8 = @splat(scalar);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.mul_i8x16(simd.load16_i8(a, i), s));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% scalar;
    }
}

// --- Tests ---

test "mul_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 2, 3, 4, 5, 6 };
    const b = [_]f64{ 3, 4, 5, 6, 7 };
    var out: [5]f64 = undefined;
    mul_f64(&a, &b, &out, 5);
    try testing.expectApproxEqAbs(out[0], 6.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 42.0, 1e-10);
}

test "mul_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, 20, 30, 40 };
    const b = [_]i8{ 2, 3, 4, 5 };
    var out: [4]i8 = undefined;
    mul_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 20);
    try testing.expectEqual(out[1], 60);
    const expected2: i8 = @truncate(@as(i32, 120));
    try testing.expectEqual(out[2], expected2);
}

test "mul_c128 basic" {
    const testing = @import("std").testing;
    // (1+2i) * (3+4i) = (-5+10i)
    const a = [_]f64{ 1, 2 };
    const b = [_]f64{ 3, 4 };
    var out: [2]f64 = undefined;
    mul_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-10);
}

test "mul_scalar_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4, 5 };
    var out: [5]f32 = undefined;
    mul_scalar_f32(&a, &out, 5, 3.0);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 15.0, 1e-5);
}

test "mul_f64 SIMD boundary N=1" {
    const testing = @import("std").testing;
    const a = [_]f64{3.0};
    const b = [_]f64{4.0};
    var out: [1]f64 = undefined;
    mul_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 12.0, 1e-10);
}

test "mul_f64 SIMD boundary N=3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 2.0, 3.0, 4.0 };
    const b = [_]f64{ 5.0, 6.0, 7.0 };
    var out: [3]f64 = undefined;
    mul_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 18.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 28.0, 1e-10);
}

test "mul_f64 edge values" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    const a = [_]f64{ math.inf(f64), 0.0, -1.0 };
    const b = [_]f64{ 2.0, 5.0, -1.0 };
    var out: [3]f64 = undefined;
    mul_f64(&a, &b, &out, 3);
    try testing.expect(math.isInf(out[0]));
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
}

test "mul_f32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    var a: [7]f32 = undefined;
    var b: [7]f32 = undefined;
    for (0..7) |i| {
        a[i] = @floatFromInt(i + 1);
        b[i] = 2.0;
    }
    var out: [7]f32 = undefined;
    mul_f32(&a, &b, &out, 7);
    for (0..7) |i| {
        const expected: f32 = @as(f32, @floatFromInt(i + 1)) * 2.0;
        try testing.expectApproxEqAbs(out[i], expected, 1e-5);
    }
}

test "mul_i32 SIMD boundary N=7" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, -2, 3, -4, 5, -6, 7 };
    const b = [_]i32{ -1, 2, -3, 4, -5, 6, -7 };
    var out: [7]i32 = undefined;
    mul_i32(&a, &b, &out, 7);
    try testing.expectEqual(out[0], -1);
    try testing.expectEqual(out[1], -4);
    try testing.expectEqual(out[2], -9);
    try testing.expectEqual(out[6], -49);
}

test "mul_i16 SIMD boundary N=9" {
    const testing = @import("std").testing;
    var a: [9]i16 = undefined;
    var b: [9]i16 = undefined;
    for (0..9) |i| {
        a[i] = @intCast(i + 1);
        b[i] = 3;
    }
    var out: [9]i16 = undefined;
    mul_i16(&a, &b, &out, 9);
    for (0..9) |i| {
        const expected: i16 = @intCast((i + 1) * 3);
        try testing.expectEqual(out[i], expected);
    }
}

test "mul_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    var a: [17]i8 = undefined;
    var b: [17]i8 = undefined;
    for (0..17) |i| {
        a[i] = 2;
        b[i] = @intCast(i);
    }
    var out: [17]i8 = undefined;
    mul_i8(&a, &b, &out, 17);
    for (0..17) |i| {
        const expected: i8 = @truncate(@as(i32, @intCast(i)) * 2);
        try testing.expectEqual(out[i], expected);
    }
}

test "mul_scalar_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    var out: [3]f64 = undefined;
    mul_scalar_f64(&a, &out, 3, 5.0);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 15.0, 1e-10);
}

test "mul_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, -2, 3, -4, 5 };
    var out: [5]i32 = undefined;
    mul_scalar_i32(&a, &out, 5, -3);
    try testing.expectEqual(out[0], -3);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], -9);
}

test "mul_c128 three elements" {
    const testing = @import("std").testing;
    // (0+1i)*(0+1i) = -1+0i
    const a = [_]f64{ 0, 1, 1, 0, 2, 3 };
    const b = [_]f64{ 0, 1, 1, 0, 4, 5 };
    var out: [6]f64 = undefined;
    mul_c128(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
    // (2+3i)*(4+5i) = 8+10i+12i-15 = -7+22i
    try testing.expectApproxEqAbs(out[4], -7.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 22.0, 1e-10);
}

test "mul_c64 basic" {
    const testing = @import("std").testing;
    // (1+2i)*(3+4i) = -5+10i
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4 };
    var out: [2]f32 = undefined;
    mul_c64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-5);
}

test "mul_scalar_c128 basic" {
    const testing = @import("std").testing;
    // (1+2i)*3 = (3+6i)
    const a = [_]f64{ 1, 2 };
    var out: [2]f64 = undefined;
    mul_scalar_c128(&a, &out, 1, 3.0);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 6.0, 1e-10);
}

test "mul_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 3, -4, 5 };
    const b = [_]i64{ 2, 3, -4 };
    var out: [3]i64 = undefined;
    mul_i64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], -12);
    try testing.expectEqual(out[2], -20);
}
