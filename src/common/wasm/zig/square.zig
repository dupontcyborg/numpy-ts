//! WASM element-wise square kernels for all numeric types.
//!
//! Unary: out[i] = a[i] * a[i]
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise square for f64 using 2-wide SIMD: out[i] = a[i] * a[i].
export fn square_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        simd.store2_f64(out, i, v * v);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * a[i];
    }
}

/// Element-wise square for f32 using 4-wide SIMD: out[i] = a[i] * a[i].
export fn square_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        simd.store4_f32(out, i, v * v);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * a[i];
    }
}

/// Element-wise complex128 square using 2-wide f64 SIMD.
/// Input/output are interleaved [re0, im0, re1, im1, ...], N = number of complex elements.
export fn square_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    var k: u32 = 0;
    while (k + 2 <= N) : (k += 2) {
        const idx = k * 2;
        const a0 = simd.load2_f64(a, idx);
        const a1 = simd.load2_f64(a, idx + 2);
        const re = @shuffle(f64, a0, a1, [2]i32{ 0, -1 });
        const im = @shuffle(f64, a0, a1, [2]i32{ 1, -2 });
        const out_re = re * re - im * im;
        const out_im = re * im + re * im; // 2·re·im
        const lo = @shuffle(f64, out_re, out_im, [2]i32{ 0, -1 });
        const hi = @shuffle(f64, out_re, out_im, [2]i32{ 1, -2 });
        simd.store2_f64(out, idx, lo);
        simd.store2_f64(out, idx + 2, hi);
    }
    while (k < N) : (k += 1) {
        const idx = k * 2;
        const re = a[idx];
        const im = a[idx + 1];
        out[idx] = re * re - im * im;
        out[idx + 1] = 2 * re * im;
    }
}

/// Element-wise complex64 square using 4-wide f32 SIMD.
/// Input/output are interleaved [re0, im0, re1, im1, ...], N = number of complex elements.
export fn square_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    var k: u32 = 0;
    while (k + 4 <= N) : (k += 4) {
        const idx = k * 2;
        const v0 = simd.load4_f32(a, idx);
        const v1 = simd.load4_f32(a, idx + 4);
        const re = @shuffle(f32, v0, v1, [4]i32{ 0, 2, -1, -3 });
        const im = @shuffle(f32, v0, v1, [4]i32{ 1, 3, -2, -4 });
        const out_re = re * re - im * im;
        const out_im = re * im + re * im;
        const lo = @shuffle(f32, out_re, out_im, [4]i32{ 0, -1, 1, -2 });
        const hi = @shuffle(f32, out_re, out_im, [4]i32{ 2, -3, 3, -4 });
        simd.store4_f32(out, idx, lo);
        simd.store4_f32(out, idx + 4, hi);
    }
    while (k < N) : (k += 1) {
        const idx = k * 2;
        const re = a[idx];
        const im = a[idx + 1];
        out[idx] = re * re - im * im;
        out[idx + 1] = 2 * re * im;
    }
}

/// Element-wise square for i64, scalar loop (no i64x2.mul in WASM SIMD).
/// Handles both signed (i64) and unsigned (u64).
export fn square_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i] *% a[i];
    }
}

/// Element-wise square for i32 using 4-wide SIMD with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn square_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_i32(a, i);
        simd.store4_i32(out, i, v *% v);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% a[i];
    }
}

/// Element-wise square for i16 using 8-wide SIMD with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn square_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = simd.load8_i16(a, i);
        simd.store8_i16(out, i, v *% v);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% a[i];
    }
}

/// Element-wise square for i8 using 16-wide SIMD with wrapping arithmetic.
/// Uses widened i16 multiply since WASM SIMD has no i8x16.mul.
/// Handles both signed (i8) and unsigned (u8).
export fn square_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const v = simd.load16_i8(a, i);
        const zero: simd.V16i8 = @splat(0);
        simd.store16_i8(out, i, simd.muladd_i8x16(zero, v, v));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] *% a[i];
    }
}

// --- Tests ---

test "square_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 2, 3, -4 };
    var out: [3]f64 = undefined;
    square_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 16.0, 1e-10);
}

test "square_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 2, 3, 10, 15 };
    var out: [4]i8 = undefined;
    square_i8(&a, &out, 4);
    try testing.expectEqual(out[0], 4);
    try testing.expectEqual(out[1], 9);
    try testing.expectEqual(out[2], 100);
    const expected: i8 = @truncate(@as(i32, 225));
    try testing.expectEqual(out[3], expected);
}

test "square_c128 basic" {
    const testing = @import("std").testing;
    // (3+4i)² = 9-16 + i(24) = -7+24i
    // (1+2i)² = 1-4 + i(4) = -3+4i
    // (0+1i)² = -1+0i
    const a = [_]f64{ 3, 4, 1, 2, 0, 1 };
    var out: [6]f64 = undefined;
    square_c128(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], -7.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 24.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 0.0, 1e-10);
}

test "square_c64 basic" {
    const testing = @import("std").testing;
    // (3+4i)² = -7+24i
    // (1+2i)² = -3+4i
    // (0+1i)² = -1+0i
    // (2+0i)² = 4+0i
    // (1+1i)² = 0+2i
    const a = [_]f32{ 3, 4, 1, 2, 0, 1, 2, 0, 1, 1 };
    var out: [10]f32 = undefined;
    square_c64(&a, &out, 5);
    try testing.expectApproxEqAbs(out[0], -7.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 24.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[6], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[7], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[8], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[9], 2.0, 1e-5);
}
