//! WASM 1D dot product kernels for all numeric types.
//!
//! Computes out[0] = sum_k a[k] * b[k] for vectors of length K..
//! Both a and b are accessed sequentially — ideal for SIMD.

const simd = @import("simd.zig");

/// Computes the dot product of two f64 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
export fn dot_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2f64 (2-wide)
    var acc: simd.V2f64 = @splat(0);

    // SIMD loop: 2 f64s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 2) {
        acc += simd.load2_f64(a, k) * simd.load2_f64(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: f64 = acc[0] + acc[1];
    while (k < K) : (k += 1) {
        sum += a[k] * b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two f32 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
export fn dot_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4f32 (4-wide)
    var acc: simd.V4f32 = @splat(0);

    // SIMD loop: 4 f32s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 4) {
        acc += simd.load4_f32(a, k) * simd.load4_f32(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: f32 = acc[0] + acc[1] + acc[2] + acc[3];
    while (k < K) : (k += 1) {
        sum += a[k] * b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two complex128 vectors of length K.
/// a and b are interleaved [re0, im0, re1, im1, ...]
/// out is also interleaved [re_out, im_out]
/// K is the number of complex elements (each = 2 f64s).
/// Uses SIMD deinterleaving: processes 2 complex elements (4 f64s) per iteration.
export fn dot_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, K: u32) void {
    var acc_re: simd.V2f64 = @splat(0);
    var acc_im: simd.V2f64 = @splat(0);

    // SIMD loop: 2 complex elements per iteration
    var k: usize = 0;
    while (k + 2 <= K) : (k += 2) {
        const idx = k * 2;
        const a0 = simd.load2_f64(a, idx); // [a0_re, a0_im]
        const a1 = simd.load2_f64(a, idx + 2); // [a1_re, a1_im]
        const b0 = simd.load2_f64(b, idx);
        const b1 = simd.load2_f64(b, idx + 2);
        const a_re = @shuffle(f64, a0, a1, [2]i32{ 0, -1 }); // [a0_re, a1_re]
        const a_im = @shuffle(f64, a0, a1, [2]i32{ 1, -2 }); // [a0_im, a1_im]
        const b_re = @shuffle(f64, b0, b1, [2]i32{ 0, -1 });
        const b_im = @shuffle(f64, b0, b1, [2]i32{ 1, -2 });
        acc_re += a_re * b_re - a_im * b_im;
        acc_im += a_re * b_im + a_im * b_re;
    }

    // Horizontal sum + scalar remainder
    var sum_re: f64 = acc_re[0] + acc_re[1];
    var sum_im: f64 = acc_im[0] + acc_im[1];
    while (k < K) : (k += 1) {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        sum_re += a_re * b_re - a_im * b_im;
        sum_im += a_re * b_im + a_im * b_re;
    }
    out[0] = sum_re;
    out[1] = sum_im;
}

/// Computes the dot product of two complex64 vectors of length K.
/// a and b are interleaved [re0, im0, re1, im1, ...]
/// out is also interleaved [re_out, im_out]
/// K is the number of complex elements (each = 2 f32s).
/// Uses SIMD deinterleaving: processes 4 complex elements (8 f32s) per iteration.
export fn dot_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, K: u32) void {
    var acc_re: simd.V4f32 = @splat(0);
    var acc_im: simd.V4f32 = @splat(0);

    // SIMD loop: 4 complex elements per iteration
    var k: usize = 0;
    while (k + 4 <= K) : (k += 4) {
        const idx = k * 2;
        const a0 = simd.load4_f32(a, idx); // [a0_re, a0_im, a1_re, a1_im]
        const a1 = simd.load4_f32(a, idx + 4); // [a2_re, a2_im, a3_re, a3_im]
        const b0 = simd.load4_f32(b, idx);
        const b1 = simd.load4_f32(b, idx + 4);
        const a_re = @shuffle(f32, a0, a1, [4]i32{ 0, 2, -1, -3 });
        const a_im = @shuffle(f32, a0, a1, [4]i32{ 1, 3, -2, -4 });
        const b_re = @shuffle(f32, b0, b1, [4]i32{ 0, 2, -1, -3 });
        const b_im = @shuffle(f32, b0, b1, [4]i32{ 1, 3, -2, -4 });
        acc_re += a_re * b_re - a_im * b_im;
        acc_im += a_re * b_im + a_im * b_re;
    }

    // Horizontal sum + scalar remainder
    var sum_re: f32 = acc_re[0] + acc_re[1] + acc_re[2] + acc_re[3];
    var sum_im: f32 = acc_im[0] + acc_im[1] + acc_im[2] + acc_im[3];
    while (k < K) : (k += 1) {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        sum_re += a_re * b_re - a_im * b_im;
        sum_im += a_re * b_im + a_im * b_re;
    }
    out[0] = sum_re;
    out[1] = sum_im;
}

/// Computes the dot product of two i64 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn dot_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2i64 (2-wide)
    var acc: simd.V2i64 = @splat(0);

    // SIMD loop: 2 i64s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 2) {
        acc +%= simd.load2_i64(a, k) *% simd.load2_i64(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: i64 = acc[0] +% acc[1];
    while (k < K) : (k += 1) {
        sum +%= a[k] *% b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two i64 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn dot_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4i32 (4-wide)
    var acc: simd.V4i32 = @splat(0);

    // SIMD loop: 4 i32s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 4) {
        acc +%= simd.load4_i32(a, k) *% simd.load4_i32(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
    while (k < K) : (k += 1) {
        sum +%= a[k] *% b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two i16 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
export fn dot_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, K: u32) void {
    const k_simd = K & ~@as(u32, 7); // floor to V8i16 (8-wide)
    var acc: simd.V8i16 = @splat(0);

    // SIMD loop: 8 i16s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 8) {
        acc +%= simd.load8_i16(a, k) *% simd.load8_i16(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: i16 = acc[0] +% acc[1] +% acc[2] +% acc[3] +% acc[4] +% acc[5] +% acc[6] +% acc[7];
    while (k < K) : (k += 1) {
        sum +%= a[k] *% b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two i8 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
export fn dot_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, K: u32) void {
    const k_simd = K & ~@as(u32, 15); // floor to V16i8 (16-wide)
    var acc: simd.V16i8 = @splat(0);

    // SIMD loop: 16 i8s per iteration (widened i16 multiply via muladd)
    var k: u32 = 0;
    while (k < k_simd) : (k += 16) {
        acc = simd.muladd_i8x16(acc, simd.load16_i8(a, k), simd.load16_i8(b, k));
    }

    // Horizontal sum of 16 lanes + scalar remainder
    var sum: i8 = 0;
    for (0..16) |lane| {
        sum +%= acc[lane];
    }
    while (k < K) : (k += 1) {
        sum +%= a[k] *% b[k];
    }
    out[0] = sum;
}

// --- Tests ---

test "dot_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4, 5 };
    const b = [_]f64{ 5, 4, 3, 2, 1 };
    var out: [1]f64 = undefined;
    dot_f64(&a, &b, &out, 5);
    // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 35
    try testing.expectApproxEqAbs(out[0], 35.0, 1e-10);
}

test "dot_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var out: [1]f32 = undefined;
    dot_f32(&a, &b, &out, 8);
    // 1*8+2*7+3*6+4*5+5*4+6*3+7*2+8*1 = 120
    try testing.expectApproxEqAbs(out[0], 120.0, 1e-5);
}

test "dot_c128 basic" {
    const testing = @import("std").testing;
    // (1+2i)*(3+4i) = 3+4i+6i-8 = -5+10i
    const a = [_]f64{ 1, 2 };
    const b = [_]f64{ 3, 4 };
    var out: [2]f64 = undefined;
    dot_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-10);
}

test "dot_c64 basic" {
    const testing = @import("std").testing;
    // (1+2i)*(3+4i) = -5+10i
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4 };
    var out: [2]f32 = undefined;
    dot_c64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-5);
}

test "dot_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 4, 5, 6 };
    var out: [1]i64 = undefined;
    dot_i64(&a, &b, &out, 3);
    // 1*4 + 2*5 + 3*6 = 32
    try testing.expectEqual(out[0], 32);
}

test "dot_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 4, 5, 6 };
    var out: [1]i32 = undefined;
    dot_i32(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 32);
}

test "dot_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i16{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var out: [1]i16 = undefined;
    dot_i16(&a, &b, &out, 8);
    // 1*8+2*7+3*6+4*5+5*4+6*3+7*2+8*1 = 120
    try testing.expectEqual(out[0], 120);
}

test "dot_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, 10, 10, 10 };
    const b = [_]i8{ 10, 10, 10, 10 };
    var out: [1]i8 = undefined;
    dot_i8(&a, &b, &out, 4);
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 4);
    try testing.expectEqual(out[0], expected);
}
