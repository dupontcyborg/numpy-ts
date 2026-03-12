//! WASM batched vector dot product kernels for all numeric types.
//!
//! Computes out[i] = sum_k a[i*K+k] * b[i*K+k] for batch size B and vector length K.
//! Both a and b are accessed sequentially — ideal for SIMD.
//! This is the "paired" dot product (vecdot), not the all-pairs inner product.

const simd = @import("simd.zig");

/// Computes the dot product of B pairs of f64 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
export fn vecdot_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2f64 (2-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V2f64 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 2 f64s per iteration
        while (k < k_simd) : (k += 2) {
            acc += simd.load2_f64(a, row + k) * simd.load2_f64(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: f64 = acc[0] + acc[1];
        while (k < K) : (k += 1) {
            sum += a[row + k] * b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes the dot product of B pairs of f32 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
export fn vecdot_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4f32 (4-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V4f32 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 4 f32s per iteration
        while (k < k_simd) : (k += 4) {
            acc += simd.load4_f32(a, row + k) * simd.load4_f32(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: f32 = acc[0] + acc[1] + acc[2] + acc[3];
        while (k < K) : (k += 1) {
            sum += a[row + k] * b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes vecdot of B pairs of complex128 vectors of length K.
/// vecdot computes sum(conj(a) * b), i.e. the conjugate of a is used.
/// a and b are interleaved [re0, im0, re1, im1, ...]
/// out is also interleaved [re_out0, im_out0, re_out1, im_out1, ...]
/// K is the number of complex elements (each = 2 f64s).
export fn vecdot_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, B: u32, K: u32) void {
    for (0..B) |i| {
        const row = i * K * 2; // 2 f64s per complex element
        var sum_re: f64 = 0;
        var sum_im: f64 = 0;
        // Scalar loop: conj(a) * b multiply-accumulate
        for (0..K) |k| {
            const idx = k * 2;
            const a_re = a[row + idx];
            const a_im = a[row + idx + 1];
            const b_re = b[row + idx];
            const b_im = b[row + idx + 1];
            // conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
            sum_re += a_re * b_re + a_im * b_im;
            sum_im += a_re * b_im - a_im * b_re;
        }
        out[i * 2] = sum_re;
        out[i * 2 + 1] = sum_im;
    }
}

/// Computes vecdot of B pairs of complex64 vectors of length K.
/// vecdot computes sum(conj(a) * b), i.e. the conjugate of a is used.
/// a and b are interleaved [re0, im0, re1, im1, ...]
/// out is also interleaved [re_out0, im_out0, re_out1, im_out1, ...]
/// K is the number of complex elements (each = 2 f32s).
export fn vecdot_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, B: u32, K: u32) void {
    for (0..B) |i| {
        const row = i * K * 2; // 2 f32s per complex element
        var sum_re: f32 = 0;
        var sum_im: f32 = 0;
        // Scalar loop: conj(a) * b multiply-accumulate
        for (0..K) |k| {
            const idx = k * 2;
            const a_re = a[row + idx];
            const a_im = a[row + idx + 1];
            const b_re = b[row + idx];
            const b_im = b[row + idx + 1];
            // conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
            sum_re += a_re * b_re + a_im * b_im;
            sum_im += a_re * b_im - a_im * b_re;
        }
        out[i * 2] = sum_re;
        out[i * 2 + 1] = sum_im;
    }
}

/// Computes the dot product of B pairs of i64 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn vecdot_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2i64 (2-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V2i64 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 2 i64s per iteration
        while (k < k_simd) : (k += 2) {
            acc +%= simd.load2_i64(a, row + k) *% simd.load2_i64(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: i64 = acc[0] +% acc[1];
        while (k < K) : (k += 1) {
            sum +%= a[row + k] *% b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes the dot product of B pairs of i32 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn vecdot_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4i32 (4-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V4i32 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 4 i32s per iteration
        while (k < k_simd) : (k += 4) {
            acc +%= simd.load4_i32(a, row + k) *% simd.load4_i32(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
        while (k < K) : (k += 1) {
            sum +%= a[row + k] *% b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes the dot product of B pairs of i16 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
export fn vecdot_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 7); // floor to V8i16 (8-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V8i16 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 8 i16s per iteration
        while (k < k_simd) : (k += 8) {
            acc +%= simd.load8_i16(a, row + k) *% simd.load8_i16(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: i16 = acc[0] +% acc[1] +% acc[2] +% acc[3] +% acc[4] +% acc[5] +% acc[6] +% acc[7];
        while (k < K) : (k += 1) {
            sum +%= a[row + k] *% b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes the dot product of B pairs of i8 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
export fn vecdot_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 15); // floor to V16i8 (16-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V16i8 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 16 i8s per iteration (widened i16 multiply via muladd)
        while (k < k_simd) : (k += 16) {
            acc = simd.muladd_i8x16(acc, simd.load16_i8(a, row + k), simd.load16_i8(b, row + k));
        }
        // Horizontal sum of 16 lanes + scalar remainder
        var sum: i8 = 0;
        for (0..16) |lane| {
            sum +%= acc[lane];
        }
        while (k < K) : (k += 1) {
            sum +%= a[row + k] *% b[row + k];
        }
        out[i] = sum;
    }
}

// --- Tests ---

test "vecdot_f64 batch=3 K=2" {
    const testing = @import("std").testing;
    // a = [[1,2],[3,4],[5,6]], b = [[7,8],[9,10],[11,12]]
    // out[0] = 1*7+2*8 = 23, out[1] = 3*9+4*10 = 67, out[2] = 5*11+6*12 = 127
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var out: [3]f64 = undefined;
    vecdot_f64(&a, &b, &out, 3, 2);
    try testing.expectApproxEqAbs(out[0], 23.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 67.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 127.0, 1e-10);
}

test "vecdot_f32 batch=2 K=4" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 1, 0, 0, 1, 1, 1, 1, 1 };
    var out: [2]f32 = undefined;
    vecdot_f32(&a, &b, &out, 2, 4);
    // out[0] = 1*1+2*0+3*0+4*1 = 5
    // out[1] = 5*1+6*1+7*1+8*1 = 26
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 26.0, 1e-5);
}

test "vecdot_c128 basic" {
    const testing = @import("std").testing;
    // a = [(1+2i), (3+4i)], b = [(5+6i), (7+8i)], batch=1, K=2
    // conj(1+2i)*(5+6i) = (1-2i)(5+6i) = 5+6i-10i+12 = 17-4i
    // conj(3+4i)*(7+8i) = (3-4i)(7+8i) = 21+24i-28i+32 = 53-4i
    // sum = 70-8i
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var out: [2]f64 = undefined;
    vecdot_c128(&a, &b, &out, 1, 2);
    try testing.expectApproxEqAbs(out[0], 70.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -8.0, 1e-10);
}

test "vecdot_c64 basic" {
    const testing = @import("std").testing;
    // Same as c128 but f32
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [2]f32 = undefined;
    vecdot_c64(&a, &b, &out, 1, 2);
    try testing.expectApproxEqAbs(out[0], 70.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -8.0, 1e-5);
}

test "vecdot_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i64{ 7, 8, 9, 10, 11, 12 };
    var out: [3]i64 = undefined;
    vecdot_i64(&a, &b, &out, 3, 2);
    try testing.expectEqual(out[0], 23);
    try testing.expectEqual(out[1], 67);
    try testing.expectEqual(out[2], 127);
}

test "vecdot_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i32{ 7, 8, 9, 10, 11, 12 };
    var out: [3]i32 = undefined;
    vecdot_i32(&a, &b, &out, 3, 2);
    try testing.expectEqual(out[0], 23);
    try testing.expectEqual(out[1], 67);
    try testing.expectEqual(out[2], 127);
}

test "vecdot_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i16{ 7, 8, 9, 10, 11, 12 };
    var out: [3]i16 = undefined;
    vecdot_i16(&a, &b, &out, 3, 2);
    try testing.expectEqual(out[0], 23);
    try testing.expectEqual(out[1], 67);
    try testing.expectEqual(out[2], 127);
}

test "vecdot_i8 wrapping" {
    const testing = @import("std").testing;
    // batch=2, K=2: a = [[10,10],[10,10]], b = [[10,10],[10,10]]
    // each dot = 10*10 + 10*10 = 200, truncated to i8
    const a = [_]i8{ 10, 10, 10, 10 };
    const b = [_]i8{ 10, 10, 10, 10 };
    var out: [2]i8 = undefined;
    vecdot_i8(&a, &b, &out, 2, 2);
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 2);
    try testing.expectEqual(out[0], expected);
    try testing.expectEqual(out[1], expected);
}
