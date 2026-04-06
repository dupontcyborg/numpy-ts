//! WASM batched vector dot product kernels for float types (f64, f32, c128, c64).
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
            acc = simd.mulAdd_f64x2(simd.load2_f64(a, row + k), simd.load2_f64(b, row + k), acc);
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
            acc = simd.mulAdd_f32x4(simd.load4_f32(a, row + k), simd.load4_f32(b, row + k), acc);
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

// --- Tests ---

test "vecdot_f64 batch=3 K=2" {
    const testing = @import("std").testing;
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
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 26.0, 1e-5);
}

test "vecdot_c128 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var out: [2]f64 = undefined;
    vecdot_c128(&a, &b, &out, 1, 2);
    try testing.expectApproxEqAbs(out[0], 70.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -8.0, 1e-10);
}

test "vecdot_c64 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [2]f32 = undefined;
    vecdot_c64(&a, &b, &out, 1, 2);
    try testing.expectApproxEqAbs(out[0], 70.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -8.0, 1e-5);
}
