//! WASM matrix-vector product kernels for all numeric types.
//!
//! Computes y[i] = sum_k A[i*K+k] * x[k] for A[M,K] and x[K].
//! A is accessed row-wise, x is accessed sequentially — both cache-friendly.

const simd = @import("simd.zig");

/// Computes the matrix-vector product of an M×K f64 matrix A and a K-length f64 vector x.
/// y is an M-length f64 vector output.
export fn matvec_f64(A: [*]const f64, x: [*]const f64, y: [*]f64, M: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2f64 (2-wide)
    for (0..M) |i| {
        const a_row = i * K;
        var acc: simd.V2f64 = @splat(0);

        // SIMD loop: 2 f64s per iteration
        var k: u32 = 0;
        while (k < k_simd) : (k += 2) {
            acc += simd.load2_f64(A, a_row + k) * simd.load2_f64(x, k);
        }

        // Horizontal sum + scalar remainder
        var sum: f64 = acc[0] + acc[1];
        while (k < K) : (k += 1) {
            sum += A[a_row + k] * x[k];
        }
        y[i] = sum;
    }
}

/// Computes the matrix-vector product of an M×K f32 matrix A and a K-length f32 vector x.
/// y is an M-length f32 vector output.
export fn matvec_f32(A: [*]const f32, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4f32 (4-wide)
    for (0..M) |i| {
        const a_row = i * K;
        var acc: simd.V4f32 = @splat(0);

        // SIMD loop: 4 f32s per iteration
        var k: u32 = 0;
        while (k < k_simd) : (k += 4) {
            acc += simd.load4_f32(A, a_row + k) * simd.load4_f32(x, k);
        }

        // Horizontal sum + scalar remainder
        var sum: f32 = acc[0] + acc[1] + acc[2] + acc[3];
        while (k < K) : (k += 1) {
            sum += A[a_row + k] * x[k];
        }
        y[i] = sum;
    }
}

/// Computes the matrix-vector product of an M×K complex128 matrix A and a K-length complex128 vector x.
/// A, x, and y are interleaved [re0, im0, re1, im1, ...].
/// y[i] = sum_k A[i*K+k] * x[k] with complex multiplication.
export fn matvec_c128(A: [*]const f64, x: [*]const f64, y: [*]f64, M: u32, K: u32) void {
    for (0..M) |i| {
        const a_row = i * K * 2; // 2 f64s per complex element
        var sum_re: f64 = 0;
        var sum_im: f64 = 0;

        // Scalar loop: complex multiply-accumulate
        for (0..K) |k| {
            const idx = k * 2;
            const a_re = A[a_row + idx];
            const a_im = A[a_row + idx + 1];
            const x_re = x[idx];
            const x_im = x[idx + 1];
            // (a_re + a_im*i) * (x_re + x_im*i)
            sum_re += a_re * x_re - a_im * x_im;
            sum_im += a_re * x_im + a_im * x_re;
        }
        y[i * 2] = sum_re;
        y[i * 2 + 1] = sum_im;
    }
}

/// Computes the matrix-vector product of an M×K complex64 matrix A and a K-length complex64 vector x.
/// A, x, and y are interleaved [re0, im0, re1, im1, ...].
/// y[i] = sum_k A[i*K+k] * x[k] with complex multiplication.
export fn matvec_c64(A: [*]const f32, x: [*]const f32, y: [*]f32, M: u32, K: u32) void {
    for (0..M) |i| {
        const a_row = i * K * 2; // 2 f32s per complex element
        var sum_re: f32 = 0;
        var sum_im: f32 = 0;

        // Scalar loop: complex multiply-accumulate
        for (0..K) |k| {
            const idx = k * 2;
            const a_re = A[a_row + idx];
            const a_im = A[a_row + idx + 1];
            const x_re = x[idx];
            const x_im = x[idx + 1];
            // (a_re + a_im*i) * (x_re + x_im*i)
            sum_re += a_re * x_re - a_im * x_im;
            sum_im += a_re * x_im + a_im * x_re;
        }
        y[i * 2] = sum_re;
        y[i * 2 + 1] = sum_im;
    }
}

/// Computes the matrix-vector product of an M×K i64 matrix A and a K-length i64 vector x.
/// y is an M-length i64 vector output.
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn matvec_i64(A: [*]const i64, x: [*]const i64, y: [*]i64, M: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2i64 (2-wide)
    for (0..M) |i| {
        const a_row = i * K;
        var acc: simd.V2i64 = @splat(0);

        // SIMD loop: 2 i64s per iteration
        var k: u32 = 0;
        while (k < k_simd) : (k += 2) {
            acc +%= simd.load2_i64(A, a_row + k) *% simd.load2_i64(x, k);
        }

        // Horizontal sum + scalar remainder
        var sum: i64 = acc[0] +% acc[1];
        while (k < K) : (k += 1) {
            sum +%= A[a_row + k] *% x[k];
        }
        y[i] = sum;
    }
}

/// Computes the matrix-vector product of an M×K i32 matrix A and a K-length i32 vector x.
/// y is an M-length i32 vector output.
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn matvec_i32(A: [*]const i32, x: [*]const i32, y: [*]i32, M: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4i32 (4-wide)
    for (0..M) |i| {
        const a_row = i * K;
        var acc: simd.V4i32 = @splat(0);

        // SIMD loop: 4 i32s per iteration
        var k: u32 = 0;
        while (k < k_simd) : (k += 4) {
            acc +%= simd.load4_i32(A, a_row + k) *% simd.load4_i32(x, k);
        }

        // Horizontal sum + scalar remainder
        var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
        while (k < K) : (k += 1) {
            sum +%= A[a_row + k] *% x[k];
        }
        y[i] = sum;
    }
}

/// Computes the matrix-vector product of an M×K i16 matrix A and a K-length i16 vector x.
/// y is an M-length i16 vector output.
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
export fn matvec_i16(A: [*]const i16, x: [*]const i16, y: [*]i16, M: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 7); // floor to V8i16 (8-wide)
    for (0..M) |i| {
        const a_row = i * K;
        var acc: simd.V8i16 = @splat(0);

        // SIMD loop: 8 i16s per iteration
        var k: u32 = 0;
        while (k < k_simd) : (k += 8) {
            acc +%= simd.load8_i16(A, a_row + k) *% simd.load8_i16(x, k);
        }

        // Horizontal sum + scalar remainder
        var sum: i16 = acc[0] +% acc[1] +% acc[2] +% acc[3] +% acc[4] +% acc[5] +% acc[6] +% acc[7];
        while (k < K) : (k += 1) {
            sum +%= A[a_row + k] *% x[k];
        }
        y[i] = sum;
    }
}

/// Computes the matrix-vector product of an M×K i8 matrix A and a K-length i8 vector x.
/// y is an M-length i8 vector output.
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
export fn matvec_i8(A: [*]const i8, x: [*]const i8, y: [*]i8, M: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 15); // floor to V16i8 (16-wide)
    for (0..M) |i| {
        const a_row = i * K;
        var acc: simd.V16i8 = @splat(0);

        // SIMD loop: 16 i8s per iteration (widened i16 multiply via muladd)
        var k: u32 = 0;
        while (k < k_simd) : (k += 16) {
            acc = simd.muladd_i8x16(acc, simd.load16_i8(A, a_row + k), simd.load16_i8(x, k));
        }

        // Horizontal sum of 16 lanes + scalar remainder
        var sum: i8 = 0;
        for (0..16) |lane| {
            sum +%= acc[lane];
        }
        while (k < K) : (k += 1) {
            sum +%= A[a_row + k] *% x[k];
        }
        y[i] = sum;
    }
}

// --- Tests ---

test "matvec_f64 2x3 · 3 → 2" {
    const testing = @import("std").testing;
    // A = [[1,2,3],[4,5,6]], x = [1,2,3]
    // y[0] = 1+4+9 = 14, y[1] = 4+10+18 = 32
    const A = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const x = [_]f64{ 1, 2, 3 };
    var y: [2]f64 = undefined;
    matvec_f64(&A, &x, &y, 2, 3);
    try testing.expectApproxEqAbs(y[0], 14.0, 1e-10);
    try testing.expectApproxEqAbs(y[1], 32.0, 1e-10);
}

test "matvec_f32 basic" {
    const testing = @import("std").testing;
    const A = [_]f32{ 1, 0, 0, 1, 1, 1 };
    const x = [_]f32{ 3, 4 };
    var y: [3]f32 = undefined;
    matvec_f32(&A, &x, &y, 3, 2);
    try testing.expectApproxEqAbs(y[0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(y[1], 4.0, 1e-5);
    try testing.expectApproxEqAbs(y[2], 7.0, 1e-5);
}

test "matvec_c128 2x2 · 2 → 2" {
    const testing = @import("std").testing;
    // A = [[(1+0i),(0+0i)],[(0+0i),(1+0i)]], x = [(3+4i),(5+6i)]
    // Identity matrix → y = x
    const A = [_]f64{ 1, 0, 0, 0, 0, 0, 1, 0 };
    const x = [_]f64{ 3, 4, 5, 6 };
    var y: [4]f64 = undefined;
    matvec_c128(&A, &x, &y, 2, 2);
    try testing.expectApproxEqAbs(y[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(y[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(y[2], 5.0, 1e-10);
    try testing.expectApproxEqAbs(y[3], 6.0, 1e-10);
}

test "matvec_c64 basic" {
    const testing = @import("std").testing;
    // A = [[(1+0i),(0+0i)],[(0+0i),(1+0i)]], x = [(3+4i),(5+6i)]
    const A = [_]f32{ 1, 0, 0, 0, 0, 0, 1, 0 };
    const x = [_]f32{ 3, 4, 5, 6 };
    var y: [4]f32 = undefined;
    matvec_c64(&A, &x, &y, 2, 2);
    try testing.expectApproxEqAbs(y[0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(y[1], 4.0, 1e-5);
    try testing.expectApproxEqAbs(y[2], 5.0, 1e-5);
    try testing.expectApproxEqAbs(y[3], 6.0, 1e-5);
}

test "matvec_i64 basic" {
    const testing = @import("std").testing;
    const A = [_]i64{ 1, 2, 3, 4, 5, 6 };
    const x = [_]i64{ 1, 2, 3 };
    var y: [2]i64 = undefined;
    matvec_i64(&A, &x, &y, 2, 3);
    // y[0] = 1+4+9 = 14, y[1] = 4+10+18 = 32
    try testing.expectEqual(y[0], 14);
    try testing.expectEqual(y[1], 32);
}

test "matvec_i32 basic" {
    const testing = @import("std").testing;
    const A = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const x = [_]i32{ 1, 2, 3 };
    var y: [2]i32 = undefined;
    matvec_i32(&A, &x, &y, 2, 3);
    try testing.expectEqual(y[0], 14);
    try testing.expectEqual(y[1], 32);
}

test "matvec_i16 basic" {
    const testing = @import("std").testing;
    const A = [_]i16{ 1, 2, 3, 4, 5, 6 };
    const x = [_]i16{ 1, 2, 3 };
    var y: [2]i16 = undefined;
    matvec_i16(&A, &x, &y, 2, 3);
    try testing.expectEqual(y[0], 14);
    try testing.expectEqual(y[1], 32);
}

test "matvec_i8 wrapping" {
    const testing = @import("std").testing;
    const A = [_]i8{ 10, 10, 10, 10 };
    const x = [_]i8{ 10, 10 };
    var y: [2]i8 = undefined;
    matvec_i8(&A, &x, &y, 2, 2);
    // 10*10 + 10*10 = 200 → wraps in i8
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 2);
    try testing.expectEqual(y[0], expected);
    try testing.expectEqual(y[1], expected);
}
