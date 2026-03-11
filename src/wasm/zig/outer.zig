//! WASM outer product kernels for all numeric types.
//!
//! Computes C[i*N+j] = a[i] * b[j] for a[M] and b[N].
//! No contraction axis — pure broadcast multiply.

const simd = @import("simd.zig");

/// Computes the outer product of an M-length f64 vector a and an N-length f64 vector b.
/// C is an M×N f64 matrix output.
export fn outer_f64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32) void {
    const n_simd = N & ~@as(u32, 1); // floor to V2f64 (2-wide)
    for (0..M) |i| {
        const a_i: simd.V2f64 = @splat(a[i]);
        const c_row = i * N;

        // Vectorized j loop: 2 f64s per step
        var j: u32 = 0;
        while (j < n_simd) : (j += 2) {
            simd.store2_f64(c, c_row + j, a_i * simd.load2_f64(b, j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            c[c_row + j] = a[i] * b[j];
        }
    }
}

/// Computes the outer product of an M-length f32 vector a and an N-length f32 vector b.
/// C is an M×N f32 matrix output.
export fn outer_f32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32) void {
    const n_simd = N & ~@as(u32, 3); // floor to V4f32 (4-wide)
    for (0..M) |i| {
        const a_i: simd.V4f32 = @splat(a[i]);
        const c_row = i * N;

        // Vectorized j loop: 4 f32s per step
        var j: u32 = 0;
        while (j < n_simd) : (j += 4) {
            simd.store4_f32(c, c_row + j, a_i * simd.load4_f32(b, j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            c[c_row + j] = a[i] * b[j];
        }
    }
}

/// Computes the outer product of an M-length complex128 vector a and an N-length complex128 vector b.
/// a, b, and c are interleaved [re0, im0, re1, im1, ...] for each element.
/// C[i,j] = a[i] * b[j] with complex multiplication.
export fn outer_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32) void {
    for (0..M) |i| {
        const a_re = a[i * 2];
        const a_im = a[i * 2 + 1];
        const c_row = i * N * 2; // 2 f64s per complex element

        // Scalar loop: complex multiply
        for (0..N) |j| {
            const idx = j * 2;
            const b_re = b[idx];
            const b_im = b[idx + 1];
            // (a_re + a_im*i) * (b_re + b_im*i)
            c[c_row + idx] = a_re * b_re - a_im * b_im;
            c[c_row + idx + 1] = a_re * b_im + a_im * b_re;
        }
    }
}

/// Computes the outer product of an M-length complex64 vector a and an N-length complex64 vector b.
/// a, b, and c are interleaved [re0, im0, re1, im1, ...] for each element.
/// C[i,j] = a[i] * b[j] with complex multiplication.
export fn outer_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32) void {
    for (0..M) |i| {
        const a_re = a[i * 2];
        const a_im = a[i * 2 + 1];
        const c_row = i * N * 2; // 2 f32s per complex element

        // Scalar loop: complex multiply
        for (0..N) |j| {
            const idx = j * 2;
            const b_re = b[idx];
            const b_im = b[idx + 1];
            // (a_re + a_im*i) * (b_re + b_im*i)
            c[c_row + idx] = a_re * b_re - a_im * b_im;
            c[c_row + idx + 1] = a_re * b_im + a_im * b_re;
        }
    }
}

/// Computes the outer product of an M-length i64 vector a and an N-length i64 vector b.
/// C is an M×N i64 matrix output.
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn outer_i64(a: [*]const i64, b: [*]const i64, c: [*]i64, M: u32, N: u32) void {
    const n_simd = N & ~@as(u32, 1); // floor to V2i64 (2-wide)
    for (0..M) |i| {
        const a_i: simd.V2i64 = @splat(a[i]);
        const c_row = i * N;

        // Vectorized j loop: 2 i64s per step
        var j: u32 = 0;
        while (j < n_simd) : (j += 2) {
            simd.store2_i64(c, c_row + j, a_i *% simd.load2_i64(b, j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            c[c_row + j] = a[i] *% b[j];
        }
    }
}

/// Computes the outer product of an M-length i32 vector a and an N-length i32 vector b.
/// C is an M×N i32 matrix output.
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn outer_i32(a: [*]const i32, b: [*]const i32, c: [*]i32, M: u32, N: u32) void {
    const n_simd = N & ~@as(u32, 3); // floor to V4i32 (4-wide)
    for (0..M) |i| {
        const a_i: simd.V4i32 = @splat(a[i]);
        const c_row = i * N;

        // Vectorized j loop: 4 i32s per step
        var j: u32 = 0;
        while (j < n_simd) : (j += 4) {
            simd.store4_i32(c, c_row + j, a_i *% simd.load4_i32(b, j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            c[c_row + j] = a[i] *% b[j];
        }
    }
}

/// Computes the outer product of an M-length i16 vector a and an N-length i16 vector b.
/// C is an M×N i16 matrix output.
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
export fn outer_i16(a: [*]const i16, b: [*]const i16, c: [*]i16, M: u32, N: u32) void {
    const n_simd = N & ~@as(u32, 7); // floor to V8i16 (8-wide)
    for (0..M) |i| {
        const a_i: simd.V8i16 = @splat(a[i]);
        const c_row = i * N;

        // Vectorized j loop: 8 i16s per step
        var j: u32 = 0;
        while (j < n_simd) : (j += 8) {
            simd.store8_i16(c, c_row + j, a_i *% simd.load8_i16(b, j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            c[c_row + j] = a[i] *% b[j];
        }
    }
}

/// Computes the outer product of an M-length i8 vector a and an N-length i8 vector b.
/// C is an M×N i8 matrix output.
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
export fn outer_i8(a: [*]const i8, b: [*]const i8, c: [*]i8, M: u32, N: u32) void {
    // Each row is just a[i] * b[:] — broadcast + muladd with acc=0
    const n_simd = N & ~@as(u32, 15); // floor to V16i8 (16-wide)
    const zero: simd.V16i8 = @splat(0);
    for (0..M) |i| {
        const a_i: simd.V16i8 = @splat(a[i]);
        const c_row = i * N;

        // Vectorized j loop: 16 i8s per step (widened i16 multiply via muladd)
        var j: u32 = 0;
        while (j < n_simd) : (j += 16) {
            simd.store16_i8(c, c_row + j, simd.muladd_i8x16(zero, a_i, simd.load16_i8(b, j)));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            c[c_row + j] = a[i] *% b[j];
        }
    }
}

// --- Tests ---

test "outer_f64 3 x 2 → 3x2" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    const b = [_]f64{ 4, 5 };
    var c: [6]f64 = undefined;
    outer_f64(&a, &b, &c, 3, 2);
    try testing.expectApproxEqAbs(c[0], 4.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 5.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 8.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 10.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 12.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 15.0, 1e-10);
}

test "outer_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4, 5 };
    var c: [6]f32 = undefined;
    outer_f32(&a, &b, &c, 2, 3);
    try testing.expectApproxEqAbs(c[0], 3.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 4.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 5.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 6.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 8.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 10.0, 1e-5);
}

test "outer_c128 2 x 2 → 2x2" {
    const testing = @import("std").testing;
    // a = [(1+2i), (3+0i)], b = [(0+1i), (1+0i)]
    // [0,0] = (1+2i)*(0+1i) = -2+1i
    // [0,1] = (1+2i)*(1+0i) = 1+2i
    // [1,0] = (3+0i)*(0+1i) = 0+3i
    // [1,1] = (3+0i)*(1+0i) = 3+0i
    const a = [_]f64{ 1, 2, 3, 0 };
    const b = [_]f64{ 0, 1, 1, 0 };
    var c: [8]f64 = undefined;
    outer_c128(&a, &b, &c, 2, 2);
    try testing.expectApproxEqAbs(c[0], -2.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 1.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 0.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 3.0, 1e-10);
    try testing.expectApproxEqAbs(c[6], 3.0, 1e-10);
    try testing.expectApproxEqAbs(c[7], 0.0, 1e-10);
}

test "outer_c64 basic" {
    const testing = @import("std").testing;
    // a = [(1+0i)], b = [(2+3i)]
    // [0,0] = (1+0i)*(2+3i) = 2+3i
    const a = [_]f32{ 1, 0 };
    const b = [_]f32{ 2, 3 };
    var c: [2]f32 = undefined;
    outer_c64(&a, &b, &c, 1, 1);
    try testing.expectApproxEqAbs(c[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 3.0, 1e-5);
}

test "outer_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 4, 5 };
    var c: [6]i64 = undefined;
    outer_i64(&a, &b, &c, 3, 2);
    try testing.expectEqual(c[0], 4);
    try testing.expectEqual(c[1], 5);
    try testing.expectEqual(c[2], 8);
    try testing.expectEqual(c[3], 10);
    try testing.expectEqual(c[4], 12);
    try testing.expectEqual(c[5], 15);
}

test "outer_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 4, 5 };
    var c: [6]i32 = undefined;
    outer_i32(&a, &b, &c, 3, 2);
    try testing.expectEqual(c[0], 4);
    try testing.expectEqual(c[1], 5);
    try testing.expectEqual(c[2], 8);
    try testing.expectEqual(c[3], 10);
    try testing.expectEqual(c[4], 12);
    try testing.expectEqual(c[5], 15);
}

test "outer_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3 };
    const b = [_]i16{ 4, 5 };
    var c: [6]i16 = undefined;
    outer_i16(&a, &b, &c, 3, 2);
    try testing.expectEqual(c[0], 4);
    try testing.expectEqual(c[1], 5);
    try testing.expectEqual(c[2], 8);
    try testing.expectEqual(c[3], 10);
    try testing.expectEqual(c[4], 12);
    try testing.expectEqual(c[5], 15);
}

test "outer_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{20};
    const b = [_]i8{20};
    var c: [1]i8 = undefined;
    outer_i8(&a, &b, &c, 1, 1);
    // 20*20 = 400 → wraps in i8
    const expected: i8 = @truncate(@as(i32, 20) * 20);
    try testing.expectEqual(c[0], expected);
}
