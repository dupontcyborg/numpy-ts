//! WASM vector-matrix product kernels for integer types.
//!
//! Computes y[j] = sum_k x[k] * A[k*N+j] for x[K] and A[K,N].
//! A is accessed column-wise, so we use the i-k-j broadcast pattern:
//! for each k, broadcast x[k] and accumulate x[k]*A[k,:] into y.

const simd = @import("simd.zig");

/// Computes the vector-matrix product of a K-length i64 vector and a KxN i64 matrix.
/// y[j] = sum_k x[k] * A[k*N+j]
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn vecmat_i64(x: [*]const i64, A: [*]const i64, y: [*]i64, K: u32, N: u32) void {
    // Zero output
    @memset(y[0..@as(usize, N)], 0);
    const n_simd = N & ~@as(u32, 1); // floor to V2i64 (2-wide)
    for (0..K) |k| {
        const x_k: simd.V2i64 = @splat(x[k]);
        const a_row = k * N;
        var j: u32 = 0;
        // Vectorized j loop: 2 i64s per step
        while (j < n_simd) : (j += 2) {
            simd.store2_i64(y, j, simd.load2_i64(y, j) +% x_k *% simd.load2_i64(A, a_row + j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            y[j] +%= x[k] *% A[a_row + j];
        }
    }
}

/// Computes the vector-matrix product of a K-length i32 vector and a KxN i32 matrix.
/// y[j] = sum_k x[k] * A[k*N+j]
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn vecmat_i32(x: [*]const i32, A: [*]const i32, y: [*]i32, K: u32, N: u32) void {
    // Zero output
    @memset(y[0..@as(usize, N)], 0);
    const n_simd = N & ~@as(u32, 3); // floor to V4i32 (4-wide)
    for (0..K) |k| {
        const x_k: simd.V4i32 = @splat(x[k]);
        const a_row = k * N;
        var j: u32 = 0;
        // Vectorized j loop: 4 i32s per step
        while (j < n_simd) : (j += 4) {
            simd.store4_i32(y, j, simd.load4_i32(y, j) +% x_k *% simd.load4_i32(A, a_row + j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            y[j] +%= x[k] *% A[a_row + j];
        }
    }
}

/// Computes the vector-matrix product of a K-length i16 vector and a KxN i16 matrix.
/// y[j] = sum_k x[k] * A[k*N+j]
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
export fn vecmat_i16(x: [*]const i16, A: [*]const i16, y: [*]i16, K: u32, N: u32) void {
    // Zero output
    @memset(y[0..@as(usize, N)], 0);
    const n_simd = N & ~@as(u32, 7); // floor to V8i16 (8-wide)
    for (0..K) |k| {
        const x_k: simd.V8i16 = @splat(x[k]);
        const a_row = k * N;
        var j: u32 = 0;
        // Vectorized j loop: 8 i16s per step
        while (j < n_simd) : (j += 8) {
            simd.store8_i16(y, j, simd.load8_i16(y, j) +% x_k *% simd.load8_i16(A, a_row + j));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            y[j] +%= x[k] *% A[a_row + j];
        }
    }
}

/// Computes the vector-matrix product of a K-length i8 vector and a KxN i8 matrix.
/// y[j] = sum_k x[k] * A[k*N+j]
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
export fn vecmat_i8(x: [*]const i8, A: [*]const i8, y: [*]i8, K: u32, N: u32) void {
    // Zero output
    @memset(y[0..@as(usize, N)], 0);
    const n_simd = N & ~@as(u32, 15); // floor to V16i8 (16-wide)
    for (0..K) |k| {
        const a_row = k * N;
        // Broadcast x[k] to 16 lanes
        const x_k: simd.V16i8 = @splat(x[k]);
        var j: u32 = 0;
        // Vectorized j loop: 16 i8s per step (widened i16 multiply via muladd)
        while (j < n_simd) : (j += 16) {
            const y_load = simd.load16_i8(y, j);
            simd.store16_i8(y, j, simd.muladd_i8x16(y_load, x_k, simd.load16_i8(A, a_row + j)));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            y[j] +%= x[k] *% A[a_row + j];
        }
    }
}

// --- Tests ---

test "vecmat_i64 basic" {
    const testing = @import("std").testing;
    const x = [_]i64{ 1, 2, 3 };
    const A = [_]i64{ 1, 2, 3, 4, 5, 6 };
    var y: [2]i64 = undefined;
    vecmat_i64(&x, &A, &y, 3, 2);
    try testing.expectEqual(y[0], 22);
    try testing.expectEqual(y[1], 28);
}

test "vecmat_i32 basic" {
    const testing = @import("std").testing;
    const x = [_]i32{ 1, 2, 3 };
    const A = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var y: [2]i32 = undefined;
    vecmat_i32(&x, &A, &y, 3, 2);
    try testing.expectEqual(y[0], 22);
    try testing.expectEqual(y[1], 28);
}

test "vecmat_i16 basic" {
    const testing = @import("std").testing;
    const x = [_]i16{ 1, 2, 3 };
    const A = [_]i16{ 1, 2, 3, 4, 5, 6 };
    var y: [2]i16 = undefined;
    vecmat_i16(&x, &A, &y, 3, 2);
    try testing.expectEqual(y[0], 22);
    try testing.expectEqual(y[1], 28);
}

test "vecmat_i8 wrapping" {
    const testing = @import("std").testing;
    // x = [10, 10], A = [[10, 10], [10, 10]], K=2, N=2
    // y[j] = 10*10 + 10*10 = 200, truncated to i8
    const x = [_]i8{ 10, 10 };
    const A = [_]i8{ 10, 10, 10, 10 };
    var y: [2]i8 = undefined;
    vecmat_i8(&x, &A, &y, 2, 2);
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 2);
    try testing.expectEqual(y[0], expected);
    try testing.expectEqual(y[1], expected);
}
