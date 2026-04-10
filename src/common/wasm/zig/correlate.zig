//! WASM 1D cross-correlation kernel for float types.
//!
//! Computes the full cross-correlation: out[k] = sum_n(a[n] * v[n - k + vLen - 1])
//! for k = 0..aLen+vLen-2.
//! The JS layer handles mode slicing (full/same/valid), complex types,
//! and dtype conversion (int→float) before calling these kernels.
//!
//! Uses dual SIMD accumulators for instruction-level parallelism.

const simd = @import("simd.zig");

/// Full 1D cross-correlation for f64.
/// out must have length na + nb - 1.
export fn correlate_f64(a: [*]const f64, na: u32, b: [*]const f64, nb: u32, out: [*]f64, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var acc0: simd.V2f64 = @splat(0.0);
        var acc1: simd.V2f64 = @splat(0.0);

        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;
        // b_off may wrap (usize underflow) but j + b_off is always valid
        const b_off = (n_b - 1) -% k;

        var j = j_start;

        // 4-wide: two V2f64 accumulators
        while (j + 4 <= j_end) : (j += 4) {
            const bi = j +% b_off;
            acc0 += simd.load2_f64(a, j) * simd.load2_f64(b, bi);
            acc1 += simd.load2_f64(a, j + 2) * simd.load2_f64(b, bi + 2);
        }
        // 2-wide remainder
        while (j + 2 <= j_end) : (j += 2) {
            acc0 += simd.load2_f64(a, j) * simd.load2_f64(b, j +% b_off);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        // Scalar remainder
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[j +% b_off];
        }
        out[k] = sum;
    }
}

/// Full 1D cross-correlation for f32.
/// out must have length na + nb - 1.
export fn correlate_f32(a: [*]const f32, na: u32, b: [*]const f32, nb: u32, out: [*]f32, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var acc0: simd.V4f32 = @splat(0.0);
        var acc1: simd.V4f32 = @splat(0.0);

        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;
        const b_off = (n_b - 1) -% k;

        var j = j_start;

        // 8-wide: two V4f32 accumulators
        while (j + 8 <= j_end) : (j += 8) {
            const bi = j +% b_off;
            acc0 += simd.load4_f32(a, j) * simd.load4_f32(b, bi);
            acc1 += simd.load4_f32(a, j + 4) * simd.load4_f32(b, bi + 4);
        }
        // 4-wide remainder
        while (j + 4 <= j_end) : (j += 4) {
            acc0 += simd.load4_f32(a, j) * simd.load4_f32(b, j +% b_off);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        // Scalar remainder
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[j +% b_off];
        }
        out[k] = sum;
    }
}

// --- Integer correlate kernels (scalar loop, same-type output) ---

fn correlateInt(comptime T: type, a: [*]const T, na: u32, b: [*]const T, nb: u32, out: [*]T, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var sum: T = 0;
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;
        var j = j_start;
        while (j < j_end) : (j += 1) {
            sum +%= a[j] *% b[j +% ((n_b - 1) -% k)];
        }
        out[k] = sum;
    }
}

export fn correlate_i32(a: [*]const i32, na: u32, b: [*]const i32, nb: u32, out: [*]i32, outLen: u32) void {
    correlateInt(i32, a, na, b, nb, out, outLen);
}
export fn correlate_u32(a: [*]const u32, na: u32, b: [*]const u32, nb: u32, out: [*]u32, outLen: u32) void {
    correlateInt(u32, a, na, b, nb, out, outLen);
}
export fn correlate_i16(a: [*]const i16, na: u32, b: [*]const i16, nb: u32, out: [*]i16, outLen: u32) void {
    correlateInt(i16, a, na, b, nb, out, outLen);
}
export fn correlate_u16(a: [*]const u16, na: u32, b: [*]const u16, nb: u32, out: [*]u16, outLen: u32) void {
    correlateInt(u16, a, na, b, nb, out, outLen);
}
export fn correlate_i8(a: [*]const i8, na: u32, b: [*]const i8, nb: u32, out: [*]i8, outLen: u32) void {
    correlateInt(i8, a, na, b, nb, out, outLen);
}
export fn correlate_u8(a: [*]const u8, na: u32, b: [*]const u8, nb: u32, out: [*]u8, outLen: u32) void {
    correlateInt(u8, a, na, b, nb, out, outLen);
}

// --- Tests ---

test "correlate_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    const v = [_]f64{ 0, 1, 0.5 };
    var out: [5]f64 = undefined;
    correlate_f64(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.5, 1e-10);
    try testing.expectApproxEqAbs(out[3], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 0.0, 1e-10);
}

test "correlate_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3 };
    const v = [_]f32{ 0, 1, 0.5 };
    var out: [5]f32 = undefined;
    correlate_f32(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 3.5, 1e-5);
}

test "correlate_f64 autocorrelation" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    var out: [5]f64 = undefined;
    correlate_f64(&a, 3, &a, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 8.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 14.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 8.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
}

test "correlate_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    const v = [_]i32{ 0, 1, 2 };
    var out: [5]i32 = undefined;
    correlate_i32(&a, 3, &v, 3, &out, 5);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 5);
    try testing.expectEqual(out[2], 8);
    try testing.expectEqual(out[3], 3);
    try testing.expectEqual(out[4], 0);
}

test "correlate_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3 };
    const v = [_]i8{ 1, 1 };
    var out: [4]i8 = undefined;
    correlate_i8(&a, 3, &v, 2, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], 3);
}

test "correlate_f64 different lengths" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4, 5 };
    const v = [_]f64{ 1, 1 };
    var out: [6]f64 = undefined;
    correlate_f64(&a, 5, &v, 2, &out, 6);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 7.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 5.0, 1e-10);
}
