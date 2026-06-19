//! WASM 1D convolution kernel for float types.
//!
//! Computes full linear convolution: out[k] = sum_n(a[n] * v[k - n])
//! for k = 0..aLen+vLen-2.
//! The JS layer handles mode slicing (full/same/valid), complex types,
//! and dtype conversion (int→float) before calling these kernels.
//!
//! Convolution accesses v in reverse order, preventing contiguous SIMD loads.
//! Kept as scalar inner loop (same approach as wasm-bench).

const simd = @import("simd.zig");

/// Full 1D convolution for f64.
/// out must have length na + nb - 1.
export fn convolve_f64(a: [*]const f64, na: u32, b: [*]const f64, nb: u32, out: [*]f64, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        // SIMD inner product: a forward, b reversed (b[k-j] descends as j grows).
        // Pair (j, j+1) reads b at {k-j, k-j-1}; load2 at (k-j-1) then swap lanes.
        // Constrained to j+2<=j_end (full pair) and j+1<=k (lower b index ≥0).
        var acc: simd.V2f64 = .{ 0, 0 };
        var j = j_start;
        while (j + 2 <= j_end and j + 1 <= k) : (j += 2) {
            const av = simd.load2_f64(a, j);
            const bpair = simd.load2_f64(b, k - j - 1); // [b[k-j-1], b[k-j]]
            const brev = @shuffle(f64, bpair, undefined, [2]i32{ 1, 0 });
            acc = simd.mulAdd_f64x2(av, brev, acc);
        }
        var sum: f64 = acc[0] + acc[1];
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[k - j];
        }
        out[k] = sum;
    }
}

/// Full 1D convolution for f32.
/// out must have length na + nb - 1.
export fn convolve_f32(a: [*]const f32, na: u32, b: [*]const f32, nb: u32, out: [*]f32, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        // SIMD inner product (4-wide): pair-of-4 reads b at {k-j .. k-j-3};
        // load4 at (k-j-3) then reverse lanes. Needs j+4<=j_end and j+3<=k.
        var acc: simd.V4f32 = .{ 0, 0, 0, 0 };
        var j = j_start;
        while (j + 4 <= j_end and j + 3 <= k) : (j += 4) {
            const av = simd.load4_f32(a, j);
            const bquad = simd.load4_f32(b, k - j - 3); // [b[k-j-3]..b[k-j]]
            const brev = @shuffle(f32, bquad, undefined, [4]i32{ 3, 2, 1, 0 });
            acc = simd.mulAdd_f32x4(av, brev, acc);
        }
        var sum: f32 = acc[0] + acc[1] + acc[2] + acc[3];
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[k - j];
        }
        out[k] = sum;
    }
}

// --- Integer convolve kernels (scalar loop, same-type output) ---

fn convolveInt(comptime T: type, a: [*]const T, na: u32, b: [*]const T, nb: u32, out: [*]T, outLen: u32) void {
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
            sum +%= a[j] *% b[k - j];
        }
        out[k] = sum;
    }
}

export fn convolve_i32(a: [*]const i32, na: u32, b: [*]const i32, nb: u32, out: [*]i32, outLen: u32) void {
    convolveInt(i32, a, na, b, nb, out, outLen);
}
export fn convolve_u32(a: [*]const u32, na: u32, b: [*]const u32, nb: u32, out: [*]u32, outLen: u32) void {
    convolveInt(u32, a, na, b, nb, out, outLen);
}
export fn convolve_i16(a: [*]const i16, na: u32, b: [*]const i16, nb: u32, out: [*]i16, outLen: u32) void {
    convolveInt(i16, a, na, b, nb, out, outLen);
}
export fn convolve_u16(a: [*]const u16, na: u32, b: [*]const u16, nb: u32, out: [*]u16, outLen: u32) void {
    convolveInt(u16, a, na, b, nb, out, outLen);
}
export fn convolve_i8(a: [*]const i8, na: u32, b: [*]const i8, nb: u32, out: [*]i8, outLen: u32) void {
    convolveInt(i8, a, na, b, nb, out, outLen);
}
export fn convolve_u8(a: [*]const u8, na: u32, b: [*]const u8, nb: u32, out: [*]u8, outLen: u32) void {
    convolveInt(u8, a, na, b, nb, out, outLen);
}

// --- Tests ---

test "convolve_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    const v = [_]i32{ 1, 1 };
    var out: [4]i32 = undefined;
    convolve_i32(&a, 3, &v, 2, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], 3);
}

test "convolve_f64 basic" {
    const testing = @import("std").testing;
    // np.convolve([1, 2, 3], [0, 1, 0.5]) = [0, 1, 2.5, 4, 1.5]
    const a = [_]f64{ 1, 2, 3 };
    const v = [_]f64{ 0, 1, 0.5 };
    var out: [5]f64 = undefined;
    convolve_f64(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.5, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 1.5, 1e-10);
}

test "convolve_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3 };
    const v = [_]f32{ 0, 1, 0.5 };
    var out: [5]f32 = undefined;
    convolve_f32(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 2.5, 1e-5);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 1.5, 1e-5);
}

test "convolve_f64 identity" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    const v = [_]f64{1};
    var out: [4]f64 = undefined;
    convolve_f64(&a, 4, &v, 1, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
}

test "convolve_f64 different lengths" {
    const testing = @import("std").testing;
    // np.convolve([1, 2, 3, 4, 5], [1, 1]) = [1, 3, 5, 7, 9, 5]
    const a = [_]f64{ 1, 2, 3, 4, 5 };
    const v = [_]f64{ 1, 1 };
    var out: [6]f64 = undefined;
    convolve_f64(&a, 5, &v, 2, &out, 6);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 7.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 5.0, 1e-10);
}

test "convolve_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 1, 2, 3 };
    const v = [_]u32{ 1, 1 };
    var out: [4]u32 = undefined;
    convolve_u32(&a, 3, &v, 2, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], 3);
}

test "convolve_i16 negatives" {
    const testing = @import("std").testing;
    // [1, -2, 3] * [1, 1] = [1, -1, 1, 3]
    const a = [_]i16{ 1, -2, 3 };
    const v = [_]i16{ 1, 1 };
    var out: [4]i16 = undefined;
    convolve_i16(&a, 3, &v, 2, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], -1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 3);
}

test "convolve_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 10, 20, 30 };
    const v = [_]u16{ 1, 2 };
    var out: [4]u16 = undefined;
    convolve_u16(&a, 3, &v, 2, &out, 4);
    try testing.expectEqual(out[0], 10); // 10*1
    try testing.expectEqual(out[1], 40); // 20*1 + 10*2
    try testing.expectEqual(out[2], 70); // 30*1 + 20*2
    try testing.expectEqual(out[3], 60); // 30*2
}

test "convolve_i8 wrapping arithmetic" {
    const testing = @import("std").testing;
    // Use values that wrap on overflow; verify two's complement +%/*% behavior
    const a = [_]i8{ 100, 50, -100 };
    const v = [_]i8{ 1, 1 };
    var out: [4]i8 = undefined;
    convolve_i8(&a, 3, &v, 2, &out, 4);
    try testing.expectEqual(out[0], 100);
    try testing.expectEqual(out[1], -106); // 100 + 50 = 150 → wraps to -106 in i8
    try testing.expectEqual(out[2], -50); // 50 + -100 = -50
    try testing.expectEqual(out[3], -100);
}

test "convolve_u8 basic" {
    const testing = @import("std").testing;
    // Stay within u8 range to avoid wrap noise
    const a = [_]u8{ 1, 2, 3, 4 };
    const v = [_]u8{ 1, 0, 1 };
    var out: [6]u8 = undefined;
    convolve_u8(&a, 4, &v, 3, &out, 6);
    try testing.expectEqual(out[0], 1); // a[0]*v[0]
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 4); // 1+3
    try testing.expectEqual(out[3], 6); // 2+4
    try testing.expectEqual(out[4], 3);
    try testing.expectEqual(out[5], 4);
}
