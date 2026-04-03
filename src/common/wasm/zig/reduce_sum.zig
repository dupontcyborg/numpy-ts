//! WASM reduction sum kernels for all numeric types.
//!
//! Reduction: result = sum(a[0..N])
//! No unsigned variants needed — wrapping addition gives the same bits.

const simd = @import("simd.zig");

/// Sum f64 array using 2-wide SIMD accumulation.
export fn reduce_sum_f64(a: [*]const f64, N: u32) f64 {
    var acc: simd.V2f64 = .{ 0, 0 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc += simd.load2_f64(a, i);
    }
    var total: f64 = acc[0] + acc[1];
    while (i < N) : (i += 1) {
        total += a[i];
    }
    return total;
}

/// Sum f32 array using 4-wide SIMD accumulation.
export fn reduce_sum_f32(a: [*]const f32, N: u32) f32 {
    var acc: simd.V4f32 = .{ 0, 0, 0, 0 };
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        acc += simd.load4_f32(a, i);
    }
    var total: f32 = acc[0] + acc[1] + acc[2] + acc[3];
    while (i < N) : (i += 1) {
        total += a[i];
    }
    return total;
}

/// Sum i64 array, scalar. Handles both signed (i64) and unsigned (u64).
export fn reduce_sum_i64(a: [*]const i64, N: u32) i64 {
    var total: i64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total +%= a[i];
    }
    return total;
}

/// Sum i32 array using 4-wide SIMD accumulation.
/// Handles both signed (i32) and unsigned (u32).
export fn reduce_sum_i32(a: [*]const i32, N: u32) i32 {
    var acc: simd.V4i32 = .{ 0, 0, 0, 0 };
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        acc +%= simd.load4_i32(a, i);
    }
    var total: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
    while (i < N) : (i += 1) {
        total +%= a[i];
    }
    return total;
}

/// Sum i16 array using 8-wide SIMD accumulation.
/// Handles both signed (i16) and unsigned (u16).
export fn reduce_sum_i16(a: [*]const i16, N: u32) i16 {
    var acc: simd.V8i16 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        acc +%= simd.load8_i16(a, i);
    }
    var total: i16 = acc[0] +% acc[1] +% acc[2] +% acc[3] +% acc[4] +% acc[5] +% acc[6] +% acc[7];
    while (i < N) : (i += 1) {
        total +%= a[i];
    }
    return total;
}

/// Sum i8 array using 16-wide SIMD accumulation.
/// Handles both signed (i8) and unsigned (u8).
export fn reduce_sum_i8(a: [*]const i8, N: u32) i8 {
    var acc: simd.V16i8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        acc +%= simd.load16_i8(a, i);
    }
    var total: i8 = 0;
    inline for (0..16) |lane| {
        total +%= acc[lane];
    }
    while (i < N) : (i += 1) {
        total +%= a[i];
    }
    return total;
}

/// Strided sum for f64 input → f64 output.
export fn reduce_sum_strided_f64(a: [*]const f64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            const src = a + base + ax * I;
            const dst = out + ob;
            const n = I & ~@as(usize, 1);
            var j: usize = 0;
            while (j < n) : (j += 2) {
                const simd_mod = @import("simd.zig");
                simd_mod.store2_f64(dst, j, simd_mod.load2_f64(dst, j) + simd_mod.load2_f64(src, j));
            }
            while (j < I) : (j += 1) dst[j] += src[j];
        }
    }
}

/// Strided sum for f32 input → f64 output (promote to avoid precision loss).
export fn reduce_sum_strided_f32(a: [*]const f32, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, a[base + i]);
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, a[base + ax * I + i]);
        }
    }
}

/// Strided sum for i64 input → f64 output.
export fn reduce_sum_strided_i64(a: [*]const i64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for u64 input → f64 output.
export fn reduce_sum_strided_u64(a: [*]const u64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for i32 input → f64 output.
export fn reduce_sum_strided_i32(a: [*]const i32, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for u32 input → f64 output.
export fn reduce_sum_strided_u32(a: [*]const u32, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for i16 input → i64 output.
export fn reduce_sum_strided_i16(a: [*]const i16, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for u16 input → f64 output.
export fn reduce_sum_strided_u16(a: [*]const u16, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for i8 input → f64 output.
export fn reduce_sum_strided_i8(a: [*]const i8, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Strided sum for u8 input → f64 output.
export fn reduce_sum_strided_u8(a: [*]const u8, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
    }
}

/// Sum i8 array, accumulate in i64 to avoid overflow.
export fn reduce_sum_i8_to_i64(a: [*]const i8, N: u32) i64 {
    var sum: i64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        sum += @as(i64, a[i]);
    }
    return sum;
}

/// Sum i16 array, accumulate in i64 to avoid overflow.
export fn reduce_sum_i16_to_i64(a: [*]const i16, N: u32) i64 {
    var sum: i64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        sum += @as(i64, a[i]);
    }
    return sum;
}

/// Sum u8 array, accumulate in u64 to avoid overflow.
export fn reduce_sum_u8_to_u64(a: [*]const u8, N: u32) u64 {
    var sum: u64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        sum += @as(u64, a[i]);
    }
    return sum;
}

/// Sum u16 array, accumulate in u64 to avoid overflow.
export fn reduce_sum_u16_to_u64(a: [*]const u16, N: u32) u64 {
    var sum: u64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        sum += @as(u64, a[i]);
    }
    return sum;
}

// --- Complex sum kernels ---
// Complex data is interleaved [re0, im0, re1, im1, ...].
// N = number of complex elements. The buffer has 2*N floats.

/// Sum complex128 (f64 pairs). Returns (re, im) via out[0], out[1].
export fn reduce_sum_c128(a: [*]const f64, N: u32, out: [*]f64) void {
    const simd_mod = @import("simd.zig");
    var acc: simd_mod.V2f64 = @splat(0.0);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        acc += simd_mod.load2_f64(a, i * 2);
    }
    simd_mod.store2_f64(out, 0, acc);
}

/// Sum complex64 (f32 pairs). Returns (re, im) via out[0], out[1].
export fn reduce_sum_c64(a: [*]const f32, N: u32, out: [*]f32) void {
    const simd_mod = @import("simd.zig");
    // Process 2 complex elements at a time (4 floats = 1 f32x4)
    const n2 = N & ~@as(u32, 1);
    var acc: simd_mod.V4f32 = @splat(0.0);
    var i: u32 = 0;
    while (i < n2) : (i += 2) {
        acc += simd_mod.load4_f32(a, i * 2);
    }
    // Reduce 4-wide to 2-wide: lanes [0,1] + [2,3]
    var re = acc[0] + acc[2];
    var im = acc[1] + acc[3];
    // Scalar tail
    if (i < N) {
        re += a[i * 2];
        im += a[i * 2 + 1];
    }
    out[0] = re;
    out[1] = im;
}

/// Strided sum for complex128. Input is interleaved f64 pairs.
/// outer × axis × inner layout, each "element" is 2 f64s.
export fn reduce_sum_strided_c128(a: [*]const f64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = (o * S) * 2;
        const ob = o * I * 2;
        // Init from first axis slice
        for (0..I) |i| {
            out[ob + i * 2] = a[base + i * 2];
            out[ob + i * 2 + 1] = a[base + i * 2 + 1];
        }
        for (1..A) |ax| {
            const row = base + ax * I * 2;
            for (0..I) |i| {
                out[ob + i * 2] += a[row + i * 2];
                out[ob + i * 2 + 1] += a[row + i * 2 + 1];
            }
        }
    }
}

/// Strided sum for complex64. Input is interleaved f32 pairs.
export fn reduce_sum_strided_c64(a: [*]const f32, out: [*]f32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = (o * S) * 2;
        const ob = o * I * 2;
        for (0..I) |i| {
            out[ob + i * 2] = a[base + i * 2];
            out[ob + i * 2 + 1] = a[base + i * 2 + 1];
        }
        for (1..A) |ax| {
            const row = base + ax * I * 2;
            for (0..I) |i| {
                out[ob + i * 2] += a[row + i * 2];
                out[ob + i * 2 + 1] += a[row + i * 2 + 1];
            }
        }
    }
}

// --- Tests ---

test "reduce_sum_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_sum_f64(&a, 5), 15.0, 1e-10);
}

test "reduce_sum_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try testing.expectEqual(reduce_sum_i32(&a, 9), 45);
}

test "reduce_sum_f64 empty and single" {
    const testing = @import("std").testing;
    const a = [_]f64{};
    try testing.expectApproxEqAbs(reduce_sum_f64(&a, 0), 0.0, 1e-10);
    const b = [_]f64{7.0};
    try testing.expectApproxEqAbs(reduce_sum_f64(&b, 1), 7.0, 1e-10);
}

test "reduce_sum_f64 negatives" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.0, -2.0, 3.0 };
    try testing.expectApproxEqAbs(reduce_sum_f64(&a, 3), 0.0, 1e-10);
}

test "reduce_sum_f32 simd boundary" {
    // 4-wide SIMD: test N=4 (full SIMD), N=5 (one scalar tail)
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_sum_f32(&a, 4), 10.0, 1e-6);
    try testing.expectApproxEqAbs(reduce_sum_f32(&a, 5), 15.0, 1e-6);
    try testing.expectApproxEqAbs(reduce_sum_f32(&a, 3), 6.0, 1e-6);
}

test "reduce_sum_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 100, 200, 300 };
    try testing.expectEqual(reduce_sum_i64(&a, 3), 600);
}

test "reduce_sum_i64 negatives" {
    const testing = @import("std").testing;
    const a = [_]i64{ -100, 50, -25 };
    try testing.expectEqual(reduce_sum_i64(&a, 3), -75);
}

test "reduce_sum_i16 simd boundary" {
    // 8-wide SIMD: test N=8, N=9, N=7
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try testing.expectEqual(reduce_sum_i16(&a, 8), 36);
    try testing.expectEqual(reduce_sum_i16(&a, 9), 45);
    try testing.expectEqual(reduce_sum_i16(&a, 7), 28);
}

test "reduce_sum_i8 simd boundary" {
    // 16-wide SIMD: test N=16, N=17
    const testing = @import("std").testing;
    // All 1s so result is easy to verify
    const a = [_]i8{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 };
    try testing.expectEqual(reduce_sum_i8(&a, 16), 16);
    try testing.expectEqual(reduce_sum_i8(&a, 17), 18);
}

test "reduce_sum_i32 negatives" {
    const testing = @import("std").testing;
    const a = [_]i32{ -5, -3, 8 };
    try testing.expectEqual(reduce_sum_i32(&a, 3), 0);
}
