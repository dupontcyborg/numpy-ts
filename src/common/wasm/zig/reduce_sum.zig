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

/// Strided sum for i16 input → f64 output.
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
