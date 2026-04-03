//! WASM reduction min kernels for all numeric types.
//!
//! Reduction: result = min(a[0..N])
//! Unsigned variants needed — comparison is sign-dependent.

const simd = @import("simd.zig");

/// Returns the mimimum f64 element. Returns 0 if N=0.
/// Note: NaN is considered less than any number, so reduce_min_f64([NaN], 1) returns NaN.
export fn reduce_min_f64(a: [*]const f64, N: u32) f64 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 1);
    var result: f64 = a[0];
    var i: u32 = 1;
    if (n_simd >= 2) {
        var acc = simd.load2_f64(a, 0);
        i = 2;
        while (i < n_simd) : (i += 2) {
            const v = simd.load2_f64(a, i);
            acc = @select(f64, acc < v, acc, v);
        }
        result = if (acc[0] < acc[1]) acc[0] else acc[1];
        i = n_simd;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum f32 element. Returns 0 if N=0.
/// Note: NaN is considered less than any number, so reduce_min_f32([NaN], 1) returns NaN.
export fn reduce_min_f32(a: [*]const f32, N: u32) f32 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 3);
    var result: f32 = a[0];
    var i: u32 = 1;
    if (n_simd >= 4) {
        var acc = simd.load4_f32(a, 0);
        i = 4;
        while (i < n_simd) : (i += 4) {
            const v = simd.load4_f32(a, i);
            acc = simd.min_f32x4(acc, v);
        }
        result = acc[0];
        inline for (1..4) |lane| {
            if (acc[lane] < result) result = acc[lane];
        }
        i = n_simd;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum i64 element. Returns 0 if N=0.
export fn reduce_min_i64(a: [*]const i64, N: u32) i64 {
    if (N == 0) return 0;
    var result: i64 = a[0];
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum u64 element. Returns 0 if N=0.
/// Uses signed i64 comparison with sign-bit flip (XOR 0x8000…) so LLVM
/// emits the native i64x2.gt_s SIMD instruction instead of scalarising.
export fn reduce_min_u64(a: [*]const u64, N: u32) u64 {
    if (N == 0) return 0;

    const FLIP: i64 = @bitCast(@as(u64, 0x8000000000000000));
    const ptr: [*]const i64 = @ptrCast(a);

    var result: i64 = ptr[0] +% FLIP;
    var i: u32 = 1;

    // 2-wide SIMD path
    const n_simd = N & ~@as(u32, 1);
    if (n_simd >= 2) {
        const flip_vec = @as(simd.V2i64, @splat(FLIP));
        var acc = simd.load2_i64(ptr, 0) +% flip_vec;
        i = 2;
        while (i < n_simd) : (i += 2) {
            const v = simd.load2_i64(ptr, i) +% flip_vec;
            acc = @select(i64, acc < v, acc, v); // uses i64x2.gt_s (inverted)
        }
        result = if (acc[0] < acc[1]) acc[0] else acc[1];
        i = n_simd;
    }

    // Scalar tail
    while (i < N) : (i += 1) {
        const biased = ptr[i] +% FLIP;
        if (biased < result) result = biased;
    }

    return @bitCast(@as(i64, result -% FLIP));
}

/// Returns the mimimum i32 element. Returns 0 if N=0. Uses 4-wide SIMD.
export fn reduce_min_i32(a: [*]const i32, N: u32) i32 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    var result: i32 = a[0];
    if (n_simd >= 4) {
        var acc = simd.load4_i32(a, 0);
        i = 4;
        while (i < n_simd) : (i += 4) {
            acc = @min(acc, simd.load4_i32(a, i));
        }
        result = @min(@min(acc[0], acc[1]), @min(acc[2], acc[3]));
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum u32 element. Returns 0 if N=0. Uses 4-wide SIMD.
export fn reduce_min_u32(a: [*]const u32, N: u32) u32 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    var result: u32 = a[0];
    if (n_simd >= 4) {
        var acc: simd.V4u32 = @as(*align(1) const simd.V4u32, @ptrCast(a)).*;
        i = 4;
        while (i < n_simd) : (i += 4) {
            const v: simd.V4u32 = @as(*align(1) const simd.V4u32, @ptrCast(a + i)).*;
            acc = @min(acc, v);
        }
        result = @min(@min(acc[0], acc[1]), @min(acc[2], acc[3]));
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum i16 element. Returns 0 if N=0. Uses 8-wide SIMD.
export fn reduce_min_i16(a: [*]const i16, N: u32) i16 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    var result: i16 = a[0];
    if (n_simd >= 8) {
        var acc = simd.load8_i16(a, 0);
        i = 8;
        while (i < n_simd) : (i += 8) {
            acc = @min(acc, simd.load8_i16(a, i));
        }
        result = acc[0];
        inline for (1..8) |lane| {
            if (acc[lane] < result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum u16 element. Returns 0 if N=0. Uses 8-wide SIMD.
export fn reduce_min_u16(a: [*]const u16, N: u32) u16 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    var result: u16 = a[0];
    if (n_simd >= 8) {
        var acc: simd.V8u16 = @as(*align(1) const simd.V8u16, @ptrCast(a)).*;
        i = 8;
        while (i < n_simd) : (i += 8) {
            const v: simd.V8u16 = @as(*align(1) const simd.V8u16, @ptrCast(a + i)).*;
            acc = @min(acc, v);
        }
        result = acc[0];
        inline for (1..8) |lane| {
            if (acc[lane] < result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum i8 element. Returns 0 if N=0. Uses 16-wide SIMD.
export fn reduce_min_i8(a: [*]const i8, N: u32) i8 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    var result: i8 = a[0];
    if (n_simd >= 16) {
        var acc = simd.load16_i8(a, 0);
        i = 16;
        while (i < n_simd) : (i += 16) {
            acc = @min(acc, simd.load16_i8(a, i));
        }
        result = acc[0];
        inline for (1..16) |lane| {
            if (acc[lane] < result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

/// Returns the mimimum u8 element. Returns 0 if N=0. Uses 16-wide SIMD.
export fn reduce_min_u8(a: [*]const u8, N: u32) u8 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    var result: u8 = a[0];
    if (n_simd >= 16) {
        var acc: simd.V16u8 = @as(*align(1) const simd.V16u8, @ptrCast(a)).*;
        i = 16;
        while (i < n_simd) : (i += 16) {
            const v: simd.V16u8 = @as(*align(1) const simd.V16u8, @ptrCast(a + i)).*;
            acc = @min(acc, v);
        }
        result = acc[0];
        inline for (1..16) |lane| {
            if (acc[lane] < result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] < result) result = a[i];
    }
    return result;
}

// --- Generic SIMD strided min helpers ---

fn stridedMinInt(comptime T: type, comptime W: comptime_int, a: [*]const T, out: [*]T, outer: u32, axis: u32, inner: u32) void {
    const VT = @Vector(W, T);
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const IW = I & ~@as(usize, W - 1);
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        var i: usize = 0;
        while (i < IW) : (i += W) @as(*align(1) VT, @ptrCast(out + ob + i)).* = @as(*align(1) const VT, @ptrCast(a + base + i)).*;
        while (i < I) : (i += 1) out[ob + i] = a[base + i];
        for (1..A) |ax| {
            const row = base + ax * I;
            i = 0;
            while (i < IW) : (i += W) {
                const acc: VT = @as(*align(1) const VT, @ptrCast(out + ob + i)).*;
                const v: VT = @as(*align(1) const VT, @ptrCast(a + row + i)).*;
                @as(*align(1) VT, @ptrCast(out + ob + i)).* = @min(acc, v);
            }
            while (i < I) : (i += 1) {
                if (a[row + i] < out[ob + i]) out[ob + i] = a[row + i];
            }
        }
    }
}

fn stridedMinFloat(comptime T: type, comptime W: comptime_int, a: [*]const T, out: [*]T, outer: u32, axis: u32, inner: u32) void {
    const VT = @Vector(W, T);
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const IW = I & ~@as(usize, W - 1);
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        var i: usize = 0;
        while (i < IW) : (i += W) @as(*align(1) VT, @ptrCast(out + ob + i)).* = @as(*align(1) const VT, @ptrCast(a + base + i)).*;
        while (i < I) : (i += 1) out[ob + i] = a[base + i];
        for (1..A) |ax| {
            const row = base + ax * I;
            i = 0;
            while (i < IW) : (i += W) {
                const acc: VT = @as(*align(1) const VT, @ptrCast(out + ob + i)).*;
                const v: VT = @as(*align(1) const VT, @ptrCast(a + row + i)).*;
                @as(*align(1) VT, @ptrCast(out + ob + i)).* = @select(T, acc < v, acc, v);
            }
            while (i < I) : (i += 1) {
                if (a[row + i] < out[ob + i]) out[ob + i] = a[row + i];
            }
        }
    }
}

fn stridedMinScalar(comptime T: type, a: [*]const T, out: [*]T, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| {
                if (a[base + ax * I + i] < out[ob + i]) out[ob + i] = a[base + ax * I + i];
            }
        }
    }
}

export fn reduce_min_strided_f64(a: [*]const f64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    stridedMinFloat(f64, 2, a, out, outer, axis, inner);
}
export fn reduce_min_strided_f32(a: [*]const f32, out: [*]f32, outer: u32, axis: u32, inner: u32) void {
    stridedMinFloat(f32, 4, a, out, outer, axis, inner);
}
export fn reduce_min_strided_i64(a: [*]const i64, out: [*]i64, outer: u32, axis: u32, inner: u32) void {
    stridedMinScalar(i64, a, out, outer, axis, inner);
}
export fn reduce_min_strided_u64(a: [*]const u64, out: [*]u64, outer: u32, axis: u32, inner: u32) void {
    stridedMinScalar(u64, a, out, outer, axis, inner);
}
export fn reduce_min_strided_i32(a: [*]const i32, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    stridedMinInt(i32, 4, a, out, outer, axis, inner);
}
export fn reduce_min_strided_u32(a: [*]const u32, out: [*]u32, outer: u32, axis: u32, inner: u32) void {
    stridedMinInt(u32, 4, a, out, outer, axis, inner);
}
export fn reduce_min_strided_i16(a: [*]const i16, out: [*]i16, outer: u32, axis: u32, inner: u32) void {
    stridedMinInt(i16, 8, a, out, outer, axis, inner);
}
export fn reduce_min_strided_u16(a: [*]const u16, out: [*]u16, outer: u32, axis: u32, inner: u32) void {
    stridedMinInt(u16, 8, a, out, outer, axis, inner);
}
export fn reduce_min_strided_i8(a: [*]const i8, out: [*]i8, outer: u32, axis: u32, inner: u32) void {
    stridedMinInt(i8, 16, a, out, outer, axis, inner);
}
export fn reduce_min_strided_u8(a: [*]const u8, out: [*]u8, outer: u32, axis: u32, inner: u32) void {
    stridedMinInt(u8, 16, a, out, outer, axis, inner);
}

// --- Tests ---

test "reduce_min_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_min_f64(&a, 5), 1.0, 1e-10);
}

test "reduce_min_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    try testing.expectEqual(reduce_min_i32(&a, 8), 1);
}

test "reduce_min_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 3, 200, 4, 1, 5 };
    try testing.expectEqual(reduce_min_u8(&a, 5), 1);
}

test "reduce_min_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{-7.0};
    try testing.expectApproxEqAbs(reduce_min_f64(&a, 1), -7.0, 1e-10);
}

test "reduce_min_f64 all negatives" {
    const testing = @import("std").testing;
    const a = [_]f64{ -5.0, -1.0, -3.0 };
    try testing.expectApproxEqAbs(reduce_min_f64(&a, 3), -5.0, 1e-10);
}

test "reduce_min_f32 simd boundary" {
    // 4-wide SIMD: N=4, N=5, N=3
    const testing = @import("std").testing;
    const a = [_]f32{ 5.0, 1.0, 3.0, 4.0, 0.5 };
    try testing.expectApproxEqAbs(reduce_min_f32(&a, 4), 1.0, 1e-6);
    try testing.expectApproxEqAbs(reduce_min_f32(&a, 5), 0.5, 1e-6);
    try testing.expectApproxEqAbs(reduce_min_f32(&a, 3), 1.0, 1e-6);
}

test "reduce_min_i64 negatives" {
    const testing = @import("std").testing;
    const a = [_]i64{ -100, -50, -200 };
    try testing.expectEqual(reduce_min_i64(&a, 3), -200);
}

test "reduce_min_i16 simd boundary" {
    // 8-wide SIMD: N=8, N=9, N=7
    const testing = @import("std").testing;
    const a = [_]i16{ 8, 7, 6, 5, 4, 3, 2, 1, -99 };
    try testing.expectEqual(reduce_min_i16(&a, 8), 1);
    try testing.expectEqual(reduce_min_i16(&a, 9), -99);
    try testing.expectEqual(reduce_min_i16(&a, 7), 2);
}

test "reduce_min_i8 simd boundary" {
    // 16-wide SIMD: N=16, N=17
    const testing = @import("std").testing;
    const a = [_]i8{ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -100 };
    try testing.expectEqual(reduce_min_i8(&a, 16), 1);
    try testing.expectEqual(reduce_min_i8(&a, 17), -100);
}

test "reduce_min_u32 unsigned correctness" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0xFFFFFFFF, 0, 5 };
    try testing.expectEqual(reduce_min_u32(&a, 3), 0);
}

test "reduce_min_u16 simd boundary" {
    const testing = @import("std").testing;
    const a = [_]u16{ 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    try testing.expectEqual(reduce_min_u16(&a, 8), 1);
    try testing.expectEqual(reduce_min_u16(&a, 9), 0);
}

test "reduce_min_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 10, 1, 999 };
    try testing.expectEqual(reduce_min_u64(&a, 3), 1);
}

test "reduce_min_i32 negatives" {
    const testing = @import("std").testing;
    const a = [_]i32{ -10, -3, -7, -1 };
    try testing.expectEqual(reduce_min_i32(&a, 4), -10);
}
