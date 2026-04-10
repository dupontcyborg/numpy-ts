//! WASM reduction max kernels for all numeric types.
//!
//! Reduction: result = max(a[0..N])
//! Unsigned variants needed — comparison is sign-dependent.

const simd = @import("simd.zig");

/// Returns the maximum f64 element, scalar.
/// Uses sing 2-wide SIMD (select-based to avoid LLVM scalarization).
export fn reduce_max_f64(a: [*]const f64, N: u32) f64 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 1);
    var result: f64 = a[0];
    var i: u32 = 1;
    if (n_simd >= 2) {
        var acc = simd.load2_f64(a, 0);
        i = 2;
        while (i < n_simd) : (i += 2) {
            const v = simd.load2_f64(a, i);
            acc = @select(f64, acc > v, acc, v);
        }
        result = if (acc[0] > acc[1]) acc[0] else acc[1];
        i = n_simd;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum i64 element, scalar.
/// Max f32 array using 4-wide SIMD (select-based to avoid LLVM scalarization).
export fn reduce_max_f32(a: [*]const f32, N: u32) f32 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 3);
    var result: f32 = a[0];
    var i: u32 = 1;
    if (n_simd >= 4) {
        var acc = simd.load4_f32(a, 0);
        i = 4;
        while (i < n_simd) : (i += 4) {
            const v = simd.load4_f32(a, i);
            acc = simd.max_f32x4(acc, v);
        }
        result = acc[0];
        inline for (1..4) |lane| {
            if (acc[lane] > result) result = acc[lane];
        }
        i = n_simd;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum i64 element. Returns 0 if N=0.
export fn reduce_max_i64(a: [*]const i64, N: u32) i64 {
    if (N == 0) return 0;
    var result: i64 = a[0];
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum u64 element, scalar.
/// Uses signed i64 comparison with sign-bit flip (XOR 0x8000…) so LLVM
/// emits the native i64x2.gt_s SIMD instruction instead of scalarising.
export fn reduce_max_u64(a: [*]const u64, N: u32) u64 {
    if (N == 0) return 0;

    const FLIP: i64 = @bitCast(@as(u64, 0x8000000000000000));
    const ptr: [*]const i64 = @ptrCast(a);

    var result: i64 = ptr[0] +% FLIP; // wrapping add == XOR for sign bit
    var i: u32 = 1;

    // 2-wide SIMD path
    const n_simd = N & ~@as(u32, 1);
    if (n_simd >= 2) {
        const flip_vec = @as(simd.V2i64, @splat(FLIP));
        var acc = simd.load2_i64(ptr, 0) +% flip_vec;
        i = 2;
        while (i < n_simd) : (i += 2) {
            const v = simd.load2_i64(ptr, i) +% flip_vec;
            acc = @select(i64, acc > v, acc, v); // uses i64x2.gt_s
        }
        result = if (acc[0] > acc[1]) acc[0] else acc[1];
        i = n_simd;
    }

    // Scalar tail
    while (i < N) : (i += 1) {
        const biased = ptr[i] +% FLIP;
        if (biased > result) result = biased;
    }

    return @bitCast(@as(i64, result -% FLIP));
}

/// Returns the maximum i32 element, scalar. Uses 4-wide SIMD.
export fn reduce_max_i32(a: [*]const i32, N: u32) i32 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    var result: i32 = a[0];
    if (n_simd >= 4) {
        var acc = simd.load4_i32(a, 0);
        i = 4;
        while (i < n_simd) : (i += 4) {
            acc = @max(acc, simd.load4_i32(a, i));
        }
        result = @max(@max(acc[0], acc[1]), @max(acc[2], acc[3]));
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum u32 element, scalar. Uses 4-wide SIMD.
export fn reduce_max_u32(a: [*]const u32, N: u32) u32 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    var result: u32 = a[0];
    if (n_simd >= 4) {
        var acc: simd.V4u32 = @as(*align(1) const simd.V4u32, @ptrCast(a)).*;
        i = 4;
        while (i < n_simd) : (i += 4) {
            const v: simd.V4u32 = @as(*align(1) const simd.V4u32, @ptrCast(a + i)).*;
            acc = @max(acc, v);
        }
        result = @max(@max(acc[0], acc[1]), @max(acc[2], acc[3]));
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum i16 element, scalar. Uses 8-wide SIMD.
export fn reduce_max_i16(a: [*]const i16, N: u32) i16 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    var result: i16 = a[0];
    if (n_simd >= 8) {
        var acc = simd.load8_i16(a, 0);
        i = 8;
        while (i < n_simd) : (i += 8) {
            acc = @max(acc, simd.load8_i16(a, i));
        }
        result = acc[0];
        inline for (1..8) |lane| {
            if (acc[lane] > result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum u16 element, scalar. Uses 8-wide SIMD.
export fn reduce_max_u16(a: [*]const u16, N: u32) u16 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    var result: u16 = a[0];
    if (n_simd >= 8) {
        var acc: simd.V8u16 = @as(*align(1) const simd.V8u16, @ptrCast(a)).*;
        i = 8;
        while (i < n_simd) : (i += 8) {
            const v: simd.V8u16 = @as(*align(1) const simd.V8u16, @ptrCast(a + i)).*;
            acc = @max(acc, v);
        }
        result = acc[0];
        inline for (1..8) |lane| {
            if (acc[lane] > result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum i8 element, scalar. Uses 16-wide SIMD.
export fn reduce_max_i8(a: [*]const i8, N: u32) i8 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    var result: i8 = a[0];
    if (n_simd >= 16) {
        var acc = simd.load16_i8(a, 0);
        i = 16;
        while (i < n_simd) : (i += 16) {
            acc = @max(acc, simd.load16_i8(a, i));
        }
        result = acc[0];
        inline for (1..16) |lane| {
            if (acc[lane] > result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

/// Returns the maximum u8 element, scalar. Uses 16-wide SIMD.
export fn reduce_max_u8(a: [*]const u8, N: u32) u8 {
    if (N == 0) return 0;
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    var result: u8 = a[0];
    if (n_simd >= 16) {
        var acc: simd.V16u8 = @as(*align(1) const simd.V16u8, @ptrCast(a)).*;
        i = 16;
        while (i < n_simd) : (i += 16) {
            const v: simd.V16u8 = @as(*align(1) const simd.V16u8, @ptrCast(a + i)).*;
            acc = @max(acc, v);
        }
        result = acc[0];
        inline for (1..16) |lane| {
            if (acc[lane] > result) result = acc[lane];
        }
    } else {
        i = 1;
    }
    while (i < N) : (i += 1) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

// --- Generic SIMD strided max helpers ---

fn stridedMaxInt(comptime T: type, comptime W: comptime_int, a: [*]const T, out: [*]T, outer: u32, axis: u32, inner: u32) void {
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
                @as(*align(1) VT, @ptrCast(out + ob + i)).* = @max(acc, v);
            }
            while (i < I) : (i += 1) {
                if (a[row + i] > out[ob + i]) out[ob + i] = a[row + i];
            }
        }
    }
}

fn stridedMaxFloat(comptime T: type, comptime W: comptime_int, a: [*]const T, out: [*]T, outer: u32, axis: u32, inner: u32) void {
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
                @as(*align(1) VT, @ptrCast(out + ob + i)).* = @select(T, acc > v, acc, v);
            }
            while (i < I) : (i += 1) {
                if (a[row + i] > out[ob + i]) out[ob + i] = a[row + i];
            }
        }
    }
}

fn stridedMaxScalar(comptime T: type, a: [*]const T, out: [*]T, outer: u32, axis: u32, inner: u32) void {
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
                if (a[base + ax * I + i] > out[ob + i]) out[ob + i] = a[base + ax * I + i];
            }
        }
    }
}

export fn reduce_max_strided_f64(a: [*]const f64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    stridedMaxFloat(f64, 2, a, out, outer, axis, inner);
}
export fn reduce_max_strided_f32(a: [*]const f32, out: [*]f32, outer: u32, axis: u32, inner: u32) void {
    stridedMaxFloat(f32, 4, a, out, outer, axis, inner);
}
export fn reduce_max_strided_i64(a: [*]const i64, out: [*]i64, outer: u32, axis: u32, inner: u32) void {
    stridedMaxScalar(i64, a, out, outer, axis, inner);
}
export fn reduce_max_strided_u64(a: [*]const u64, out: [*]u64, outer: u32, axis: u32, inner: u32) void {
    stridedMaxScalar(u64, a, out, outer, axis, inner);
}
export fn reduce_max_strided_i32(a: [*]const i32, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    stridedMaxInt(i32, 4, a, out, outer, axis, inner);
}
export fn reduce_max_strided_u32(a: [*]const u32, out: [*]u32, outer: u32, axis: u32, inner: u32) void {
    stridedMaxInt(u32, 4, a, out, outer, axis, inner);
}
export fn reduce_max_strided_i16(a: [*]const i16, out: [*]i16, outer: u32, axis: u32, inner: u32) void {
    stridedMaxInt(i16, 8, a, out, outer, axis, inner);
}
export fn reduce_max_strided_u16(a: [*]const u16, out: [*]u16, outer: u32, axis: u32, inner: u32) void {
    stridedMaxInt(u16, 8, a, out, outer, axis, inner);
}

export fn reduce_max_strided_i8(a: [*]const i8, out: [*]i8, outer: u32, axis: u32, inner: u32) void {
    stridedMaxInt(i8, 16, a, out, outer, axis, inner);
}
export fn reduce_max_strided_u8(a: [*]const u8, out: [*]u8, outer: u32, axis: u32, inner: u32) void {
    stridedMaxInt(u8, 16, a, out, outer, axis, inner);
}

// --- Tests ---

test "reduce_max_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_max_f64(&a, 5), 5.0, 1e-10);
}

test "reduce_max_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    try testing.expectEqual(reduce_max_i32(&a, 8), 9);
}

test "reduce_max_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 3, 200, 4, 1, 5 };
    try testing.expectEqual(reduce_max_u8(&a, 5), 200);
}

test "reduce_max_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{-7.0};
    try testing.expectApproxEqAbs(reduce_max_f64(&a, 1), -7.0, 1e-10);
}

test "reduce_max_f64 all negatives" {
    const testing = @import("std").testing;
    const a = [_]f64{ -5.0, -1.0, -3.0 };
    try testing.expectApproxEqAbs(reduce_max_f64(&a, 3), -1.0, 1e-10);
}

test "reduce_max_f32 simd boundary" {
    // 4-wide SIMD: N=4, N=5, N=3
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 9.0, 3.0, 4.0, 10.0 };
    try testing.expectApproxEqAbs(reduce_max_f32(&a, 4), 9.0, 1e-6);
    try testing.expectApproxEqAbs(reduce_max_f32(&a, 5), 10.0, 1e-6);
    try testing.expectApproxEqAbs(reduce_max_f32(&a, 3), 9.0, 1e-6);
}

test "reduce_max_i64 negatives" {
    const testing = @import("std").testing;
    const a = [_]i64{ -100, -50, -200 };
    try testing.expectEqual(reduce_max_i64(&a, 3), -50);
}

test "reduce_max_i16 simd boundary" {
    // 8-wide SIMD: N=8, N=9, N=7
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6, 7, 8, 99 };
    try testing.expectEqual(reduce_max_i16(&a, 8), 8);
    try testing.expectEqual(reduce_max_i16(&a, 9), 99);
    try testing.expectEqual(reduce_max_i16(&a, 7), 7);
}

test "reduce_max_i8 simd boundary" {
    // 16-wide SIMD: N=16, N=17
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100 };
    try testing.expectEqual(reduce_max_i8(&a, 16), 16);
    try testing.expectEqual(reduce_max_i8(&a, 17), 100);
}

test "reduce_max_u32 unsigned correctness" {
    // u32 max value should beat any signed interpretation
    const testing = @import("std").testing;
    const a = [_]u32{ 1, 2, 0xFFFFFFFF, 4 };
    try testing.expectEqual(reduce_max_u32(&a, 4), 0xFFFFFFFF);
}

test "reduce_max_u16 simd boundary" {
    const testing = @import("std").testing;
    const a = [_]u16{ 1, 2, 3, 4, 5, 6, 7, 8, 1000 };
    try testing.expectEqual(reduce_max_u16(&a, 8), 8);
    try testing.expectEqual(reduce_max_u16(&a, 9), 1000);
}

test "reduce_max_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 10, 999, 5 };
    try testing.expectEqual(reduce_max_u64(&a, 3), 999);
}

test "reduce_max_i32 negatives" {
    const testing = @import("std").testing;
    const a = [_]i32{ -10, -3, -7, -1 };
    try testing.expectEqual(reduce_max_i32(&a, 4), -1);
}

test "reduce_max_strided_f64 basic" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1.0, 5.0, 3.0 };
    var out = [_]f64{0.0};
    reduce_max_strided_f64(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5.0);
}

test "reduce_max_strided_f32 basic" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1.0, 5.0, 3.0 };
    var out = [_]f32{0.0};
    reduce_max_strided_f32(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5.0);
}

test "reduce_max_strided_i64 basic" {
    const testing = @import("std").testing;
    var a = [_]i64{ 1, 5, 3 };
    var out = [_]i64{0};
    reduce_max_strided_i64(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_u64 basic" {
    const testing = @import("std").testing;
    var a = [_]u64{ 1, 5, 3 };
    var out = [_]u64{0};
    reduce_max_strided_u64(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_i32 basic" {
    const testing = @import("std").testing;
    var a = [_]i32{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_max_strided_i32(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_u32 basic" {
    const testing = @import("std").testing;
    var a = [_]u32{ 1, 5, 3 };
    var out = [_]u32{0};
    reduce_max_strided_u32(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_i16 basic" {
    const testing = @import("std").testing;
    var a = [_]i16{ 1, 5, 3 };
    var out = [_]i16{0};
    reduce_max_strided_i16(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_u16 basic" {
    const testing = @import("std").testing;
    var a = [_]u16{ 1, 5, 3 };
    var out = [_]u16{0};
    reduce_max_strided_u16(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_i8 basic" {
    const testing = @import("std").testing;
    var a = [_]i8{ 1, 5, 3 };
    var out = [_]i8{0};
    reduce_max_strided_i8(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}

test "reduce_max_strided_u8 basic" {
    const testing = @import("std").testing;
    var a = [_]u8{ 1, 5, 3 };
    var out = [_]u8{0};
    reduce_max_strided_u8(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 5);
}
