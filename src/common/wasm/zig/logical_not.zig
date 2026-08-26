//! WASM element-wise logical NOT kernels: out[i] = (a[i] == 0) ? 1 : 0.
//!
//! Truthiness is `v != 0` for every numeric type — NaN is truthy, -0.0 is not,
//! which is what NumPy does. float16 arrives as raw u16 bits, so it masks the
//! sign bit instead of comparing as a float.
//!
//! `@intFromBool` on a vector yields one byte per lane, which is already the bool
//! output layout. That replaces the select / bitcast / shuffle / per-lane-extract
//! chain the narrow dtypes used to do, and lets f64 and f32 vectorize at all —
//! they were plain scalar loops.

/// One v128 worth of lanes for T: 16 for i8 ... 2 for f64/i64.
inline fn Lanes(comptime T: type) comptime_int {
    return 16 / @sizeOf(T);
}

/// Lane-wise truthiness of one v128 group.
inline fn truthy(comptime T: type, p: [*]const T, i: u32) @Vector(Lanes(T), bool) {
    const V = @Vector(Lanes(T), T);
    const z: V = @splat(0);
    return @as(*align(1) const V, @ptrCast(p + i)).* != z;
}

/// float16 truthiness from raw bits: mask the sign so -0.0 is false.
inline fn truthyF16(p: [*]const u16, i: u32) @Vector(8, bool) {
    const V = @Vector(8, u16);
    const mask: V = @splat(0x7FFF);
    const z: V = @splat(0);
    return (@as(*align(1) const V, @ptrCast(p + i)).* & mask) != z;
}

/// One byte per lane, straight into the bool output.
inline fn storeBool(comptime L: comptime_int, out: [*]u8, i: u32, m: @Vector(L, bool)) void {
    @as(*align(1) @Vector(L, u8), @ptrCast(out + i)).* = @intFromBool(m);
}

/// Generic body.
inline fn notT(comptime T: type, a: [*]const T, out: [*]u8, N: u32) void {
    const L = Lanes(T);
    const n = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n) : (i += L) storeBool(L, out, i, ~truthy(T, a, i));
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] == 0);
}

export fn logical_not_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    notT(f64, a, out, N);
}

export fn logical_not_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    notT(f32, a, out, N);
}

export fn logical_not_i64(a: [*]const i64, out: [*]u8, N: u32) void {
    notT(i64, a, out, N);
}

export fn logical_not_i32(a: [*]const i32, out: [*]u8, N: u32) void {
    notT(i32, a, out, N);
}

export fn logical_not_i16(a: [*]const i16, out: [*]u8, N: u32) void {
    notT(i16, a, out, N);
}

export fn logical_not_i8(a: [*]const i8, out: [*]u8, N: u32) void {
    notT(i8, a, out, N);
}

/// float16, taking raw u16 bit patterns.
export fn logical_not_f16(a: [*]const u16, out: [*]u8, N: u32) void {
    const n = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n) : (i += 8) storeBool(8, out, i, ~truthyF16(a, i));
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] & 0x7FFF == 0);
}

// --- Tests ---

test "logical_not_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.5, 0.0, -2.0 };
    var out: [4]u8 = undefined;
    logical_not_f64(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_not_i8 large SIMD" {
    const testing = @import("std").testing;
    var a: [20]i8 = undefined;
    for (0..20) |idx| {
        a[idx] = if (idx % 3 == 0) 0 else @intCast(idx);
    }
    var out: [20]u8 = undefined;
    logical_not_i8(&a, &out, 20);
    for (0..20) |idx| {
        const expected: u8 = if (idx % 3 == 0) 1 else 0;
        try testing.expectEqual(out[idx], expected);
    }
}

test "logical_not_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 5, -3, 0, 7 };
    var out: [5]u8 = undefined;
    logical_not_i32(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
}

test "logical_not_f64 edge zero types" {
    const testing = @import("std").testing;
    const a = [_]f64{ -0.0, 0.0, 1e-300 };
    var out: [3]u8 = undefined;
    logical_not_f64(&a, &out, 3);
    try testing.expectEqual(out[0], 1); // -0.0 == 0
    try testing.expectEqual(out[1], 1); // 0.0 == 0
    try testing.expectEqual(out[2], 0); // tiny but nonzero
}

test "logical_not_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -1.0, 0.0, 0.5 };
    var out: [5]u8 = undefined;
    logical_not_f32(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
}

test "logical_not_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 1, -1, 0, 100 };
    var out: [5]u8 = undefined;
    logical_not_i64(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
}

test "logical_not_i16 SIMD boundary N=9" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 1, 0, -1, 0, 5, 0, -3, 7 };
    var out: [9]u8 = undefined;
    logical_not_i16(&a, &out, 9);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
    try testing.expectEqual(out[8], 0);
}

test "logical_not_f16 basic" {
    const testing = @import("std").testing;
    // 1.0=0x3C00, 0.0=0x0000, -0.0=0x8000, -1.0=0xBC00
    const a = [_]u16{ 0x3C00, 0x0000, 0x8000, 0xBC00 };
    var out: [4]u8 = undefined;
    logical_not_f16(&a, &out, 4);
    try testing.expectEqual(out[0], 0); // !1.0 = 0
    try testing.expectEqual(out[1], 1); // !0.0 = 1
    try testing.expectEqual(out[2], 1); // !(-0.0) = 1
    try testing.expectEqual(out[3], 0); // !(-1.0) = 0
}

test "logical_not_i64 odd length exercises the 2-wide body and the tail" {
    const testing = @import("std").testing;
    const MIN = @import("std").math.minInt(i64);
    const MAX = @import("std").math.maxInt(i64);
    const a = [_]i64{ 0, 1, MIN, MAX, 0, -1, 0 };
    var out = [_]u8{9} ** 7;
    logical_not_i64(&a, &out, 7);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 0, 0, 0, 1, 0, 1 }, &out);
}

test "logical_not float and f16 truthiness edges" {
    const testing = @import("std").testing;
    const nan = @import("std").math.nan(f64);
    const inf = @import("std").math.inf(f64);
    // -0.0 is falsy so NOT(-0.0) is true; NaN and inf are truthy so NOT is false.
    const a = [_]f64{ 0.0, -0.0, nan, inf, 1.0, -1.0, 0.0 };
    var out = [_]u8{9} ** 7;
    logical_not_f64(&a, &out, 7);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 0, 0, 0, 0, 1 }, &out);

    const h = [_]u16{ 0x0000, 0x8000, 0x3C00, 0x7E00, 0x7C00, 0xBC00, 0x0001, 0x8001, 0x0000 };
    var o2 = [_]u8{9} ** 9;
    logical_not_f16(&h, &o2, 9);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 0, 0, 0, 0, 0, 0, 1 }, &o2);
}
