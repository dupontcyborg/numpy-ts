//! WASM element-wise logical OR kernels.
//! Two same-dtype arrays (or one array and a scalar) in, one u8 (bool) array out.
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
    @as(*align(1) [L]u8, @ptrCast(out + i)).* = @intFromBool(m);
}

/// Generic two-array body.
inline fn binT(comptime T: type, a: [*]const T, b: [*]const T, out: [*]u8, N: u32) void {
    const L = Lanes(T);
    const n = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n) : (i += L) storeBool(L, out, i, truthy(T, a, i) | truthy(T, b, i));
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] != 0 or b[i] != 0);
}

/// A truthy scalar makes the whole result true, so it short-circuits to a fill.
inline fn binScalarT(comptime T: type, a: [*]const T, out: [*]u8, N: u32, scalar: T) void {
    if (scalar != 0) {
        @memset(out[0..N], 1);
        return;
    }
    const L = Lanes(T);
    const n = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n) : (i += L) storeBool(L, out, i, truthy(T, a, i));
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] != 0);
}

export fn logical_or_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    binT(f64, a, b, out, N);
}

export fn logical_or_scalar_f64(a: [*]const f64, out: [*]u8, N: u32, scalar: f64) void {
    binScalarT(f64, a, out, N, scalar);
}

export fn logical_or_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    binT(f32, a, b, out, N);
}

export fn logical_or_scalar_f32(a: [*]const f32, out: [*]u8, N: u32, scalar: f32) void {
    binScalarT(f32, a, out, N, scalar);
}

export fn logical_or_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    binT(i64, a, b, out, N);
}

export fn logical_or_scalar_i64(a: [*]const i64, out: [*]u8, N: u32, scalar: i64) void {
    binScalarT(i64, a, out, N, scalar);
}

export fn logical_or_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    binT(i32, a, b, out, N);
}

export fn logical_or_scalar_i32(a: [*]const i32, out: [*]u8, N: u32, scalar: i32) void {
    binScalarT(i32, a, out, N, scalar);
}

export fn logical_or_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    binT(i16, a, b, out, N);
}

export fn logical_or_scalar_i16(a: [*]const i16, out: [*]u8, N: u32, scalar: i16) void {
    binScalarT(i16, a, out, N, scalar);
}

export fn logical_or_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    binT(i8, a, b, out, N);
}

export fn logical_or_scalar_i8(a: [*]const i8, out: [*]u8, N: u32, scalar: i8) void {
    binScalarT(i8, a, out, N, scalar);
}

/// float16 pair, taking raw u16 bit patterns.
export fn logical_or_f16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    const n = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n) : (i += 8) storeBool(8, out, i, truthyF16(a, i) | truthyF16(b, i));
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] & 0x7FFF != 0 or b[i] & 0x7FFF != 0);
}

/// float16 against a scalar whose truthiness the caller has already resolved.
export fn logical_or_scalar_f16(a: [*]const u16, out: [*]u8, N: u32, scalar_truthy: u32) void {
    if (scalar_truthy != 0) {
        @memset(out[0..N], 1);
        return;
    }
    const n = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n) : (i += 8) storeBool(8, out, i, truthyF16(a, i));
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] & 0x7FFF != 0);
}

// --- Tests ---

test "logical_or_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.5, 0.0, -2.0 };
    const b = [_]f64{ 1.0, 0.0, 0.0, -3.0 };
    var out: [4]u8 = undefined;
    logical_or_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_scalar_i8 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 0, 0, 0, 0 };
    var out: [5]u8 = undefined;
    logical_or_scalar_i8(&a, &out, 5, 3);
    for (0..5) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
}

test "logical_or_scalar_i8 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, -1, 0, 5 };
    var out: [5]u8 = undefined;
    logical_or_scalar_i8(&a, &out, 5, 0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
}

test "logical_or_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    // N=17: 16 elements via SIMD + 1 remainder element
    var a: [17]i8 = undefined;
    var b: [17]i8 = undefined;
    for (0..17) |idx| {
        a[idx] = 0;
        b[idx] = 0;
    }
    // Make last element (remainder) have a=1 to test scalar fallback
    a[16] = 1;
    var out: [17]u8 = undefined;
    logical_or_i8(&a, &b, &out, 17);
    for (0..16) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
    try testing.expectEqual(out[16], 1); // nonzero OR 0 = 1
}

test "logical_or_i8 large SIMD" {
    const testing = @import("std").testing;
    var a: [20]i8 = undefined;
    var b: [20]i8 = undefined;
    for (0..20) |idx| {
        a[idx] = if (idx % 2 == 0) 1 else 0;
        b[idx] = if (idx % 3 == 0) 1 else 0;
    }
    var out: [20]u8 = undefined;
    logical_or_i8(&a, &b, &out, 20);
    for (0..20) |idx| {
        const expected: u8 = if (idx % 2 == 0 or idx % 3 == 0) 1 else 0;
        try testing.expectEqual(out[idx], expected);
    }
}

test "logical_or_f64 truth table" {
    const testing = @import("std").testing;
    // (0,0)->0, (0,nonzero)->1, (nonzero,0)->1, (nonzero,nonzero)->1
    const a = [_]f64{ 0.0, 0.0, 5.0, 5.0 };
    const b = [_]f64{ 0.0, 3.0, 0.0, 3.0 };
    var out: [4]u8 = undefined;
    logical_or_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_f32 truth table" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 0.0, 5.0, 5.0 };
    const b = [_]f32{ 0.0, 3.0, 0.0, 3.0 };
    var out: [4]u8 = undefined;
    logical_or_f32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_i64 truth table" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 0, 7, 7 };
    const b = [_]i64{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_or_i64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_i32 truth table" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 0, 7, 7 };
    const b = [_]i32{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_or_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_i16 truth table" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 0, 7, 7 };
    const b = [_]i16{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_or_i16(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_i8 truth table" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 0, 7, 7 };
    const b = [_]i8{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_or_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_f64 mixed positive/negative nonzero" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.0, -2.5, 0.0, 0.0 };
    const b = [_]f64{ 0.0, -1.0, -3.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_or_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_or_i32 mixed positive/negative nonzero" {
    const testing = @import("std").testing;
    const a = [_]i32{ -1, 0, 0, -7 };
    const b = [_]i32{ 0, -1, 0, 50 };
    var out: [4]u8 = undefined;
    logical_or_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_f64 NaN Inf neg_zero as nonzero" {
    const testing = @import("std").testing;
    const inf = @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const neg_zero = @as(f64, @bitCast(@as(u64, 0x8000000000000000)));
    // NaN != 0 is false in Zig (NaN compares false), Inf != 0 is true, -0.0 == 0
    const a = [_]f64{ nan, inf, neg_zero, nan };
    const b = [_]f64{ 0.0, 0.0, 0.0, nan };
    var out: [4]u8 = undefined;
    logical_or_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // NaN OR 0 = 1 (NaN != 0 is true)
    try testing.expectEqual(out[1], 1); // Inf OR 0 = 1
    try testing.expectEqual(out[2], 0); // -0.0 OR 0 = 0
    try testing.expectEqual(out[3], 1); // NaN OR NaN = 1
}

test "logical_or_f32 NaN Inf neg_zero as nonzero" {
    const testing = @import("std").testing;
    const inf = @as(f32, @bitCast(@as(u32, 0x7F800000)));
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const neg_zero = @as(f32, @bitCast(@as(u32, 0x80000000)));
    const a = [_]f32{ nan, inf, neg_zero, nan };
    const b = [_]f32{ 0.0, 0.0, 0.0, nan };
    var out: [4]u8 = undefined;
    logical_or_f32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // NaN OR 0 = 1
    try testing.expectEqual(out[1], 1); // Inf OR 0 = 1
    try testing.expectEqual(out[2], 0); // -0.0 OR 0 = 0
    try testing.expectEqual(out[3], 1); // NaN OR NaN = 1
}

test "logical_or_scalar_f64 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_f64(&a, &out, 4, 5.0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
}

test "logical_or_scalar_f64 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, -2.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_f64(&a, &out, 4, 0.0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_or_scalar_f32 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_f32(&a, &out, 4, 5.0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
}

test "logical_or_scalar_f32 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -2.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_f32(&a, &out, 4, 0.0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_or_scalar_i64 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 0, 0, 0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_i64(&a, &out, 4, 5);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
}

test "logical_or_scalar_i64 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 1, -2, 0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_i64(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_or_scalar_i32 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 0, 0, 0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_i32(&a, &out, 4, 5);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
}

test "logical_or_scalar_i32 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, -2, 0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_i32(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_or_scalar_i16 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 0, 0, 0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_i16(&a, &out, 4, 5);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
}

test "logical_or_scalar_i16 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 1, -2, 0 };
    var out: [4]u8 = undefined;
    logical_or_scalar_i16(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_or_i8 mixed positive/negative" {
    const testing = @import("std").testing;
    const a = [_]i8{ -128, 0, 0, 127 };
    const b = [_]i8{ 0, -1, 0, -1 };
    var out: [4]u8 = undefined;
    logical_or_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_or_f16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0x3C00, 0x0000, 0x3C00, 0x0000 };
    const b = [_]u16{ 0x3C00, 0x3C00, 0x0000, 0x0000 };
    var out: [4]u8 = undefined;
    logical_or_f16(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // 1 OR 1 = 1
    try testing.expectEqual(out[1], 1); // 0 OR 1 = 1
    try testing.expectEqual(out[2], 1); // 1 OR 0 = 1
    try testing.expectEqual(out[3], 0); // 0 OR 0 = 0
}

test "logical_or_scalar_f16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0x3C00, 0x0000, 0x8000, 0xBC00 };
    var out: [4]u8 = undefined;
    // scalar_truthy=1 (nonzero scalar) -> all 1
    logical_or_scalar_f16(&a, &out, 4, 1);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    // scalar_truthy=0 -> depends on a
    logical_or_scalar_f16(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 1); // 1.0
    try testing.expectEqual(out[1], 0); // 0.0
    try testing.expectEqual(out[2], 0); // -0.0
    try testing.expectEqual(out[3], 1); // -1.0
}

test "logical_or_i64 odd length exercises the 2-wide body and the tail" {
    const testing = @import("std").testing;
    const MIN = @import("std").math.minInt(i64);
    const MAX = @import("std").math.maxInt(i64);
    const a = [_]i64{ 0, 1, MIN, MAX, 0, -1, 0 };
    const b = [_]i64{ 1, 0, 0, 0, 0, 0, 0 };
    var out = [_]u8{9} ** 7;
    logical_or_i64(&a, &b, &out, 7);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1, 1, 0, 1, 0 }, &out);

    var o2 = [_]u8{9} ** 7;
    logical_or_scalar_i64(&a, &o2, 7, 0);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 1, 1, 0, 1, 0 }, &o2);
    logical_or_scalar_i64(&a, &o2, 7, MAX);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1, 1, 1, 1, 1 }, &o2);
}

test "logical_or float truthiness: NaN truthy, -0.0 falsy" {
    const testing = @import("std").testing;
    // Reaches the new vector body for f64, which used to be a scalar loop. NaN
    // and inf are truthy, -0.0 is falsy — matching NumPy. Odd length so the
    // scalar tail runs too.
    const nan = @import("std").math.nan(f64);
    const inf = @import("std").math.inf(f64);
    //                 nan   0.0   -0.0   inf   1.0  -0.0   2.0
    const a = [_]f64{ nan, 0.0, -0.0, inf, 1.0, -0.0, 2.0 };
    const b = [_]f64{ 1.0, 1.0, 5.0, 1.0, 0.0, -0.0, 3.0 };
    // truthy(a) = T F F T T F T
    // truthy(b) = T T T T F F T
    var out = [_]u8{9} ** 7;
    logical_or_f64(&a, &b, &out, 7);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1, 1, 1, 0, 1 }, &out);
}

test "logical_or f16 masks the sign bit so -0.0 is falsy" {
    const testing = @import("std").testing;
    // f16 arrives as raw u16 bits: 0x8000 is -0.0 and must read false, while
    // 0x7E00 (NaN) and 0x7C00 (inf) read true. 9 elements exercises the tail.
    const a = [_]u16{ 0x0000, 0x8000, 0x3C00, 0x7E00, 0x7C00, 0xBC00, 0x0001, 0x8001, 0x0000 };
    const b = [_]u16{ 0x3C00, 0x3C00, 0x0000, 0x3C00, 0x0000, 0x0000, 0x3C00, 0x0000, 0x0000 };
    var out = [_]u8{9} ** 9;
    logical_or_f16(&a, &b, &out, 9);
    var expect = [_]u8{0} ** 9;
    for (0..9) |i| {
        const at = (a[i] & 0x7FFF) != 0;
        const bt = (b[i] & 0x7FFF) != 0;
        expect[i] = @intFromBool(at or bt);
    }
    try testing.expectEqualSlices(u8, &expect, &out);
}
