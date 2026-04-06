//! WASM element-wise logical AND kernels for all numeric types.
//!
//! Binary: out[i] = (a[i] != 0) & (b[i] != 0)
//! Scalar: out[i] = (a[i] != 0) & (scalar != 0)
//! Output is always u8 (0 or 1). Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise logical AND for f64: out[i] = (a[i] != 0) & (b[i] != 0).
export fn logical_and_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0 and b[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND scalar for f64: out[i] = (a[i] != 0) & (scalar != 0).
export fn logical_and_scalar_f64(a: [*]const f64, out: [*]u8, N: u32, scalar: f64) void {
    if (scalar == 0) {
        @memset(out[0..N], 0);
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND for f32: out[i] = (a[i] != 0) & (b[i] != 0).
export fn logical_and_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0 and b[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND scalar for f32: out[i] = (a[i] != 0) & (scalar != 0).
export fn logical_and_scalar_f32(a: [*]const f32, out: [*]u8, N: u32, scalar: f32) void {
    if (scalar == 0) {
        @memset(out[0..N], 0);
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 else 0;
    }
}

/// Float16 logical and (array): both nonzero?
export fn logical_and_f16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = if (a[i] & 0x7FFF != 0 and b[i] & 0x7FFF != 0) 1 else 0;
}

export fn logical_and_scalar_f16(a: [*]const u16, out: [*]u8, N: u32, scalar_truthy: u32) void {
    if (scalar_truthy == 0) {
        @memset(out[0..N], 0);
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = if (a[i] & 0x7FFF != 0) 1 else 0;
}

/// Element-wise logical AND for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn logical_and_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0 and b[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND scalar for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn logical_and_scalar_i64(a: [*]const i64, out: [*]u8, N: u32, scalar: i64) void {
    if (scalar == 0) {
        @memset(out[0..N], 0);
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND for i32 using 4-wide SIMD: out[i] = (a[i] != 0) & (b[i] != 0).
export fn logical_and_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    const zero_i32: simd.V4i32 = @splat(0);
    const one_i32: simd.V4i32 = @splat(1);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va_bool = @select(i32, simd.load4_i32(a, i) != zero_i32, one_i32, zero_i32);
        const vb_bool = @select(i32, simd.load4_i32(b, i) != zero_i32, one_i32, zero_i32);
        const result = va_bool & vb_bool;
        const result_bytes: @Vector(16, u8) = @bitCast(result);
        out[i] = result_bytes[0];
        out[i + 1] = result_bytes[4];
        out[i + 2] = result_bytes[8];
        out[i + 3] = result_bytes[12];
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0 and b[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND scalar for i32: out[i] = (a[i] != 0) & (scalar != 0).
export fn logical_and_scalar_i32(a: [*]const i32, out: [*]u8, N: u32, scalar: i32) void {
    if (scalar == 0) {
        @memset(out[0..N], 0);
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND for i16 using 8-wide SIMD: out[i] = (a[i] != 0) & (b[i] != 0).
export fn logical_and_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    const zero_i16: simd.V8i16 = @splat(0);
    const one_i16: simd.V8i16 = @splat(1);
    const narrow = @Vector(8, i32){ 0, 2, 4, 6, 8, 10, 12, 14 };
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va_bool = @select(i16, simd.load8_i16(a, i) != zero_i16, one_i16, zero_i16);
        const vb_bool = @select(i16, simd.load8_i16(b, i) != zero_i16, one_i16, zero_i16);
        const result = va_bool & vb_bool;
        const result_bytes: @Vector(16, u8) = @bitCast(result);
        const narrowed: @Vector(8, u8) = @shuffle(u8, result_bytes, undefined, narrow);
        inline for (0..8) |lane| {
            out[i + lane] = narrowed[lane];
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0 and b[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND scalar for i16: out[i] = (a[i] != 0) & (scalar != 0).
export fn logical_and_scalar_i16(a: [*]const i16, out: [*]u8, N: u32, scalar: i16) void {
    if (scalar == 0) {
        @memset(out[0..N], 0);
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND for i8 using 16-wide SIMD: out[i] = (a[i] != 0) & (b[i] != 0).
/// Input and output are both byte-width, enabling natural 16-wide vectorization.
export fn logical_and_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const one: simd.V16u8 = @splat(1);
    const zero_u8: simd.V16u8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va_bool: simd.V16u8 = @select(u8, simd.load16_i8(a, i) != zero, one, zero_u8);
        const vb_bool: simd.V16u8 = @select(u8, simd.load16_i8(b, i) != zero, one, zero_u8);
        simd.store16_u8(out, i, va_bool & vb_bool);
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0 and b[i] != 0) 1 else 0;
    }
}

/// Element-wise logical AND scalar for i8 using 16-wide SIMD: out[i] = (a[i] != 0) & (scalar != 0).
export fn logical_and_scalar_i8(a: [*]const i8, out: [*]u8, N: u32, scalar: i8) void {
    if (scalar == 0) {
        @memset(out[0..N], 0);
        return;
    }
    const zero: simd.V16i8 = @splat(0);
    const one: simd.V16u8 = @splat(1);
    const zero_u8: simd.V16u8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_u8(out, i, @select(u8, simd.load16_i8(a, i) != zero, one, zero_u8));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] != 0) 1 else 0;
    }
}

// --- Tests ---

test "logical_and_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.5, 0.0, -2.0 };
    const b = [_]f64{ 1.0, 0.0, 0.0, -3.0 };
    var out: [4]u8 = undefined;
    logical_and_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_scalar_i8 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5 };
    var out: [5]u8 = undefined;
    logical_and_scalar_i8(&a, &out, 5, 0);
    for (0..5) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
}

test "logical_and_scalar_i8 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, -1, 0, 5 };
    var out: [5]u8 = undefined;
    logical_and_scalar_i8(&a, &out, 5, 3);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
}

test "logical_and_i8 large SIMD" {
    const testing = @import("std").testing;
    var a: [20]i8 = undefined;
    var b: [20]i8 = undefined;
    for (0..20) |idx| {
        a[idx] = if (idx % 2 == 0) 1 else 0;
        b[idx] = if (idx % 3 == 0) 1 else 0;
    }
    var out: [20]u8 = undefined;
    logical_and_i8(&a, &b, &out, 20);
    for (0..20) |idx| {
        const expected: u8 = if (idx % 2 == 0 and idx % 3 == 0) 1 else 0;
        try testing.expectEqual(out[idx], expected);
    }
}

test "logical_and_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    // N=17: 16 elements via SIMD + 1 remainder element
    var a: [17]i8 = undefined;
    var b: [17]i8 = undefined;
    for (0..17) |idx| {
        a[idx] = @intCast(@as(i32, @intCast(idx)) + 1); // all nonzero
        b[idx] = @intCast(@as(i32, @intCast(idx)) + 1);
    }
    // Make last element (remainder) have b=0 to test scalar fallback
    b[16] = 0;
    var out: [17]u8 = undefined;
    logical_and_i8(&a, &b, &out, 17);
    for (0..16) |idx| {
        try testing.expectEqual(out[idx], 1);
    }
    try testing.expectEqual(out[16], 0); // nonzero AND 0 = 0
}

test "logical_and_f64 truth table" {
    const testing = @import("std").testing;
    // (0,0)->0, (0,nonzero)->0, (nonzero,0)->0, (nonzero,nonzero)->1
    const a = [_]f64{ 0.0, 0.0, 5.0, 5.0 };
    const b = [_]f64{ 0.0, 3.0, 0.0, 3.0 };
    var out: [4]u8 = undefined;
    logical_and_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_f32 truth table" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 0.0, 5.0, 5.0 };
    const b = [_]f32{ 0.0, 3.0, 0.0, 3.0 };
    var out: [4]u8 = undefined;
    logical_and_f32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_i64 truth table" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 0, 7, 7 };
    const b = [_]i64{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_and_i64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_i32 truth table" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 0, 7, 7 };
    const b = [_]i32{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_and_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_i16 truth table" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 0, 7, 7 };
    const b = [_]i16{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_and_i16(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_i8 truth table" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 0, 7, 7 };
    const b = [_]i8{ 0, 3, 0, 3 };
    var out: [4]u8 = undefined;
    logical_and_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "logical_and_f64 mixed positive/negative nonzero" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.0, -2.5, 3.0, -4.0 };
    const b = [_]f64{ 1.0, -1.0, -3.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_and_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_i32 mixed positive/negative nonzero" {
    const testing = @import("std").testing;
    const a = [_]i32{ -1, -100, 50, -7 };
    const b = [_]i32{ 1, -1, -3, 0 };
    var out: [4]u8 = undefined;
    logical_and_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_f64 NaN Inf neg_zero as nonzero" {
    const testing = @import("std").testing;
    const inf = @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const neg_zero = @as(f64, @bitCast(@as(u64, 0x8000000000000000)));
    // NaN != 0 is true, Inf != 0 is true, -0.0 == 0 is true
    const a = [_]f64{ nan, inf, neg_zero, nan };
    const b = [_]f64{ 1.0, 1.0, 1.0, nan };
    var out: [4]u8 = undefined;
    logical_and_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // NaN != 0 is true (NaN is unequal to everything)
    try testing.expectEqual(out[1], 1); // Inf AND 1 = 1
    try testing.expectEqual(out[2], 0); // -0.0 AND 1 = 0 (-0.0 == 0)
    try testing.expectEqual(out[3], 1); // NaN AND NaN = 1 (both NaN != 0)
}

test "logical_and_f32 NaN Inf neg_zero as nonzero" {
    const testing = @import("std").testing;
    const inf = @as(f32, @bitCast(@as(u32, 0x7F800000)));
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const neg_zero = @as(f32, @bitCast(@as(u32, 0x80000000)));
    const a = [_]f32{ nan, inf, neg_zero, nan };
    const b = [_]f32{ 1.0, 1.0, 1.0, nan };
    var out: [4]u8 = undefined;
    logical_and_f32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // NaN != 0 is true
    try testing.expectEqual(out[1], 1); // Inf AND 1 = 1
    try testing.expectEqual(out[2], 0); // -0.0 AND 1 = 0
    try testing.expectEqual(out[3], 1); // NaN AND NaN = 1
}

test "logical_and_scalar_f64 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_f64(&a, &out, 4, 0.0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
}

test "logical_and_scalar_f64 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, -2.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_f64(&a, &out, 4, 5.0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_scalar_f32 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_f32(&a, &out, 4, 0.0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
}

test "logical_and_scalar_f32 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -2.0, 0.0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_f32(&a, &out, 4, 5.0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_scalar_i64 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3, 4 };
    var out: [4]u8 = undefined;
    logical_and_scalar_i64(&a, &out, 4, 0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
}

test "logical_and_scalar_i64 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 1, -2, 0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_i64(&a, &out, 4, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_scalar_i32 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4 };
    var out: [4]u8 = undefined;
    logical_and_scalar_i32(&a, &out, 4, 0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
}

test "logical_and_scalar_i32 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, -2, 0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_i32(&a, &out, 4, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_scalar_i16 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4 };
    var out: [4]u8 = undefined;
    logical_and_scalar_i16(&a, &out, 4, 0);
    for (0..4) |idx| {
        try testing.expectEqual(out[idx], 0);
    }
}

test "logical_and_scalar_i16 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 1, -2, 0 };
    var out: [4]u8 = undefined;
    logical_and_scalar_i16(&a, &out, 4, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_i8 mixed positive/negative" {
    const testing = @import("std").testing;
    const a = [_]i8{ -128, 127, -1, 1 };
    const b = [_]i8{ 1, -1, -1, 0 };
    var out: [4]u8 = undefined;
    logical_and_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_and_f16 basic" {
    const testing = @import("std").testing;
    // 1.0=0x3C00, 0.0=0x0000, -0.0=0x8000, -1.0=0xBC00
    const a = [_]u16{ 0x3C00, 0x0000, 0x3C00, 0x0000 };
    const b = [_]u16{ 0x3C00, 0x3C00, 0x0000, 0x0000 };
    var out: [4]u8 = undefined;
    logical_and_f16(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // 1 AND 1 = 1
    try testing.expectEqual(out[1], 0); // 0 AND 1 = 0
    try testing.expectEqual(out[2], 0); // 1 AND 0 = 0
    try testing.expectEqual(out[3], 0); // 0 AND 0 = 0
}

test "logical_and_scalar_f16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0x3C00, 0x0000, 0x8000, 0xBC00 };
    var out: [4]u8 = undefined;
    // scalar_truthy=1 (nonzero scalar)
    logical_and_scalar_f16(&a, &out, 4, 1);
    try testing.expectEqual(out[0], 1); // 1.0 AND true
    try testing.expectEqual(out[1], 0); // 0.0 AND true
    try testing.expectEqual(out[2], 0); // -0.0 AND true
    try testing.expectEqual(out[3], 1); // -1.0 AND true
    // scalar_truthy=0
    logical_and_scalar_f16(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
}
