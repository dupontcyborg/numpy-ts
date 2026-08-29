//! WASM population count (bitwise_count) kernels for all integer types.
//!
//! Unary: out[i] = popcount(a[i])
//! For signed types, counts bits of abs(value) to match NumPy behavior.
//! For unsigned types, counts bits of the raw value.
//! Output is always u8.
//!
//! All widths vectorize through `@popCount` on a @Vector, which lowers to the
//! native `i8x16.popcnt` instruction (plus a lane fold for widths > 8 bits).
//! Signed inputs use `@abs`, which returns the same-width unsigned type, so
//! abs(minInt) is representable (e.g. abs(i8 -128) = u8 128, one bit set).

/// Vector popcount over `L` lanes of `T` (unsigned counterpart `U`), narrowing
/// the per-lane count to u8. Handles the SIMD body; callers run the scalar tail.
inline fn countVec(comptime T: type, comptime U: type, comptime L: comptime_int, a: [*]const T, out: [*]u8, N: u32) u32 {
    const W = @sizeOf(U); // bytes per element
    const VT = @Vector(L, T);
    const VU = @Vector(L, U);
    const VOut = @Vector(L, u8);
    const VBytes = @Vector(L * W, u8);
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const v = @as(*align(1) const VT, @ptrCast(a + i)).*;
        const mag: VU = if (T == U) v else @abs(v);
        // Count every byte at once (i8x16.popcnt), then fold each element's W
        // byte-counts together with a shuffle/add ladder. Applying @popCount to
        // the wide vector directly instead makes LLVM emit a per-lane
        // extract/replace chain that is slower than the scalar SWAR loop.
        // Max total is 64, so a u8 accumulator cannot overflow.
        const counts: VBytes = @popCount(@as(VBytes, @bitCast(mag)));
        var sum: VOut = @splat(0);
        inline for (0..W) |k| {
            const pick: @Vector(L, i32) = comptime blk: {
                var idx: [L]i32 = undefined;
                for (&idx, 0..) |*p, j| p.* = @intCast(j * W + k);
                break :blk idx;
            };
            sum += @shuffle(u8, counts, undefined, pick);
        }
        @as(*align(1) VOut, @ptrCast(out + i)).* = sum;
    }
    return i;
}

/// Bitwise count for signed i64 — counts bits of abs(value). Scalar
/// `@popCount` on a u64 is the native `i64.popcnt`, and at 8 bytes in per 1
/// byte out this loop is store-bound, so the SIMD shuffle-fold used for the
/// narrower widths buys nothing here and is skipped in favor of the plain
/// scalar loop.
export fn bitwise_count_i64(a: [*]const i64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @popCount(@abs(a[i]));
}

/// Bitwise count for unsigned u64 — counts bits of raw value.
export fn bitwise_count_u64(a: [*]const u64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @popCount(a[i]);
}

/// Bitwise count for signed i32 — counts bits of abs(value).
export fn bitwise_count_i32(a: [*]const i32, out: [*]u8, N: u32) void {
    var i = countVec(i32, u32, 4, a, out, N);
    while (i < N) : (i += 1) out[i] = @popCount(@abs(a[i]));
}

/// Bitwise count for unsigned u32 — counts bits of raw value.
export fn bitwise_count_u32(a: [*]const u32, out: [*]u8, N: u32) void {
    var i = countVec(u32, u32, 4, a, out, N);
    while (i < N) : (i += 1) out[i] = @popCount(a[i]);
}

/// Bitwise count for signed i16 — counts bits of abs(value).
export fn bitwise_count_i16(a: [*]const i16, out: [*]u8, N: u32) void {
    var i = countVec(i16, u16, 8, a, out, N);
    while (i < N) : (i += 1) out[i] = @popCount(@abs(a[i]));
}

/// Bitwise count for unsigned u16 — counts bits of raw value.
export fn bitwise_count_u16(a: [*]const u16, out: [*]u8, N: u32) void {
    var i = countVec(u16, u16, 8, a, out, N);
    while (i < N) : (i += 1) out[i] = @popCount(a[i]);
}

/// Bitwise count for signed i8 — counts bits of abs(value).
export fn bitwise_count_i8(a: [*]const i8, out: [*]u8, N: u32) void {
    var i = countVec(i8, u8, 16, a, out, N);
    while (i < N) : (i += 1) out[i] = @popCount(@abs(a[i]));
}

/// Bitwise count for unsigned u8 — counts bits of raw value.
export fn bitwise_count_u8(a: [*]const u8, out: [*]u8, N: u32) void {
    var i = countVec(u8, u8, 16, a, out, N);
    while (i < N) : (i += 1) out[i] = @popCount(a[i]);
}

// --- Tests ---

test "bitwise_count_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 0, 1, 255, 128, 15 };
    var out: [5]u8 = undefined;
    bitwise_count_u8(&a, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 8);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 4);
}

test "bitwise_count_i8 signed" {
    const testing = @import("std").testing;
    // NumPy: bitwise_count on signed counts bits of abs(value)
    const a = [_]i8{ 0, 1, -1, -128, 127 };
    var out: [5]u8 = undefined;
    bitwise_count_i8(&a, &out, 5);
    try testing.expectEqual(out[0], 0); // abs(0) = 0
    try testing.expectEqual(out[1], 1); // abs(1) = 1 bit
    try testing.expectEqual(out[2], 1); // abs(-1) = 1 bit
    try testing.expectEqual(out[3], 1); // abs(-128) = 128 = 1 bit
    try testing.expectEqual(out[4], 7); // abs(127) = 0b1111111 = 7 bits
}

test "bitwise_count_i32 signed" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, -1, 7, -2147483648 };
    var out: [5]u8 = undefined;
    bitwise_count_i32(&a, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1); // abs(-1) = 1 bit
    try testing.expectEqual(out[3], 3);
    try testing.expectEqual(out[4], 1); // abs(-2147483648) = 2147483648 = 1 bit
}

test "bitwise_count_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0, 1, 0xFFFFFFFF, 0x80000000 };
    var out: [4]u8 = undefined;
    bitwise_count_u32(&a, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 32);
    try testing.expectEqual(out[3], 1);
}

test "bitwise_count_i64 signed" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, -1, 255 };
    var out: [3]u8 = undefined;
    bitwise_count_i64(&a, &out, 3);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1); // abs(-1) = 1 bit
    try testing.expectEqual(out[2], 8);
}

test "bitwise_count_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0xFF};
    var out: [1]u8 = undefined;
    bitwise_count_u64(&a, &out, 1);
    try testing.expectEqual(out[0], 8);
}

test "bitwise_count_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0x000F};
    var out: [1]u8 = undefined;
    bitwise_count_i16(&a, &out, 1);
    try testing.expectEqual(out[0], 4);
}

test "bitwise_count_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0x00FF};
    var out: [1]u8 = undefined;
    bitwise_count_u16(&a, &out, 1);
    try testing.expectEqual(out[0], 8);
}

// SIMD-path coverage: lengths that exercise the vector body plus a ragged tail.

test "bitwise_count_u8 simd body and tail" {
    const testing = @import("std").testing;
    var a: [37]u8 = undefined;
    var out: [37]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = @intCast(i * 7 % 256);
    bitwise_count_u8(&a, &out, 37);
    for (0..37) |i| try testing.expectEqual(out[i], @popCount(a[i]));
}

test "bitwise_count_i8 simd body handles minInt" {
    const testing = @import("std").testing;
    var a: [35]i8 = undefined;
    var out: [35]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = if (i % 5 == 0) -128 else @intCast(@as(i32, @intCast(i)) - 17);
    bitwise_count_i8(&a, &out, 35);
    for (0..35) |i| try testing.expectEqual(out[i], @popCount(@abs(a[i])));
}

test "bitwise_count_i16 simd body and tail" {
    const testing = @import("std").testing;
    var a: [19]i16 = undefined;
    var out: [19]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = if (i % 4 == 0) -32768 else @intCast(@as(i32, @intCast(i)) * -301);
    bitwise_count_i16(&a, &out, 19);
    for (0..19) |i| try testing.expectEqual(out[i], @popCount(@abs(a[i])));
}

test "bitwise_count_i32 simd body and tail" {
    const testing = @import("std").testing;
    var a: [11]i32 = undefined;
    var out: [11]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = if (i % 3 == 0) -2147483648 else @as(i32, @intCast(i)) * -70001;
    bitwise_count_i32(&a, &out, 11);
    for (0..11) |i| try testing.expectEqual(out[i], @popCount(@abs(a[i])));
}

test "bitwise_count_i64 simd body and tail" {
    const testing = @import("std").testing;
    var a: [7]i64 = undefined;
    var out: [7]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = if (i % 3 == 0) -9223372036854775808 else @as(i64, @intCast(i)) * -1234567891;
    bitwise_count_i64(&a, &out, 7);
    for (0..7) |i| try testing.expectEqual(out[i], @popCount(@abs(a[i])));
}

test "bitwise_count_u64 simd body and tail" {
    const testing = @import("std").testing;
    var a: [7]u64 = undefined;
    var out: [7]u8 = undefined;
    for (&a, 0..) |*p, i| p.* = @as(u64, 0xFFFF_0000_1234_5678) >> @intCast(i);
    bitwise_count_u64(&a, &out, 7);
    for (0..7) |i| try testing.expectEqual(out[i], @popCount(a[i]));
}
