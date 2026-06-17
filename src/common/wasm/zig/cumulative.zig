//! WASM cumulative scan: cumsum / cumprod over a flattened contiguous array.
//!
//! Sequential prefix scan (no SIMD — the recurrence is serial). Floats
//! accumulate in their own dtype to match NumPy. Integers accumulate in a
//! widened i64/u64 with wrapping arithmetic (`+%`/`*%`), matching NumPy's
//! modulo-wrap behavior for integer cumsum/cumprod.

// --- cumsum ---

inline fn cumsumFloat(comptime T: type, a: [*]const T, out: [*]T, N: u32) void {
    var acc: T = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        acc += a[i];
        out[i] = acc;
    }
}

// Widen each element to the wider integer accumulator, then wrapping-add.
inline fn cumsumInt(comptime T: type, comptime A: type, a: [*]const T, out: [*]A, N: u32) void {
    var acc: A = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        acc +%= @intCast(a[i]);
        out[i] = acc;
    }
}

export fn cumsum_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    cumsumFloat(f64, a, out, N);
}
export fn cumsum_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    cumsumFloat(f32, a, out, N);
}
export fn cumsum_i8(a: [*]const i8, out: [*]i64, N: u32) void {
    cumsumInt(i8, i64, a, out, N);
}
export fn cumsum_i16(a: [*]const i16, out: [*]i64, N: u32) void {
    cumsumInt(i16, i64, a, out, N);
}
export fn cumsum_i32(a: [*]const i32, out: [*]i64, N: u32) void {
    cumsumInt(i32, i64, a, out, N);
}
export fn cumsum_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    cumsumInt(i64, i64, a, out, N);
}
export fn cumsum_u8(a: [*]const u8, out: [*]u64, N: u32) void {
    cumsumInt(u8, u64, a, out, N);
}
export fn cumsum_u16(a: [*]const u16, out: [*]u64, N: u32) void {
    cumsumInt(u16, u64, a, out, N);
}
export fn cumsum_u32(a: [*]const u32, out: [*]u64, N: u32) void {
    cumsumInt(u32, u64, a, out, N);
}
export fn cumsum_u64(a: [*]const u64, out: [*]u64, N: u32) void {
    cumsumInt(u64, u64, a, out, N);
}

// --- cumprod ---

inline fn cumprodFloat(comptime T: type, a: [*]const T, out: [*]T, N: u32) void {
    var acc: T = 1;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        acc *= a[i];
        out[i] = acc;
    }
}

inline fn cumprodInt(comptime T: type, comptime A: type, a: [*]const T, out: [*]A, N: u32) void {
    var acc: A = 1;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        acc *%= @intCast(a[i]);
        out[i] = acc;
    }
}

export fn cumprod_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    cumprodFloat(f64, a, out, N);
}
export fn cumprod_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    cumprodFloat(f32, a, out, N);
}
export fn cumprod_i8(a: [*]const i8, out: [*]i64, N: u32) void {
    cumprodInt(i8, i64, a, out, N);
}
export fn cumprod_i16(a: [*]const i16, out: [*]i64, N: u32) void {
    cumprodInt(i16, i64, a, out, N);
}
export fn cumprod_i32(a: [*]const i32, out: [*]i64, N: u32) void {
    cumprodInt(i32, i64, a, out, N);
}
export fn cumprod_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    cumprodInt(i64, i64, a, out, N);
}
export fn cumprod_u8(a: [*]const u8, out: [*]u64, N: u32) void {
    cumprodInt(u8, u64, a, out, N);
}
export fn cumprod_u16(a: [*]const u16, out: [*]u64, N: u32) void {
    cumprodInt(u16, u64, a, out, N);
}
export fn cumprod_u32(a: [*]const u32, out: [*]u64, N: u32) void {
    cumprodInt(u32, u64, a, out, N);
}
export fn cumprod_u64(a: [*]const u64, out: [*]u64, N: u32) void {
    cumprodInt(u64, u64, a, out, N);
}

// --- Tests ---

test "cumsum_f64 basic" {
    const t = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    var o: [3]f64 = undefined;
    cumsum_f64(&a, &o, 3);
    try t.expectEqual(@as(f64, 1), o[0]);
    try t.expectEqual(@as(f64, 3), o[1]);
    try t.expectEqual(@as(f64, 6), o[2]);
}

test "cumsum_i8 widens to i64 (no overflow)" {
    const t = @import("std").testing;
    const a = [_]i8{ 100, 100 };
    var o: [2]i64 = undefined;
    cumsum_i8(&a, &o, 2);
    try t.expectEqual(@as(i64, 100), o[0]);
    try t.expectEqual(@as(i64, 200), o[1]);
}

test "cumprod_f64 basic" {
    const t = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    var o: [4]f64 = undefined;
    cumprod_f64(&a, &o, 4);
    try t.expectEqual(@as(f64, 1), o[0]);
    try t.expectEqual(@as(f64, 2), o[1]);
    try t.expectEqual(@as(f64, 6), o[2]);
    try t.expectEqual(@as(f64, 24), o[3]);
}

test "cumprod_i32 widens to i64" {
    const t = @import("std").testing;
    const a = [_]i32{ 2, 3, 4 };
    var o: [3]i64 = undefined;
    cumprod_i32(&a, &o, 3);
    try t.expectEqual(@as(i64, 2), o[0]);
    try t.expectEqual(@as(i64, 6), o[1]);
    try t.expectEqual(@as(i64, 24), o[2]);
}

test "cumsum_f32 basic" {
    const t = @import("std").testing;
    const a = [_]f32{ 1, 2, 3 };
    var o: [3]f32 = undefined;
    cumsum_f32(&a, &o, 3);
    try t.expectApproxEqAbs(@as(f32, 1), o[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 3), o[1], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 6), o[2], 1e-5);
}

test "cumsum signed ints widen to i64" {
    const t = @import("std").testing;
    var o: [3]i64 = undefined;
    const a16 = [_]i16{ 1, 2, 3 };
    cumsum_i16(&a16, &o, 3);
    try t.expectEqual(@as(i64, 1), o[0]);
    try t.expectEqual(@as(i64, 3), o[1]);
    try t.expectEqual(@as(i64, 6), o[2]);
    const a32 = [_]i32{ 1, 2, 3 };
    cumsum_i32(&a32, &o, 3);
    try t.expectEqual(@as(i64, 6), o[2]);
    const a64 = [_]i64{ 1, 2, 3 };
    cumsum_i64(&a64, &o, 3);
    try t.expectEqual(@as(i64, 6), o[2]);
}

test "cumsum unsigned ints widen to u64" {
    const t = @import("std").testing;
    var o: [3]u64 = undefined;
    const a8 = [_]u8{ 1, 2, 3 };
    cumsum_u8(&a8, &o, 3);
    try t.expectEqual(@as(u64, 1), o[0]);
    try t.expectEqual(@as(u64, 3), o[1]);
    try t.expectEqual(@as(u64, 6), o[2]);
    const a16 = [_]u16{ 1, 2, 3 };
    cumsum_u16(&a16, &o, 3);
    try t.expectEqual(@as(u64, 6), o[2]);
    const a32 = [_]u32{ 1, 2, 3 };
    cumsum_u32(&a32, &o, 3);
    try t.expectEqual(@as(u64, 6), o[2]);
    const a64 = [_]u64{ 1, 2, 3 };
    cumsum_u64(&a64, &o, 3);
    try t.expectEqual(@as(u64, 6), o[2]);
}

test "cumprod_f32 basic" {
    const t = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4 };
    var o: [4]f32 = undefined;
    cumprod_f32(&a, &o, 4);
    try t.expectApproxEqAbs(@as(f32, 1), o[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 2), o[1], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 6), o[2], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 24), o[3], 1e-5);
}

test "cumprod signed ints widen to i64" {
    const t = @import("std").testing;
    var o: [3]i64 = undefined;
    const a8 = [_]i8{ 2, 3, 4 };
    cumprod_i8(&a8, &o, 3);
    try t.expectEqual(@as(i64, 2), o[0]);
    try t.expectEqual(@as(i64, 6), o[1]);
    try t.expectEqual(@as(i64, 24), o[2]);
    const a16 = [_]i16{ 2, 3, 4 };
    cumprod_i16(&a16, &o, 3);
    try t.expectEqual(@as(i64, 24), o[2]);
    const a64 = [_]i64{ 2, 3, 4 };
    cumprod_i64(&a64, &o, 3);
    try t.expectEqual(@as(i64, 24), o[2]);
}

test "cumprod unsigned ints widen to u64" {
    const t = @import("std").testing;
    var o: [3]u64 = undefined;
    const a8 = [_]u8{ 2, 3, 4 };
    cumprod_u8(&a8, &o, 3);
    try t.expectEqual(@as(u64, 2), o[0]);
    try t.expectEqual(@as(u64, 6), o[1]);
    try t.expectEqual(@as(u64, 24), o[2]);
    const a16 = [_]u16{ 2, 3, 4 };
    cumprod_u16(&a16, &o, 3);
    try t.expectEqual(@as(u64, 24), o[2]);
    const a32 = [_]u32{ 2, 3, 4 };
    cumprod_u32(&a32, &o, 3);
    try t.expectEqual(@as(u64, 24), o[2]);
    const a64 = [_]u64{ 2, 3, 4 };
    cumprod_u64(&a64, &o, 3);
    try t.expectEqual(@as(u64, 24), o[2]);
}
