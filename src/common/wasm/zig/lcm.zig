//! WASM element-wise LCM (least common multiple) kernels: out[i] = lcm(a[i],
//! b[i]) (binary) or lcm(a[i], scalar) (scalar), on contiguous 1D buffers,
//! preserving the promoted integer dtype.
//!
//! lcm(x, y) = |x| / gcd(x, y) * |y|, dividing first so the intermediate never
//! overflows for inputs whose true lcm fits. Where it does not fit, the multiply
//! wraps, which is what NumPy's integer lcm does.

/// Absolute value that wraps rather than trapping on the most negative value.
/// `-@as(i8, -128)` has no representable result; NumPy wraps there too.
fn absWrap(comptime T: type, v: T) T {
    if (@typeInfo(T).int.signedness == .signed) {
        return if (v < 0) -%v else v;
    }
    return v;
}

fn gcdGeneric(comptime T: type, x_in: T, y_in: T) T {
    var x = absWrap(T, x_in);
    var y = absWrap(T, y_in);
    while (y != 0) {
        const temp = y;
        y = @rem(x, y);
        x = temp;
    }
    return x;
}

fn lcmGeneric(comptime T: type, x_in: T, y_in: T) T {
    const g = gcdGeneric(T, x_in, y_in);
    // NumPy: lcm(0, y) == lcm(x, 0) == 0.
    if (g == 0) return 0;
    const ax = absWrap(T, x_in);
    const ay = absWrap(T, y_in);
    return @divTrunc(ax, g) *% ay;
}

/// Binary LCM for i8: out[i] = lcm(a[i], b[i]).
export fn lcm_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i8, a[i], b[i]);
}

/// Binary LCM for u8: out[i] = lcm(a[i], b[i]).
export fn lcm_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u8, a[i], b[i]);
}

/// Binary LCM for i16: out[i] = lcm(a[i], b[i]).
export fn lcm_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i16, a[i], b[i]);
}

/// Binary LCM for u16: out[i] = lcm(a[i], b[i]).
export fn lcm_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u16, a[i], b[i]);
}

/// Binary LCM for i32: out[i] = lcm(a[i], b[i]).
export fn lcm_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i32, a[i], b[i]);
}

/// Binary LCM for u32: out[i] = lcm(a[i], b[i]).
export fn lcm_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u32, a[i], b[i]);
}

/// Binary LCM for i64: out[i] = lcm(a[i], b[i]).
export fn lcm_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i64, a[i], b[i]);
}

/// Binary LCM for u64: out[i] = lcm(a[i], b[i]).
export fn lcm_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u64, a[i], b[i]);
}

/// Scalar LCM for i8: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i8, a[i], scalar);
}

/// Scalar LCM for u8: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, scalar: u8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u8, a[i], scalar);
}

/// Scalar LCM for i16: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i16, a[i], scalar);
}

/// Scalar LCM for u16: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, scalar: u16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u16, a[i], scalar);
}

/// Scalar LCM for i32: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i32, a[i], scalar);
}

/// Scalar LCM for u32: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, scalar: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u32, a[i], scalar);
}

/// Scalar LCM for i64: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(i64, a[i], scalar);
}

/// Scalar LCM for u64: out[i] = lcm(a[i], scalar).
export fn lcm_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, scalar: u64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = lcmGeneric(u64, a[i], scalar);
}

// --- Tests ---

test "lcm_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 4, 7, 0, -4 };
    const b = [_]i32{ 18, 6, 5, 3, 6 };
    var out: [5]i32 = undefined;
    lcm_i32(&a, &b, &out, 5);
    try testing.expectEqual(out[0], 36);
    try testing.expectEqual(out[1], 12);
    try testing.expectEqual(out[2], 35);
    try testing.expectEqual(out[3], 0); // lcm(0, x) == 0
    try testing.expectEqual(out[4], 12); // sign is dropped
}

test "lcm_i64 beyond the f64 exact range" {
    const testing = @import("std").testing;
    const a = [_]i64{ 9007199254740993, 1000000007, 0 };
    const b = [_]i64{ 2, 1000000009, 5 };
    var out: [3]i64 = undefined;
    lcm_i64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 18014398509481986);
    try testing.expectEqual(out[1], 1000000016000000063); // both prime
    try testing.expectEqual(out[2], 0);
}

test "lcm_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 12, 1, 0 };
    const b = [_]u64{ 18, 7, 9 };
    var out: [3]u64 = undefined;
    lcm_u64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 36);
    try testing.expectEqual(out[1], 7);
    try testing.expectEqual(out[2], 0);
}

test "lcm_u8 wraps like NumPy on overflow" {
    const testing = @import("std").testing;
    // lcm(100, 3) == 300, which does not fit in u8 and wraps to 44.
    const a = [_]u8{ 100, 12 };
    const b = [_]u8{ 3, 18 };
    var out: [2]u8 = undefined;
    lcm_u8(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 44);
    try testing.expectEqual(out[1], 36);
}

test "lcm_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 4, 0, -9 };
    var out: [4]i32 = undefined;
    lcm_scalar_i32(&a, &out, 4, 6);
    try testing.expectEqual(out[0], 12);
    try testing.expectEqual(out[1], 12);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 18);
}

test "lcm_i8 most negative input does not trap" {
    const testing = @import("std").testing;
    const a = [_]i8{-128};
    const b = [_]i8{2};
    var out: [1]i8 = undefined;
    lcm_i8(&a, &b, &out, 1);
    try testing.expectEqual(out[0], -128); // |−128| wraps to −128, lcm wraps with it
}

test "lcm_i16 and lcm_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 12, 4, 7, 0, -4 };
    const b = [_]i16{ 18, 6, 5, 3, 6 };
    var out: [5]i16 = undefined;
    lcm_i16(&a, &b, &out, 5);
    try testing.expectEqualSlices(i16, &[_]i16{ 36, 12, 35, 0, 12 }, &out);

    const c = [_]u16{ 12, 4, 7, 0 };
    const d = [_]u16{ 18, 6, 5, 3 };
    var uout: [4]u16 = undefined;
    lcm_u16(&c, &d, &uout, 4);
    try testing.expectEqualSlices(u16, &[_]u16{ 36, 12, 35, 0 }, &uout);
}

test "lcm_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 12, 1000000, 0 };
    const b = [_]u32{ 18, 3000000, 9 };
    var out: [3]u32 = undefined;
    lcm_u32(&a, &b, &out, 3);
    try testing.expectEqualSlices(u32, &[_]u32{ 36, 3000000, 0 }, &out);
}

test "lcm_scalar over the narrow dtypes" {
    const testing = @import("std").testing;
    const si8 = [_]i8{ 12, 4, 0, -9 };
    var oi8: [4]i8 = undefined;
    lcm_scalar_i8(&si8, &oi8, 4, 6);
    try testing.expectEqualSlices(i8, &[_]i8{ 12, 12, 0, 18 }, &oi8);

    const su8 = [_]u8{ 12, 4, 0, 9 };
    var ou8: [4]u8 = undefined;
    lcm_scalar_u8(&su8, &ou8, 4, 6);
    try testing.expectEqualSlices(u8, &[_]u8{ 12, 12, 0, 18 }, &ou8);

    const si16 = [_]i16{ 12, 4, 0, -9 };
    var oi16: [4]i16 = undefined;
    lcm_scalar_i16(&si16, &oi16, 4, 6);
    try testing.expectEqualSlices(i16, &[_]i16{ 12, 12, 0, 18 }, &oi16);

    const su16 = [_]u16{ 12, 4, 0, 9 };
    var ou16: [4]u16 = undefined;
    lcm_scalar_u16(&su16, &ou16, 4, 6);
    try testing.expectEqualSlices(u16, &[_]u16{ 12, 12, 0, 18 }, &ou16);

    const su32 = [_]u32{ 12, 4, 0, 9 };
    var ou32: [4]u32 = undefined;
    lcm_scalar_u32(&su32, &ou32, 4, 6);
    try testing.expectEqualSlices(u32, &[_]u32{ 12, 12, 0, 18 }, &ou32);
}

test "lcm_scalar_i64 and lcm_scalar_u64 beyond the f64 exact range" {
    const testing = @import("std").testing;
    const a = [_]i64{ 9007199254740993, 0, 7 };
    var out: [3]i64 = undefined;
    lcm_scalar_i64(&a, &out, 3, 2);
    try testing.expectEqualSlices(i64, &[_]i64{ 18014398509481986, 0, 14 }, &out);

    const b = [_]u64{ 12, 0, 7 };
    var uout: [3]u64 = undefined;
    lcm_scalar_u64(&b, &uout, 3, 18);
    try testing.expectEqualSlices(u64, &[_]u64{ 36, 0, 126 }, &uout);
}
