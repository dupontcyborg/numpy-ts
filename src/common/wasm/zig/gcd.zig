//! WASM element-wise GCD (greatest common divisor) kernels.
//!
//! Scalar: out[i] = gcd(a[i], scalar)
//! Binary: out[i] = gcd(a[i], b[i])
//! Uses Euclidean algorithm. Operates on contiguous 1D buffers.

/// Absolute value that wraps rather than trapping on the most negative value.
/// Plain `-v` on `minInt` panics under ReleaseSafe and is undefined under the
/// ReleaseFast build this ships with; `-%` wraps instead, so `absWrap(i8, -128)`
/// stays `-128`. The Euclidean loop below still works with a negative `x`:
/// `@rem` keeps the sign of the dividend and the loop terminates on `y`, which
/// reaches 0 either way, matching NumPy's result at that boundary.
fn absWrap(comptime T: type, v: T) T {
    if (@typeInfo(T).int.signedness == .signed) {
        return if (v < 0) -%v else v;
    }
    return v;
}

/// Scalar GCD for i32: out[i] = gcd(abs(a[i]), abs(scalar)).
export fn gcd_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const b = absWrap(i32, scalar);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var x = absWrap(i32, a[i]);
        var y = b;
        while (y != 0) {
            const temp = y;
            y = @rem(x, y);
            x = temp;
        }
        out[i] = x;
    }
}

/// Binary GCD for i32: out[i] = gcd(abs(a[i]), abs(b[i])).
export fn gcd_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var x = absWrap(i32, a[i]);
        var y = absWrap(i32, b[i]);
        while (y != 0) {
            const temp = y;
            y = @rem(x, y);
            x = temp;
        }
        out[i] = x;
    }
}

// --- Small-int native-dtype kernels ---

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

/// Binary GCD for i16: out[i] = gcd(a[i], b[i]), preserving i16 dtype.
export fn gcd_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(i16, a[i], b[i]);
}

/// Binary GCD for u16: out[i] = gcd(a[i], b[i]), preserving u16 dtype.
export fn gcd_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u16, a[i], b[i]);
}

/// Binary GCD for i8: out[i] = gcd(a[i], b[i]), preserving i8 dtype.
export fn gcd_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(i8, a[i], b[i]);
}

/// Binary GCD for u8: out[i] = gcd(a[i], b[i]), preserving u8 dtype.
export fn gcd_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u8, a[i], b[i]);
}

/// Scalar GCD for i16: out[i] = gcd(a[i], scalar), preserving i16 dtype.
export fn gcd_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(i16, a[i], scalar);
}

/// Scalar GCD for u16: out[i] = gcd(a[i], scalar), preserving u16 dtype.
export fn gcd_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, scalar: u16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u16, a[i], scalar);
}

/// Scalar GCD for i8: out[i] = gcd(a[i], scalar), preserving i8 dtype.
export fn gcd_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(i8, a[i], scalar);
}

/// Scalar GCD for u8: out[i] = gcd(a[i], scalar), preserving u8 dtype.
export fn gcd_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, scalar: u8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u8, a[i], scalar);
}

/// Binary GCD for u32: out[i] = gcd(a[i], b[i]), preserving u32 dtype.
export fn gcd_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u32, a[i], b[i]);
}

/// Scalar GCD for u32: out[i] = gcd(a[i], scalar), preserving u32 dtype.
export fn gcd_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, scalar: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u32, a[i], scalar);
}

/// Binary GCD for i64: out[i] = gcd(a[i], b[i]), preserving i64 dtype.
export fn gcd_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(i64, a[i], b[i]);
}

/// Scalar GCD for i64: out[i] = gcd(a[i], scalar), preserving i64 dtype.
export fn gcd_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(i64, a[i], scalar);
}

/// Binary GCD for u64: out[i] = gcd(a[i], b[i]), preserving u64 dtype.
export fn gcd_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u64, a[i], b[i]);
}

/// Scalar GCD for u64: out[i] = gcd(a[i], scalar), preserving u64 dtype.
export fn gcd_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, scalar: u64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = gcdGeneric(u64, a[i], scalar);
}

// --- Tests ---

test "gcd_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, 7, 0, -15 };
    var out: [5]i32 = undefined;
    gcd_scalar_i32(&a, &out, 5, 6);
    try testing.expectEqual(out[0], 6); // gcd(12,6)
    try testing.expectEqual(out[1], 6); // gcd(18,6)
    try testing.expectEqual(out[2], 1); // gcd(7,6)
    try testing.expectEqual(out[3], 6); // gcd(0,6)
    try testing.expectEqual(out[4], 3); // gcd(15,6)
}

test "gcd_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 12, 18 };
    const b = [_]i16{ 8, 12 };
    var out: [2]i16 = undefined;
    gcd_i16(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 4);
    try testing.expectEqual(out[1], 6);
}

test "gcd_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 12, 18 };
    const b = [_]u8{ 8, 12 };
    var out: [2]u8 = undefined;
    gcd_u8(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 4);
    try testing.expectEqual(out[1], 6);
}

test "gcd_i32 binary" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, 7, 0 };
    const b = [_]i32{ 8, 12, 5, 3 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 4); // gcd(12,8)
    try testing.expectEqual(out[1], 6); // gcd(18,12)
    try testing.expectEqual(out[2], 1); // gcd(7,5)
    try testing.expectEqual(out[3], 3); // gcd(0,3)
}

test "gcd_i32 edge case gcd(0,0)" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    const b = [_]i32{0};
    var out: [1]i32 = undefined;
    gcd_i32(&a, &b, &out, 1);
    try testing.expectEqual(out[0], 0); // gcd(0,0) = 0
}

test "gcd_i32 edge case gcd(x,0) and gcd(0,x)" {
    const testing = @import("std").testing;
    const a = [_]i32{ 15, 0, 0, 42 };
    const b = [_]i32{ 0, 15, 0, 0 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 15); // gcd(15,0)
    try testing.expectEqual(out[1], 15); // gcd(0,15)
    try testing.expectEqual(out[2], 0); // gcd(0,0)
    try testing.expectEqual(out[3], 42); // gcd(42,0)
}

test "gcd_i32 gcd(1,x) = 1" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 1, 1, 1 };
    const b = [_]i32{ 7, 100, 999, 1 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "gcd_i32 large coprime numbers" {
    const testing = @import("std").testing;
    const a = [_]i32{ 97, 1000003 };
    const b = [_]i32{ 89, 999979 };
    var out: [2]i32 = undefined;
    gcd_i32(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 1); // 97 and 89 are both prime
    try testing.expectEqual(out[1], 1); // coprime large numbers
}

test "gcd_i32 identity gcd(x,x) = x" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 7, 42, 100 };
    const b = [_]i32{ 1, 7, 42, 100 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 7);
    try testing.expectEqual(out[2], 42);
    try testing.expectEqual(out[3], 100);
}

test "gcd_i32 negative inputs" {
    const testing = @import("std").testing;
    const a = [_]i32{ -12, 12, -12, -18 };
    const b = [_]i32{ 8, -8, -8, -12 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 4); // gcd(|-12|,|8|)
    try testing.expectEqual(out[1], 4); // gcd(|12|,|-8|)
    try testing.expectEqual(out[2], 4); // gcd(|-12|,|-8|)
    try testing.expectEqual(out[3], 6); // gcd(|-18|,|-12|)
}

test "gcd_scalar_i32 gcd(x,0) returns abs(x)" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, -7, 0, 13 };
    var out: [4]i32 = undefined;
    gcd_scalar_i32(&a, &out, 4, 0);
    try testing.expectEqual(out[0], 5);
    try testing.expectEqual(out[1], 7);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 13);
}

test "gcd_scalar_i32 gcd(x,1) = 1" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, 100, -7, 0 };
    var out: [4]i32 = undefined;
    gcd_scalar_i32(&a, &out, 4, 1);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "gcd_scalar_i32 negative scalar" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, -15 };
    var out: [3]i32 = undefined;
    gcd_scalar_i32(&a, &out, 3, -6);
    try testing.expectEqual(out[0], 6); // gcd(12,6)
    try testing.expectEqual(out[1], 6); // gcd(18,6)
    try testing.expectEqual(out[2], 3); // gcd(15,6)
}

test "gcd_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 48, 100, 17, 0 };
    const b = [_]u16{ 18, 75, 13, 7 };
    var out: [4]u16 = undefined;
    gcd_u16(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], 25);
    try testing.expectEqual(out[2], 1); // both prime
    try testing.expectEqual(out[3], 7); // gcd(0, 7)
}

test "gcd_i8 negative inputs" {
    const testing = @import("std").testing;
    const a = [_]i8{ -12, 12, -24, 0 };
    const b = [_]i8{ 8, -8, -16, -7 };
    var out: [4]i8 = undefined;
    gcd_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 4);
    try testing.expectEqual(out[1], 4);
    try testing.expectEqual(out[2], 8);
    try testing.expectEqual(out[3], 7);
}

test "gcd_scalar_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 12, 18, 7, -15 };
    var out: [4]i16 = undefined;
    gcd_scalar_i16(&a, &out, 4, 6);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 3);
}

test "gcd_scalar_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 100, 75, 1, 0 };
    var out: [4]u16 = undefined;
    gcd_scalar_u16(&a, &out, 4, 25);
    try testing.expectEqual(out[0], 25);
    try testing.expectEqual(out[1], 25);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 25); // gcd(0, x) = x
}

test "gcd_scalar_i8 negative scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 12, -18, 0, 7 };
    var out: [4]i8 = undefined;
    gcd_scalar_i8(&a, &out, 4, -6);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], 6); // gcd(0, |-6|)
    try testing.expectEqual(out[3], 1);
}

test "gcd_scalar_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 12, 18, 7, 0 };
    var out: [4]u8 = undefined;
    gcd_scalar_u8(&a, &out, 4, 6);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 6);
}

test "gcd_i32 various known values" {
    const testing = @import("std").testing;
    const a = [_]i32{ 48, 54, 35, 100, 17, 144, 1000 };
    const b = [_]i32{ 18, 24, 15, 75, 13, 89, 600 };
    var out: [7]i32 = undefined;
    gcd_i32(&a, &b, &out, 7);
    try testing.expectEqual(out[0], 6); // gcd(48,18)
    try testing.expectEqual(out[1], 6); // gcd(54,24)
    try testing.expectEqual(out[2], 5); // gcd(35,15)
    try testing.expectEqual(out[3], 25); // gcd(100,75)
    try testing.expectEqual(out[4], 1); // gcd(17,13) both prime
    try testing.expectEqual(out[5], 1); // gcd(144,89) 89 is prime
    try testing.expectEqual(out[6], 200); // gcd(1000,600)
}

test "gcd_i64 large values beyond f64 exact range" {
    const testing = @import("std").testing;
    // Values above 2^53, where routing through f64 would lose the answer.
    const a = [_]i64{ 9007199254740994, 123456789012345678, -48 };
    const b = [_]i64{ 4503599627370497, 987654321098765432, 18 };
    var out: [3]i64 = undefined;
    gcd_i64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 4503599627370497);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 6);
}

test "gcd_u64 near the top of the range" {
    const testing = @import("std").testing;
    const a = [_]u64{ 18446744073709551614, 0, 1000000007 };
    const b = [_]u64{ 2, 7, 1000000007 };
    var out: [3]u64 = undefined;
    gcd_u64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 7);
    try testing.expectEqual(out[2], 1000000007);
}

test "gcd_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 48, 4294967294, 17 };
    const b = [_]u32{ 18, 2, 13 };
    var out: [3]u32 = undefined;
    gcd_u32(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 1);
}

test "gcd_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 12, 18, 7, -15 };
    var out: [4]i64 = undefined;
    gcd_scalar_i64(&a, &out, 4, 6);
    try testing.expectEqual(out[0], 6);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 3);
}

test "gcd most negative input does not trap" {
    const testing = @import("std").testing;
    // |minInt| is not representable; absWrap leaves it negative rather than
    // invoking illegal behaviour. NumPy agrees on the results below.
    const a8 = [_]i8{ -128, -128 };
    const b8 = [_]i8{ 2, 3 };
    var o8: [2]i8 = undefined;
    gcd_i8(&a8, &b8, &o8, 2);
    try testing.expectEqual(o8[0], 2);
    try testing.expectEqual(o8[1], 1);

    const a32 = [_]i32{ -2147483648, -2147483648, -2147483648 };
    const b32 = [_]i32{ 2, 3, 0 };
    var o32: [3]i32 = undefined;
    gcd_i32(&a32, &b32, &o32, 3);
    try testing.expectEqual(o32[0], 2);
    try testing.expectEqual(o32[1], 1);
    // gcd(x, 0) short-circuits and hands back the wrapped |x|.
    try testing.expectEqual(o32[2], -2147483648);

    var os: [1]i32 = undefined;
    gcd_scalar_i32(&[_]i32{-2147483648}, &os, 1, -2147483648);
    try testing.expectEqual(os[0], -2147483648);

    const a64 = [_]i64{-9223372036854775808};
    const b64 = [_]i64{4};
    var o64: [1]i64 = undefined;
    gcd_i64(&a64, &b64, &o64, 1);
    try testing.expectEqual(o64[0], 4);
}
