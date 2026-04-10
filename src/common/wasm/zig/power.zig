//! WASM element-wise power kernels for all numeric types.
//!
//! Binary: out[i] = a[i] ^ b[i]
//! Scalar: out[i] = a[i] ^ scalar
//! Both operate on contiguous 1D buffers of length N.
//! For integer types, uses integer exponentiation (exponent >= 0).
//! For float types, uses std.math.pow.

const math = @import("std").math;
const simd = @import("simd.zig");

/// Element-wise power for f64: out[i] = a[i] ^ b[i].
export fn power_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.pow(f64, a[i], b[i]);
    }
}

/// Element-wise power scalar for f64: out[i] = a[i] ^ scalar.
export fn power_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.pow(f64, a[i], scalar);
    }
}

/// Element-wise power for f32: out[i] = a[i] ^ b[i].
export fn power_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.pow(f64, @as(f64, a[i]), @as(f64, b[i])));
    }
}

/// Element-wise power scalar for f32: out[i] = a[i] ^ scalar.
export fn power_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.pow(f64, @as(f64, a[i]), @as(f64, scalar)));
    }
}

/// Element-wise integer power for i64: out[i] = a[i] ^ b[i].
/// Scalar loop (no i64x2.mul in WASM SIMD).
export fn power_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i64(a[i], b[i]);
    }
}

/// Element-wise integer power scalar for i64: out[i] = a[i] ^ scalar.
export fn power_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i64(a[i], scalar);
    }
}

/// Element-wise integer power for i32: out[i] = a[i] ^ b[i].
export fn power_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i32(a[i], b[i]);
    }
}

/// Element-wise integer power scalar for i32: out[i] = a[i] ^ scalar.
export fn power_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i32(a[i], scalar);
    }
}

/// Element-wise integer power for i16: out[i] = a[i] ^ b[i].
export fn power_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i16(a[i], b[i]);
    }
}

/// Element-wise integer power scalar for i16: out[i] = a[i] ^ scalar.
export fn power_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i16(a[i], scalar);
    }
}

/// Element-wise integer power for i8: out[i] = a[i] ^ b[i].
export fn power_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i8(a[i], b[i]);
    }
}

/// Element-wise integer power scalar for i8: out[i] = a[i] ^ scalar.
export fn power_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = intPow_i8(a[i], scalar);
    }
}

fn intPow_i64(base: i64, exp: i64) i64 {
    if (exp == 0) return 1;
    var result: i64 = 1;
    var b = base;
    var e = exp;
    while (e > 0) {
        if (e & 1 == 1) result *%= b;
        b *%= b;
        e >>= 1;
    }
    return result;
}

fn intPow_i32(base: i32, exp: i32) i32 {
    if (exp == 0) return 1;
    var result: i32 = 1;
    var b = base;
    var e = exp;
    while (e > 0) {
        if (e & 1 == 1) result *%= b;
        b *%= b;
        e >>= 1;
    }
    return result;
}

fn intPow_i16(base: i16, exp: i16) i16 {
    if (exp == 0) return 1;
    var result: i16 = 1;
    var b = base;
    var e = exp;
    while (e > 0) {
        if (e & 1 == 1) result *%= b;
        b *%= b;
        e >>= 1;
    }
    return result;
}

fn intPow_i8(base: i8, exp: i8) i8 {
    if (exp == 0) return 1;
    var result: i8 = 1;
    var b = base;
    var e = exp;
    while (e > 0) {
        if (e & 1 == 1) result *%= b;
        b *%= b;
        e >>= 1;
    }
    return result;
}

// --- Tests ---

test "power_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 2, 3, 4 };
    const b = [_]f64{ 3, 2, 0.5 };
    var out: [3]f64 = undefined;
    power_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 8.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}

test "power_scalar_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    var out: [4]f64 = undefined;
    power_scalar_f64(&a, &out, 4, 2.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 16.0, 1e-10);
}

test "power_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 2, 3, 4 };
    const b = [_]f32{ 3, 2, 0.5 };
    var out: [3]f32 = undefined;
    power_f32(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 8.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 9.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-5);
}

test "power_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 2, 3, 5 };
    const b = [_]i32{ 10, 5, 3 };
    var out: [3]i32 = undefined;
    power_i32(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 1024);
    try testing.expectEqual(out[1], 243);
    try testing.expectEqual(out[2], 125);
}

test "power_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    var out: [3]i32 = undefined;
    power_scalar_i32(&a, &out, 3, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 8);
    try testing.expectEqual(out[2], 27);
}

test "power_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 2, 3 };
    const b = [_]i64{ 20, 10 };
    var out: [2]i64 = undefined;
    power_i64(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 1048576);
    try testing.expectEqual(out[1], 59049);
}

test "power_i8 zero exponent" {
    const testing = @import("std").testing;
    const a = [_]i8{ 5, -3, 0, 7 };
    const b = [_]i8{ 0, 0, 0, 0 };
    var out: [4]i8 = undefined;
    power_i8(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
}

test "power_f64 edge zero base" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.0 };
    const b = [_]f64{ 0.0, 5.0 };
    var out: [2]f64 = undefined;
    power_f64(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10); // 0^0 = 1
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10); // 0^5 = 0
}

test "power_scalar_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{2.0};
    var out: [1]f32 = undefined;
    power_scalar_f32(&a, &out, 1, 3.0);
    try testing.expectApproxEqAbs(out[0], 8.0, 1e-6);
}

test "power_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{2};
    var out: [1]i64 = undefined;
    power_scalar_i64(&a, &out, 1, 3);
    try testing.expectEqual(out[0], 8);
}

test "power_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{2};
    const b = [_]i16{3};
    var out: [1]i16 = undefined;
    power_i16(&a, &b, &out, 1);
    try testing.expectEqual(out[0], 8);
}

test "power_scalar_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{2};
    var out: [1]i16 = undefined;
    power_scalar_i16(&a, &out, 1, 3);
    try testing.expectEqual(out[0], 8);
}

test "power_scalar_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{2};
    var out: [1]i8 = undefined;
    power_scalar_i8(&a, &out, 1, 3);
    try testing.expectEqual(out[0], 8);
}
