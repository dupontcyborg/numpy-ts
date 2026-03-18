//! WASM element-wise copysign kernels for all numeric types.
//!
//! Binary: out[i] = copysign(x1[i], x2[i])  (magnitude of x1, sign of x2)
//! Scalar: out[i] = copysign(x1[i], scalar)
//! Output is always f64. Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

const sign_mask_64: u64 = @as(u64, 1) << 63;
const mag_mask_64: u64 = ~sign_mask_64;

/// Element-wise copysign for f64 using 2-wide SIMD: out[i] = copysign(x1[i], x2[i]).
/// Uses bitwise manipulation to combine magnitude of x1 with sign of x2.
export fn copysign_f64(x1: [*]const f64, x2: [*]const f64, out: [*]f64, N: u32) void {
    const sign_v: simd.V2u64 = @splat(sign_mask_64);
    const mag_v: simd.V2u64 = @splat(mag_mask_64);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const mag: simd.V2u64 = @as(simd.V2u64, @bitCast(simd.load2_f64(x1, i))) & mag_v;
        const sign: simd.V2u64 = @as(simd.V2u64, @bitCast(simd.load2_f64(x2, i))) & sign_v;
        simd.store2_f64(out, i, @as(simd.V2f64, @bitCast(mag | sign)));
    }
    while (i < N) : (i += 1) {
        const mag = @as(u64, @bitCast(x1[i])) & mag_mask_64;
        const sign = @as(u64, @bitCast(x2[i])) & sign_mask_64;
        out[i] = @as(f64, @bitCast(mag | sign));
    }
}

/// Element-wise copysign for f32, output f64: out[i] = copysign(x1[i], x2[i]).
export fn copysign_f32(x1: [*]const f32, x2: [*]const f32, out: [*]f64, N: u32) void {
    const sign_mask_32: u32 = @as(u32, 1) << 31;
    const mag_mask_32: u32 = ~sign_mask_32;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag = @as(u32, @bitCast(x1[i])) & mag_mask_32;
        const sign = @as(u32, @bitCast(x2[i])) & sign_mask_32;
        out[i] = @as(f64, @as(f32, @bitCast(mag | sign)));
    }
}

/// Element-wise copysign for i64, output is f64.
export fn copysign_i64(x1: [*]const i64, x2: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for u64, output is f64.
export fn copysign_u64(x1: [*]const u64, x2: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @as(f64, @floatFromInt(x1[i]));
        const sign: f64 = if (x2[i] > 0) 1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for i32, output is f64.
export fn copysign_i32(x1: [*]const i32, x2: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for u32, output is f64.
export fn copysign_u32(x1: [*]const u32, x2: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @as(f64, @floatFromInt(x1[i]));
        const sign: f64 = if (x2[i] > 0) 1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for i16, output is f64.
export fn copysign_i16(x1: [*]const i16, x2: [*]const i16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for u16, output is f64.
export fn copysign_u16(x1: [*]const u16, x2: [*]const u16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @as(f64, @floatFromInt(x1[i]));
        const sign: f64 = if (x2[i] > 0) 1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for i8, output is f64.
export fn copysign_i8(x1: [*]const i8, x2: [*]const i8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign for u8, output is f64.
export fn copysign_u8(x1: [*]const u8, x2: [*]const u8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @as(f64, @floatFromInt(x1[i]));
        const sign: f64 = if (x2[i] > 0) 1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign scalar for f64 using 2-wide SIMD: out[i] = copysign(x1[i], scalar).
/// Uses bitwise manipulation to combine magnitude of x1 with sign of scalar.
export fn copysign_scalar_f64(x1: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const sign_bit = @as(u64, @bitCast(scalar)) & sign_mask_64;
    const sign_v: simd.V2u64 = @splat(sign_bit);
    const mag_v: simd.V2u64 = @splat(mag_mask_64);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const mag: simd.V2u64 = @as(simd.V2u64, @bitCast(simd.load2_f64(x1, i))) & mag_v;
        simd.store2_f64(out, i, @as(simd.V2f64, @bitCast(mag | sign_v)));
    }
    while (i < N) : (i += 1) {
        const mag = @as(u64, @bitCast(x1[i])) & mag_mask_64;
        out[i] = @as(f64, @bitCast(mag | sign_bit));
    }
}

/// Element-wise copysign scalar for f32, output f64: out[i] = copysign(x1[i], scalar).
export fn copysign_scalar_f32(x1: [*]const f32, out: [*]f64, N: u32, scalar: f32) void {
    const sign_mask_32: u32 = @as(u32, 1) << 31;
    const mag_mask_32: u32 = ~sign_mask_32;
    const sign_bit = @as(u32, @bitCast(scalar)) & sign_mask_32;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag = @as(u32, @bitCast(x1[i])) & mag_mask_32;
        out[i] = @as(f64, @as(f32, @bitCast(mag | sign_bit)));
    }
}

/// Element-wise copysign for i64 scalar, output is f64.
export fn copysign_scalar_i64(x1: [*]const i64, out: [*]f64, N: u32, scalar: i64) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for u64 scalar, output is f64.
export fn copysign_scalar_u64(x1: [*]const u64, out: [*]f64, N: u32, scalar: f64) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @as(f64, @floatFromInt(x1[i]));
    }
}

/// Element-wise copysign for i32 scalar, output is f64.
export fn copysign_scalar_i32(x1: [*]const i32, out: [*]f64, N: u32, scalar: i32) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for u32 scalar, output is f64.
export fn copysign_scalar_u32(x1: [*]const u32, out: [*]f64, N: u32, scalar: f64) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @as(f64, @floatFromInt(x1[i]));
    }
}

/// Element-wise copysign for i16 scalar, output is f64.
export fn copysign_scalar_i16(x1: [*]const i16, out: [*]f64, N: u32, scalar: i16) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for u16 scalar, output is f64.
export fn copysign_scalar_u16(x1: [*]const u16, out: [*]f64, N: u32, scalar: f64) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @as(f64, @floatFromInt(x1[i]));
    }
}

/// Element-wise copysign for i8 scalar, output is f64.
export fn copysign_scalar_i8(x1: [*]const i8, out: [*]f64, N: u32, scalar: i8) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for u8 scalar, output is f64.
export fn copysign_scalar_u8(x1: [*]const u8, out: [*]f64, N: u32, scalar: f64) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @as(f64, @floatFromInt(x1[i]));
    }
}

// --- Tests ---

test "copysign_f64 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ 1.0, -2.0, 3.0, -4.0, 0.0 };
    const x2 = [_]f64{ -1.0, 1.0, -1.0, 1.0, -1.0 };
    var out: [5]f64 = undefined;
    copysign_f64(&x1, &x2, &out, 5);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
}

test "copysign_scalar_f64 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ 1.0, -2.0, 3.0 };
    var out: [3]f64 = undefined;
    copysign_scalar_f64(&x1, &out, 3, -1.0);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
}

test "copysign_i32 basic" {
    const testing = @import("std").testing;
    const x1 = [_]i32{ 5, -3, 7 };
    const x2 = [_]i32{ -1, 2, -3 };
    var out: [3]f64 = undefined;
    copysign_i32(&x1, &x2, &out, 3);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -7.0, 1e-10);
}

test "copysign_scalar_i8 basic" {
    const testing = @import("std").testing;
    const x1 = [_]i8{ 5, -3, 0, 7 };
    var out: [4]f64 = undefined;
    copysign_scalar_i8(&x1, &out, 4, -1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -7.0, 1e-10);
}

test "copysign_f64 SIMD boundary N=3" {
    const testing = @import("std").testing;
    const x1 = [_]f64{ 5.0, -3.0, 0.0 };
    const x2 = [_]f64{ -1.0, -1.0, -1.0 };
    var out: [3]f64 = undefined;
    copysign_f64(&x1, &x2, &out, 3);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-10);
    // copysign(0, -1) = -0.0 (sign bit set)
}

test "copysign_f32 basic" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ 1.0, -2.0, 3.0 };
    const x2 = [_]f32{ -1.0, 1.0, 1.0 };
    var out: [3]f64 = undefined;
    copysign_f32(&x1, &x2, &out, 3);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-5);
}

test "copysign_i64 basic" {
    const testing = @import("std").testing;
    const x1 = [_]i64{ 5, -3, 0 };
    const x2 = [_]i64{ -1, 2, -3 };
    var out: [3]f64 = undefined;
    copysign_i64(&x1, &x2, &out, 3);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
}

test "copysign_scalar_i32 positive scalar" {
    const testing = @import("std").testing;
    const x1 = [_]i32{ 5, -3, 0, 7 };
    var out: [4]f64 = undefined;
    copysign_scalar_i32(&x1, &out, 4, 1);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 7.0, 1e-10);
}

test "copysign_i16 basic" {
    const testing = @import("std").testing;
    const x1 = [_]i16{ 100, -200, 300 };
    const x2 = [_]i16{ -1, -1, 1 };
    var out: [3]f64 = undefined;
    copysign_i16(&x1, &x2, &out, 3);
    try testing.expectApproxEqAbs(out[0], -100.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -200.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 300.0, 1e-10);
}

test "copysign_f64 SIMD boundary N=3 remainder" {
    const testing = @import("std").testing;
    // N=3: 2 by SIMD (V2f64), 1 by scalar remainder
    const x1 = [_]f64{ 10.0, 20.0, 30.0 };
    const x2 = [_]f64{ -1.0, 1.0, -1.0 };
    var out: [3]f64 = undefined;
    copysign_f64(&x1, &x2, &out, 3);
    try testing.expectApproxEqAbs(out[0], -10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 20.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -30.0, 1e-10);
}

test "copysign_f32 various signs" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ 5.0, -3.0, 7.0, -1.0, 0.0 };
    const x2 = [_]f32{ -1.0, -1.0, 1.0, 1.0, 1.0 };
    var out: [5]f64 = undefined;
    copysign_f32(&x1, &x2, &out, 5);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 7.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 0.0, 1e-5);
}

test "copysign_f64 with negative zero sign" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    // -0.0 has sign bit set, so copysign(x, -0.0) should give -|x|
    const neg_zero: f64 = -0.0;
    const x1 = [_]f64{ 5.0, 3.0 };
    const x2 = [_]f64{ neg_zero, neg_zero };
    var out: [2]f64 = undefined;
    copysign_f64(&x1, &x2, &out, 2);
    // Should be -5.0 and -3.0 (sign bit of -0.0 is set)
    try testing.expect(math.signbit(out[0]));
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expect(math.signbit(out[1]));
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-10);
}

test "copysign_f64 positive to negative" {
    const testing = @import("std").testing;
    // copysign(positive, negative) -> negative
    const x1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const x2 = [_]f64{ -5.0, -6.0, -7.0, -8.0 };
    var out: [4]f64 = undefined;
    copysign_f64(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -4.0, 1e-10);
}

test "copysign_f64 negative to positive" {
    const testing = @import("std").testing;
    // copysign(negative, positive) -> positive
    const x1 = [_]f64{ -1.0, -2.0, -3.0, -4.0 };
    const x2 = [_]f64{ 5.0, 6.0, 7.0, 8.0 };
    var out: [4]f64 = undefined;
    copysign_f64(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
}

test "copysign_i32 sign combinations" {
    const testing = @import("std").testing;
    // positive magnitude, negative sign
    const x1 = [_]i32{ 10, -20, 30, -40 };
    const x2 = [_]i32{ -1, 1, 1, -1 };
    var out: [4]f64 = undefined;
    copysign_i32(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], -10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 20.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 30.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -40.0, 1e-10);
}

test "copysign_i16 sign combinations" {
    const testing = @import("std").testing;
    const x1 = [_]i16{ 50, -60, 70, -80 };
    const x2 = [_]i16{ 1, -1, -1, 1 };
    var out: [4]f64 = undefined;
    copysign_i16(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], 50.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -60.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -70.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 80.0, 1e-10);
}

test "copysign_i8 sign combinations" {
    const testing = @import("std").testing;
    const x1 = [_]i8{ 5, -3, 7, -1 };
    const x2 = [_]i8{ -1, 1, -1, 1 };
    var out: [4]f64 = undefined;
    copysign_i8(&x1, &x2, &out, 4);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -7.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 1.0, 1e-10);
}

test "copysign_scalar_f64 with negative zero" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    const x1 = [_]f64{ 5.0, 3.0, 1.0 };
    var out: [3]f64 = undefined;
    copysign_scalar_f64(&x1, &out, 3, -0.0);
    // -0.0 has sign bit set, so all outputs should be negative
    try testing.expect(math.signbit(out[0]));
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expect(math.signbit(out[1]));
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-10);
    try testing.expect(math.signbit(out[2]));
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-10);
}

test "copysign_scalar_f32 positive scalar" {
    const testing = @import("std").testing;
    const x1 = [_]f32{ -5.0, 3.0, -1.0 };
    var out: [3]f64 = undefined;
    copysign_scalar_f32(&x1, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-5);
}

test "copysign_scalar_i64 negative scalar" {
    const testing = @import("std").testing;
    const x1 = [_]i64{ 5, -3, 10 };
    var out: [3]f64 = undefined;
    copysign_scalar_i64(&x1, &out, 3, -1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -10.0, 1e-10);
}

test "copysign_scalar_i16 positive scalar" {
    const testing = @import("std").testing;
    const x1 = [_]i16{ -100, 200, -300 };
    var out: [3]f64 = undefined;
    copysign_scalar_i16(&x1, &out, 3, 1);
    try testing.expectApproxEqAbs(out[0], 100.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 200.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 300.0, 1e-10);
}
