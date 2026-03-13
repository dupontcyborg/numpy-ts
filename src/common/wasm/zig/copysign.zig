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

/// Element-wise copysign for i64, scalar loop (no i64x2 compare in WASM SIMD).
/// Output is f64.
export fn copysign_i64(x1: [*]const i64, x2: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign scalar for i64, scalar loop (no i64x2 compare in WASM SIMD).
/// Output is f64.
export fn copysign_scalar_i64(x1: [*]const i64, out: [*]f64, N: u32, scalar: i64) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for i32, output f64: out[i] = copysign(x1[i], x2[i]).
export fn copysign_i32(x1: [*]const i32, x2: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign scalar for i32, output f64: out[i] = copysign(x1[i], scalar).
export fn copysign_scalar_i32(x1: [*]const i32, out: [*]f64, N: u32, scalar: i32) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for i16, output f64: out[i] = copysign(x1[i], x2[i]).
export fn copysign_i16(x1: [*]const i16, x2: [*]const i16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign scalar for i16, output f64: out[i] = copysign(x1[i], scalar).
export fn copysign_scalar_i16(x1: [*]const i16, out: [*]f64, N: u32, scalar: i16) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
    }
}

/// Element-wise copysign for i8, output f64: out[i] = copysign(x1[i], x2[i]).
export fn copysign_i8(x1: [*]const i8, x2: [*]const i8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const mag: f64 = @abs(@as(f64, @floatFromInt(x1[i])));
        const sign: f64 = if (x2[i] > 0) 1.0 else if (x2[i] < 0) -1.0 else 0.0;
        out[i] = sign * mag;
    }
}

/// Element-wise copysign scalar for i8, output f64: out[i] = copysign(x1[i], scalar).
export fn copysign_scalar_i8(x1: [*]const i8, out: [*]f64, N: u32, scalar: i8) void {
    const sign: f64 = if (scalar > 0) 1.0 else if (scalar < 0) -1.0 else 0.0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = sign * @abs(@as(f64, @floatFromInt(x1[i])));
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
