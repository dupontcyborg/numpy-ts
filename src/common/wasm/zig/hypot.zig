//! WASM element-wise hypotenuse kernels for float types.
//!
//! Binary: out[i] = sqrt(a[i]^2 + b[i]^2)
//! Scalar: out[i] = sqrt(a[i]^2 + scalar^2)
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");
const math = @import("std").math;

/// Element-wise hypotenuse for f64: out[i] = sqrt(a[i]^2 + b[i]^2).
export fn hypot_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @sqrt(a[i] * a[i] + b[i] * b[i]);
    }
}

/// Element-wise hypotenuse scalar for f64: out[i] = sqrt(a[i]^2 + scalar^2).
export fn hypot_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s2 = scalar * scalar;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @sqrt(a[i] * a[i] + s2);
    }
}

/// Element-wise hypotenuse for f32: out[i] = sqrt(a[i]^2 + b[i]^2).
export fn hypot_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @sqrt(a[i] * a[i] + b[i] * b[i]);
    }
}

/// Element-wise hypotenuse scalar for f32: out[i] = sqrt(a[i]^2 + scalar^2).
export fn hypot_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s2 = scalar * scalar;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @sqrt(a[i] * a[i] + s2);
    }
}

/// Element-wise hypotenuse for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn hypot_i64(a: [*]const i64, b: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        out[i] = @sqrt(af * af + bf * bf);
    }
}

/// Element-wise hypotenuse scalar for i64 → f64 output.
export fn hypot_scalar_i64(a: [*]const i64, out: [*]f64, N: u32, scalar: i64) void {
    const sf = @as(f64, @floatFromInt(scalar));
    const s2 = sf * sf;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        out[i] = @sqrt(af * af + s2);
    }
}

/// Element-wise hypotenuse for i32 → f64 output.
/// Uses SIMD: load i32x4, convert to f32x4, compute, widen to f64 for accumulation.
export fn hypot_i32(a: [*]const i32, b: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        out[i] = @sqrt(af * af + bf * bf);
    }
}

/// Element-wise hypotenuse scalar for i32 → f64 output.
export fn hypot_scalar_i32(a: [*]const i32, out: [*]f64, N: u32, scalar: i32) void {
    const sf = @as(f64, @floatFromInt(scalar));
    const s2 = sf * sf;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        out[i] = @sqrt(af * af + s2);
    }
}

/// Element-wise hypotenuse for i16 → f32 output.
export fn hypot_i16(a: [*]const i16, b: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        out[i] = @floatCast(@sqrt(af * af + bf * bf));
    }
}

/// Element-wise hypotenuse scalar for i16 → f32 output.
export fn hypot_scalar_i16(a: [*]const i16, out: [*]f32, N: u32, scalar: i16) void {
    const sf = @as(f64, @floatFromInt(scalar));
    const s2 = sf * sf;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        out[i] = @floatCast(@sqrt(af * af + s2));
    }
}

/// Element-wise hypotenuse for i8 → f32 output.
export fn hypot_i8(a: [*]const i8, b: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        out[i] = @floatCast(@sqrt(af * af + bf * bf));
    }
}

/// Element-wise hypotenuse scalar for i8 → f32 output.
export fn hypot_scalar_i8(a: [*]const i8, out: [*]f32, N: u32, scalar: i8) void {
    const sf = @as(f64, @floatFromInt(scalar));
    const s2 = sf * sf;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        out[i] = @floatCast(@sqrt(af * af + s2));
    }
}

// --- Tests ---

test "hypot_f64 basic (3,4)=5 and (5,12)=13" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 5.0, 0.0 };
    const b = [_]f64{ 4.0, 12.0, 0.0 };
    var out: [3]f64 = undefined;
    hypot_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 13.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
}

test "hypot_f32 basic (3,4)=5 and (5,12)=13" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 5.0, 0.0 };
    const b = [_]f32{ 4.0, 12.0, 0.0 };
    var out: [3]f32 = undefined;
    hypot_f32(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 13.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-5);
}

test "hypot_scalar_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 0.0 };
    var out: [2]f64 = undefined;
    hypot_scalar_f64(&a, &out, 2, 4.0);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
}

test "hypot_scalar_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 0.0 };
    var out: [2]f32 = undefined;
    hypot_scalar_f32(&a, &out, 2, 4.0);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-5);
}

test "hypot_f64 N=1 boundary" {
    const testing = @import("std").testing;
    const a = [_]f64{5.0};
    const b = [_]f64{12.0};
    var out: [1]f64 = undefined;
    hypot_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 13.0, 1e-10);
}

test "hypot_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 3, 5, 0 };
    const b = [_]i64{ 4, 12, 0 };
    var out: [3]f64 = undefined;
    hypot_i64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 13.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
}

test "hypot_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 3, 0 };
    var out: [2]f64 = undefined;
    hypot_scalar_i64(&a, &out, 2, 4);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
}

test "hypot_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{3};
    const b = [_]i32{4};
    var out: [1]f64 = undefined;
    hypot_i32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
}

test "hypot_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{3};
    var out: [1]f64 = undefined;
    hypot_scalar_i32(&a, &out, 1, 4);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
}

test "hypot_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{3};
    const b = [_]i16{4};
    var out: [1]f32 = undefined;
    hypot_i16(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
}

test "hypot_scalar_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{3};
    var out: [1]f32 = undefined;
    hypot_scalar_i16(&a, &out, 1, 4);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
}

test "hypot_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{3};
    const b = [_]i8{4};
    var out: [1]f32 = undefined;
    hypot_i8(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
}

test "hypot_scalar_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{3};
    var out: [1]f32 = undefined;
    hypot_scalar_i8(&a, &out, 1, 4);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
}
