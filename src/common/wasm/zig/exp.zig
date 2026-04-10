//! WASM element-wise exp kernels for float types.
//!
//! Unary: out[i] = exp(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//! WASM SIMD has no native exp instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise exp for f64: out[i] = exp(a[i]).
/// No SIMD — WASM has no f64x2.exp instruction.
export fn exp_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp(a[i]);
    }
}

/// Element-wise exp for f32: out[i] = exp(a[i]).
/// No SIMD — WASM has no f32x4.exp instruction.
export fn exp_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.exp(@as(f64, a[i])));
    }
}

/// Element-wise exp for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn exp_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise exp for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn exp_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise exp for i32 → f64 output.
export fn exp_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise exp for u32 → f64 output.
export fn exp_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise exp for i16 → f32 output.
export fn exp_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise exp for u16 → f32 output.
export fn exp_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise exp for i8 → f32 output.
export fn exp_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise exp for u8 → f32 output.
export fn exp_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

// --- Tests ---

test "exp_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    exp_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.718281828459045, 1e-10);
}

test "exp_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    exp_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.7183, 1e-4);
}

test "exp_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    exp_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    exp_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    exp_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    exp_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    exp_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    exp_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    exp_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    exp_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}
