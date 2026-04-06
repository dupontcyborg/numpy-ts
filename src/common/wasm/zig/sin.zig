//! WASM element-wise sine kernels for float types.
//!
//! Unary: out[i] = sin(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.
//! WASM SIMD has no native sin instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise sin for f64: out[i] = sin(a[i]).
export fn sin_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.sin(a[i]);
    }
}

/// Element-wise sin for f32: out[i] = sin(a[i]).
export fn sin_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.sin(@as(f64, a[i])));
    }
}

// --- Integer-to-f64 variants (avoid JS conversion loop) ---

/// Element-wise sin for i64 → f64 output: out[i] = sin(float(a[i])).
export fn sin_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for u64 → f64 output: out[i] = sin(float(a[i])).
export fn sin_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for i32 → f64 output: out[i] = sin(float(a[i])).
export fn sin_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for u32 → f64 output: out[i] = sin(float(a[i])).
export fn sin_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for i16 → f64 output: out[i] = sin(float(a[i])).
export fn sin_i16_f64(a: [*]const i16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for u16 → f64 output: out[i] = sin(float(a[i])).
export fn sin_u16_f64(a: [*]const u16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for i8 → f64 output: out[i] = sin(float(a[i])).
export fn sin_i8_f64(a: [*]const i8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}
/// Element-wise sin for u8 → f64 output: out[i] = sin(float(a[i])).
export fn sin_u8_f64(a: [*]const u8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}

// --- Tests ---

test "sin_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0 };
    var out: [4]f64 = undefined;
    sin_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -1.0, 1e-10);
}

test "sin_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0, math.pi / 2.0, math.pi };
    var out: [3]f32 = undefined;
    sin_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-5);
}
