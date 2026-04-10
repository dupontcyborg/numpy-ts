//! WASM element-wise tangent kernels for float types.
//!
//! Unary: out[i] = tan(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.
//! WASM SIMD has no native tan instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise tan for f64: out[i] = tan(a[i]).
/// No SIMD — WASM has no f64x2.tan instruction.
export fn tan_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.tan(a[i]);
    }
}

/// Element-wise tan for f32: out[i] = tan(a[i]).
/// No SIMD — WASM has no f32x4.tan instruction.
export fn tan_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.tan(@as(f64, a[i])));
    }
}

// --- Integer-to-float variants ---

/// Element-wise tan for i64 → f64 output.
export fn tan_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.tan(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise tan for u64 → f64 output.
export fn tan_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.tan(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise tan for i32 → f64 output.
export fn tan_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.tan(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise tan for u32 → f64 output.
export fn tan_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.tan(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise tan for i16 → f32 output.
export fn tan_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.tan(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise tan for u16 → f32 output.
export fn tan_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.tan(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise tan for i8 → f32 output.
export fn tan_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.tan(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise tan for u8 → f32 output.
export fn tan_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.tan(@as(f64, @floatFromInt(a[i]))));
}

// --- Tests ---

test "tan_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 4.0, -math.pi / 4.0 };
    var out: [3]f64 = undefined;
    tan_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-10);
}

test "tan_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0, math.pi / 4.0, -math.pi / 4.0 };
    var out: [3]f32 = undefined;
    tan_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-5);
}

test "tan_f64 at pi" {
    const testing = @import("std").testing;
    const a = [_]f64{math.pi};
    var out: [1]f64 = undefined;
    tan_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    tan_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    tan_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    tan_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    tan_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    tan_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "tan_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    tan_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "tan_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    tan_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "tan_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    tan_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
