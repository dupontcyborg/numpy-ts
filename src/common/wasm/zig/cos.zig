//! WASM element-wise cosine kernels for float types.
//!
//! Unary: out[i] = cos(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.
//! WASM SIMD has no native cos instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise cos for f64: out[i] = cos(a[i]).
/// No SIMD — WASM has no f64x2.cos instruction.
export fn cos_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.cos(a[i]);
    }
}

/// Element-wise cos for f32: out[i] = cos(a[i]).
/// No SIMD — WASM has no f32x4.cos instruction.
export fn cos_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.cos(@as(f64, a[i])));
    }
}

// --- Integer-to-f64 variants (avoid JS conversion loop) ---

/// Element-wise cos for i64 → f64 output: out[i] = cos(float(a[i])).
export fn cos_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.cos(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise cos for u64 → f64 output: out[i] = cos(float(a[i])).
export fn cos_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.cos(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise cos for i32 → f64 output: out[i] = cos(float(a[i])).
export fn cos_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.cos(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise cos for u32 → f64 output: out[i] = cos(float(a[i])).
export fn cos_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.cos(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise cos for i16 → f32 output: out[i] = cos(float(a[i])).
export fn cos_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cos(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise cos for u16 → f32 output: out[i] = cos(float(a[i])).
export fn cos_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cos(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise cos for i8 → f32 output: out[i] = cos(float(a[i])).
export fn cos_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cos(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise cos for u8 → f32 output: out[i] = cos(float(a[i])).
export fn cos_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cos(@as(f64, @floatFromInt(a[i]))));
}

// --- Tests ---

test "cos_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0 };
    var out: [4]f64 = undefined;
    cos_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
}

test "cos_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0, math.pi / 2.0, math.pi };
    var out: [3]f32 = undefined;
    cos_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-5);
}

test "cos_f64 negative values" {
    const testing = @import("std").testing;
    const a = [_]f64{ -math.pi / 2.0, -math.pi };
    var out: [2]f64 = undefined;
    cos_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -1.0, 1e-10);
}

test "cos_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    cos_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cos_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    cos_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cos_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    cos_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cos_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    cos_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cos_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    cos_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "cos_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    cos_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "cos_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    cos_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "cos_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    cos_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}
