//! WASM element-wise exp2 kernels for float types.
//!
//! Unary: out[i] = exp2(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//! WASM SIMD has no native exp2 instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise exp2 for f64: out[i] = exp2(a[i]).
export fn exp2_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(a[i]);
    }
}

/// Element-wise exp2 for f32: out[i] = exp2(a[i]).
export fn exp2_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(exp2_via_exp(a[i]));
    }
}

/// Element-wise exp2 for i64 → f64 output.
export fn exp2_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(@floatFromInt(a[i]));
    }
}

/// Element-wise exp2 for u64 → f64 output.
export fn exp2_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(@floatFromInt(a[i]));
    }
}

/// Compute exp2 via exp(x * ln2) to avoid ldexp dependency in WASM.
inline fn exp2_via_exp(x: f64) f64 {
    return math.exp(x * math.ln2);
}

/// Element-wise exp2 for i32 → f64 output.
export fn exp2_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = exp2_via_exp(@floatFromInt(a[i]));
}

/// Element-wise exp2 for u32 → f64 output.
export fn exp2_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = exp2_via_exp(@floatFromInt(a[i]));
}

/// Element-wise exp2 for i16 → f32 output.
export fn exp2_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

/// Element-wise exp2 for u16 → f32 output.
export fn exp2_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

/// Element-wise exp2 for i8 → f32 output.
export fn exp2_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

/// Element-wise exp2 for u8 → f32 output.
export fn exp2_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

// --- Tests ---

test "exp2_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, 3.0 };
    var out: [3]f64 = undefined;
    exp2_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 8.0, 1e-10);
}

test "exp2_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 3.0 };
    var out: [3]f32 = undefined;
    exp2_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 8.0, 1e-5);
}

test "exp2_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    exp2_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    exp2_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    exp2_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    exp2_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    exp2_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp2_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    exp2_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp2_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    exp2_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp2_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    exp2_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}
