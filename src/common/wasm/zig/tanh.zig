//! WASM element-wise tanh kernels for float types.
//!
//! Unary: out[i] = tanh(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//! WASM SIMD has no native tanh instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise tanh for f64: out[i] = tanh(a[i]).
/// No SIMD — WASM has no f64x2.tanh instruction.
export fn tanh_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.tanh(a[i]));
    }
}

/// Element-wise tanh for f32: out[i] = tanh(a[i]).
/// No SIMD — WASM has no f32x4.tanh instruction.
export fn tanh_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.tanh(@as(f64, a[i])));
    }
}

/// Element-wise tanh for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn tanh_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.tanh(@as(f64, @floatFromInt(a[i]))));
    }
}

/// Element-wise tanh for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn tanh_u64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.tanh(@as(f64, @floatFromInt(a[i]))));
    }
}

// --- Tests ---

test "tanh_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    tanh_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.7615941559557649, 1e-10);
}

test "tanh_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    tanh_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.7616, 1e-4);
}
