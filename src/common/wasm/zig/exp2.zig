//! WASM element-wise exp2 kernels for float types.
//!
//! Unary: out[i] = exp2(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//! WASM SIMD has no native exp2 instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise exp2 for f64: out[i] = exp2(a[i]).
/// No SIMD — WASM has no f64x2.exp2 instruction.
export fn exp2_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp2(a[i]);
    }
}

/// Element-wise exp2 for f32: out[i] = exp2(a[i]).
/// No SIMD — WASM has no f32x4.exp2 instruction.
export fn exp2_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.exp2(@as(f64, a[i])));
    }
}

/// Element-wise exp2 for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn exp2_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp2(@as(f64, @floatFromInt(a[i])));
    }
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
