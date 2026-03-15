//! WASM element-wise sinh kernels for float types.
//!
//! Unary: out[i] = sinh(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//! WASM SIMD has no native sinh instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise sinh for f64: out[i] = sinh(a[i]).
/// No SIMD — WASM has no f64x2.sinh instruction.
export fn sinh_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.sinh(a[i]);
    }
}

/// Element-wise sinh for f32: out[i] = sinh(a[i]).
/// No SIMD — WASM has no f32x4.sinh instruction.
export fn sinh_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.sinh(@as(f64, a[i])));
    }
}

/// Element-wise sinh for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn sinh_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.sinh(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise sinh for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn sinh_u64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.sinh(@as(f64, @floatFromInt(a[i])));
    }
}

// --- Tests ---

test "sinh_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    sinh_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.1752011936438014, 1e-10);
}

test "sinh_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    sinh_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.1752, 1e-4);
}
