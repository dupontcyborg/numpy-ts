//! WASM element-wise arctan kernels for float types.
//!
//! Unary: out[i] = arctan(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//! WASM SIMD has no native arctan instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise arctan for f64: out[i] = atan(a[i]).
/// No SIMD — WASM has no f64x2.atan instruction.
export fn arctan_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan(a[i]);
    }
}

/// Element-wise arctan for f32: out[i] = atan(a[i]).
/// No SIMD — WASM has no f32x4.atan instruction.
export fn arctan_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.atan(@as(f64, a[i])));
    }
}

/// Element-wise arctan for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn arctan_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise arctan for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn arctan_u64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan(@as(f64, @floatFromInt(a[i])));
    }
}

// --- Tests ---

test "arctan_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, -1.0 };
    var out: [3]f64 = undefined;
    arctan_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.7853981633974483, 1e-10);
    try testing.expectApproxEqAbs(out[2], -0.7853981633974483, 1e-10);
}

test "arctan_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -1.0 };
    var out: [3]f32 = undefined;
    arctan_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.7854, 1e-4);
    try testing.expectApproxEqAbs(out[2], -0.7854, 1e-4);
}
