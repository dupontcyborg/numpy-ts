//! WASM element-wise arccos kernels for float types.
//!
//! Unary: out[i] = acos(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.

const math = @import("std").math;

/// Element-wise arccos for f64: out[i] = acos(a[i]).
export fn arccos_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.acos(a[i]);
    }
}

/// Element-wise arccos for f32: out[i] = acos(a[i]).
export fn arccos_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.acos(@as(f64, a[i])));
    }
}

/// Element-wise arccos for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn arccos_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.acos(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise arccos for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn arccos_u64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.acos(@as(f64, @floatFromInt(a[i])));
    }
}

// --- Tests ---

test "arccos_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.5, 0.0, -1.0 };
    var out: [4]f64 = undefined;
    arccos_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0471975511965976, 1e-10);
    try testing.expectApproxEqAbs(out[2], math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], math.pi, 1e-10);
}

test "arccos_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 0.5, 0.0 };
    var out: [3]f32 = undefined;
    arccos_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0472, 1e-4);
    try testing.expectApproxEqAbs(out[2], math.pi / 2.0, 1e-5);
}
