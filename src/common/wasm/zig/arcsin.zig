//! WASM element-wise arcsin kernels for float types.
//!
//! Unary: out[i] = asin(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.

const math = @import("std").math;

/// Element-wise arcsin for f64: out[i] = asin(a[i]).
export fn arcsin_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.asin(a[i]);
    }
}

/// Element-wise arcsin for f32: out[i] = asin(a[i]).
export fn arcsin_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.asin(@as(f64, a[i])));
    }
}

/// Element-wise arcsin for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn arcsin_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.asin(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise arcsin for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn arcsin_u64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.asin(@as(f64, @floatFromInt(a[i])));
    }
}

// --- Tests ---

test "arcsin_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.5, 1.0, -1.0 };
    var out: [4]f64 = undefined;
    arcsin_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.5235987755982988, 1e-10);
    try testing.expectApproxEqAbs(out[2], math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -math.pi / 2.0, 1e-10);
}

test "arcsin_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 0.5, 1.0 };
    var out: [3]f32 = undefined;
    arcsin_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.5236, 1e-4);
    try testing.expectApproxEqAbs(out[2], math.pi / 2.0, 1e-5);
}
