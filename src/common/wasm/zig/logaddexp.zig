//! WASM element-wise logaddexp kernels for float types.
//!
//! Binary: out[i] = log(exp(a[i]) + exp(b[i]))
//! Scalar: out[i] = log(exp(a[i]) + exp(scalar))
//! Uses numerically stable formula: max(a,b) + log1p(exp(min-max)).
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");
const math = @import("std").math;

/// Element-wise logaddexp for f64: out[i] = log(exp(a[i]) + exp(b[i])).
export fn logaddexp_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const va = a[i];
        const vb = b[i];
        const mx = @max(va, vb);
        const mn = @min(va, vb);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp scalar for f64: out[i] = log(exp(a[i]) + exp(scalar)).
export fn logaddexp_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const va = a[i];
        const mx = @max(va, scalar);
        const mn = @min(va, scalar);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp for f32: out[i] = log(exp(a[i]) + exp(b[i])).
/// Computes in f64 for numerical stability, casts back to f32.
export fn logaddexp_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const va: f64 = @floatCast(a[i]);
        const vb: f64 = @floatCast(b[i]);
        const mx = @max(va, vb);
        const mn = @min(va, vb);
        out[i] = @floatCast(mx + math.log1p(math.exp(mn - mx)));
    }
}

/// Element-wise logaddexp scalar for f32: out[i] = log(exp(a[i]) + exp(scalar)).
/// Computes in f64 for numerical stability, casts back to f32.
export fn logaddexp_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: f64 = @floatCast(scalar);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const va: f64 = @floatCast(a[i]);
        const mx = @max(va, s);
        const mn = @min(va, s);
        out[i] = @floatCast(mx + math.log1p(math.exp(mn - mx)));
    }
}

/// Element-wise logaddexp for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn logaddexp_i64(a: [*]const i64, b: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        const mx = @max(af, bf);
        const mn = @min(af, bf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp scalar for i64 → f64 output.
export fn logaddexp_scalar_i64(a: [*]const i64, out: [*]f64, N: u32, scalar: i64) void {
    const sf = @as(f64, @floatFromInt(scalar));
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const mx = @max(af, sf);
        const mn = @min(af, sf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp for i32 → f64 output.
export fn logaddexp_i32(a: [*]const i32, b: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        const mx = @max(af, bf);
        const mn = @min(af, bf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp scalar for i32 → f64 output.
export fn logaddexp_scalar_i32(a: [*]const i32, out: [*]f64, N: u32, scalar: i32) void {
    const sf = @as(f64, @floatFromInt(scalar));
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const mx = @max(af, sf);
        const mn = @min(af, sf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp for i16 → f64 output.
export fn logaddexp_i16(a: [*]const i16, b: [*]const i16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        const mx = @max(af, bf);
        const mn = @min(af, bf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp scalar for i16 → f64 output.
export fn logaddexp_scalar_i16(a: [*]const i16, out: [*]f64, N: u32, scalar: i16) void {
    const sf = @as(f64, @floatFromInt(scalar));
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const mx = @max(af, sf);
        const mn = @min(af, sf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp for i8 → f64 output.
export fn logaddexp_i8(a: [*]const i8, b: [*]const i8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const bf = @as(f64, @floatFromInt(b[i]));
        const mx = @max(af, bf);
        const mn = @min(af, bf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

/// Element-wise logaddexp scalar for i8 → f64 output.
export fn logaddexp_scalar_i8(a: [*]const i8, out: [*]f64, N: u32, scalar: i8) void {
    const sf = @as(f64, @floatFromInt(scalar));
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const af = @as(f64, @floatFromInt(a[i]));
        const mx = @max(af, sf);
        const mn = @min(af, sf);
        out[i] = mx + math.log1p(math.exp(mn - mx));
    }
}

// --- Tests ---

test "logaddexp_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.0, 1.0 };
    const b = [_]f64{ 1.0, 0.0, 2.0 };
    var out: [3]f64 = undefined;
    logaddexp_f64(&a, &b, &out, 3);
    // log(exp(1)+exp(1)) = 1 + log(2) ≈ 1.6931
    try testing.expectApproxEqAbs(out[0], 1.6931471805599454, 1e-10);
    // log(exp(0)+exp(0)) = log(2) ≈ 0.6931
    try testing.expectApproxEqAbs(out[1], 0.6931471805599453, 1e-10);
    // log(exp(1)+exp(2)) ≈ 2.3133
    try testing.expectApproxEqAbs(out[2], 2.3132616875182228, 1e-10);
}

test "logaddexp_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 0.0, 1.0 };
    const b = [_]f32{ 1.0, 0.0, 2.0 };
    var out: [3]f32 = undefined;
    logaddexp_f32(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.6931, 1e-3);
    try testing.expectApproxEqAbs(out[1], 0.6931, 1e-3);
    try testing.expectApproxEqAbs(out[2], 2.3133, 1e-3);
}

test "logaddexp_scalar_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.0 };
    var out: [2]f64 = undefined;
    logaddexp_scalar_f64(&a, &out, 2, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.6931471805599454, 1e-10);
    // log(exp(0)+exp(1)) = log(1+e) ≈ 1.3133
    try testing.expectApproxEqAbs(out[1], 1.3132616875182228, 1e-10);
}

test "logaddexp_scalar_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 0.0 };
    var out: [2]f32 = undefined;
    logaddexp_scalar_f32(&a, &out, 2, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.6931, 1e-3);
    try testing.expectApproxEqAbs(out[1], 1.3133, 1e-3);
}

test "logaddexp_f64 N=1 boundary" {
    const testing = @import("std").testing;
    const a = [_]f64{0.0};
    const b = [_]f64{0.0};
    var out: [1]f64 = undefined;
    logaddexp_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 0, 1 };
    const b = [_]i64{ 1, 0, 2 };
    var out: [3]f64 = undefined;
    logaddexp_i64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.6931471805599454, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.6931471805599453, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.3132616875182228, 1e-10);
}

test "logaddexp_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 0 };
    var out: [2]f64 = undefined;
    logaddexp_scalar_i64(&a, &out, 2, 1);
    try testing.expectApproxEqAbs(out[0], 1.6931471805599454, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.3132616875182228, 1e-10);
}
