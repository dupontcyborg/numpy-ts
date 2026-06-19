//! WASM element-wise logaddexp kernels for float types.
//!
//! Binary: out[i] = log(exp(a[i]) + exp(b[i]))
//! Scalar: out[i] = log(exp(a[i]) + exp(scalar))
//! Uses numerically stable formula: max(a,b) + log1p(exp(min-max)).
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");
const t = @import("transcend.zig");
const math = @import("std").math;

/// logaddexp of a 2-wide f64 lane: max(a,b) + softplus(min−max).
/// `softplusv_f64` fuses the log1p∘exp into one polynomial for |min−max| ≤ 2
/// (no exp/log/divide) and falls back to the general path for spread lanes.
inline fn lae_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const mx = simd.max_f64x2(a, b);
    const mn = simd.min_f64x2(a, b);
    return mx + t.softplusv_f64(mn - mx);
}

/// Element-wise logaddexp for f64: out[i] = log(exp(a[i]) + exp(b[i])).
export fn logaddexp_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, lae_f64(simd.load2_f64(a, i), simd.load2_f64(b, i)));
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        const bv: simd.V2f64 = .{ b[i], b[i] };
        out[i] = lae_f64(av, bv)[0];
    }
}

/// Element-wise logaddexp for f32: out[i] = log(exp(a[i]) + exp(b[i])).
/// Computes in f64 for numerical stability, casts back to f32.
export fn logaddexp_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ a[i], a[i + 1] };
        const bv: simd.V2f64 = .{ b[i], b[i + 1] };
        const r = lae_f64(av, bv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        const bv: simd.V2f64 = .{ b[i], b[i] };
        out[i] = @floatCast(lae_f64(av, bv)[0]);
    }
}

// Integer inputs are widened to f64 lanes and run through the same fused SIMD
// core (`lae_f64` → `softplusv_f64`), so they get the transcendental-free fast
// path too. i32/u32 vectorize the int→f64 convert; i64/u64 scalarize it (no
// i64x2→f64x2 convert in WASM SIMD) but still vectorize the softplus math.
inline fn laeIntToF64(comptime T: type, a: [*]const T, b: [*]const T, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]) };
        const bv: simd.V2f64 = .{ @floatFromInt(b[i]), @floatFromInt(b[i + 1]) };
        simd.store2_f64(out, i, lae_f64(av, bv));
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = @splat(@floatFromInt(a[i]));
        const bv: simd.V2f64 = @splat(@floatFromInt(b[i]));
        out[i] = lae_f64(av, bv)[0];
    }
}

inline fn laeIntToF32(comptime T: type, a: [*]const T, b: [*]const T, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]) };
        const bv: simd.V2f64 = .{ @floatFromInt(b[i]), @floatFromInt(b[i + 1]) };
        const r = lae_f64(av, bv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = @splat(@floatFromInt(a[i]));
        const bv: simd.V2f64 = @splat(@floatFromInt(b[i]));
        out[i] = @floatCast(lae_f64(av, bv)[0]);
    }
}

/// Element-wise logaddexp for i64 → f64 output.
export fn logaddexp_i64(a: [*]const i64, b: [*]const i64, out: [*]f64, N: u32) void {
    laeIntToF64(i64, a, b, out, N);
}

/// Element-wise logaddexp for u64 → f64 output.
export fn logaddexp_u64(a: [*]const u64, b: [*]const u64, out: [*]f64, N: u32) void {
    laeIntToF64(u64, a, b, out, N);
}

/// Element-wise logaddexp for i32 → f64 output.
export fn logaddexp_i32(a: [*]const i32, b: [*]const i32, out: [*]f64, N: u32) void {
    laeIntToF64(i32, a, b, out, N);
}

/// Element-wise logaddexp for u32 → f64 output.
export fn logaddexp_u32(a: [*]const u32, b: [*]const u32, out: [*]f64, N: u32) void {
    laeIntToF64(u32, a, b, out, N);
}

/// Element-wise logaddexp for i16 → f32 output.
export fn logaddexp_i16(a: [*]const i16, b: [*]const i16, out: [*]f32, N: u32) void {
    laeIntToF32(i16, a, b, out, N);
}

/// Element-wise logaddexp for u16 → f32 output.
export fn logaddexp_u16(a: [*]const u16, b: [*]const u16, out: [*]f32, N: u32) void {
    laeIntToF32(u16, a, b, out, N);
}

/// Element-wise logaddexp for i8 → f32 output.
export fn logaddexp_i8(a: [*]const i8, b: [*]const i8, out: [*]f32, N: u32) void {
    laeIntToF32(i8, a, b, out, N);
}

/// Element-wise logaddexp for u8 → f32 output.
export fn logaddexp_u8(a: [*]const u8, b: [*]const u8, out: [*]f32, N: u32) void {
    laeIntToF32(u8, a, b, out, N);
}

/// Element-wise logaddexp scalar for f64: out[i] = log(exp(a[i]) + exp(scalar)).
export fn logaddexp_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const bv: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, lae_f64(simd.load2_f64(a, i), bv));
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        out[i] = lae_f64(av, bv)[0];
    }
}

/// Element-wise logaddexp scalar for f32: out[i] = log(exp(a[i]) + exp(scalar)).
/// Computes in f64 for numerical stability, casts back to f32.
export fn logaddexp_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: f64 = @floatCast(scalar);
    const bv: simd.V2f64 = @splat(s);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = lae_f64(av, bv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(lae_f64(av, bv)[0]);
    }
}

inline fn laeScalarIntToF64(comptime T: type, a: [*]const T, out: [*]f64, N: u32, scalar: T) void {
    const bv: simd.V2f64 = @splat(@floatFromInt(scalar));
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]) };
        simd.store2_f64(out, i, lae_f64(av, bv));
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = @splat(@floatFromInt(a[i]));
        out[i] = lae_f64(av, bv)[0];
    }
}

inline fn laeScalarIntToF32(comptime T: type, a: [*]const T, out: [*]f32, N: u32, scalar: T) void {
    const bv: simd.V2f64 = @splat(@floatFromInt(scalar));
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const av: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i + 1]) };
        const r = lae_f64(av, bv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const av: simd.V2f64 = @splat(@floatFromInt(a[i]));
        out[i] = @floatCast(lae_f64(av, bv)[0]);
    }
}

/// Element-wise logaddexp scalar for i64 → f64 output.
export fn logaddexp_scalar_i64(a: [*]const i64, out: [*]f64, N: u32, scalar: i64) void {
    laeScalarIntToF64(i64, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for u64 → f64 output.
export fn logaddexp_scalar_u64(a: [*]const u64, out: [*]f64, N: u32, scalar: u64) void {
    laeScalarIntToF64(u64, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for i32 → f64 output.
export fn logaddexp_scalar_i32(a: [*]const i32, out: [*]f64, N: u32, scalar: i32) void {
    laeScalarIntToF64(i32, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for u32 → f64 output.
export fn logaddexp_scalar_u32(a: [*]const u32, out: [*]f64, N: u32, scalar: u32) void {
    laeScalarIntToF64(u32, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for i16 → f32 output.
export fn logaddexp_scalar_i16(a: [*]const i16, out: [*]f32, N: u32, scalar: i16) void {
    laeScalarIntToF32(i16, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for u16 → f32 output.
export fn logaddexp_scalar_u16(a: [*]const u16, out: [*]f32, N: u32, scalar: u16) void {
    laeScalarIntToF32(u16, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for i8 → f32 output.
export fn logaddexp_scalar_i8(a: [*]const i8, out: [*]f32, N: u32, scalar: i8) void {
    laeScalarIntToF32(i8, a, out, N, scalar);
}

/// Element-wise logaddexp scalar for u8 → f32 output.
export fn logaddexp_scalar_u8(a: [*]const u8, out: [*]f32, N: u32, scalar: u8) void {
    laeScalarIntToF32(u8, a, out, N, scalar);
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

test "logaddexp_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    const b = [_]u64{0};
    var out: [1]f64 = undefined;
    logaddexp_u64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    const b = [_]i32{0};
    var out: [1]f64 = undefined;
    logaddexp_i32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    const b = [_]u32{0};
    var out: [1]f64 = undefined;
    logaddexp_u32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    const b = [_]i16{0};
    var out: [1]f32 = undefined;
    logaddexp_i16(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    const b = [_]u16{0};
    var out: [1]f32 = undefined;
    logaddexp_u16(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    const b = [_]i8{0};
    var out: [1]f32 = undefined;
    logaddexp_i8(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    const b = [_]u8{0};
    var out: [1]f32 = undefined;
    logaddexp_u8(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_scalar_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    logaddexp_scalar_u64(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    logaddexp_scalar_i32(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_scalar_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    logaddexp_scalar_u32(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-10);
}

test "logaddexp_scalar_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    logaddexp_scalar_i16(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_scalar_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    logaddexp_scalar_u16(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_scalar_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    logaddexp_scalar_i8(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}

test "logaddexp_scalar_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    logaddexp_scalar_u8(&a, &out, 1, 0);
    try testing.expectApproxEqAbs(out[0], 0.6931471805599453, 1e-5);
}
