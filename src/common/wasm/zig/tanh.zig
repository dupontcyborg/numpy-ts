//! WASM element-wise tanh kernels for float / int types.
//!
//! Unary: out[i] = tanh(a[i])
//!
//! SIMD via the Cephes scheme, branchless + a per-vector early-out:
//!   |x| < 0.625 : rational  x + x·s·P(s)/Q(s),  s = x²   (no cancellation near 0)
//!   |x| ≥ 0.625 : 1 − 2/(e^{2x}+1)  (sign-correct; expv saturates → ±1 for big x)
//!   whole 2-lane group with |x| > 20 : store copysign(1,x), skip the exp entirely
//! The 1−2/(e^{2x}+1) form needs no abs/sign, so the old signed/unsigned codegen
//! split (int16 was ~3.4x slower than uint16) is gone. f32 and integer outputs
//! run through the 2-wide f64 core then narrow. Measured 1.4–9.4x faster than the
//! previous scalar kernels across all dtypes. See SIMD_VECTORIZATION_AUDIT.md.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

// Cephes tanh rational (|x| < 0.625), double precision.
const TP0: f64 = -9.64399179425052238628e-1;
const TP1: f64 = -9.92877231001918586564e1;
const TP2: f64 = -1.61468768441708447952e3;
const TQ0: f64 = 1.12811678491632931402e2;
const TQ1: f64 = 2.23548839060100448583e3;
const TQ2: f64 = 4.84406305325125486048e3;

const TANH_LO: f64 = 0.625;
const TANH_SAT: f64 = 20.0; // |x| > this → tanh = ±1 to f64 precision

/// Vectorized tanh for a 2-wide f64 lane (branchless rational/exp select).
inline fn tanhv_f64(x: simd.V2f64) simd.V2f64 {
    const one: simd.V2f64 = @splat(1.0);
    const two: simd.V2f64 = @splat(2.0);

    // exp branch: 1 - 2/(e^{2x}+1) — sign-correct for all x; expv clamps to ±inf
    // so this saturates to ±1 for large |x|.
    const e = t.expv_f64(two * x);
    const expB = one - two / (e + one);

    // rational branch: x + x·s·P(s)/Q(s), s = x²
    const s = x * x;
    var num: simd.V2f64 = @splat(TP0);
    num = simd.mulAdd_f64x2(num, s, @splat(TP1));
    num = simd.mulAdd_f64x2(num, s, @splat(TP2));
    var den: simd.V2f64 = s + @as(simd.V2f64, @splat(TQ0));
    den = simd.mulAdd_f64x2(den, s, @splat(TQ1));
    den = simd.mulAdd_f64x2(den, s, @splat(TQ2));
    const ratB = simd.mulAdd_f64x2(x * s, num / den, x); // x·s·(P/Q) + x

    return @select(f64, @abs(x) < @as(simd.V2f64, @splat(TANH_LO)), ratB, expB);
}

/// tanh of a 2-lane group with the per-vector early-out folded in.
inline fn tanhVec(x: simd.V2f64) simd.V2f64 {
    if (@reduce(.And, @abs(x) > @as(simd.V2f64, @splat(TANH_SAT)))) {
        const one: simd.V2f64 = @splat(1.0);
        return @select(f64, x < @as(simd.V2f64, @splat(0.0)), -one, one);
    }
    return tanhv_f64(x);
}

export fn tanh_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, tanhVec(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = tanhVec(v)[0];
    }
}

export fn tanh_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = tanhVec(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(tanhVec(v)[0]);
    }
}

// --- Integer inputs (widen to f64 via the 2-wide core, then narrow as needed) ---

inline fn tanhInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        simd.store2_f64(out, i, tanhVec(@floatFromInt(vi)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = tanhVec(v)[0];
    }
}

inline fn tanhInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = tanhVec(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(tanhVec(v)[0]);
    }
}

export fn tanh_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    tanhInt_f64(i64, a, out, N);
}
export fn tanh_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    tanhInt_f64(u64, a, out, N);
}
export fn tanh_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    tanhInt_f64(i32, a, out, N);
}
export fn tanh_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    tanhInt_f64(u32, a, out, N);
}
export fn tanh_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    tanhInt_f32(i16, a, out, N);
}
export fn tanh_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    tanhInt_f32(u16, a, out, N);
}
export fn tanh_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    tanhInt_f32(i8, a, out, N);
}
export fn tanh_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    tanhInt_f32(u8, a, out, N);
}

// --- Tests ---

test "tanh_f64 matches std over a wide range" {
    const testing = @import("std").testing;
    const xs = [_]f64{ 0.0, 0.1, 0.3, 0.5, 0.624, 0.7, 1.0, 2.0, 5.0, 19.0, 25.0, -0.4, -1.5, -8.0, -30.0 };
    var out: [16]f64 = undefined;
    var buf: [16]f64 = undefined;
    @memcpy(buf[0..xs.len], &xs);
    tanh_f64(&buf, &out, xs.len);
    for (xs, 0..) |x, i| try testing.expectApproxEqAbs(out[i], math.tanh(x), 1e-13);
}

test "tanh_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    tanh_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-12);
    try testing.expectApproxEqAbs(out[1], 0.7615941559557649, 1e-12);
}

test "tanh_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -2.0 };
    var out: [3]f32 = undefined;
    tanh_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-6);
    try testing.expectApproxEqAbs(out[1], 0.7615942, 1e-5);
    try testing.expectApproxEqAbs(out[2], -0.9640276, 1e-5);
}

test "tanh int variants" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 2, -3 };
    var out: [4]f64 = undefined;
    tanh_i32_f64(&a, &out, 4);
    for (a, 0..) |x, i| try testing.expectApproxEqAbs(out[i], math.tanh(@as(f64, @floatFromInt(x))), 1e-13);
    const b = [_]u8{ 0, 1, 2 };
    var o2: [3]f32 = undefined;
    tanh_u8_f32(&b, &o2, 3);
    try testing.expectApproxEqAbs(o2[2], @as(f32, @floatCast(math.tanh(@as(f64, 2.0)))), 1e-6);
}

test "tanh remaining int variants" {
    const testing = @import("std").testing;

    // f64-widening: i64, u64, u32
    const i64s = [_]i64{ 0, 1, 2, -3 };
    var o64: [4]f64 = undefined;
    tanh_i64_f64(&i64s, &o64, 4);
    for (i64s, 0..) |x, i| try testing.expectApproxEqAbs(o64[i], math.tanh(@as(f64, @floatFromInt(x))), 1e-13);

    const u64s = [_]u64{ 0, 1, 2, 3 };
    tanh_u64_f64(&u64s, &o64, 4);
    for (u64s, 0..) |x, i| try testing.expectApproxEqAbs(o64[i], math.tanh(@as(f64, @floatFromInt(x))), 1e-13);

    const u32s = [_]u32{ 0, 1, 2, 3 };
    tanh_u32_f64(&u32s, &o64, 4);
    for (u32s, 0..) |x, i| try testing.expectApproxEqAbs(o64[i], math.tanh(@as(f64, @floatFromInt(x))), 1e-13);

    // f32-widening: i16, u16, i8
    const i16s = [_]i16{ 0, 1, 2, -3 };
    var o32: [4]f32 = undefined;
    tanh_i16_f32(&i16s, &o32, 4);
    for (i16s, 0..) |x, i| try testing.expectApproxEqAbs(o32[i], @as(f32, @floatCast(math.tanh(@as(f64, @floatFromInt(x))))), 1e-6);

    const u16s = [_]u16{ 0, 1, 2, 3 };
    tanh_u16_f32(&u16s, &o32, 4);
    for (u16s, 0..) |x, i| try testing.expectApproxEqAbs(o32[i], @as(f32, @floatCast(math.tanh(@as(f64, @floatFromInt(x))))), 1e-6);

    const i8s = [_]i8{ 0, 1, 2, -3 };
    tanh_i8_f32(&i8s, &o32, 4);
    for (i8s, 0..) |x, i| try testing.expectApproxEqAbs(o32[i], @as(f32, @floatCast(math.tanh(@as(f64, @floatFromInt(x))))), 1e-6);
}
