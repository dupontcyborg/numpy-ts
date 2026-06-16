//! WASM element-wise log1p (log(1 + x)) kernels for float / int types.
//!
//! Unary: out[i] = log1p(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Float/int paths use the shared SIMD core (transcend.log1pv_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow.

const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// Element-wise log1p for f64 using 2-wide SIMD.
export fn log1p_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.log1pv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = t.log1pv_f64(v)[0];
    }
}

/// Element-wise log1p for f32 via the 2-wide f64 core, then narrowed.
export fn log1p_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = t.log1pv_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(t.log1pv_f64(v)[0]);
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn log1pInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, t.log1pv_f64(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = t.log1pv_f64(v)[0];
    }
}

inline fn log1pInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = t.log1pv_f64(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(t.log1pv_f64(v)[0]);
    }
}

export fn log1p_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    log1pInt_f64(i64, a, out, N);
}
export fn log1p_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    log1pInt_f64(u64, a, out, N);
}
export fn log1p_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    log1pInt_f64(i32, a, out, N);
}
export fn log1p_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    log1pInt_f64(u32, a, out, N);
}
export fn log1p_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    log1pInt_f32(i16, a, out, N);
}
export fn log1p_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    log1pInt_f32(u16, a, out, N);
}
export fn log1p_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    log1pInt_f32(i8, a, out, N);
}
export fn log1p_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    log1pInt_f32(u8, a, out, N);
}

// --- Tests ---

test "log1p_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, -0.5, 0.5 };
    var out: [4]f64 = undefined;
    log1p_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-12);
    try testing.expectApproxEqAbs(out[1], 0.6931471805599453, 1e-12);
    try testing.expectApproxEqAbs(out[2], -0.6931471805599453, 1e-12);
    try testing.expectApproxEqAbs(out[3], 0.4054651081085771, 1e-12);
}

test "log1p_f64 near zero accuracy" {
    const testing = @import("std").testing;
    const a = [_]f64{1e-8};
    var out: [1]f64 = undefined;
    log1p_f64(&a, &out, 1);
    try testing.expectApproxEqRel(out[0], 9.9999999500000003e-9, 1e-10);
}

test "log1p_f64 domain" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f64{ -1.0, -2.0 };
    var out: [2]f64 = undefined;
    log1p_f64(&a, &out, 2);
    try testing.expect(out[0] == -std.math.inf(f64));
    try testing.expect(std.math.isNan(out[1]));
}

test "log1p_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1 };
    var out: [2]f64 = undefined;
    log1p_i32_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-12);
    try testing.expectApproxEqAbs(out[1], 0.6931471805599453, 1e-12);
}
