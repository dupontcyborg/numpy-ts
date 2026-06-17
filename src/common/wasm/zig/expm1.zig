//! WASM element-wise expm1 (e^x − 1) kernels for float / int types.
//!
//! Unary: out[i] = expm1(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Float/int paths use the shared SIMD core (transcend.expm1v_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow.

const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// Element-wise expm1 for f64 using 2-wide SIMD.
export fn expm1_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.expm1v_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = t.expm1v_f64(v)[0];
    }
}

/// Element-wise expm1 for f32 via the 2-wide f64 core, then narrowed.
export fn expm1_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = t.expm1v_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(t.expm1v_f64(v)[0]);
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn expm1Int_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, t.expm1v_f64(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = t.expm1v_f64(v)[0];
    }
}

inline fn expm1Int_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = t.expm1v_f64(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(t.expm1v_f64(v)[0]);
    }
}

export fn expm1_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    expm1Int_f64(i64, a, out, N);
}
export fn expm1_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    expm1Int_f64(u64, a, out, N);
}
export fn expm1_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    expm1Int_f64(i32, a, out, N);
}
export fn expm1_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    expm1Int_f64(u32, a, out, N);
}
export fn expm1_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    expm1Int_f32(i16, a, out, N);
}
export fn expm1_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    expm1Int_f32(u16, a, out, N);
}
export fn expm1_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    expm1Int_f32(i8, a, out, N);
}
export fn expm1_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    expm1Int_f32(u8, a, out, N);
}

// --- Tests ---

test "expm1_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, -1.0, 0.5 };
    var out: [4]f64 = undefined;
    expm1_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-12);
    try testing.expectApproxEqAbs(out[1], 1.718281828459045, 1e-12);
    try testing.expectApproxEqAbs(out[2], -0.6321205588285577, 1e-12);
    try testing.expectApproxEqAbs(out[3], 0.6487212707001282, 1e-12);
}

test "expm1_f64 near zero accuracy" {
    const testing = @import("std").testing;
    const a = [_]f64{1e-8};
    var out: [1]f64 = undefined;
    expm1_f64(&a, &out, 1);
    try testing.expectApproxEqRel(out[0], 1.0000000050000000e-8, 1e-10);
}

test "expm1_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    expm1_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.7182817, 1e-5);
}

test "expm1_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 2 };
    var out: [2]f64 = undefined;
    expm1_i32_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-12);
    try testing.expectApproxEqAbs(out[1], 6.38905609893065, 1e-12);
}

test "expm1 int variants" {
    const testing = @import("std").testing;
    const expected: f64 = 6.38905609893065; // e^2 - 1
    const ai = [_]i64{ 0, 2 };
    const au = [_]u64{ 0, 2 };
    const au32 = [_]u32{ 0, 2 };
    var o64: [2]f64 = undefined;
    expm1_i64_f64(&ai, &o64, 2);
    try testing.expectApproxEqAbs(o64[1], expected, 1e-12);
    expm1_u64_f64(&au, &o64, 2);
    try testing.expectApproxEqAbs(o64[1], expected, 1e-12);
    expm1_u32_f64(&au32, &o64, 2);
    try testing.expectApproxEqAbs(o64[1], expected, 1e-12);

    const ai16 = [_]i16{ 0, 2 };
    const au16 = [_]u16{ 0, 2 };
    const ai8 = [_]i8{ 0, 2 };
    const au8 = [_]u8{ 0, 2 };
    var o32: [2]f32 = undefined;
    expm1_i16_f32(&ai16, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @as(f32, @floatCast(expected)), 1e-4);
    expm1_u16_f32(&au16, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @as(f32, @floatCast(expected)), 1e-4);
    expm1_i8_f32(&ai8, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @as(f32, @floatCast(expected)), 1e-4);
    expm1_u8_f32(&au8, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @as(f32, @floatCast(expected)), 1e-4);
}
