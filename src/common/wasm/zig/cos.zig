//! WASM element-wise cosine kernels for float / int / complex types.
//!
//! Float/int paths use the shared SIMD Cephes core (transcend.cosv_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow. Complex uses
//! cos(a+bi) = cos a·cosh b − i·sin a·sinh b composed from the shared cores.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// Element-wise cos for f64 using 2-wide SIMD.
export fn cos_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.cosv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = t.cosv_f64(v)[0];
    }
}

/// Element-wise cos for f32 via the 2-wide f64 core, then narrowed.
export fn cos_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = t.cosv_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.cos(@as(f64, a[i])));
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn cosInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, t.cosv_f64(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = t.cosv_f64(v)[0];
    }
}

inline fn cosInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = t.cosv_f64(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(t.cosv_f64(v)[0]);
    }
}

export fn cos_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    cosInt_f64(i64, a, out, N);
}
export fn cos_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    cosInt_f64(u64, a, out, N);
}
export fn cos_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    cosInt_f64(i32, a, out, N);
}
export fn cos_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    cosInt_f64(u32, a, out, N);
}
export fn cos_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    cosInt_f32(i16, a, out, N);
}
export fn cos_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    cosInt_f32(u16, a, out, N);
}
export fn cos_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    cosInt_f32(i8, a, out, N);
}
export fn cos_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    cosInt_f32(u8, a, out, N);
}

// --- Complex: cos(a+bi) = cos a·cosh b − i·sin a·sinh b ---
inline fn opCos(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    var sh: simd.V2f64 = undefined;
    var ch: simd.V2f64 = undefined;
    t.sinhcosh_f64(im, &sh, &ch);
    out_re.* = t.cosv_f64(re) * ch;
    out_im.* = -(t.sinv_f64(re) * sh);
}

export fn cos_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opCos);
}
export fn cos_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opCos);
}

// --- Tests ---

test "cos_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0 };
    var out: [4]f64 = undefined;
    cos_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
}

test "cos_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1 };
    var out: [2]f64 = undefined;
    cos_i32_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqRel(out[1], @cos(1.0), 1e-12);
}

test "cos_c128 matches identity" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f64{ 0.7, 1.2 };
    const b: f64 = 1.2;
    var c: [2]f64 = undefined;
    cos_c128(&a, &c, 1);
    try testing.expectApproxEqRel(c[0], @cos(0.7) * std.math.cosh(b), 1e-12);
    try testing.expectApproxEqRel(c[1], -@sin(0.7) * std.math.sinh(b), 1e-12);
}

test "cos_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 2.0 };
    var out: [3]f32 = undefined;
    cos_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-6);
    try testing.expectApproxEqAbs(out[1], @cos(@as(f32, 1.0)), 1e-6);
    try testing.expectApproxEqAbs(out[2], @cos(@as(f32, 2.0)), 1e-6);
}

test "cos int variants" {
    const testing = @import("std").testing;
    const ai = [_]i64{ 0, 1 };
    const au = [_]u64{ 0, 1 };
    const au32 = [_]u32{ 0, 1 };
    var o64: [2]f64 = undefined;
    cos_i64_f64(&ai, &o64, 2);
    try testing.expectApproxEqRel(o64[1], @cos(1.0), 1e-12);
    cos_u64_f64(&au, &o64, 2);
    try testing.expectApproxEqRel(o64[1], @cos(1.0), 1e-12);
    cos_u32_f64(&au32, &o64, 2);
    try testing.expectApproxEqRel(o64[1], @cos(1.0), 1e-12);

    const ai16 = [_]i16{ 0, 1 };
    const au16 = [_]u16{ 0, 1 };
    const ai8 = [_]i8{ 0, 1 };
    const au8 = [_]u8{ 0, 1 };
    var o32: [2]f32 = undefined;
    cos_i16_f32(&ai16, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @cos(@as(f32, 1.0)), 1e-6);
    cos_u16_f32(&au16, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @cos(@as(f32, 1.0)), 1e-6);
    cos_i8_f32(&ai8, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @cos(@as(f32, 1.0)), 1e-6);
    cos_u8_f32(&au8, &o32, 2);
    try testing.expectApproxEqAbs(o32[1], @cos(@as(f32, 1.0)), 1e-6);
}

test "cos_c64 matches identity" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f32{ 0.7, 1.2 };
    var c: [2]f32 = undefined;
    cos_c64(&a, &c, 1);
    try testing.expectApproxEqAbs(c[0], @cos(@as(f32, 0.7)) * std.math.cosh(@as(f32, 1.2)), 1e-5);
    try testing.expectApproxEqAbs(c[1], -@sin(@as(f32, 0.7)) * std.math.sinh(@as(f32, 1.2)), 1e-5);
}
