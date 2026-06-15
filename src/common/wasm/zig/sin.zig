//! WASM element-wise sine kernels for float / int / complex types.
//!
//! Float/int paths use the shared SIMD Cephes core (transcend.sinv_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow. Complex uses
//! sin(a+bi) = sin a·cosh b + i·cos a·sinh b composed from the shared cores.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// Element-wise sin for f64 using 2-wide SIMD.
export fn sin_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.sinv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = t.sinv_f64(v)[0];
    }
}

/// Element-wise sin for f32 via the 2-wide f64 core, then narrowed.
export fn sin_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = t.sinv_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.sin(@as(f64, a[i])));
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn sinInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, t.sinv_f64(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = t.sinv_f64(v)[0];
    }
}

inline fn sinInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = t.sinv_f64(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(t.sinv_f64(v)[0]);
    }
}

export fn sin_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    sinInt_f64(i64, a, out, N);
}
export fn sin_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    sinInt_f64(u64, a, out, N);
}
export fn sin_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    sinInt_f64(i32, a, out, N);
}
export fn sin_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    sinInt_f64(u32, a, out, N);
}
export fn sin_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    sinInt_f32(i16, a, out, N);
}
export fn sin_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    sinInt_f32(u16, a, out, N);
}
export fn sin_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    sinInt_f32(i8, a, out, N);
}
export fn sin_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    sinInt_f32(u8, a, out, N);
}

// --- Complex: sin(a+bi) = sin a·cosh b + i·cos a·sinh b ---
inline fn opSin(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    var sh: simd.V2f64 = undefined;
    var ch: simd.V2f64 = undefined;
    t.sinhcosh_f64(im, &sh, &ch);
    out_re.* = t.sinv_f64(re) * ch;
    out_im.* = t.cosv_f64(re) * sh;
}

export fn sin_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opSin);
}
export fn sin_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opSin);
}

// --- Tests ---

test "sin_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0 };
    var out: [4]f64 = undefined;
    sin_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -1.0, 1e-10);
}

test "sin_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1 };
    var out: [2]f64 = undefined;
    sin_i32_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqRel(out[1], @sin(1.0), 1e-12);
}

test "sin_c128 matches identity" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f64{ 0.7, 1.2 };
    const b: f64 = 1.2;
    var s: [2]f64 = undefined;
    sin_c128(&a, &s, 1);
    try testing.expectApproxEqRel(s[0], @sin(0.7) * std.math.cosh(b), 1e-12);
    try testing.expectApproxEqRel(s[1], @cos(0.7) * std.math.sinh(b), 1e-12);
}
