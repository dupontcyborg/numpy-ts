//! WASM element-wise tangent kernels for float / int types.
//!
//! Unary: out[i] = tan(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Float/int paths use the shared SIMD sin/cos cores (tan = sin/cos); f32 and
//! integer outputs route through the 2-wide f64 core then narrow.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// tan for a 2-wide f64 lane: sin(x)/cos(x) via the shared cores.
inline fn tanv(x: simd.V2f64) simd.V2f64 {
    return t.sinv_f64(x) / t.cosv_f64(x);
}

/// Element-wise tan for f64 using 2-wide SIMD.
export fn tan_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, tanv(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = tanv(v)[0];
    }
}

/// Element-wise tan for f32 via the 2-wide f64 core, then narrowed.
export fn tan_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = tanv(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(tanv(v)[0]);
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn tanInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, tanv(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = tanv(v)[0];
    }
}

inline fn tanInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = tanv(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(tanv(v)[0]);
    }
}

export fn tan_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    tanInt_f64(i64, a, out, N);
}
export fn tan_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    tanInt_f64(u64, a, out, N);
}
export fn tan_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    tanInt_f64(i32, a, out, N);
}
export fn tan_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    tanInt_f64(u32, a, out, N);
}
export fn tan_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    tanInt_f32(i16, a, out, N);
}
export fn tan_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    tanInt_f32(u16, a, out, N);
}
export fn tan_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    tanInt_f32(i8, a, out, N);
}
export fn tan_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    tanInt_f32(u8, a, out, N);
}

// --- Tests ---

test "tan_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 4.0, -math.pi / 4.0 };
    var out: [3]f64 = undefined;
    tan_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-10);
}

test "tan_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0, math.pi / 4.0, -math.pi / 4.0 };
    var out: [3]f32 = undefined;
    tan_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -1.0, 1e-5);
}

test "tan_f64 at pi" {
    const testing = @import("std").testing;
    const a = [_]f64{math.pi};
    var out: [1]f64 = undefined;
    tan_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    tan_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    tan_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    tan_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    tan_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "tan_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    tan_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "tan_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    tan_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "tan_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    tan_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "tan_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    tan_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
