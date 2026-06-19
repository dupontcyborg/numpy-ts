//! WASM element-wise sinh kernels for float / int types.
//!
//! Unary: out[i] = sinh(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Float/int paths use the shared SIMD core (transcend.sinhcosh_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow.

const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// sinh for a 2-wide f64 lane via the shared sinh/cosh core.
inline fn sinhv(x: simd.V2f64) simd.V2f64 {
    var sh: simd.V2f64 = undefined;
    var ch: simd.V2f64 = undefined;
    t.sinhcosh_f64(x, &sh, &ch);
    return sh;
}

/// Element-wise sinh for f64 using 2-wide SIMD.
export fn sinh_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, sinhv(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = sinhv(v)[0];
    }
}

/// Element-wise sinh for f32 via the 2-wide f64 core, then narrowed.
export fn sinh_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = sinhv(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(sinhv(v)[0]);
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn sinhInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, sinhv(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = sinhv(v)[0];
    }
}

inline fn sinhInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = sinhv(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(sinhv(v)[0]);
    }
}

export fn sinh_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    sinhInt_f64(i64, a, out, N);
}
export fn sinh_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    sinhInt_f64(u64, a, out, N);
}
export fn sinh_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    sinhInt_f64(i32, a, out, N);
}
export fn sinh_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    sinhInt_f64(u32, a, out, N);
}
export fn sinh_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    sinhInt_f32(i16, a, out, N);
}
export fn sinh_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    sinhInt_f32(u16, a, out, N);
}
export fn sinh_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    sinhInt_f32(i8, a, out, N);
}
export fn sinh_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    sinhInt_f32(u8, a, out, N);
}

// --- Tests ---

test "sinh_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    sinh_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.1752011936438014, 1e-10);
}

test "sinh_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    sinh_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.1752, 1e-4);
}

test "sinh_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    sinh_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sinh_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    sinh_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sinh_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    sinh_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sinh_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    sinh_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sinh_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    sinh_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "sinh_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    sinh_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "sinh_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    sinh_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "sinh_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    sinh_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
