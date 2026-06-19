//! WASM element-wise arctan kernels for float / int types.
//!
//! Unary: out[i] = arctan(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Float/int paths use the shared SIMD core (transcend.atanv_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow.

const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// Element-wise arctan for f64 using 2-wide SIMD.
export fn arctan_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.atanv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = t.atanv_f64(v)[0];
    }
}

/// Element-wise arctan for f32 via the 2-wide f64 core, then narrowed.
export fn arctan_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = t.atanv_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = @floatCast(t.atanv_f64(v)[0]);
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn arctanInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, t.atanv_f64(xf));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = t.atanv_f64(v)[0];
    }
}

inline fn arctanInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        const r = t.atanv_f64(xf);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        out[i] = @floatCast(t.atanv_f64(v)[0]);
    }
}

export fn arctan_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    arctanInt_f64(i64, a, out, N);
}
export fn arctan_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    arctanInt_f64(u64, a, out, N);
}
export fn arctan_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    arctanInt_f64(i32, a, out, N);
}
export fn arctan_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    arctanInt_f64(u32, a, out, N);
}
export fn arctan_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    arctanInt_f32(i16, a, out, N);
}
export fn arctan_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    arctanInt_f32(u16, a, out, N);
}
export fn arctan_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    arctanInt_f32(i8, a, out, N);
}
export fn arctan_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    arctanInt_f32(u8, a, out, N);
}

// --- Tests ---

test "arctan_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, -1.0 };
    var out: [3]f64 = undefined;
    arctan_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.7853981633974483, 1e-10);
    try testing.expectApproxEqAbs(out[2], -0.7853981633974483, 1e-10);
}

test "arctan_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -1.0 };
    var out: [3]f32 = undefined;
    arctan_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.7854, 1e-4);
    try testing.expectApproxEqAbs(out[2], -0.7854, 1e-4);
}

test "arctan_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    arctan_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    arctan_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    arctan_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    arctan_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    arctan_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    arctan_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    arctan_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    arctan_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
