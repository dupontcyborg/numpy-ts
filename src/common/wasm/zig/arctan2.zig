//! WASM element-wise arctan2 kernels for float / int types.
//!
//! Binary: out[i] = atan2(a[i], b[i])  (a = y, b = x)
//! Operates on contiguous 1D buffers of length N.
//! Float/int paths use the shared SIMD core (transcend.atan2v_f64); f32 and
//! integer outputs route through the 2-wide f64 core then narrow.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

/// Element-wise atan2 for f64 using 2-wide SIMD.
export fn arctan2_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.atan2v_f64(simd.load2_f64(a, i), simd.load2_f64(b, i)));
    }
    while (i < N) : (i += 1) {
        const yv: simd.V2f64 = .{ a[i], a[i] };
        const xv: simd.V2f64 = .{ b[i], b[i] };
        out[i] = t.atan2v_f64(yv, xv)[0];
    }
}

/// Element-wise atan2 for f32 via the 2-wide f64 core, then narrowed.
export fn arctan2_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const yv: simd.V2f64 = .{ a[i], a[i + 1] };
        const xv: simd.V2f64 = .{ b[i], b[i + 1] };
        const r = t.atan2v_f64(yv, xv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const yv: simd.V2f64 = .{ a[i], a[i] };
        const xv: simd.V2f64 = .{ b[i], b[i] };
        out[i] = @floatCast(t.atan2v_f64(yv, xv)[0]);
    }
}

// --- Integer inputs (both f32 and f64 outputs go through the f64 core) ---
inline fn arctan2Int_f64(comptime I: type, a: [*]const I, b: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const yi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xi = @as(*align(1) const @Vector(2, I), @ptrCast(b + i)).*;
        const yv: simd.V2f64 = @floatFromInt(yi);
        const xv: simd.V2f64 = @floatFromInt(xi);
        simd.store2_f64(out, i, t.atan2v_f64(yv, xv));
    }
    while (i < N) : (i += 1) {
        const yv: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        const xv: simd.V2f64 = .{ @floatFromInt(b[i]), @floatFromInt(b[i]) };
        out[i] = t.atan2v_f64(yv, xv)[0];
    }
}

inline fn arctan2Int_f32(comptime I: type, a: [*]const I, b: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const yi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xi = @as(*align(1) const @Vector(2, I), @ptrCast(b + i)).*;
        const yv: simd.V2f64 = @floatFromInt(yi);
        const xv: simd.V2f64 = @floatFromInt(xi);
        const r = t.atan2v_f64(yv, xv);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const yv: simd.V2f64 = .{ @floatFromInt(a[i]), @floatFromInt(a[i]) };
        const xv: simd.V2f64 = .{ @floatFromInt(b[i]), @floatFromInt(b[i]) };
        out[i] = @floatCast(t.atan2v_f64(yv, xv)[0]);
    }
}

export fn arctan2_i64_f64(a: [*]const i64, b: [*]const i64, out: [*]f64, N: u32) void {
    arctan2Int_f64(i64, a, b, out, N);
}
export fn arctan2_u64_f64(a: [*]const u64, b: [*]const u64, out: [*]f64, N: u32) void {
    arctan2Int_f64(u64, a, b, out, N);
}
export fn arctan2_i32_f64(a: [*]const i32, b: [*]const i32, out: [*]f64, N: u32) void {
    arctan2Int_f64(i32, a, b, out, N);
}
export fn arctan2_u32_f64(a: [*]const u32, b: [*]const u32, out: [*]f64, N: u32) void {
    arctan2Int_f64(u32, a, b, out, N);
}
export fn arctan2_i16_f32(a: [*]const i16, b: [*]const i16, out: [*]f32, N: u32) void {
    arctan2Int_f32(i16, a, b, out, N);
}
export fn arctan2_u16_f32(a: [*]const u16, b: [*]const u16, out: [*]f32, N: u32) void {
    arctan2Int_f32(u16, a, b, out, N);
}
export fn arctan2_i8_f32(a: [*]const i8, b: [*]const i8, out: [*]f32, N: u32) void {
    arctan2Int_f32(i8, a, b, out, N);
}
export fn arctan2_u8_f32(a: [*]const u8, b: [*]const u8, out: [*]f32, N: u32) void {
    arctan2Int_f32(u8, a, b, out, N);
}

// --- Tests ---

test "arctan2_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.0, -1.0, 0.0 };
    const b = [_]f64{ 0.0, 1.0, 0.0, -1.0 };
    var out: [4]f64 = undefined;
    arctan2_f64(&a, &b, &out, 4);
    try testing.expectApproxEqAbs(out[0], math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], math.pi, 1e-10);
}

test "arctan2_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 0.0, -1.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0 };
    var out: [3]f32 = undefined;
    arctan2_f32(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], math.pi / 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -math.pi / 2.0, 1e-5);
}

test "arctan2_f64 negative values" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.0, 1.0 };
    const b = [_]f64{ -1.0, 1.0 };
    var out: [2]f64 = undefined;
    arctan2_f64(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], -3.0 * math.pi / 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], math.pi / 4.0, 1e-10);
}

test "arctan2_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    const b = [_]i64{1};
    var out: [1]f64 = undefined;
    arctan2_i64_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    const b = [_]u64{1};
    var out: [1]f64 = undefined;
    arctan2_u64_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    const b = [_]i32{1};
    var out: [1]f64 = undefined;
    arctan2_i32_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    const b = [_]u32{1};
    var out: [1]f64 = undefined;
    arctan2_u32_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    const b = [_]i16{1};
    var out: [1]f32 = undefined;
    arctan2_i16_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan2_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    const b = [_]u16{1};
    var out: [1]f32 = undefined;
    arctan2_u16_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan2_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    const b = [_]i8{1};
    var out: [1]f32 = undefined;
    arctan2_i8_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan2_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    const b = [_]u8{1};
    var out: [1]f32 = undefined;
    arctan2_u8_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
