//! WASM element-wise signbit kernels with SIMD acceleration.
//!
//! Unary: out[i] = 1 if a[i] has negative sign bit, 0 otherwise.
//! Returns a uint8 boolean array.
//! Uses SIMD compare + narrow for high throughput.

const math = @import("std").math;
const simd = @import("simd.zig");

/// signbit for f64 — 2-wide SIMD (check sign bit via < -0.0 comparison handles -0.0 correctly).
export fn signbit_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    const zero: simd.V2f64 = @splat(0.0);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        // Use signbit on each lane (handles -0.0 and NaN correctly)
        out[i] = if (math.signbit(v[0])) 1 else 0;
        out[i + 1] = if (math.signbit(v[1])) 1 else 0;
        _ = zero;
    }
    while (i < N) : (i += 1) {
        out[i] = if (math.signbit(a[i])) 1 else 0;
    }
}

/// signbit for f32 — 4-wide SIMD.
export fn signbit_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        out[i] = if (math.signbit(v[0])) 1 else 0;
        out[i + 1] = if (math.signbit(v[1])) 1 else 0;
        out[i + 2] = if (math.signbit(v[2])) 1 else 0;
        out[i + 3] = if (math.signbit(v[3])) 1 else 0;
    }
    while (i < N) : (i += 1) {
        out[i] = if (math.signbit(a[i])) 1 else 0;
    }
}

/// Float16 signbit via bit extraction: (u16 >> 15).
export fn signbit_f16(a: [*]const u16, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @intCast(a[i] >> 15);
}

/// signbit for i64 — 2-wide SIMD compare < 0.
export fn signbit_i64(a: [*]const i64, out: [*]u8, N: u32) void {
    const zero: simd.V2i64 = @splat(0);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = @as(*align(1) const simd.V2i64, @ptrCast(a + i)).*;
        const mask = v < zero;
        out[i] = if (mask[0]) 1 else 0;
        out[i + 1] = if (mask[1]) 1 else 0;
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < 0) 1 else 0;
    }
}

/// signbit for i32 — 4-wide SIMD compare < 0.
export fn signbit_i32(a: [*]const i32, out: [*]u8, N: u32) void {
    const zero: simd.V4i32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = @as(*align(1) const simd.V4i32, @ptrCast(a + i)).*;
        const mask = v < zero;
        out[i] = if (mask[0]) 1 else 0;
        out[i + 1] = if (mask[1]) 1 else 0;
        out[i + 2] = if (mask[2]) 1 else 0;
        out[i + 3] = if (mask[3]) 1 else 0;
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < 0) 1 else 0;
    }
}

/// signbit for i16 — 8-wide SIMD compare < 0.
export fn signbit_i16(a: [*]const i16, out: [*]u8, N: u32) void {
    const zero: simd.V8i16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = @as(*align(1) const simd.V8i16, @ptrCast(a + i)).*;
        const mask = v < zero;
        inline for (0..8) |lane| {
            out[i + lane] = if (mask[lane]) 1 else 0;
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < 0) 1 else 0;
    }
}

/// signbit for i8 — 16-wide SIMD compare < 0 (processes 16 elements per SIMD op).
export fn signbit_i8(a: [*]const i8, out: [*]u8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const v = @as(*align(1) const simd.V16i8, @ptrCast(a + i)).*;
        const mask = v < zero;
        inline for (0..16) |lane| {
            out[i + lane] = if (mask[lane]) 1 else 0;
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < 0) 1 else 0;
    }
}

// --- Tests ---

test "signbit_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, -1.0, 0.0, -0.0, math.inf(f64), -math.inf(f64), math.nan(f64) };
    var out: [7]u8 = undefined;
    signbit_f64(&a, &out, 7);
    try testing.expectEqual(out[0], 0); // 1.0
    try testing.expectEqual(out[1], 1); // -1.0
    try testing.expectEqual(out[2], 0); // 0.0
    try testing.expectEqual(out[3], 1); // -0.0
    try testing.expectEqual(out[4], 0); // inf
    try testing.expectEqual(out[5], 1); // -inf
}

test "signbit_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, -1.0, 0.0, -0.0 };
    var out: [4]u8 = undefined;
    signbit_f32(&a, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "signbit_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, -1, 0, -100 };
    var out: [4]u8 = undefined;
    signbit_i64(&a, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
}

test "signbit_i8 16-wide" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, -1, 0, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9 };
    var out: [18]u8 = undefined;
    signbit_i8(&a, &out, 18);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[16], 0);
    try testing.expectEqual(out[17], 1);
}

test "signbit_f16 basic" {
    const testing = @import("std").testing;
    // 1.0=0x3C00, -1.0=0xBC00, 0.0=0x0000, -0.0=0x8000
    const a = [_]u16{ 0x3C00, 0xBC00, 0x0000, 0x8000 };
    var out: [4]u8 = undefined;
    signbit_f16(&a, &out, 4);
    try testing.expectEqual(out[0], 0); // 1.0
    try testing.expectEqual(out[1], 1); // -1.0
    try testing.expectEqual(out[2], 0); // 0.0
    try testing.expectEqual(out[3], 1); // -0.0
}

test "signbit_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ -1, 0, 1 };
    var out: [3]u8 = undefined;
    signbit_i32(&a, &out, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
}

test "signbit_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ -1, 0, 1 };
    var out: [3]u8 = undefined;
    signbit_i16(&a, &out, 3);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
}
