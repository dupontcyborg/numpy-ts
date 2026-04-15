//! WASM element-wise deg2rad and rad2deg kernels for float types.
//!
//! deg2rad: out[i] = a[i] * (π / 180)
//! rad2deg: out[i] = a[i] * (180 / π)
//! Operates on contiguous 1D buffers of length N.

const std = @import("std");
const simd = @import("simd.zig");

/// Element-wise deg2rad for f64 using 2-wide SIMD: out[i] = a[i] * (π / 180).
export fn deg2rad_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const factor: simd.V2f64 = @splat(std.math.pi / 180.0);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) * factor);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * (std.math.pi / 180.0);
    }
}

/// Element-wise deg2rad for f32 using 4-wide SIMD: out[i] = a[i] * (π / 180).
export fn deg2rad_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const factor: simd.V4f32 = @splat(std.math.pi / 180.0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) * factor);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * (std.math.pi / 180.0);
    }
}

/// Element-wise rad2deg for f64 using 2-wide SIMD: out[i] = a[i] * (180 / π).
export fn rad2deg_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const factor: simd.V2f64 = @splat(180.0 / std.math.pi);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) * factor);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * (180.0 / std.math.pi);
    }
}

/// Element-wise rad2deg for f32 using 4-wide SIMD: out[i] = a[i] * (180 / π).
export fn rad2deg_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const factor: simd.V4f32 = @splat(180.0 / std.math.pi);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) * factor);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] * (180.0 / std.math.pi);
    }
}

// --- Tests ---

test "deg2rad_f64 basic" {
    const testing = std.testing;
    const a = [_]f64{ 0, 90, 180, 270, 360 };
    var out: [5]f64 = undefined;
    deg2rad_f64(&a, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], std.math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], std.math.pi, 1e-10);
    try testing.expectApproxEqAbs(out[3], 3.0 * std.math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 2.0 * std.math.pi, 1e-10);
}

test "deg2rad_f32 basic" {
    const testing = std.testing;
    const a = [_]f32{ 0, 90, 180 };
    var out: [3]f32 = undefined;
    deg2rad_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], std.math.pi / 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], std.math.pi, 1e-5);
}

test "rad2deg_f64 basic" {
    const testing = std.testing;
    const a = [_]f64{ 0, std.math.pi / 2.0, std.math.pi, 2.0 * std.math.pi };
    var out: [4]f64 = undefined;
    rad2deg_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 90.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 180.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 360.0, 1e-10);
}

test "rad2deg_f32 basic" {
    const testing = std.testing;
    const a = [_]f32{ 0, std.math.pi / 2.0, std.math.pi };
    var out: [3]f32 = undefined;
    rad2deg_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-4);
    try testing.expectApproxEqAbs(out[1], 90.0, 1e-4);
    try testing.expectApproxEqAbs(out[2], 180.0, 1e-4);
}

test "deg2rad_f64 SIMD boundary N=1" {
    const testing = std.testing;
    const a = [_]f64{180.0};
    var out: [1]f64 = undefined;
    deg2rad_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], std.math.pi, 1e-10);
}

test "deg2rad_f64 SIMD boundary N=3" {
    const testing = std.testing;
    const a = [_]f64{ 0.0, 90.0, 45.0 };
    var out: [3]f64 = undefined;
    deg2rad_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], std.math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], std.math.pi / 4.0, 1e-10);
}

test "rad2deg_f64 SIMD boundary N=3" {
    const testing = std.testing;
    const a = [_]f64{ 0.0, std.math.pi / 4.0, std.math.pi };
    var out: [3]f64 = undefined;
    rad2deg_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 45.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 180.0, 1e-10);
}

test "deg2rad_f32 SIMD boundary N=7" {
    const testing = std.testing;
    var a: [7]f32 = undefined;
    for (0..7) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 30.0;
    }
    var out: [7]f32 = undefined;
    deg2rad_f32(&a, &out, 7);
    for (0..7) |i| {
        const deg: f32 = @as(f32, @floatFromInt(i)) * 30.0;
        const expected: f32 = deg * (std.math.pi / 180.0);
        try testing.expectApproxEqAbs(out[i], expected, 1e-5);
    }
}

test "roundtrip deg2rad then rad2deg f64" {
    const testing = std.testing;
    const a = [_]f64{ 0, 45, 90, 135, 180, 270, 360 };
    var rad: [7]f64 = undefined;
    var deg: [7]f64 = undefined;
    deg2rad_f64(&a, &rad, 7);
    rad2deg_f64(&rad, &deg, 7);
    for (0..7) |i| {
        try testing.expectApproxEqAbs(deg[i], a[i], 1e-10);
    }
}
