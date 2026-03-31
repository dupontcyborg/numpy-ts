//! WASM flat repeat kernels for all numeric types.
//!
//! repeat: Each element a[i] is written `reps` times to output.
//! Output length is N * reps. Operates on contiguous 1D buffers.

const simd = @import("simd.zig");

/// Flat repeat for f64: each element repeated `reps` times.
/// For reps=2, splats each f64 into a V2f64 SIMD store (one 128-bit write per element).
export fn repeat_f64(a: [*]const f64, out: [*]f64, N: u32, reps: u32) void {
    const n = @as(usize, N);
    if (reps == 2) {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const v: simd.V2f64 = @splat(a[i]);
            simd.store2_f64(out, i * 2, v);
        }
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const val = a[i];
        const base = i * reps;
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            out[base + r] = val;
        }
    }
}

/// Flat repeat for f32: each element repeated `reps` times.
/// For reps=2, packs two f32 copies into a single i64 store.
export fn repeat_f32(a: [*]const f32, out: [*]f32, N: u32, reps: u32) void {
    const n = @as(usize, N);
    if (reps == 2) {
        const out64: [*]u64 = @alignCast(@ptrCast(out));
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const bits: u32 = @bitCast(a[i]);
            out64[i] = @as(u64, bits) | (@as(u64, bits) << 32);
        }
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const val = a[i];
        const base = i * reps;
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            out[base + r] = val;
        }
    }
}

/// Flat repeat for i64: each element repeated `reps` times.
/// For reps=2, splats each i64 into a V2i64 SIMD store.
export fn repeat_i64(a: [*]const i64, out: [*]i64, N: u32, reps: u32) void {
    const n = @as(usize, N);
    if (reps == 2) {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const v: @Vector(2, i64) = @splat(a[i]);
            @as(*align(1) @Vector(2, i64), @ptrCast(out + i * 2)).* = v;
        }
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const val = a[i];
        const base = i * reps;
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            out[base + r] = val;
        }
    }
}

/// Flat repeat for i32: each element repeated `reps` times.
/// For reps=2, packs two i32 copies into a single i64 store.
export fn repeat_i32(a: [*]const i32, out: [*]i32, N: u32, reps: u32) void {
    const n = @as(usize, N);
    if (reps == 2) {
        const out64: [*]u64 = @alignCast(@ptrCast(out));
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const bits: u32 = @bitCast(a[i]);
            out64[i] = @as(u64, bits) | (@as(u64, bits) << 32);
        }
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const val = a[i];
        const base = i * reps;
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            out[base + r] = val;
        }
    }
}

/// Flat repeat for i16: each element repeated `reps` times.
/// For reps=2, packs two i16 copies into a single i32 store.
export fn repeat_i16(a: [*]const i16, out: [*]i16, N: u32, reps: u32) void {
    const n = @as(usize, N);
    if (reps == 2) {
        const out32: [*]u32 = @alignCast(@ptrCast(out));
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const v: u16 = @bitCast(a[i]);
            out32[i] = @as(u32, v) | (@as(u32, v) << 16);
        }
        return;
    }
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const val = a[i];
        const base = i * reps;
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            out[base + r] = val;
        }
    }
}

/// Flat repeat for i8: each element repeated `reps` times.
/// For reps=2, packs two copies into a single i16 store (2x fewer memory ops).
export fn repeat_i8(a: [*]const i8, out: [*]i8, N: u32, reps: u32) void {
    const n = @as(usize, N);
    const r = @as(usize, reps);

    // Fast path: reps=2 — write pairs as i16
    if (reps == 2) {
        const out16: [*]u16 = @alignCast(@ptrCast(out));
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const v: u8 = @bitCast(a[i]);
            out16[i] = @as(u16, v) | (@as(u16, v) << 8);
        }
        return;
    }

    // General path
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const val = a[i];
        const base = i * r;
        var ri: usize = 0;
        while (ri < r) : (ri += 1) {
            out[base + ri] = val;
        }
    }
}

// --- Tests ---

test "repeat_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    var out: [6]f64 = undefined;
    repeat_f64(&a, &out, 3, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 3.0, 1e-10);
}

test "repeat_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 5, -3, 7 };
    var out: [9]i8 = undefined;
    repeat_i8(&a, &out, 3, 3);
    try testing.expectEqual(out[0], 5);
    try testing.expectEqual(out[1], 5);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], -3);
    try testing.expectEqual(out[4], -3);
    try testing.expectEqual(out[5], -3);
    try testing.expectEqual(out[6], 7);
    try testing.expectEqual(out[7], 7);
    try testing.expectEqual(out[8], 7);
}

test "repeat_f64 single rep" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    var out: [3]f64 = undefined;
    repeat_f64(&a, &out, 3, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
}

test "repeat_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 10.0, 20.0 };
    var out: [6]f32 = undefined;
    repeat_f32(&a, &out, 2, 3);
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 10.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 20.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 20.0, 1e-5);
    try testing.expectApproxEqAbs(out[5], 20.0, 1e-5);
}

test "repeat_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, -2, 3 };
    var out: [6]i32 = undefined;
    repeat_i32(&a, &out, 3, 2);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], -2);
    try testing.expectEqual(out[3], -2);
    try testing.expectEqual(out[4], 3);
    try testing.expectEqual(out[5], 3);
}

test "repeat_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 100, 200 };
    var out: [8]i16 = undefined;
    repeat_i16(&a, &out, 2, 4);
    for (0..4) |i| {
        try testing.expectEqual(out[i], 100);
    }
    for (4..8) |i| {
        try testing.expectEqual(out[i], 200);
    }
}

test "repeat_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{42};
    var out: [3]i64 = undefined;
    repeat_i64(&a, &out, 1, 3);
    try testing.expectEqual(out[0], 42);
    try testing.expectEqual(out[1], 42);
    try testing.expectEqual(out[2], 42);
}
