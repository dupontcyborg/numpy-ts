//! WASM flat repeat kernels for all numeric types.
//!
//! repeat: Each element a[i] is written `reps` times to output.
//! Output length is N * reps. Operates on contiguous 1D buffers.

const simd = @import("simd.zig");

/// Flat repeat for f64: each element repeated `reps` times.
export fn repeat_f64(a: [*]const f64, out: [*]f64, N: u32, reps: u32) void {
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
export fn repeat_f32(a: [*]const f32, out: [*]f32, N: u32, reps: u32) void {
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

/// Flat repeat for i64, scalar loop (no i64x2 in WASM SIMD).
export fn repeat_i64(a: [*]const i64, out: [*]i64, N: u32, reps: u32) void {
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
export fn repeat_i32(a: [*]const i32, out: [*]i32, N: u32, reps: u32) void {
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
export fn repeat_i16(a: [*]const i16, out: [*]i16, N: u32, reps: u32) void {
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
export fn repeat_i8(a: [*]const i8, out: [*]i8, N: u32, reps: u32) void {
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
