//! WASM flat array roll (circular shift) kernels for all numeric types.
//!
//! roll: out[i] = a[(i - shift + N) % N]  (circular shift by `shift` positions)
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Flat roll for f64 using 2-wide SIMD: circular shift by `shift` positions.
export fn roll_f64(a: [*]const f64, out: [*]f64, N: u32, shift: i32) void {
    if (N == 0) return;
    // Normalize shift to [0, N)
    const s: u32 = @intCast(@mod(@as(i64, shift), @as(i64, N)));
    if (s == 0) {
        // Just copy
        const n_simd = N & ~@as(u32, 1);
        var i: u32 = 0;
        while (i < n_simd) : (i += 2) {
            simd.store2_f64(out, i, simd.load2_f64(a, i));
        }
        while (i < N) : (i += 1) {
            out[i] = a[i];
        }
        return;
    }
    // Copy last `s` elements to beginning of output
    const tail_start = N - s;
    var i: u32 = 0;
    while (i < s) : (i += 1) {
        out[i] = a[tail_start + i];
    }
    // Copy first `N-s` elements after
    i = 0;
    const head_len = N - s;
    const n_simd = head_len & ~@as(u32, 1);
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out + s, i, simd.load2_f64(a, i));
    }
    while (i < head_len) : (i += 1) {
        (out + s)[i] = a[i];
    }
}

/// Flat roll for f32 using 4-wide SIMD: circular shift by `shift` positions.
export fn roll_f32(a: [*]const f32, out: [*]f32, N: u32, shift: i32) void {
    if (N == 0) return;
    const s: u32 = @intCast(@mod(@as(i64, shift), @as(i64, N)));
    if (s == 0) {
        const n_simd = N & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_f32(out, i, simd.load4_f32(a, i));
        }
        while (i < N) : (i += 1) {
            out[i] = a[i];
        }
        return;
    }
    const tail_start = N - s;
    var i: u32 = 0;
    while (i < s) : (i += 1) {
        out[i] = a[tail_start + i];
    }
    i = 0;
    const head_len = N - s;
    const n_simd = head_len & ~@as(u32, 3);
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out + s, i, simd.load4_f32(a, i));
    }
    while (i < head_len) : (i += 1) {
        (out + s)[i] = a[i];
    }
}

/// Flat roll for i64, scalar loop (no i64x2 in WASM SIMD).
export fn roll_i64(a: [*]const i64, out: [*]i64, N: u32, shift: i32) void {
    if (N == 0) return;
    const s: u32 = @intCast(@mod(@as(i64, shift), @as(i64, N)));
    if (s == 0) {
        var i: u32 = 0;
        while (i < N) : (i += 1) {
            out[i] = a[i];
        }
        return;
    }
    const tail_start = N - s;
    var i: u32 = 0;
    while (i < s) : (i += 1) {
        out[i] = a[tail_start + i];
    }
    i = 0;
    while (i < N - s) : (i += 1) {
        (out + s)[i] = a[i];
    }
}

/// Flat roll for i32 using 4-wide SIMD: circular shift by `shift` positions.
export fn roll_i32(a: [*]const i32, out: [*]i32, N: u32, shift: i32) void {
    if (N == 0) return;
    const s: u32 = @intCast(@mod(@as(i64, shift), @as(i64, N)));
    if (s == 0) {
        const n_simd = N & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_i32(out, i, simd.load4_i32(a, i));
        }
        while (i < N) : (i += 1) {
            out[i] = a[i];
        }
        return;
    }
    const tail_start = N - s;
    var i: u32 = 0;
    while (i < s) : (i += 1) {
        out[i] = a[tail_start + i];
    }
    i = 0;
    const head_len = N - s;
    const n_simd = head_len & ~@as(u32, 3);
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out + s, i, simd.load4_i32(a, i));
    }
    while (i < head_len) : (i += 1) {
        (out + s)[i] = a[i];
    }
}

/// Flat roll for i16 using 8-wide SIMD: circular shift by `shift` positions.
export fn roll_i16(a: [*]const i16, out: [*]i16, N: u32, shift: i32) void {
    if (N == 0) return;
    const s: u32 = @intCast(@mod(@as(i64, shift), @as(i64, N)));
    if (s == 0) {
        const n_simd = N & ~@as(u32, 7);
        var i: u32 = 0;
        while (i < n_simd) : (i += 8) {
            simd.store8_i16(out, i, simd.load8_i16(a, i));
        }
        while (i < N) : (i += 1) {
            out[i] = a[i];
        }
        return;
    }
    const tail_start = N - s;
    var i: u32 = 0;
    while (i < s) : (i += 1) {
        out[i] = a[tail_start + i];
    }
    i = 0;
    const head_len = N - s;
    const n_simd = head_len & ~@as(u32, 7);
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out + s, i, simd.load8_i16(a, i));
    }
    while (i < head_len) : (i += 1) {
        (out + s)[i] = a[i];
    }
}

/// Flat roll for i8 using 16-wide SIMD: circular shift by `shift` positions.
export fn roll_i8(a: [*]const i8, out: [*]i8, N: u32, shift: i32) void {
    if (N == 0) return;
    const s: u32 = @intCast(@mod(@as(i64, shift), @as(i64, N)));
    if (s == 0) {
        const n_simd = N & ~@as(u32, 15);
        var i: u32 = 0;
        while (i < n_simd) : (i += 16) {
            simd.store16_i8(out, i, simd.load16_i8(a, i));
        }
        while (i < N) : (i += 1) {
            out[i] = a[i];
        }
        return;
    }
    const tail_start = N - s;
    var i: u32 = 0;
    while (i < s) : (i += 1) {
        out[i] = a[tail_start + i];
    }
    i = 0;
    const head_len = N - s;
    const n_simd = head_len & ~@as(u32, 15);
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out + s, i, simd.load16_i8(a, i));
    }
    while (i < head_len) : (i += 1) {
        (out + s)[i] = a[i];
    }
}

// --- Tests ---

test "roll_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out: [5]f64 = undefined;
    roll_f64(&a, &out, 5, 2);
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
}

test "roll_i32 negative shift" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    var out: [5]i32 = undefined;
    roll_i32(&a, &out, 5, -2);
    try testing.expectEqual(out[0], 3);
    try testing.expectEqual(out[1], 4);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 2);
}

test "roll_i8 zero shift" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5 };
    var out: [5]i8 = undefined;
    roll_i8(&a, &out, 5, 0);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[4], 5);
}
