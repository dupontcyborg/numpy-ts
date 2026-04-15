//! WASM argwhere kernels — find flat indices of non-zero elements.
//!
//! argwhere_fill_<T>(a, out, N) → fills `out` with flat indices, returns count.
//!
//! The TS wrapper allocates a worst-case (N u32) output buffer in scratch,
//! then calls fill once and copies out the populated prefix.
//! Scalar loops for all types — the win is avoiding JS overhead.

/// Write flat indices of non-zero f64 elements into `out`. Returns count written.
export fn argwhere_fill_f64(a: [*]const f64, out: [*]u32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) {
            out[count] = i;
            count += 1;
        }
    }
    return count;
}

/// Write flat indices of non-zero f32 elements into `out`. Returns count written.
export fn argwhere_fill_f32(a: [*]const f32, out: [*]u32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) {
            out[count] = i;
            count += 1;
        }
    }
    return count;
}

/// Write flat indices of non-zero i64 elements into `out`. Returns count written.
/// Also handles u64 via reinterpret.
export fn argwhere_fill_i64(a: [*]const i64, out: [*]u32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) {
            out[count] = i;
            count += 1;
        }
    }
    return count;
}

/// Write flat indices of non-zero i32 elements into `out`. Returns count written.
/// Also handles u32 via reinterpret.
export fn argwhere_fill_i32(a: [*]const i32, out: [*]u32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) {
            out[count] = i;
            count += 1;
        }
    }
    return count;
}

/// Write flat indices of non-zero i16 elements into `out`. Returns count written.
/// Also handles u16 via reinterpret.
export fn argwhere_fill_i16(a: [*]const i16, out: [*]u32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) {
            out[count] = i;
            count += 1;
        }
    }
    return count;
}

/// Write flat indices of non-zero i8 elements into `out`. Returns count written.
/// Also handles u8 via reinterpret.
export fn argwhere_fill_i8(a: [*]const i8, out: [*]u32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) {
            out[count] = i;
            count += 1;
        }
    }
    return count;
}

// --- Tests ---

const testing = @import("std").testing;

test "argwhere_fill_f64 basic" {
    const a = [_]f64{ 1.0, 0.0, 3.0, 0.0, 5.0 };
    var out: [5]u32 = undefined;
    const count = argwhere_fill_f64(&a, &out, 5);
    try testing.expectEqual(count, 3);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 4);
}

test "argwhere_fill_f64 all zero" {
    const a = [_]f64{ 0.0, 0.0, 0.0 };
    var out: [3]u32 = undefined;
    const count = argwhere_fill_f64(&a, &out, 3);
    try testing.expectEqual(count, 0);
}

test "argwhere_fill_f64 all nonzero" {
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    var out: [3]u32 = undefined;
    const count = argwhere_fill_f64(&a, &out, 3);
    try testing.expectEqual(count, 3);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 2);
}

test "argwhere_fill_f64 empty" {
    const a = [_]f64{};
    var out: [0]u32 = undefined;
    const count = argwhere_fill_f64(&a, &out, 0);
    try testing.expectEqual(count, 0);
}

test "argwhere_fill_f32 basic" {
    const a = [_]f32{ 0.0, 1.0, 0.0, 2.0 };
    var out: [4]u32 = undefined;
    const count = argwhere_fill_f32(&a, &out, 4);
    try testing.expectEqual(count, 2);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
}

test "argwhere_fill_i64 negatives" {
    const a = [_]i64{ -1, 0, -2, 0 };
    var out: [4]u32 = undefined;
    const count = argwhere_fill_i64(&a, &out, 4);
    try testing.expectEqual(count, 2);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 2);
}

test "argwhere_fill_i32 basic" {
    const a = [_]i32{ 0, 0, 3, 0, 5 };
    var out: [5]u32 = undefined;
    const count = argwhere_fill_i32(&a, &out, 5);
    try testing.expectEqual(count, 2);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 4);
}

test "argwhere_fill_i16 basic" {
    const a = [_]i16{ 0, 100, 0, 200, 0 };
    var out: [5]u32 = undefined;
    const count = argwhere_fill_i16(&a, &out, 5);
    try testing.expectEqual(count, 2);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
}

test "argwhere_fill_i8 basic" {
    const a = [_]i8{ 1, 0, -1, 0 };
    var out: [4]u32 = undefined;
    const count = argwhere_fill_i8(&a, &out, 4);
    try testing.expectEqual(count, 2);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 2);
}
