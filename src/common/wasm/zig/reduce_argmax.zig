//! WASM reduction argmax kernels for all numeric types.
//!
//! Reduction: result = index of max(a[0..N])
//! Returns u32 index. Unsigned variants needed — comparison is sign-dependent.
//! i8/u8 use two-pass SIMD: find max value (16-wide), then find first index.

const simd = @import("simd.zig");

/// Returns the index of the maximum f64 element. Returns 0 if N=0.
/// Note: NaN is considered less than any number, so reduce_argmax_f64([NaN], 1) returns 0.
export fn reduce_argmax_f64(a: [*]const f64, N: u32) u32 {
    if (N == 0) return 0;
    var best: f64 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum f32 element. Returns 0 if N=0.
/// Note: NaN is considered less than any number, so reduce_argmax_f32([NaN], 1) returns 0.
export fn reduce_argmax_f32(a: [*]const f32, N: u32) u32 {
    if (N == 0) return 0;
    var best: f32 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i64 element. Returns 0 if N=0.
export fn reduce_argmax_i64(a: [*]const i64, N: u32) u32 {
    if (N == 0) return 0;
    var best: i64 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u64 element. Returns 0 if N=0.
export fn reduce_argmax_u64(a: [*]const u64, N: u32) u32 {
    if (N == 0) return 0;
    var best: u64 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i32 element. Returns 0 if N=0.
export fn reduce_argmax_i32(a: [*]const i32, N: u32) u32 {
    if (N == 0) return 0;
    var best: i32 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u32 element. Returns 0 if N=0.
export fn reduce_argmax_u32(a: [*]const u32, N: u32) u32 {
    if (N == 0) return 0;
    var best: u32 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i16 element. Returns 0 if N=0.
export fn reduce_argmax_i16(a: [*]const i16, N: u32) u32 {
    if (N == 0) return 0;
    var best: i16 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum u16 element. Returns 0 if N=0.
export fn reduce_argmax_u16(a: [*]const u16, N: u32) u32 {
    if (N == 0) return 0;
    var best: u16 = a[0];
    var idx: u32 = 0;
    var i: u32 = 1;
    while (i < N) : (i += 1) {
        if (a[i] > best) {
            best = a[i];
            idx = i;
        }
    }
    return idx;
}

/// Returns the index of the maximum i8 element. Two-pass SIMD.
export fn reduce_argmax_i8(a: [*]const i8, N: u32) u32 {
    if (N == 0) return 0;
    const n16 = N & ~@as(u32, 15);
    var max_val: i8 = a[0];
    if (n16 >= 16) {
        var acc = simd.load16_i8(a, 0);
        var i: u32 = 16;
        while (i < n16) : (i += 16) acc = @max(acc, simd.load16_i8(a, i));
        max_val = acc[0];
        inline for (1..16) |lane| {
            if (acc[lane] > max_val) max_val = acc[lane];
        }
    }
    {
        var i: u32 = n16;
        while (i < N) : (i += 1) {
            if (a[i] > max_val) max_val = a[i];
        }
    }
    const target: simd.V16i8 = @splat(max_val);
    {
        var i: u32 = 0;
        while (i < n16) : (i += 16) {
            const eq = simd.load16_i8(a, i) == target;
            inline for (0..16) |lane| {
                if (eq[lane]) return i + @as(u32, lane);
            }
        }
        var j: u32 = n16;
        while (j < N) : (j += 1) {
            if (a[j] == max_val) return j;
        }
    }
    return 0;
}

/// Returns the index of the maximum u8 element. Two-pass SIMD.
export fn reduce_argmax_u8(a: [*]const u8, N: u32) u32 {
    if (N == 0) return 0;
    const n16 = N & ~@as(u32, 15);
    var max_val: u8 = a[0];
    if (n16 >= 16) {
        var acc = simd.load16_u8(a, 0);
        var i: u32 = 16;
        while (i < n16) : (i += 16) acc = @max(acc, simd.load16_u8(a, i));
        max_val = acc[0];
        inline for (1..16) |lane| {
            if (acc[lane] > max_val) max_val = acc[lane];
        }
    }
    {
        var i: u32 = n16;
        while (i < N) : (i += 1) {
            if (a[i] > max_val) max_val = a[i];
        }
    }
    const target: simd.V16u8 = @splat(max_val);
    {
        var i: u32 = 0;
        while (i < n16) : (i += 16) {
            const eq = simd.load16_u8(a, i) == target;
            inline for (0..16) |lane| {
                if (eq[lane]) return i + @as(u32, lane);
            }
        }
        var j: u32 = n16;
        while (j < N) : (j += 1) {
            if (a[j] == max_val) return j;
        }
    }
    return 0;
}

// --- Strided axis reduction (output is always i32 indices) ---

/// Returns the index of the maximum f64 element along the axis, strided.
export fn reduce_argmax_strided_f64(a: [*]const f64, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum f32 element along the axis, strided.
export fn reduce_argmax_strided_f32(a: [*]const f32, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum i32 element along the axis, strided.
export fn reduce_argmax_strided_i32(a: [*]const i32, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum u32 element along the axis, strided.
export fn reduce_argmax_strided_u32(a: [*]const u32, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum i16 element along the axis, strided.
export fn reduce_argmax_strided_i16(a: [*]const i16, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum u16 element along the axis, strided.
export fn reduce_argmax_strided_u16(a: [*]const u16, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum i8 element along the axis, strided.
export fn reduce_argmax_strided_i8(a: [*]const i8, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

/// Returns the index of the maximum u8 element along the axis, strided.
export fn reduce_argmax_strided_u8(a: [*]const u8, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| {
            var best = a[base + i];
            var bestIdx: i32 = 0;
            for (1..A) |ax| {
                const val = a[base + ax * I + i];
                if (val > best) {
                    best = val;
                    bestIdx = @intCast(ax);
                }
            }
            out[ob + i] = bestIdx;
        }
    }
}

// --- Tests ---

test "reduce_argmax_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0 };
    try testing.expectEqual(reduce_argmax_f64(&a, 5), 4);
}

test "reduce_argmax_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 3, 200, 4, 1, 5 };
    try testing.expectEqual(reduce_argmax_u8(&a, 5), 1);
}

test "reduce_argmax_f64 max at first and single" {
    const testing = @import("std").testing;
    const a = [_]f64{ 10.0, 1.0, 2.0 };
    try testing.expectEqual(reduce_argmax_f64(&a, 3), 0);
    const b = [_]f64{42.0};
    try testing.expectEqual(reduce_argmax_f64(&b, 1), 0);
}

test "reduce_argmax_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 5.0, 3.0 };
    try testing.expectEqual(reduce_argmax_f32(&a, 3), 1);
}

test "reduce_argmax_i64 negatives" {
    const testing = @import("std").testing;
    const a = [_]i64{ -5, -1, -3 };
    try testing.expectEqual(reduce_argmax_i64(&a, 3), 1);
}

test "reduce_argmax_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 7, 2, 9, 4 };
    try testing.expectEqual(reduce_argmax_i32(&a, 4), 2);
}

test "reduce_argmax_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ -10, 100, 50 };
    try testing.expectEqual(reduce_argmax_i16(&a, 3), 1);
}

test "reduce_argmax_i8 negatives" {
    const testing = @import("std").testing;
    const a = [_]i8{ -128, -1, -64 };
    try testing.expectEqual(reduce_argmax_i8(&a, 3), 1);
}

test "reduce_argmax_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 5, 100, 3 };
    try testing.expectEqual(reduce_argmax_u64(&a, 3), 1);
}

test "reduce_argmax_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 1, 2, 0xFFFFFFFF };
    try testing.expectEqual(reduce_argmax_u32(&a, 3), 2);
}

test "reduce_argmax_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0xFFFF, 0, 1 };
    try testing.expectEqual(reduce_argmax_u16(&a, 3), 0);
}

test "reduce_argmax ties return first occurrence" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, 5, 5 };
    try testing.expectEqual(reduce_argmax_i32(&a, 3), 0);
}

test "reduce_argmax_strided_f64 basic" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1.0, 5.0, 3.0 };
    var out = [_]i32{0};
    reduce_argmax_strided_f64(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_f32 basic" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1.0, 5.0, 3.0 };
    var out = [_]i32{0};
    reduce_argmax_strided_f32(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_i32 basic" {
    const testing = @import("std").testing;
    var a = [_]i32{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_argmax_strided_i32(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_u32 basic" {
    const testing = @import("std").testing;
    var a = [_]u32{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_argmax_strided_u32(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_i16 basic" {
    const testing = @import("std").testing;
    var a = [_]i16{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_argmax_strided_i16(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_u16 basic" {
    const testing = @import("std").testing;
    var a = [_]u16{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_argmax_strided_u16(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_i8 basic" {
    const testing = @import("std").testing;
    var a = [_]i8{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_argmax_strided_i8(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}

test "reduce_argmax_strided_u8 basic" {
    const testing = @import("std").testing;
    var a = [_]u8{ 1, 5, 3 };
    var out = [_]i32{0};
    reduce_argmax_strided_u8(&a, &out, 1, 3, 1);
    try testing.expectEqual(out[0], 1);
}
