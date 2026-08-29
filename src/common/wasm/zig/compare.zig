//! WASM element-wise comparison kernels: eq, ne, lt, le, gt, ge.
//! Two same-dtype arrays in, one u8 (bool) array out.
//!
//! Lane width drives the win, so the narrow dtypes gain most: i8/u8 compare 16
//! elements per instruction while f64 manages 2. The JS fallback is penalised
//! in the same direction (Int8Array reads sign-extend per element), so the gap
//! over JS widens further for the narrow types than for f64.
//!
//! `@intFromBool` on a vector yields one byte per lane, which is exactly the
//! bool layout, so no extra packing step is needed.

const Op = enum { eq, ne, lt, le, gt, ge };

inline fn apply(comptime op: Op, va: anytype, vb: @TypeOf(va)) @TypeOf(va == vb) {
    return switch (op) {
        .eq => va == vb,
        .ne => va != vb,
        .lt => va < vb,
        .le => va <= vb,
        .gt => va > vb,
        .ge => va >= vb,
    };
}

/// Vectorised compare over `L` lanes of `T`, writing one byte per element.
inline fn cmp(comptime T: type, comptime L: comptime_int, comptime op: Op, a: [*]const T, b: [*]const T, out: [*]u8, N: u32) void {
    const V = @Vector(L, T);
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const va = @as(*align(1) const V, @ptrCast(a + i)).*;
        const vb = @as(*align(1) const V, @ptrCast(b + i)).*;
        const m: @Vector(L, u8) = @intFromBool(apply(op, va, vb));
        @as(*align(1) [L]u8, @ptrCast(out + i)).* = m;
    }
    while (i < N) : (i += 1) {
        const r = switch (op) {
            .eq => a[i] == b[i],
            .ne => a[i] != b[i],
            .lt => a[i] < b[i],
            .le => a[i] <= b[i],
            .gt => a[i] > b[i],
            .ge => a[i] >= b[i],
        };
        out[i] = if (r) 1 else 0;
    }
}

// --- Exports ---

export fn eq_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    cmp(f64, 2, .eq, a, b, out, N);
}
export fn eq_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    cmp(f32, 4, .eq, a, b, out, N);
}
export fn eq_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    cmp(i64, 2, .eq, a, b, out, N);
}
export fn eq_u64(a: [*]const u64, b: [*]const u64, out: [*]u8, N: u32) void {
    cmp(u64, 2, .eq, a, b, out, N);
}
export fn eq_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    cmp(i32, 4, .eq, a, b, out, N);
}
export fn eq_u32(a: [*]const u32, b: [*]const u32, out: [*]u8, N: u32) void {
    cmp(u32, 4, .eq, a, b, out, N);
}
export fn eq_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    cmp(i16, 8, .eq, a, b, out, N);
}
export fn eq_u16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    cmp(u16, 8, .eq, a, b, out, N);
}
export fn eq_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    cmp(i8, 16, .eq, a, b, out, N);
}
export fn eq_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    cmp(u8, 16, .eq, a, b, out, N);
}
export fn ne_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    cmp(f64, 2, .ne, a, b, out, N);
}
export fn ne_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    cmp(f32, 4, .ne, a, b, out, N);
}
export fn ne_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    cmp(i64, 2, .ne, a, b, out, N);
}
export fn ne_u64(a: [*]const u64, b: [*]const u64, out: [*]u8, N: u32) void {
    cmp(u64, 2, .ne, a, b, out, N);
}
export fn ne_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    cmp(i32, 4, .ne, a, b, out, N);
}
export fn ne_u32(a: [*]const u32, b: [*]const u32, out: [*]u8, N: u32) void {
    cmp(u32, 4, .ne, a, b, out, N);
}
export fn ne_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    cmp(i16, 8, .ne, a, b, out, N);
}
export fn ne_u16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    cmp(u16, 8, .ne, a, b, out, N);
}
export fn ne_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    cmp(i8, 16, .ne, a, b, out, N);
}
export fn ne_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    cmp(u8, 16, .ne, a, b, out, N);
}
export fn lt_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    cmp(f64, 2, .lt, a, b, out, N);
}
export fn lt_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    cmp(f32, 4, .lt, a, b, out, N);
}
export fn lt_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    cmp(i64, 2, .lt, a, b, out, N);
}
export fn lt_u64(a: [*]const u64, b: [*]const u64, out: [*]u8, N: u32) void {
    cmp(u64, 2, .lt, a, b, out, N);
}
export fn lt_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    cmp(i32, 4, .lt, a, b, out, N);
}
export fn lt_u32(a: [*]const u32, b: [*]const u32, out: [*]u8, N: u32) void {
    cmp(u32, 4, .lt, a, b, out, N);
}
export fn lt_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    cmp(i16, 8, .lt, a, b, out, N);
}
export fn lt_u16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    cmp(u16, 8, .lt, a, b, out, N);
}
export fn lt_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    cmp(i8, 16, .lt, a, b, out, N);
}
export fn lt_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    cmp(u8, 16, .lt, a, b, out, N);
}
export fn le_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    cmp(f64, 2, .le, a, b, out, N);
}
export fn le_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    cmp(f32, 4, .le, a, b, out, N);
}
export fn le_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    cmp(i64, 2, .le, a, b, out, N);
}
export fn le_u64(a: [*]const u64, b: [*]const u64, out: [*]u8, N: u32) void {
    cmp(u64, 2, .le, a, b, out, N);
}
export fn le_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    cmp(i32, 4, .le, a, b, out, N);
}
export fn le_u32(a: [*]const u32, b: [*]const u32, out: [*]u8, N: u32) void {
    cmp(u32, 4, .le, a, b, out, N);
}
export fn le_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    cmp(i16, 8, .le, a, b, out, N);
}
export fn le_u16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    cmp(u16, 8, .le, a, b, out, N);
}
export fn le_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    cmp(i8, 16, .le, a, b, out, N);
}
export fn le_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    cmp(u8, 16, .le, a, b, out, N);
}
export fn gt_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    cmp(f64, 2, .gt, a, b, out, N);
}
export fn gt_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    cmp(f32, 4, .gt, a, b, out, N);
}
export fn gt_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    cmp(i64, 2, .gt, a, b, out, N);
}
export fn gt_u64(a: [*]const u64, b: [*]const u64, out: [*]u8, N: u32) void {
    cmp(u64, 2, .gt, a, b, out, N);
}
export fn gt_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    cmp(i32, 4, .gt, a, b, out, N);
}
export fn gt_u32(a: [*]const u32, b: [*]const u32, out: [*]u8, N: u32) void {
    cmp(u32, 4, .gt, a, b, out, N);
}
export fn gt_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    cmp(i16, 8, .gt, a, b, out, N);
}
export fn gt_u16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    cmp(u16, 8, .gt, a, b, out, N);
}
export fn gt_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    cmp(i8, 16, .gt, a, b, out, N);
}
export fn gt_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    cmp(u8, 16, .gt, a, b, out, N);
}
export fn ge_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    cmp(f64, 2, .ge, a, b, out, N);
}
export fn ge_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    cmp(f32, 4, .ge, a, b, out, N);
}
export fn ge_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    cmp(i64, 2, .ge, a, b, out, N);
}
export fn ge_u64(a: [*]const u64, b: [*]const u64, out: [*]u8, N: u32) void {
    cmp(u64, 2, .ge, a, b, out, N);
}
export fn ge_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    cmp(i32, 4, .ge, a, b, out, N);
}
export fn ge_u32(a: [*]const u32, b: [*]const u32, out: [*]u8, N: u32) void {
    cmp(u32, 4, .ge, a, b, out, N);
}
export fn ge_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    cmp(i16, 8, .ge, a, b, out, N);
}
export fn ge_u16(a: [*]const u16, b: [*]const u16, out: [*]u8, N: u32) void {
    cmp(u16, 8, .ge, a, b, out, N);
}
export fn ge_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    cmp(i8, 16, .ge, a, b, out, N);
}
export fn ge_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    cmp(u8, 16, .ge, a, b, out, N);
}

// --- Tests ---

/// Compares a 19-element pair whose values cycle a > b, a < b, a == b. The
/// length leaves a scalar tail at every lane width, so both loops are exercised
/// for each dtype. `expected` gives the three output bytes for one cycle.
fn expectCmp(comptime T: type, f: anytype, expected: [3]u8) !void {
    const testing = @import("std").testing;
    const N = 19;
    var a: [N]T = undefined;
    var b: [N]T = undefined;
    var out = [_]u8{0xAA} ** N;
    for (0..N) |i| {
        a[i] = switch (i % 3) {
            0 => 3,
            1 => 1,
            else => 2,
        };
        b[i] = switch (i % 3) {
            0 => 1,
            1 => 3,
            else => 2,
        };
    }
    f(&a, &b, &out, N);
    for (0..N) |i| try testing.expectEqual(expected[i % 3], out[i]);
}

test "eq over every dtype" {
    try expectCmp(f64, eq_f64, .{ 0, 0, 1 });
    try expectCmp(f32, eq_f32, .{ 0, 0, 1 });
    try expectCmp(i64, eq_i64, .{ 0, 0, 1 });
    try expectCmp(u64, eq_u64, .{ 0, 0, 1 });
    try expectCmp(i32, eq_i32, .{ 0, 0, 1 });
    try expectCmp(u32, eq_u32, .{ 0, 0, 1 });
    try expectCmp(i16, eq_i16, .{ 0, 0, 1 });
    try expectCmp(u16, eq_u16, .{ 0, 0, 1 });
    try expectCmp(i8, eq_i8, .{ 0, 0, 1 });
    try expectCmp(u8, eq_u8, .{ 0, 0, 1 });
}

test "ne over every dtype" {
    try expectCmp(f64, ne_f64, .{ 1, 1, 0 });
    try expectCmp(f32, ne_f32, .{ 1, 1, 0 });
    try expectCmp(i64, ne_i64, .{ 1, 1, 0 });
    try expectCmp(u64, ne_u64, .{ 1, 1, 0 });
    try expectCmp(i32, ne_i32, .{ 1, 1, 0 });
    try expectCmp(u32, ne_u32, .{ 1, 1, 0 });
    try expectCmp(i16, ne_i16, .{ 1, 1, 0 });
    try expectCmp(u16, ne_u16, .{ 1, 1, 0 });
    try expectCmp(i8, ne_i8, .{ 1, 1, 0 });
    try expectCmp(u8, ne_u8, .{ 1, 1, 0 });
}

test "lt over every dtype" {
    try expectCmp(f64, lt_f64, .{ 0, 1, 0 });
    try expectCmp(f32, lt_f32, .{ 0, 1, 0 });
    try expectCmp(i64, lt_i64, .{ 0, 1, 0 });
    try expectCmp(u64, lt_u64, .{ 0, 1, 0 });
    try expectCmp(i32, lt_i32, .{ 0, 1, 0 });
    try expectCmp(u32, lt_u32, .{ 0, 1, 0 });
    try expectCmp(i16, lt_i16, .{ 0, 1, 0 });
    try expectCmp(u16, lt_u16, .{ 0, 1, 0 });
    try expectCmp(i8, lt_i8, .{ 0, 1, 0 });
    try expectCmp(u8, lt_u8, .{ 0, 1, 0 });
}

test "le over every dtype" {
    try expectCmp(f64, le_f64, .{ 0, 1, 1 });
    try expectCmp(f32, le_f32, .{ 0, 1, 1 });
    try expectCmp(i64, le_i64, .{ 0, 1, 1 });
    try expectCmp(u64, le_u64, .{ 0, 1, 1 });
    try expectCmp(i32, le_i32, .{ 0, 1, 1 });
    try expectCmp(u32, le_u32, .{ 0, 1, 1 });
    try expectCmp(i16, le_i16, .{ 0, 1, 1 });
    try expectCmp(u16, le_u16, .{ 0, 1, 1 });
    try expectCmp(i8, le_i8, .{ 0, 1, 1 });
    try expectCmp(u8, le_u8, .{ 0, 1, 1 });
}

test "gt over every dtype" {
    try expectCmp(f64, gt_f64, .{ 1, 0, 0 });
    try expectCmp(f32, gt_f32, .{ 1, 0, 0 });
    try expectCmp(i64, gt_i64, .{ 1, 0, 0 });
    try expectCmp(u64, gt_u64, .{ 1, 0, 0 });
    try expectCmp(i32, gt_i32, .{ 1, 0, 0 });
    try expectCmp(u32, gt_u32, .{ 1, 0, 0 });
    try expectCmp(i16, gt_i16, .{ 1, 0, 0 });
    try expectCmp(u16, gt_u16, .{ 1, 0, 0 });
    try expectCmp(i8, gt_i8, .{ 1, 0, 0 });
    try expectCmp(u8, gt_u8, .{ 1, 0, 0 });
}

test "ge over every dtype" {
    try expectCmp(f64, ge_f64, .{ 1, 0, 1 });
    try expectCmp(f32, ge_f32, .{ 1, 0, 1 });
    try expectCmp(i64, ge_i64, .{ 1, 0, 1 });
    try expectCmp(u64, ge_u64, .{ 1, 0, 1 });
    try expectCmp(i32, ge_i32, .{ 1, 0, 1 });
    try expectCmp(u32, ge_u32, .{ 1, 0, 1 });
    try expectCmp(i16, ge_i16, .{ 1, 0, 1 });
    try expectCmp(u16, ge_u16, .{ 1, 0, 1 });
    try expectCmp(i8, ge_i8, .{ 1, 0, 1 });
    try expectCmp(u8, ge_u8, .{ 1, 0, 1 });
}

test "signed dtypes order negatives correctly" {
    const testing = @import("std").testing;
    const a = [_]i8{ -128, -1, 0, 1, 127 };
    const b = [_]i8{ 127, 0, 0, -1, -128 };
    var out = [_]u8{0xAA} ** 5;
    lt_i8(&a, &b, &out, 5);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 0, 0, 0 }, &out);
    ge_i8(&a, &b, &out, 5);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 1, 1, 1 }, &out);
}

test "every comparison against NaN is false except ne" {
    const testing = @import("std").testing;
    const nan = @as(f64, 0.0) / @as(f64, 0.0);
    const a = [_]f64{ nan, nan, 1.0 };
    const b = [_]f64{ nan, 1.0, nan };
    var out = [_]u8{0xAA} ** 3;
    eq_f64(&a, &b, &out, 3);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, &out);
    lt_f64(&a, &b, &out, 3);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, &out);
    ge_f64(&a, &b, &out, 3);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, &out);
    ne_f64(&a, &b, &out, 3);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1 }, &out);
}

test "compare with N = 0 writes nothing" {
    const testing = @import("std").testing;
    const a = [_]i32{5};
    const b = [_]i32{5};
    var out = [_]u8{0xAA};
    eq_i32(&a, &b, &out, 0);
    try testing.expectEqual(@as(u8, 0xAA), out[0]);
}
