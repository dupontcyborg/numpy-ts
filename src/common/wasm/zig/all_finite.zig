//! WASM "all finite" check: returns 1 if every element is finite, 0 otherwise.
//! Uses the same exponent-bit logic as isfinite, but with early exit and no output array.
//!
//! For the SIMD path, we OR together all exponent-masked results across a chunk,
//! then check once — avoiding per-element branching while still enabling early exit
//! every 4/8 elements.

/// all_finite for f64 — 2-wide SIMD (WASM has i64x2.eq).
export fn all_finite_f64(a: [*]const f64, N: u32) u32 {
    const a_i64: [*]const i64 = @ptrCast(a);
    const n2 = N & ~@as(u32, 1);
    var i: u32 = 0;

    const exp_v: @Vector(2, i64) = @splat(@as(i64, @bitCast(@as(u64, 0x7FF0_0000_0000_0000))));

    while (i < n2) : (i += 2) {
        const bits: @Vector(2, i64) = @as(*align(1) const @Vector(2, i64), @ptrCast(a_i64 + i)).*;
        const is_special = (bits & exp_v) == exp_v;
        if (is_special[0] or is_special[1]) return 0;
    }
    if (i < N) {
        if ((@as(u64, @bitCast(a_i64[i])) & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000) return 0;
    }
    return 1;
}

/// all_finite for f32 — 4-wide SIMD with early exit per chunk.
export fn all_finite_f32(a: [*]const f32, N: u32) u32 {
    const a_i32: [*]const i32 = @ptrCast(a);
    const n4 = N & ~@as(u32, 3);
    var i: u32 = 0;

    const exp_v: @Vector(4, i32) = @splat(@as(i32, @bitCast(@as(u32, 0x7F80_0000))));

    while (i < n4) : (i += 4) {
        const bits: @Vector(4, i32) = @as(*align(1) const @Vector(4, i32), @ptrCast(a_i32 + i)).*;
        const is_special = (bits & exp_v) == exp_v;
        // If any lane is true (non-finite found), bail
        if (is_special[0] or is_special[1] or is_special[2] or is_special[3]) return 0;
    }
    while (i < N) : (i += 1) {
        if ((@as(u32, @bitCast(a_i32[i])) & 0x7F80_0000) == 0x7F80_0000) return 0;
    }
    return 1;
}

/// all_finite for f16 (as raw u16) — 8-wide SIMD with early exit per chunk.
export fn all_finite_u16(a: [*]const u16, N: u32) u32 {
    const a_i16: [*]const i16 = @ptrCast(a);
    const n8 = N & ~@as(u32, 7);
    var i: u32 = 0;

    const exp_v: @Vector(8, i16) = @splat(@as(i16, @bitCast(@as(u16, 0x7C00))));

    while (i < n8) : (i += 8) {
        const bits: @Vector(8, i16) = @as(*align(1) const @Vector(8, i16), @ptrCast(a_i16 + i)).*;
        const is_special = (bits & exp_v) == exp_v;
        if (is_special[0] or is_special[1] or is_special[2] or is_special[3] or
            is_special[4] or is_special[5] or is_special[6] or is_special[7]) return 0;
    }
    while (i < N) : (i += 1) {
        if ((@as(u16, @bitCast(a_i16[i])) & 0x7C00) == 0x7C00) return 0;
    }
    return 1;
}

// --- Tests ---

test "all_finite_f64" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, -2.0, 0.0, 3.14 };
    try testing.expectEqual(all_finite_f64(&a, 4), 1);
    const nan: f64 = @bitCast(@as(u64, 0x7FF8000000000000));
    const b = [_]f64{ 1.0, nan, 3.0 };
    try testing.expectEqual(all_finite_f64(&b, 3), 0);
    const inf: f64 = @bitCast(@as(u64, 0x7FF0000000000000));
    const c = [_]f64{ 1.0, 2.0, inf };
    try testing.expectEqual(all_finite_f64(&c, 3), 0);
}

test "all_finite_f32" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, -2.0, 0.0, 3.14, 5.0, 6.0, 7.0, 8.0 };
    try testing.expectEqual(all_finite_f32(&a, 8), 1);
    const nan: f32 = @bitCast(@as(u32, 0x7FC00000));
    const b = [_]f32{ 1.0, nan, 3.0, 4.0 };
    try testing.expectEqual(all_finite_f32(&b, 4), 0);
    const inf: f32 = @bitCast(@as(u32, 0x7F800000));
    const c = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, inf };
    try testing.expectEqual(all_finite_f32(&c, 8), 0);
}

test "all_finite_u16" {
    const testing = @import("std").testing;
    // all finite f16 values
    const a = [_]u16{ 0x3C00, 0x4000, 0x0000, 0x8000, 0x0001, 0x3C00, 0x4200, 0x3C00 };
    try testing.expectEqual(all_finite_u16(&a, 8), 1);
    // NaN at position 5
    const b = [_]u16{ 0x3C00, 0x4000, 0x0000, 0x8000, 0x0001, 0x7E00, 0x4200, 0x3C00 };
    try testing.expectEqual(all_finite_u16(&b, 8), 0);
    // inf at end
    const c = [_]u16{ 0x3C00, 0x7C00 };
    try testing.expectEqual(all_finite_u16(&c, 2), 0);
}
