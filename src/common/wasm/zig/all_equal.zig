//! WASM SIMD whole-array equality: does every element of `a` equal `b`?
//!
//! Backs `array_equal` and `array_equiv`. A per-element scalar loop costs the
//! same regardless of dtype, but NumPy's cost scales with bytes moved;
//! comparing a whole v128 at a time (16 lanes for i8 down to 2 for i64/f64)
//! keeps this kernel on the same per-byte footing across dtypes.
//!
//! Returns 1 when every element matches, 0 otherwise, and bails on the first
//! mismatching block.

/// Integer and default float comparison. For floats this is IEEE equality, so
/// NaN never equals NaN — which is exactly `array_equal(..., equal_nan=false)`.
inline fn allEqual(comptime T: type, comptime L: comptime_int, a: [*]const T, b: [*]const T, N: u32) u32 {
    const V = @Vector(L, T);
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const va = @as(*align(1) const V, @ptrCast(a + i)).*;
        const vb = @as(*align(1) const V, @ptrCast(b + i)).*;
        if (!@reduce(.And, va == vb)) return 0;
    }
    while (i < N) : (i += 1) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

/// Float comparison treating NaN as equal to NaN — `equal_nan=true`.
/// A lane passes when the values are equal *or* both are NaN.
inline fn allEqualNan(comptime T: type, comptime L: comptime_int, a: [*]const T, b: [*]const T, N: u32) u32 {
    const V = @Vector(L, T);
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const va = @as(*align(1) const V, @ptrCast(a + i)).*;
        const vb = @as(*align(1) const V, @ptrCast(b + i)).*;
        const both_nan = (va != va) & (vb != vb);
        if (!@reduce(.And, (va == vb) | both_nan)) return 0;
    }
    while (i < N) : (i += 1) {
        const x = a[i];
        const y = b[i];
        if (!(x == y or (x != x and y != y))) return 0;
    }
    return 1;
}

export fn all_equal_i8(a: [*]const i8, b: [*]const i8, N: u32) u32 {
    return allEqual(i8, 16, a, b, N);
}
export fn all_equal_u8(a: [*]const u8, b: [*]const u8, N: u32) u32 {
    return allEqual(u8, 16, a, b, N);
}
export fn all_equal_i16(a: [*]const i16, b: [*]const i16, N: u32) u32 {
    return allEqual(i16, 8, a, b, N);
}
export fn all_equal_u16(a: [*]const u16, b: [*]const u16, N: u32) u32 {
    return allEqual(u16, 8, a, b, N);
}
export fn all_equal_i32(a: [*]const i32, b: [*]const i32, N: u32) u32 {
    return allEqual(i32, 4, a, b, N);
}
export fn all_equal_u32(a: [*]const u32, b: [*]const u32, N: u32) u32 {
    return allEqual(u32, 4, a, b, N);
}
export fn all_equal_i64(a: [*]const i64, b: [*]const i64, N: u32) u32 {
    return allEqual(i64, 2, a, b, N);
}
export fn all_equal_u64(a: [*]const u64, b: [*]const u64, N: u32) u32 {
    return allEqual(u64, 2, a, b, N);
}
export fn all_equal_f32(a: [*]const f32, b: [*]const f32, N: u32) u32 {
    return allEqual(f32, 4, a, b, N);
}
export fn all_equal_f64(a: [*]const f64, b: [*]const f64, N: u32) u32 {
    return allEqual(f64, 2, a, b, N);
}

/// equal_nan=true variants. Integers need none: they cannot hold NaN, so the
/// plain kernel is already correct for both settings.
export fn all_equal_nan_f32(a: [*]const f32, b: [*]const f32, N: u32) u32 {
    return allEqualNan(f32, 4, a, b, N);
}
export fn all_equal_nan_f64(a: [*]const f64, b: [*]const f64, N: u32) u32 {
    return allEqualNan(f64, 2, a, b, N);
}

// --- Tests ---

test "all_equal_i8 matching and mismatching" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    var b = a;
    try testing.expectEqual(@as(u32, 1), all_equal_i8(&a, &b, 18));
    b[17] = 99; // in the scalar tail, past the 16-wide block
    try testing.expectEqual(@as(u32, 0), all_equal_i8(&a, &b, 18));
    b[17] = 18;
    b[3] = 99; // inside the vector block
    try testing.expectEqual(@as(u32, 0), all_equal_i8(&a, &b, 18));
}

test "all_equal_f64 NaN is never equal to itself" {
    const testing = @import("std").testing;
    const nan = @as(f64, 0.0) / @as(f64, 0.0);
    const a = [_]f64{ 1.0, nan, 3.0, 4.0, 5.0 };
    const b = [_]f64{ 1.0, nan, 3.0, 4.0, 5.0 };
    try testing.expectEqual(@as(u32, 0), all_equal_f64(&a, &b, 5));
    try testing.expectEqual(@as(u32, 1), all_equal_nan_f64(&a, &b, 5));
}

test "all_equal_nan_f32 still rejects genuine differences" {
    const testing = @import("std").testing;
    const nan = @as(f32, 0.0) / @as(f32, 0.0);
    const a = [_]f32{ 1.0, nan, 3.0, 4.0, 5.0, 6.0 };
    const b = [_]f32{ 1.0, nan, 3.0, 4.5, 5.0, 6.0 };
    try testing.expectEqual(@as(u32, 0), all_equal_nan_f32(&a, &b, 6));
    // NaN on one side only is not a match
    const c = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try testing.expectEqual(@as(u32, 0), all_equal_nan_f32(&a, &c, 6));
}

test "all_equal_i64 and u64 two-wide" {
    const testing = @import("std").testing;
    const a = [_]i64{ 9007199254740993, 2, 3 };
    var b = [_]i64{ 9007199254740993, 2, 3 };
    try testing.expectEqual(@as(u32, 1), all_equal_i64(&a, &b, 3));
    b[0] = 9007199254740992; // differs only below the f64 exact range
    try testing.expectEqual(@as(u32, 0), all_equal_i64(&a, &b, 3));
}

test "all_equal_u8 empty and single" {
    const testing = @import("std").testing;
    const a = [_]u8{7};
    const b = [_]u8{7};
    try testing.expectEqual(@as(u32, 1), all_equal_u8(&a, &b, 0));
    try testing.expectEqual(@as(u32, 1), all_equal_u8(&a, &b, 1));
}
