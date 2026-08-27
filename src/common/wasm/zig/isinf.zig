//! WASM SIMD isinf kernels: out[i] = 1 if a[i] is +/-infinity, else 0.
//!
//! Mirrors isnan.zig. Infinity is the only value whose magnitude equals
//! infinity, so `@abs(v) == inf` is the whole test — one absolute value and one
//! compare per vector. `@intFromBool` on the vector yields one byte per lane,
//! which is already the bool output layout, so nothing needs packing.
//!
//! Integer dtypes never reach here: they cannot be infinite, so the caller
//! fills zeros directly.

inline fn isinfFloat(comptime T: type, comptime L: comptime_int, a: [*]const T, out: [*]u8, N: u32) void {
    const V = @Vector(L, T);
    const inf: V = @splat(@as(T, 1.0) / @as(T, 0.0));
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const v = @as(*align(1) const V, @ptrCast(a + i)).*;
        const m: @Vector(L, u8) = @intFromBool(@abs(v) == inf);
        @as(*align(1) @Vector(L, u8), @ptrCast(out + i)).* = m;
    }
    const scalar_inf = @as(T, 1.0) / @as(T, 0.0);
    while (i < N) : (i += 1) out[i] = @intFromBool(@abs(a[i]) == scalar_inf);
}

/// isinf for f64 — 2-wide.
export fn isinf_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    isinfFloat(f64, 2, a, out, N);
}

/// isinf for f32 — 4-wide.
export fn isinf_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    isinfFloat(f32, 4, a, out, N);
}

/// isinf for f16, taking raw u16 bit patterns — 8-wide.
///
/// The input is not a float type, so there is nothing to take the magnitude of
/// without converting first. Infinite iff (bits & 0x7FFF) == 0x7C00.
export fn isinf_u16(a: [*]const u16, out: [*]u8, N: u32) void {
    const L = 8;
    const V = @Vector(L, u16);
    const mask: V = @splat(0x7FFF);
    const inf_bits: V = @splat(0x7C00);
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const v = @as(*align(1) const V, @ptrCast(a + i)).*;
        const m: @Vector(L, u8) = @intFromBool((v & mask) == inf_bits);
        @as(*align(1) @Vector(L, u8), @ptrCast(out + i)).* = m;
    }
    while (i < N) : (i += 1) out[i] = @intFromBool((a[i] & 0x7FFF) == 0x7C00);
}

// --- Tests ---

test "isinf_f64 basic" {
    const testing = @import("std").testing;
    const inf = @as(f64, 1.0) / @as(f64, 0.0);
    const nan = @as(f64, 0.0) / @as(f64, 0.0);
    const a = [_]f64{ inf, -inf, 0.0, 1.5, nan, -3.25, inf, 2.0, -inf };
    var out: [9]u8 = undefined;
    isinf_f64(&a, &out, 9);
    const want = [_]u8{ 1, 1, 0, 0, 0, 0, 1, 0, 1 };
    for (want, 0..) |w, i| try testing.expectEqual(w, out[i]);
}

test "isinf_f32 covers the vector tail" {
    const testing = @import("std").testing;
    const inf = @as(f32, 1.0) / @as(f32, 0.0);
    const nan = @as(f32, 0.0) / @as(f32, 0.0);
    const a = [_]f32{ 1, 2, 3, 4, 5, inf, nan, -inf, 0, -inf };
    var out: [10]u8 = undefined;
    isinf_f32(&a, &out, 10);
    const want = [_]u8{ 0, 0, 0, 0, 0, 1, 0, 1, 0, 1 };
    for (want, 0..) |w, i| try testing.expectEqual(w, out[i]);
}

test "isinf_u16 f16 bit patterns" {
    const testing = @import("std").testing;
    // 0x7C00 = +inf, 0xFC00 = -inf, 0x7E00 = NaN, 0x7BFF = max finite.
    const a = [_]u16{ 0x7C00, 0xFC00, 0x7E00, 0x7BFF, 0x0000, 0x3C00, 0x7C00, 0xFC00, 0x0001 };
    var out: [9]u8 = undefined;
    isinf_u16(&a, &out, 9);
    const want = [_]u8{ 1, 1, 0, 0, 0, 0, 1, 1, 0 };
    for (want, 0..) |w, i| try testing.expectEqual(w, out[i]);
}
