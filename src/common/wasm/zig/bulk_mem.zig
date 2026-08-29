//! WASM bulk-memory helpers (memory.copy / memory.fill) shared by the
//! copy-shaped kernels: pad, tile.
//!
//! WASM's `memory.copy`/`memory.fill` (Zig's `@memcpy`/`@memset`) carry a fixed
//! per-call cost, so they only pay off on long runs, and the crossover is the
//! same byte length for every dtype, not the same element count.
//!
//! Below the threshold the fallback must stay a vector loop rather than a
//! scalar one, or it loses to the SIMD kernels these helpers replace. Being
//! generic over T is free, since the lane count is comptime.

/// Run length in bytes at which the bulk-memory instructions win.
pub const MIN_BYTES: usize = 512;

/// True when a run of `n` elements of `T` is long enough for bulk memory ops.
/// Hoist this out of any loop — it is invariant for a whole kernel call.
pub inline fn useBulk(comptime T: type, n: u32) bool {
    return @as(usize, n) * @sizeOf(T) >= MIN_BYTES;
}

/// One v128 worth of lanes: 16 for i8 ... 2 for f64/i64.
inline fn Lanes(comptime T: type) comptime_int {
    return 16 / @sizeOf(T);
}

/// Short-run copy, v128-wide.
pub inline fn copySmall(comptime T: type, dst: [*]T, src: [*]const T, n: u32) void {
    const L = Lanes(T);
    const V = @Vector(L, T);
    const n_simd = n & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        @as(*align(1) V, @ptrCast(dst + i)).* = @as(*align(1) const V, @ptrCast(src + i)).*;
    }
    while (i < n) : (i += 1) dst[i] = src[i];
}

/// Short-run zero fill, v128-wide.
pub inline fn fillSmall(comptime T: type, dst: [*]T, n: u32) void {
    const L = Lanes(T);
    const V = @Vector(L, T);
    const z: V = @splat(0);
    const n_simd = n & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        @as(*align(1) V, @ptrCast(dst + i)).* = z;
    }
    while (i < n) : (i += 1) dst[i] = 0;
}

/// Copy `n` elements, picking bulk memory or a vector loop by run length.
pub inline fn copyRun(comptime T: type, dst: [*]T, src: [*]const T, n: u32) void {
    if (useBulk(T, n)) @memcpy(dst[0..n], src[0..n]) else copySmall(T, dst, src, n);
}

/// Zero `n` elements, picking bulk memory or a vector loop by run length.
pub inline fn fillZero(comptime T: type, dst: [*]T, n: u32) void {
    if (useBulk(T, n)) @memset(dst[0..n], 0) else fillSmall(T, dst, n);
}

test "copyRun and fillZero agree with a reference at every length either side of the threshold" {
    const testing = @import("std").testing;
    inline for (.{ i8, i16, i32, i64, f32, f64 }) |T| {
        const N = 2000; // spans both sides of MIN_BYTES for every width
        var src: [N]T = undefined;
        var dst: [N]T = undefined;
        for (&src, 0..) |*p, i| p.* = switch (@typeInfo(T)) {
            .float => @floatFromInt(i % 100),
            else => @intCast(i % 100),
        };
        for ([_]u32{ 0, 1, 2, 3, 7, 15, 16, 17, 63, 64, 255, 256, 511, 512, 513, 1024, 1999 }) |n| {
            @memset(dst[0..N], 0);
            copyRun(T, &dst, &src, n);
            try testing.expectEqualSlices(T, src[0..n], dst[0..n]);
            // Nothing past n may be touched.
            for (dst[n..N]) |v| try testing.expectEqual(@as(T, 0), v);

            for (&dst) |*p| p.* = 1;
            fillZero(T, &dst, n);
            for (dst[0..n]) |v| try testing.expectEqual(@as(T, 0), v);
            for (dst[n..N]) |v| try testing.expectEqual(@as(T, 1), v);
        }
    }
}
