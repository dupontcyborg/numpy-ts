// WASM reduction kernels: sum and max for f32/f64 with SIMD vectorization
//
// Note: max uses native WASM SIMD widths (@Vector(2,f64) / @Vector(4,f32))
// and @select instead of @max/@reduce, which generates much better codegen.
// The wider vectors (@Vector(4,f64)) produce scalar branch trees on WASM.

// ─── f64 reductions ─────────────────────────────────────────────────────────

export fn sum_f64(ptr: [*]const f64, n: u32) f64 {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var acc: @Vector(4, f64) = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: @Vector(4, f64) = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        acc += v;
    }
    var result: f64 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        result += ptr[i];
    }
    return result;
}

export fn max_f64(ptr: [*]const f64, n: u32) f64 {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    if (len == 0) return -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    // Use native WASM SIMD width: @Vector(2, f64) = one v128
    var acc: @Vector(2, f64) = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 2 <= len) : (i += 2) {
        const v: @Vector(2, f64) = .{ ptr[i], ptr[i + 1] };
        acc = @select(f64, v > acc, v, acc);
    }
    // Horizontal reduce: just 2 lanes
    var result: f64 = if (acc[0] > acc[1]) acc[0] else acc[1];
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

// ─── f32 reductions ─────────────────────────────────────────────────────────

export fn sum_f32(ptr: [*]const f32, n: u32) f32 {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var acc: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const v: @Vector(8, f32) = .{
            ptr[i],     ptr[i + 1], ptr[i + 2], ptr[i + 3],
            ptr[i + 4], ptr[i + 5], ptr[i + 6], ptr[i + 7],
        };
        acc += v;
    }
    var result: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        result += ptr[i];
    }
    return result;
}

export fn max_f32(ptr: [*]const f32, n: u32) f32 {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    if (len == 0) return -@as(f32, @bitCast(@as(u32, 0x7F800000)));
    // Use native WASM SIMD width: @Vector(4, f32) = one v128
    var acc: @Vector(4, f32) = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: @Vector(4, f32) = .{ ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3] };
        acc = @select(f32, v > acc, v, acc);
    }
    // Horizontal reduce: 4 lanes
    var result: f32 = acc[0];
    if (acc[1] > result) result = acc[1];
    if (acc[2] > result) result = acc[2];
    if (acc[3] > result) result = acc[3];
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}
