// WASM reduction kernels for f32/f64 with SIMD
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Two accumulators for sum/prod to saturate memory bandwidth.
// Pointer-cast loads to guarantee v128.load opcodes.

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);

inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}
inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}
inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

// ─── f64 reductions ─────────────────────────────────────────────────────────

export fn sum_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    var acc0: V2f64 = @splat(0.0);
    var acc1: V2f64 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc0 += load2_f64(ptr, i);
        acc1 += load2_f64(ptr, i + 2);
    }
    while (i + 2 <= len) : (i += 2) {
        acc0 += load2_f64(ptr, i);
    }
    acc0 += acc1;
    var result: f64 = acc0[0] + acc0[1];
    while (i < len) : (i += 1) {
        result += ptr[i];
    }
    return result;
}

export fn max_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    var acc: V2f64 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 2 <= len) : (i += 2) {
        const v = load2_f64(ptr, i);
        acc = @select(f64, v > acc, v, acc);
    }
    var result: f64 = if (acc[0] > acc[1]) acc[0] else acc[1];
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    if (len == 0) return @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    var acc: V2f64 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 2 <= len) : (i += 2) {
        const v = load2_f64(ptr, i);
        acc = @select(f64, v < acc, v, acc);
    }
    var result: f64 = if (acc[0] < acc[1]) acc[0] else acc[1];
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

export fn prod_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    var acc0: V2f64 = @splat(1.0);
    var acc1: V2f64 = @splat(1.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc0 *= load2_f64(ptr, i);
        acc1 *= load2_f64(ptr, i + 2);
    }
    while (i + 2 <= len) : (i += 2) {
        acc0 *= load2_f64(ptr, i);
    }
    acc0 *= acc1;
    var result: f64 = acc0[0] * acc0[1];
    while (i < len) : (i += 1) {
        result *= ptr[i];
    }
    return result;
}

export fn mean_f64(ptr: [*]const f64, n: u32) f64 {
    return sum_f64(ptr, n) / @as(f64, @floatFromInt(n));
}

// ─── f32 reductions ─────────────────────────────────────────────────────────

export fn sum_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    var acc0: V4f32 = @splat(0.0);
    var acc1: V4f32 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        acc0 += load4_f32(ptr, i);
        acc1 += load4_f32(ptr, i + 4);
    }
    while (i + 4 <= len) : (i += 4) {
        acc0 += load4_f32(ptr, i);
    }
    acc0 += acc1;
    var result: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
    while (i < len) : (i += 1) {
        result += ptr[i];
    }
    return result;
}

export fn max_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return -@as(f32, @bitCast(@as(u32, 0x7F800000)));
    var acc: V4f32 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v = load4_f32(ptr, i);
        acc = @select(f32, v > acc, v, acc);
    }
    var result: f32 = acc[0];
    if (acc[1] > result) result = acc[1];
    if (acc[2] > result) result = acc[2];
    if (acc[3] > result) result = acc[3];
    while (i < len) : (i += 1) {
        if (ptr[i] > result) result = ptr[i];
    }
    return result;
}

export fn min_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    if (len == 0) return @as(f32, @bitCast(@as(u32, 0x7F800000)));
    var acc: V4f32 = @splat(ptr[0]);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v = load4_f32(ptr, i);
        acc = @select(f32, v < acc, v, acc);
    }
    var result: f32 = acc[0];
    if (acc[1] < result) result = acc[1];
    if (acc[2] < result) result = acc[2];
    if (acc[3] < result) result = acc[3];
    while (i < len) : (i += 1) {
        if (ptr[i] < result) result = ptr[i];
    }
    return result;
}

export fn prod_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    var acc0: V4f32 = @splat(1.0);
    var acc1: V4f32 = @splat(1.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        acc0 *= load4_f32(ptr, i);
        acc1 *= load4_f32(ptr, i + 4);
    }
    while (i + 4 <= len) : (i += 4) {
        acc0 *= load4_f32(ptr, i);
    }
    acc0 *= acc1;
    var result: f32 = acc0[0] * acc0[1] * acc0[2] * acc0[3];
    while (i < len) : (i += 1) {
        result *= ptr[i];
    }
    return result;
}

export fn mean_f32(ptr: [*]const f32, n: u32) f32 {
    return sum_f32(ptr, n) / @as(f32, @floatFromInt(n));
}

// ─── diff: first-order differences ──────────────────────────────────────────
// out[i] = in[i+1] - in[i], output has n-1 elements

export fn diff_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    const out_len = len - 1;
    var i: usize = 0;
    while (i + 2 <= out_len) : (i += 2) {
        const v0 = load2_f64(in_ptr, i);
        const v1 = load2_f64(in_ptr, i + 1);
        // v1 - v0 gives [in[i+1]-in[i], in[i+2]-in[i+1]]
        store2_f64(out_ptr, i, v1 - v0);
    }
    while (i < out_len) : (i += 1) {
        out_ptr[i] = in_ptr[i + 1] - in_ptr[i];
    }
}

export fn diff_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    const out_len = len - 1;
    var i: usize = 0;
    while (i + 4 <= out_len) : (i += 4) {
        const v0 = load4_f32(in_ptr, i);
        const v1 = load4_f32(in_ptr, i + 1);
        store4_f32(out_ptr, i, v1 - v0);
    }
    while (i < out_len) : (i += 1) {
        out_ptr[i] = in_ptr[i + 1] - in_ptr[i];
    }
}
