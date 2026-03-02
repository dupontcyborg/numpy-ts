// WASM 1D cross-correlation and convolution kernels for f32/f64
//
// correlate: out[k] = sum_j a[j] * b[j + k - (nb-1)] (full mode)
// convolve:  out[k] = sum_j a[j] * b_reversed[j + k - (nb-1)]
//
// Uses SIMD for the inner dot-product loop.

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);

inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}

// ─── f64 ────────────────────────────────────────────────────────────────────

export fn correlate_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, out_ptr: [*]f64, na: u32, nb: u32) void {
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const out_len = n_a + n_b - 1;

    for (0..out_len) |k| {
        var acc0: V2f64 = @splat(0.0);
        var acc1: V2f64 = @splat(0.0);

        // Compute valid index range
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        var j = j_start;
        const b_off = n_b - 1 - k; // b index = j + b_off (with wrapping)

        while (j + 4 <= j_end) : (j += 4) {
            const bi = j + b_off;
            acc0 += load2_f64(a_ptr, j) * load2_f64(b_ptr, bi);
            acc1 += load2_f64(a_ptr, j + 2) * load2_f64(b_ptr, bi + 2);
        }
        while (j + 2 <= j_end) : (j += 2) {
            acc0 += load2_f64(a_ptr, j) * load2_f64(b_ptr, j + b_off);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        while (j < j_end) : (j += 1) {
            sum += a_ptr[j] * b_ptr[j + b_off];
        }
        out_ptr[k] = sum;
    }
}

export fn convolve_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, out_ptr: [*]f64, na: u32, nb: u32) void {
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const out_len = n_a + n_b - 1;

    for (0..out_len) |k| {
        var sum: f64 = 0.0;
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        // convolve = correlate with b reversed: b[nb-1 - (j+b_off)] = b[k - j]
        var j = j_start;
        while (j < j_end) : (j += 1) {
            sum += a_ptr[j] * b_ptr[k - j];
        }
        out_ptr[k] = sum;
    }
}

// ─── f32 ────────────────────────────────────────────────────────────────────

export fn correlate_f32(a_ptr: [*]const f32, b_ptr: [*]const f32, out_ptr: [*]f32, na: u32, nb: u32) void {
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const out_len = n_a + n_b - 1;

    for (0..out_len) |k| {
        var acc0: V4f32 = @splat(0.0);
        var acc1: V4f32 = @splat(0.0);

        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        var j = j_start;
        const b_off = n_b - 1 - k;

        while (j + 8 <= j_end) : (j += 8) {
            const bi = j + b_off;
            acc0 += load4_f32(a_ptr, j) * load4_f32(b_ptr, bi);
            acc1 += load4_f32(a_ptr, j + 4) * load4_f32(b_ptr, bi + 4);
        }
        while (j + 4 <= j_end) : (j += 4) {
            acc0 += load4_f32(a_ptr, j) * load4_f32(b_ptr, j + b_off);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        while (j < j_end) : (j += 1) {
            sum += a_ptr[j] * b_ptr[j + b_off];
        }
        out_ptr[k] = sum;
    }
}

export fn convolve_f32(a_ptr: [*]const f32, b_ptr: [*]const f32, out_ptr: [*]f32, na: u32, nb: u32) void {
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const out_len = n_a + n_b - 1;

    for (0..out_len) |k| {
        var sum: f32 = 0.0;
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        var j = j_start;
        while (j < j_end) : (j += 1) {
            sum += a_ptr[j] * b_ptr[k - j];
        }
        out_ptr[k] = sum;
    }
}
