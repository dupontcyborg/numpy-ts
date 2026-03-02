// WASM matmul kernels for f32 and f64
// Tiled i-k-j loop with SIMD vectorization
// Uses native v128 widths with pointer-cast loads/stores.

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);

inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}
inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}
inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

const TILE_F64 = 48;
const TILE_F32 = 64;

fn tiledMatmulF64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: usize, N: usize, K: usize) void {
    // Zero output
    for (0..M * N) |i| {
        c[i] = 0;
    }

    // Tiled i-k-j loop
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F64) {
        const i_end = if (ii + TILE_F64 < M) ii + TILE_F64 else M;
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F64) {
            const k_end = if (kk + TILE_F64 < K) kk + TILE_F64 else K;
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F64) {
                const j_end = if (jj + TILE_F64 < N) jj + TILE_F64 else N;

                // Inner tile
                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        const a_ik = a[i * K + k];
                        const a_vec: V2f64 = @splat(a_ik);
                        const b_row = k * N;
                        const c_row = i * N;

                        // Vectorized j loop: two v128 (2×f64) per step = 4 f64
                        var j: usize = jj;
                        while (j + 4 <= j_end) : (j += 4) {
                            store2_f64(c, c_row + j, load2_f64(c, c_row + j) + a_vec * load2_f64(b, b_row + j));
                            store2_f64(c, c_row + j + 2, load2_f64(c, c_row + j + 2) + a_vec * load2_f64(b, b_row + j + 2));
                        }
                        // One more v128 if possible
                        while (j + 2 <= j_end) : (j += 2) {
                            store2_f64(c, c_row + j, load2_f64(c, c_row + j) + a_vec * load2_f64(b, b_row + j));
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

fn tiledMatmulF32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: usize, N: usize, K: usize) void {
    // Zero output
    for (0..M * N) |i| {
        c[i] = 0;
    }

    // Tiled i-k-j loop
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = if (ii + TILE_F32 < M) ii + TILE_F32 else M;
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F32) {
            const k_end = if (kk + TILE_F32 < K) kk + TILE_F32 else K;
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F32) {
                const j_end = if (jj + TILE_F32 < N) jj + TILE_F32 else N;

                // Inner tile
                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        const a_ik = a[i * K + k];
                        const a_vec: V4f32 = @splat(a_ik);
                        const b_row = k * N;
                        const c_row = i * N;

                        // Vectorized j loop: two v128 (4×f32) per step = 8 f32
                        var j: usize = jj;
                        while (j + 8 <= j_end) : (j += 8) {
                            store4_f32(c, c_row + j, load4_f32(c, c_row + j) + a_vec * load4_f32(b, b_row + j));
                            store4_f32(c, c_row + j + 4, load4_f32(c, c_row + j + 4) + a_vec * load4_f32(b, b_row + j + 4));
                        }
                        // One more v128 if possible
                        while (j + 4 <= j_end) : (j += 4) {
                            store4_f32(c, c_row + j, load4_f32(c, c_row + j) + a_vec * load4_f32(b, b_row + j));
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

// C-ABI exports for WASM
export fn matmul_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, c_ptr: [*]f64, M: u32, N: u32, K: u32) void {
    tiledMatmulF64(a_ptr, b_ptr, c_ptr, M, N, K);
}

export fn matmul_f32(a_ptr: [*]const f32, b_ptr: [*]const f32, c_ptr: [*]f32, M: u32, N: u32, K: u32) void {
    tiledMatmulF32(a_ptr, b_ptr, c_ptr, M, N, K);
}
