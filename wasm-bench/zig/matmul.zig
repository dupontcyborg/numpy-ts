// WASM matmul kernels for f32 and f64
// Tiled i-k-j loop with SIMD vectorization
// Build: zig build-lib -target wasm32-freestanding -O ReleaseFast -mcpu=generic+simd128

const TILE_F64 = 48;
const TILE_F32 = 64;

fn tiledMatmulF64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: usize, N: usize, K: usize) void {
    @setFloatMode(.optimized);

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
                        const a_vec: @Vector(4, f64) = @splat(a_ik);

                        // Vectorized j loop (4-wide f64 SIMD)
                        var j: usize = jj;
                        while (j + 4 <= j_end) : (j += 4) {
                            const b_vec: @Vector(4, f64) = .{
                                b[k * N + j],
                                b[k * N + j + 1],
                                b[k * N + j + 2],
                                b[k * N + j + 3],
                            };
                            const c_idx = i * N + j;
                            var c_vec: @Vector(4, f64) = .{
                                c[c_idx],
                                c[c_idx + 1],
                                c[c_idx + 2],
                                c[c_idx + 3],
                            };
                            c_vec += a_vec * b_vec;
                            c[c_idx] = c_vec[0];
                            c[c_idx + 1] = c_vec[1];
                            c[c_idx + 2] = c_vec[2];
                            c[c_idx + 3] = c_vec[3];
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

fn tiledMatmulF32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: usize, N: usize, K: usize) void {
    @setFloatMode(.optimized);

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
                        const a_vec: @Vector(8, f32) = @splat(a_ik);

                        // Vectorized j loop (8-wide f32 SIMD)
                        var j: usize = jj;
                        while (j + 8 <= j_end) : (j += 8) {
                            const b_vec: @Vector(8, f32) = .{
                                b[k * N + j],
                                b[k * N + j + 1],
                                b[k * N + j + 2],
                                b[k * N + j + 3],
                                b[k * N + j + 4],
                                b[k * N + j + 5],
                                b[k * N + j + 6],
                                b[k * N + j + 7],
                            };
                            const c_idx = i * N + j;
                            var c_vec: @Vector(8, f32) = .{
                                c[c_idx],
                                c[c_idx + 1],
                                c[c_idx + 2],
                                c[c_idx + 3],
                                c[c_idx + 4],
                                c[c_idx + 5],
                                c[c_idx + 6],
                                c[c_idx + 7],
                            };
                            c_vec += a_vec * b_vec;
                            c[c_idx] = c_vec[0];
                            c[c_idx + 1] = c_vec[1];
                            c[c_idx + 2] = c_vec[2];
                            c[c_idx + 3] = c_vec[3];
                            c[c_idx + 4] = c_vec[4];
                            c[c_idx + 5] = c_vec[5];
                            c[c_idx + 6] = c_vec[6];
                            c[c_idx + 7] = c_vec[7];
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c[i * N + j] += a_ik * b[k * N + j];
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
