//! WASM Kronecker product kernels for all numeric types.
//!
//! Computes C = A ⊗ B where C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
//! for A[am×an] and B[bm×bn], output C[(am*bm)×(an*bn)].

const simd = @import("simd.zig");

/// Computes the Kronecker product of two f64 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
export fn kron_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V2f64 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;

                // Vectorized j loop: 2 f64s per step
                var jb: usize = 0;
                while (jb + 2 <= b_cols) : (jb += 2) {
                    simd.store2_f64(out_row, jb, aij * simd.load2_f64(b_row, jb));
                }
                // Scalar remainder
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] * b_row[jb];
                }
            }
        }
    }
}

/// Computes the Kronecker product of two f32 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
export fn kron_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V4f32 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;

                // Vectorized j loop: 4 f32s per step
                var jb: usize = 0;
                while (jb + 4 <= b_cols) : (jb += 4) {
                    simd.store4_f32(out_row, jb, aij * simd.load4_f32(b_row, jb));
                }
                // Scalar remainder
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] * b_row[jb];
                }
            }
        }
    }
}

/// Computes the Kronecker product of two complex128 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// A, B, and C are interleaved [re0, im0, re1, im1, ...].
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
export fn kron_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const a_re = a[(ia * a_cols + ja) * 2];
            const a_im = a[(ia * a_cols + ja) * 2 + 1];
            for (0..b_rows) |ib| {
                const out_base = ((ia * b_rows + ib) * out_cols + ja * b_cols) * 2;
                const b_base = ib * b_cols * 2;

                // Scalar loop: complex multiply
                for (0..b_cols) |jb| {
                    const b_re = b[b_base + jb * 2];
                    const b_im = b[b_base + jb * 2 + 1];
                    // (a_re + a_im*i) * (b_re + b_im*i)
                    out[out_base + jb * 2] = a_re * b_re - a_im * b_im;
                    out[out_base + jb * 2 + 1] = a_re * b_im + a_im * b_re;
                }
            }
        }
    }
}

/// Computes the Kronecker product of two complex64 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// A, B, and C are interleaved [re0, im0, re1, im1, ...].
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
export fn kron_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const a_re = a[(ia * a_cols + ja) * 2];
            const a_im = a[(ia * a_cols + ja) * 2 + 1];
            for (0..b_rows) |ib| {
                const out_base = ((ia * b_rows + ib) * out_cols + ja * b_cols) * 2;
                const b_base = ib * b_cols * 2;

                // Scalar loop: complex multiply
                for (0..b_cols) |jb| {
                    const b_re = b[b_base + jb * 2];
                    const b_im = b[b_base + jb * 2 + 1];
                    // (a_re + a_im*i) * (b_re + b_im*i)
                    out[out_base + jb * 2] = a_re * b_re - a_im * b_im;
                    out[out_base + jb * 2 + 1] = a_re * b_im + a_im * b_re;
                }
            }
        }
    }
}

/// Computes the Kronecker product of two i64 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn kron_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V2i64 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;

                // Vectorized j loop: 2 i64s per step
                var jb: usize = 0;
                while (jb + 2 <= b_cols) : (jb += 2) {
                    simd.store2_i64(out_row, jb, aij *% simd.load2_i64(b_row, jb));
                }
                // Scalar remainder
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] *% b_row[jb];
                }
            }
        }
    }
}

/// Computes the Kronecker product of two i32 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn kron_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V4i32 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;

                // Vectorized j loop: 4 i32s per step
                var jb: usize = 0;
                while (jb + 4 <= b_cols) : (jb += 4) {
                    simd.store4_i32(out_row, jb, aij *% simd.load4_i32(b_row, jb));
                }
                // Scalar remainder
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] *% b_row[jb];
                }
            }
        }
    }
}

/// Computes the Kronecker product of two i16 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
export fn kron_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V8i16 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;

                // Vectorized j loop: 8 i16s per step
                var jb: usize = 0;
                while (jb + 8 <= b_cols) : (jb += 8) {
                    simd.store8_i16(out_row, jb, aij *% simd.load8_i16(b_row, jb));
                }
                // Scalar remainder
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] *% b_row[jb];
                }
            }
        }
    }
}

/// Computes the Kronecker product of two i8 matrices A and B.
/// A is am×an, B is bm×bn, output C is (am*bm)×(an*bn).
/// C[(ia*bm+ib), (ja*bn+jb)] = A[ia,ja] * B[ib,jb]
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
export fn kron_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, am: u32, an: u32, bm: u32, bn: u32) void {
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..@as(usize, am)) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V16i8 = @splat(a[ia * a_cols + ja]);
            const zero: simd.V16i8 = @splat(0);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;

                // Vectorized j loop: 16 i8s per step (widened i16 multiply via muladd)
                var jb: usize = 0;
                while (jb + 16 <= b_cols) : (jb += 16) {
                    simd.store16_i8(out_row, jb, simd.muladd_i8x16(zero, aij, simd.load16_i8(b_row, jb)));
                }
                // Scalar remainder
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] *% b_row[jb];
                }
            }
        }
    }
}

// --- Tests ---

test "kron_f64 2x2 ⊗ 2x2" {
    const testing = @import("std").testing;
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // kron = [[5,6,10,12],[7,8,14,16],[15,18,20,24],[21,24,28,32]]
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var out: [16]f64 = undefined;
    kron_f64(&a, &b, &out, 2, 2, 2, 2);
    const expected = [_]f64{ 5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32 };
    for (0..16) |i| {
        try testing.expectApproxEqAbs(out[i], expected[i], 1e-10);
    }
}

test "kron_f32 2x2 ⊗ 2x2" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [16]f32 = undefined;
    kron_f32(&a, &b, &out, 2, 2, 2, 2);
    const expected = [_]f32{ 5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32 };
    for (0..16) |i| {
        try testing.expectApproxEqAbs(out[i], expected[i], 1e-5);
    }
}

test "kron_c128 1x1 ⊗ 1x2" {
    const testing = @import("std").testing;
    // A = [[(2+3i)]], B = [[(1+0i), (0+1i)]]
    // kron = [[(2+3i)*(1+0i), (2+3i)*(0+1i)]] = [[(2+3i), (-3+2i)]]
    const a = [_]f64{ 2, 3 };
    const b = [_]f64{ 1, 0, 0, 1 };
    var out: [4]f64 = undefined;
    kron_c128(&a, &b, &out, 1, 1, 1, 2);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
}

test "kron_c64 1x1 ⊗ 1x2" {
    const testing = @import("std").testing;
    // A = [[(2+3i)]], B = [[(1+0i), (0+1i)]]
    // kron = [[(2+3i), (-3+2i)]]
    const a = [_]f32{ 2, 3 };
    const b = [_]f32{ 1, 0, 0, 1 };
    var out: [4]f32 = undefined;
    kron_c64(&a, &b, &out, 1, 1, 1, 2);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-5);
}

test "kron_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3, 4 };
    const b = [_]i64{ 5, 6, 7, 8 };
    var out: [16]i64 = undefined;
    kron_i64(&a, &b, &out, 2, 2, 2, 2);
    const expected = [_]i64{ 5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32 };
    for (0..16) |i| {
        try testing.expectEqual(out[i], expected[i]);
    }
}

test "kron_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4 };
    const b = [_]i32{ 5, 6, 7, 8 };
    var out: [16]i32 = undefined;
    kron_i32(&a, &b, &out, 2, 2, 2, 2);
    const expected = [_]i32{ 5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32 };
    for (0..16) |i| {
        try testing.expectEqual(out[i], expected[i]);
    }
}

test "kron_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4 };
    const b = [_]i16{ 5, 6, 7, 8 };
    var out: [16]i16 = undefined;
    kron_i16(&a, &b, &out, 2, 2, 2, 2);
    const expected = [_]i16{ 5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32 };
    for (0..16) |i| {
        try testing.expectEqual(out[i], expected[i]);
    }
}

test "kron_i8 wrapping" {
    const testing = @import("std").testing;
    // 50 * 50 = 2500 → wraps in i8
    const a = [_]i8{ 50, 1 };
    const b = [_]i8{ 50, 1 };
    var out: [4]i8 = undefined;
    kron_i8(&a, &b, &out, 1, 2, 1, 2);
    const expected_00: i8 = @truncate(@as(i32, 50) * 50);
    try testing.expectEqual(out[0], expected_00);
    try testing.expectEqual(out[1], 50); // 50*1
    try testing.expectEqual(out[2], 50); // 1*50
    try testing.expectEqual(out[3], 1); // 1*1
}
