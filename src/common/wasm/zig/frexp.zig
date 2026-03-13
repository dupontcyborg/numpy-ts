//! WASM element-wise frexp kernel: decompose into mantissa and exponent.
//!
//! out_m[i], out_e[i] = frexp(x[i])
//! where x = mantissa * 2^exponent, 0.5 <= |mantissa| < 1.0
//! Operates on contiguous 1D buffers of length N.

const math = @import("std").math;

/// frexp for f64: decomposes x[i] into mantissa (f64) and exponent (i32).
/// out_m[i] = mantissa, out_e[i] = exponent such that x[i] = out_m[i] * 2^out_e[i].
export fn frexp_f64(x: [*]const f64, out_m: [*]f64, out_e: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const val = x[i];
        if (val == 0.0 or !math.isFinite(val)) {
            out_m[i] = val;
            out_e[i] = 0;
        } else {
            const result = math.frexp(val);
            out_m[i] = result.significand;
            out_e[i] = result.exponent;
        }
    }
}

/// --- Tests ---
const testing = @import("std").testing;

test "frexp_f64 basic" {
    const x = [_]f64{ 0.0, 1.0, 2.0, -4.0, 0.5 };
    var out_m: [5]f64 = undefined;
    var out_e: [5]i32 = undefined;
    frexp_f64(&x, &out_m, &out_e, 5);
    // 0 → (0, 0)
    try testing.expectApproxEqAbs(out_m[0], 0.0, 1e-10);
    try testing.expectEqual(out_e[0], 0);
    // 1.0 → (0.5, 1)
    try testing.expectApproxEqAbs(out_m[1], 0.5, 1e-10);
    try testing.expectEqual(out_e[1], 1);
    // 2.0 → (0.5, 2)
    try testing.expectApproxEqAbs(out_m[2], 0.5, 1e-10);
    try testing.expectEqual(out_e[2], 2);
    // -4.0 → (-0.5, 3)
    try testing.expectApproxEqAbs(out_m[3], -0.5, 1e-10);
    try testing.expectEqual(out_e[3], 3);
    // 0.5 → (0.5, 0)
    try testing.expectApproxEqAbs(out_m[4], 0.5, 1e-10);
    try testing.expectEqual(out_e[4], 0);
}
