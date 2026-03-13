//! WASM element-wise GCD (greatest common divisor) kernels.
//!
//! Scalar: out[i] = gcd(a[i], scalar)
//! Binary: out[i] = gcd(a[i], b[i])
//! Uses Euclidean algorithm. Operates on contiguous 1D buffers.

/// Scalar GCD for i32: out[i] = gcd(abs(a[i]), abs(scalar)).
export fn gcd_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const b = if (scalar < 0) -scalar else scalar;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var x = a[i];
        if (x < 0) x = -x;
        var y = b;
        while (y != 0) {
            const temp = y;
            y = @rem(x, y);
            x = temp;
        }
        out[i] = x;
    }
}

/// Binary GCD for i32: out[i] = gcd(abs(a[i]), abs(b[i])).
export fn gcd_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var x = a[i];
        if (x < 0) x = -x;
        var y = b[i];
        if (y < 0) y = -y;
        while (y != 0) {
            const temp = y;
            y = @rem(x, y);
            x = temp;
        }
        out[i] = x;
    }
}

// --- Tests ---

test "gcd_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, 7, 0, -15 };
    var out: [5]i32 = undefined;
    gcd_scalar_i32(&a, &out, 5, 6);
    try testing.expectEqual(out[0], 6); // gcd(12,6)
    try testing.expectEqual(out[1], 6); // gcd(18,6)
    try testing.expectEqual(out[2], 1); // gcd(7,6)
    try testing.expectEqual(out[3], 6); // gcd(0,6)
    try testing.expectEqual(out[4], 3); // gcd(15,6)
}

test "gcd_i32 binary" {
    const testing = @import("std").testing;
    const a = [_]i32{ 12, 18, 7, 0 };
    const b = [_]i32{ 8, 12, 5, 3 };
    var out: [4]i32 = undefined;
    gcd_i32(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 4); // gcd(12,8)
    try testing.expectEqual(out[1], 6); // gcd(18,12)
    try testing.expectEqual(out[2], 1); // gcd(7,5)
    try testing.expectEqual(out[3], 3); // gcd(0,3)
}
