//! WASM 1D gradient kernel using central differences.
//!
//! out[0]     = (a[1] - a[0]) / h          (forward)
//! out[i]     = (a[i+1] - a[i-1]) / (2*h)  (central)
//! out[N-1]   = (a[N-1] - a[N-2]) / h      (backward)
//!
//! Output is always f64 (gradient promotes to float).

const simd = @import("simd.zig");

/// 1D gradient for f64 input, f64 output.
export fn gradient_f64(a: [*]const f64, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    // Forward difference at start
    out[0] = (a[1] - a[0]) / h;

    // Central differences in interior (SIMD)
    const h2 = 2.0 * h;
    const h2v: simd.V2f64 = @splat(h2);
    const interior = N - 1;
    const n_simd = 1 + ((interior - 1) & ~@as(u32, 1)); // round down to even, offset by 1
    var i: u32 = 1;
    while (i < n_simd and i < interior) : (i += 2) {
        const vp = simd.load2_f64(a, i + 1);
        const vm = simd.load2_f64(a, i - 1);
        simd.store2_f64(out, i, (vp - vm) / h2v);
    }
    while (i < interior) : (i += 1) {
        out[i] = (a[i + 1] - a[i - 1]) / h2;
    }

    // Backward difference at end
    out[N - 1] = (a[N - 1] - a[N - 2]) / h;
}

/// 1D gradient for f32 input, f32 output.
export fn gradient_f32(a: [*]const f32, out: [*]f32, N: u32, h: f32) void {
    if (N < 2) return;

    out[0] = (a[1] - a[0]) / h;

    const h2 = 2.0 * h;
    const h2v: simd.V4f32 = @splat(h2);
    const interior = N - 1;
    const n_simd = 1 + ((interior - 1) & ~@as(u32, 3));
    var i: u32 = 1;
    while (i < n_simd and i < interior) : (i += 4) {
        const vp = simd.load4_f32(a, i + 1);
        const vm = simd.load4_f32(a, i - 1);
        simd.store4_f32(out, i, (vp - vm) / h2v);
    }
    while (i < interior) : (i += 1) {
        out[i] = (a[i + 1] - a[i - 1]) / h2;
    }

    out[N - 1] = (a[N - 1] - a[N - 2]) / h;
}

/// 1D gradient for i64 input → f64 output (scalar loop).
export fn gradient_i64(a: [*]const i64, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for u64 input → f64 output (scalar loop).
export fn gradient_u64(a: [*]const u64, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;
    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;
    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }
    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for i32 input → f64 output.
export fn gradient_i32(a: [*]const i32, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for u32 input → f64 output.
export fn gradient_u32(a: [*]const u32, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;
    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;
    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }
    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for i16 input → f64 output.
export fn gradient_i16(a: [*]const i16, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for u16 input → f64 output.
export fn gradient_u16(a: [*]const u16, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;
    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;
    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }
    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for i8 input → f64 output.
export fn gradient_i8(a: [*]const i8, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;

    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;

    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }

    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

/// 1D gradient for u8 input → f64 output.
export fn gradient_u8(a: [*]const u8, out: [*]f64, N: u32, h: f64) void {
    if (N < 2) return;
    out[0] = (@as(f64, @floatFromInt(a[1])) - @as(f64, @floatFromInt(a[0]))) / h;
    const h2 = 2.0 * h;
    var i: u32 = 1;
    while (i < N - 1) : (i += 1) {
        const vp = @as(f64, @floatFromInt(a[i + 1]));
        const vm = @as(f64, @floatFromInt(a[i - 1]));
        out[i] = (vp - vm) / h2;
    }
    out[N - 1] = (@as(f64, @floatFromInt(a[N - 1])) - @as(f64, @floatFromInt(a[N - 2]))) / h;
}

// --- Tests ---

test "gradient_f64 basic" {
    const testing = @import("std").testing;
    // a = [0, 1, 4, 9, 16] (x^2 at 0..4)
    const a = [_]f64{ 0.0, 1.0, 4.0, 9.0, 16.0 };
    var out: [5]f64 = undefined;
    gradient_f64(&a, &out, 5, 1.0);
    // forward:  (1-0)/1 = 1
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    // central:  (4-0)/2 = 2
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    // central:  (9-1)/2 = 4
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    // central:  (16-4)/2 = 6
    try testing.expectApproxEqAbs(out[3], 6.0, 1e-10);
    // backward: (16-9)/1 = 7
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-10);
}

test "gradient_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 4.0, 9.0, 16.0 };
    var out: [5]f32 = undefined;
    gradient_f32(&a, &out, 5, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-5);
}

test "gradient_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 4, 9, 16 };
    var out: [5]f64 = undefined;
    gradient_i32(&a, &out, 5, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-10);
}

test "gradient_f64 two elements" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 10.0 };
    var out: [2]f64 = undefined;
    gradient_f64(&a, &out, 2, 1.0);
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-10);
}

test "gradient_f64 custom h" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 2.0, 8.0 };
    var out: [3]f64 = undefined;
    gradient_f64(&a, &out, 3, 0.5);
    // forward: (2-0)/0.5 = 4
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-10);
    // central: (8-0)/(2*0.5) = 8
    try testing.expectApproxEqAbs(out[1], 8.0, 1e-10);
    // backward: (8-2)/0.5 = 12
    try testing.expectApproxEqAbs(out[2], 12.0, 1e-10);
}

test "gradient_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 1, 4, 9, 16 };
    var out: [5]f64 = undefined;
    gradient_i64(&a, &out, 5, 1.0);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-10);
}

test "gradient_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 10, 40 };
    var out: [3]f64 = undefined;
    gradient_i16(&a, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 20.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 30.0, 1e-10);
}

test "gradient_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 5, 20 };
    var out: [3]f64 = undefined;
    gradient_i8(&a, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 15.0, 1e-10);
}

test "gradient_f64 SIMD boundary N=3 (minimal interior)" {
    const testing = @import("std").testing;
    // N=3: forward, 1 central, backward - tests V2f64 with just 1 interior element
    const a = [_]f64{ 0.0, 3.0, 12.0 };
    var out: [3]f64 = undefined;
    gradient_f64(&a, &out, 3, 1.0);
    // forward: (3-0)/1 = 3
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
    // central: (12-0)/2 = 6
    try testing.expectApproxEqAbs(out[1], 6.0, 1e-10);
    // backward: (12-3)/1 = 9
    try testing.expectApproxEqAbs(out[2], 9.0, 1e-10);
}

test "gradient_f64 SIMD boundary N=7 (remainder testing)" {
    const testing = @import("std").testing;
    // a[i] = i^2: 0, 1, 4, 9, 16, 25, 36
    const a = [_]f64{ 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0 };
    var out: [7]f64 = undefined;
    gradient_f64(&a, &out, 7, 1.0);
    // forward: (1-0)/1 = 1
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    // central i=1: (4-0)/2 = 2
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    // central i=2: (9-1)/2 = 4
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    // central i=3: (16-4)/2 = 6
    try testing.expectApproxEqAbs(out[3], 6.0, 1e-10);
    // central i=4: (25-9)/2 = 8
    try testing.expectApproxEqAbs(out[4], 8.0, 1e-10);
    // central i=5: (36-16)/2 = 10
    try testing.expectApproxEqAbs(out[5], 10.0, 1e-10);
    // backward: (36-25)/1 = 11
    try testing.expectApproxEqAbs(out[6], 11.0, 1e-10);
}

test "gradient_f32 SIMD boundary N=9 (V4f32 boundary)" {
    const testing = @import("std").testing;
    // a[i] = i^2: 0,1,4,9,16,25,36,49,64
    var a: [9]f32 = undefined;
    for (0..9) |idx| {
        const v: f32 = @floatFromInt(idx);
        a[idx] = v * v;
    }
    var out: [9]f32 = undefined;
    gradient_f32(&a, &out, 9, 1.0);
    // forward: (1-0)/1 = 1
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-4);
    // central i=1: (4-0)/2 = 2
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-4);
    // central i=4: (25-9)/2 = 8
    try testing.expectApproxEqAbs(out[4], 8.0, 1e-4);
    // central i=7: (64-36)/2 = 14
    try testing.expectApproxEqAbs(out[7], 14.0, 1e-4);
    // backward: (64-49)/1 = 15
    try testing.expectApproxEqAbs(out[8], 15.0, 1e-4);
}

test "gradient_i64 larger array" {
    const testing = @import("std").testing;
    // a = [0, 10, 40, 90, 160]
    const a = [_]i64{ 0, 10, 40, 90, 160 };
    var out: [5]f64 = undefined;
    gradient_i64(&a, &out, 5, 1.0);
    // forward: (10-0)/1 = 10
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-10);
    // central i=1: (40-0)/2 = 20
    try testing.expectApproxEqAbs(out[1], 20.0, 1e-10);
    // central i=2: (90-10)/2 = 40
    try testing.expectApproxEqAbs(out[2], 40.0, 1e-10);
    // central i=3: (160-40)/2 = 60
    try testing.expectApproxEqAbs(out[3], 60.0, 1e-10);
    // backward: (160-90)/1 = 70
    try testing.expectApproxEqAbs(out[4], 70.0, 1e-10);
}

test "gradient_i16 larger array" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 1, 4, 9, 16 };
    var out: [5]f64 = undefined;
    gradient_i16(&a, &out, 5, 1.0);
    // forward: (1-0)/1 = 1
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    // central i=1: (4-0)/2 = 2
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    // central i=2: (9-1)/2 = 4
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    // central i=3: (16-4)/2 = 6
    try testing.expectApproxEqAbs(out[3], 6.0, 1e-10);
    // backward: (16-9)/1 = 7
    try testing.expectApproxEqAbs(out[4], 7.0, 1e-10);
}

test "gradient_i8 larger array" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 2, 8, 18, 32 };
    var out: [5]f64 = undefined;
    gradient_i8(&a, &out, 5, 1.0);
    // forward: (2-0)/1 = 2
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    // central i=1: (8-0)/2 = 4
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    // central i=2: (18-2)/2 = 8
    try testing.expectApproxEqAbs(out[2], 8.0, 1e-10);
    // central i=3: (32-8)/2 = 12
    try testing.expectApproxEqAbs(out[3], 12.0, 1e-10);
    // backward: (32-18)/1 = 14
    try testing.expectApproxEqAbs(out[4], 14.0, 1e-10);
}

test "gradient_f64 constant array (gradient should be 0 in interior)" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 5.0, 5.0, 5.0, 5.0 };
    var out: [5]f64 = undefined;
    gradient_f64(&a, &out, 5, 1.0);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 0.0, 1e-10);
}

test "gradient_f32 custom step size h=0.5" {
    const testing = @import("std").testing;
    // a = [0, 1, 4] with h=0.5
    const a = [_]f32{ 0.0, 1.0, 4.0 };
    var out: [3]f32 = undefined;
    gradient_f32(&a, &out, 3, 0.5);
    // forward: (1-0)/0.5 = 2
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5);
    // central: (4-0)/(2*0.5) = 4
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-5);
    // backward: (4-1)/0.5 = 6
    try testing.expectApproxEqAbs(out[2], 6.0, 1e-5);
}

test "gradient_i32 custom step size h=2.0" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 10, 40 };
    var out: [3]f64 = undefined;
    gradient_i32(&a, &out, 3, 2.0);
    // forward: (10-0)/2 = 5
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    // central: (40-0)/(2*2) = 10
    try testing.expectApproxEqAbs(out[1], 10.0, 1e-10);
    // backward: (40-10)/2 = 15
    try testing.expectApproxEqAbs(out[2], 15.0, 1e-10);
}

test "gradient_f64 N=2 edge case (only forward and backward, no central)" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 7.0 };
    var out: [2]f64 = undefined;
    gradient_f64(&a, &out, 2, 1.0);
    // forward: (7-3)/1 = 4
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-10);
    // backward: (7-3)/1 = 4
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
}

test "gradient_f32 N=2 edge case" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 7.0 };
    var out: [2]f32 = undefined;
    gradient_f32(&a, &out, 2, 1.0);
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-5);
}

test "gradient_i64 N=2 edge case" {
    const testing = @import("std").testing;
    const a = [_]i64{ 3, 7 };
    var out: [2]f64 = undefined;
    gradient_i64(&a, &out, 2, 1.0);
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
}

test "gradient_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 0, 2, 4 };
    var out: [3]f64 = undefined;
    gradient_u64(&a, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}

test "gradient_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0, 2, 4 };
    var out: [3]f64 = undefined;
    gradient_u32(&a, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}

test "gradient_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0, 2, 4 };
    var out: [3]f64 = undefined;
    gradient_u16(&a, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}

test "gradient_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 0, 2, 4 };
    var out: [3]f64 = undefined;
    gradient_u8(&a, &out, 3, 1.0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
}
