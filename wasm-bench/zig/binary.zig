// WASM binary elementwise kernels: add and mul for f32/f64 with SIMD

// ─── f64 binary ─────────────────────────────────────────────────────────────

export fn add_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const va: @Vector(4, f64) = .{ a[i], a[i + 1], a[i + 2], a[i + 3] };
        const vb: @Vector(4, f64) = .{ b[i], b[i + 1], b[i + 2], b[i + 3] };
        const r = va + vb;
        out[i] = r[0];
        out[i + 1] = r[1];
        out[i + 2] = r[2];
        out[i + 3] = r[3];
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] + b[i];
    }
}

export fn mul_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const va: @Vector(4, f64) = .{ a[i], a[i + 1], a[i + 2], a[i + 3] };
        const vb: @Vector(4, f64) = .{ b[i], b[i + 1], b[i + 2], b[i + 3] };
        const r = va * vb;
        out[i] = r[0];
        out[i + 1] = r[1];
        out[i + 2] = r[2];
        out[i + 3] = r[3];
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

// ─── f32 binary ─────────────────────────────────────────────────────────────

export fn add_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, f32) = .{
            a[i],     a[i + 1], a[i + 2], a[i + 3],
            a[i + 4], a[i + 5], a[i + 6], a[i + 7],
        };
        const vb: @Vector(8, f32) = .{
            b[i],     b[i + 1], b[i + 2], b[i + 3],
            b[i + 4], b[i + 5], b[i + 6], b[i + 7],
        };
        const r = va + vb;
        out[i] = r[0];
        out[i + 1] = r[1];
        out[i + 2] = r[2];
        out[i + 3] = r[3];
        out[i + 4] = r[4];
        out[i + 5] = r[5];
        out[i + 6] = r[6];
        out[i + 7] = r[7];
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] + b[i];
    }
}

export fn mul_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, f32) = .{
            a[i],     a[i + 1], a[i + 2], a[i + 3],
            a[i + 4], a[i + 5], a[i + 6], a[i + 7],
        };
        const vb: @Vector(8, f32) = .{
            b[i],     b[i + 1], b[i + 2], b[i + 3],
            b[i + 4], b[i + 5], b[i + 6], b[i + 7],
        };
        const r = va * vb;
        out[i] = r[0];
        out[i + 1] = r[1];
        out[i + 2] = r[2];
        out[i + 3] = r[3];
        out[i + 4] = r[4];
        out[i + 5] = r[5];
        out[i + 6] = r[6];
        out[i + 7] = r[7];
    }
    while (i < len) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}
