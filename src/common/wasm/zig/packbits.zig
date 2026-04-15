//! WASM packbits / unpackbits kernels (1D, big-endian bit order, uint8 only).
//!
//! packbits_u8:   packs N input bytes (each 0 or non-zero) into ceil(N/8) output bytes.
//! unpackbits_u8: unpacks N_in packed bytes into N_out output bytes (each 0 or 1).

/// Pack N input bytes into ceil(N/8) output bytes, big-endian bit order.
/// Each input byte is treated as boolean: 0 => bit off, non-zero => bit on.
export fn packbits_u8(a: [*]const u8, out: [*]u8, N: u32) void {
    const n_full = N / 8; // number of complete 8-element groups
    var i: u32 = 0;
    // Pack complete 8-element groups into one byte each
    while (i < n_full) : (i += 1) {
        const base = i * 8;
        var byte: u8 = 0;
        // MSB-first: bit 0 of input → bit 7 of output byte
        inline for (0..8) |bit| {
            if (a[base + bit] != 0) byte |= @as(u8, 0x80) >> @intCast(bit);
        }
        out[i] = byte;
    }
    // Pack remaining < 8 elements into a final byte (zero-padded on the right)
    const rem = N - n_full * 8;
    if (rem > 0) {
        var byte: u8 = 0;
        const base = n_full * 8;
        for (0..rem) |bit| {
            if (a[base + bit] != 0) byte |= @as(u8, 0x80) >> @intCast(bit);
        }
        out[n_full] = byte;
    }
}

/// Unpack N_in packed bytes into N_out output bytes (0 or 1), big-endian bit order.
export fn unpackbits_u8(a: [*]const u8, out: [*]u8, N_in: u32, N_out: u32) void {
    var out_idx: u32 = 0;
    var i: u32 = 0;
    while (i < N_in and out_idx < N_out) : (i += 1) {
        const byte = a[i];
        inline for (0..8) |bit| {
            if (out_idx < N_out) {
                out[out_idx] = if (byte & (@as(u8, 0x80) >> @intCast(bit)) != 0) 1 else 0;
                out_idx += 1;
            }
        }
    }
}

// --- Tests ---

const std = @import("std");
const expect = std.testing.expect;

test "packbits_u8 full byte" {
    const input = [_]u8{ 1, 0, 1, 0, 1, 0, 1, 0 };
    var out: [1]u8 = undefined;
    packbits_u8(&input, &out, 8);
    try expect(out[0] == 0b10101010);
}

test "packbits_u8 partial byte" {
    const input = [_]u8{ 1, 1, 1, 0, 0 };
    var out: [1]u8 = undefined;
    packbits_u8(&input, &out, 5);
    try expect(out[0] == 0b11100000);
}

test "packbits_u8 non-zero values treated as 1" {
    const input = [_]u8{ 255, 0, 42, 0, 0, 0, 0, 1 };
    var out: [1]u8 = undefined;
    packbits_u8(&input, &out, 8);
    try expect(out[0] == 0b10100001);
}

test "unpackbits_u8 full byte" {
    const input = [_]u8{0b10101010};
    var out: [8]u8 = undefined;
    unpackbits_u8(&input, &out, 1, 8);
    const expected = [_]u8{ 1, 0, 1, 0, 1, 0, 1, 0 };
    for (0..8) |j| {
        try expect(out[j] == expected[j]);
    }
}

test "unpackbits_u8 with count limit" {
    const input = [_]u8{0b11100000};
    var out: [5]u8 = undefined;
    unpackbits_u8(&input, &out, 1, 5);
    const expected = [_]u8{ 1, 1, 1, 0, 0 };
    for (0..5) |j| {
        try expect(out[j] == expected[j]);
    }
}

test "packbits_u8 then unpackbits_u8 roundtrip" {
    const input = [_]u8{ 1, 0, 0, 1, 1, 1, 0, 1, 1, 0 };
    var pack_out: [2]u8 = undefined;
    packbits_u8(&input, &pack_out, 10);
    // First byte: 10011101 = 0x9D, second byte: 10000000 = 0x80
    try expect(pack_out[0] == 0x9D);
    try expect(pack_out[1] == 0x80);

    var unpacked: [10]u8 = undefined;
    unpackbits_u8(&pack_out, &unpacked, 2, 10);
    for (0..10) |j| {
        try expect(unpacked[j] == input[j]);
    }
}
