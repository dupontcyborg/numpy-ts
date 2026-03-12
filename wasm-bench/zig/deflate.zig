//! WASM raw DEFLATE inflate/deflate kernel.
//!
//! Implements RFC 1951 raw DEFLATE:
//!   - Non-compressed blocks (BTYPE=00)
//!   - Fixed Huffman (BTYPE=01)
//!   - Dynamic Huffman (BTYPE=10)
//!
//! Designed for NPZ/ZIP compression/decompression.
//! Exports:
//!   inflate_raw(src, src_len, dst, dst_cap) -> bytes_written or 0 on error
//!   deflate_raw(src, src_len, dst, dst_cap) -> bytes_written or 0 on error

// ──────────────────────── Bit reader ────────────────────────

const BitReader = struct {
    src: [*]const u8,
    src_len: usize,
    pos: usize, // byte position
    bit_pos: u8, // bit position within current byte (0..7)

    fn init(src: [*]const u8, src_len: usize) BitReader {
        return .{ .src = src, .src_len = src_len, .pos = 0, .bit_pos = 0 };
    }

    fn readBits(self: *BitReader, count: u5) ?u32 {
        var result: u32 = 0;
        var shift: u5 = 0;
        var remaining: u8 = count;
        while (remaining > 0) {
            if (self.pos >= self.src_len) return null;
            const avail: u8 = 8 - self.bit_pos;
            const take: u8 = if (remaining < avail) remaining else avail;
            const take5: u5 = @intCast(take);
            const mask = (@as(u32, 1) << take5) - 1;
            result |= ((@as(u32, self.src[self.pos]) >> @intCast(self.bit_pos)) & mask) << shift;
            shift += take5;
            remaining -= take;
            self.bit_pos += take;
            if (self.bit_pos >= 8) {
                self.bit_pos = 0;
                self.pos += 1;
            }
        }
        return result;
    }

    fn alignToByte(self: *BitReader) void {
        if (self.bit_pos != 0) {
            self.bit_pos = 0;
            self.pos += 1;
        }
    }

    fn readU16LE(self: *BitReader) ?u16 {
        if (self.pos + 2 > self.src_len) return null;
        const lo = @as(u16, self.src[self.pos]);
        const hi = @as(u16, self.src[self.pos + 1]);
        self.pos += 2;
        return lo | (hi << 8);
    }
};

// ──────────────────────── Huffman decoder ────────────────────────

const MAX_BITS = 15;
const MAX_LIT = 288;
const MAX_DIST = 32;

const HuffTable = struct {
    counts: [MAX_BITS + 1]u16, // number of codes of each length
    symbols: [MAX_LIT + MAX_DIST]u16, // sorted symbols
    num_symbols: u16,
};

fn buildHuffTable(table: *HuffTable, lengths: []const u8, num_symbols: u16) void {
    table.num_symbols = num_symbols;

    // Count code lengths
    for (&table.counts) |*c| c.* = 0;
    for (0..num_symbols) |i| {
        if (lengths[i] > 0) {
            table.counts[lengths[i]] += 1;
        }
    }

    // Compute offsets for sorting
    var offsets: [MAX_BITS + 1]u16 = undefined;
    offsets[0] = 0;
    offsets[1] = 0;
    for (2..MAX_BITS + 1) |i| {
        offsets[i] = offsets[i - 1] + table.counts[i - 1];
    }

    // Fill sorted symbols
    for (0..num_symbols) |i| {
        if (lengths[i] > 0) {
            table.symbols[offsets[lengths[i]]] = @intCast(i);
            offsets[lengths[i]] += 1;
        }
    }
}

fn decodeSymbol(reader: *BitReader, table: *const HuffTable) ?u16 {
    // Standard canonical Huffman decode:
    // DEFLATE sends bits LSB-first but Huffman codes are MSB-first.
    // We read one bit at a time and build code MSB-first: code = (code << 1) | bit
    var code: i32 = 0;
    var first: i32 = 0;
    var idx: u32 = 0;

    for (1..MAX_BITS + 1) |len_usize| {
        const bit: i32 = @intCast(reader.readBits(1) orelse return null);
        code = (code << 1) | bit;
        const count: i32 = table.counts[len_usize];
        if (code - first < count) {
            return table.symbols[@intCast(idx + @as(u32, @intCast(code - first)))];
        }
        idx += @intCast(count);
        first = (first + count) << 1;
    }
    return null;
}

// ──────────────────────── Fixed Huffman tables ────────────────────────

var fixed_lit_table: HuffTable = undefined;
var fixed_dist_table: HuffTable = undefined;
var fixed_tables_built = false;

fn ensureFixedTables() void {
    if (fixed_tables_built) return;
    // Literal/length: 0-143 → 8 bits, 144-255 → 9 bits, 256-279 → 7 bits, 280-287 → 8 bits
    var lit_lens: [288]u8 = undefined;
    for (0..144) |i| lit_lens[i] = 8;
    for (144..256) |i| lit_lens[i] = 9;
    for (256..280) |i| lit_lens[i] = 7;
    for (280..288) |i| lit_lens[i] = 8;
    buildHuffTable(&fixed_lit_table, &lit_lens, 288);

    // Distance: all 5 bits
    var dist_lens: [32]u8 = undefined;
    for (&dist_lens) |*d| d.* = 5;
    buildHuffTable(&fixed_dist_table, &dist_lens, 32);

    fixed_tables_built = true;
}

// ──────────────────────── Length/distance tables ────────────────────────

const LEN_BASE = [_]u16{
    3,   4,   5,   6,   7,   8,   9,   10,  11,  13,
    15,  17,  19,  23,  27,  31,  35,  43,  51,  59,
    67,  83,  99,  115, 131, 163, 195, 227, 258,
};
const LEN_EXTRA = [_]u4{
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 0,
};
const DIST_BASE = [_]u16{
    1,    2,    3,    4,    5,    7,    9,    13,   17,   25,
    33,   49,   65,   97,   129,  193,  257,  385,  513,  769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
};
const DIST_EXTRA = [_]u4{
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
};

// Code length order for dynamic tables
const CL_ORDER = [_]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

// ──────────────────────── Inflate engine ────────────────────────

fn inflateBlock(
    reader: *BitReader,
    lit_table: *const HuffTable,
    dist_table: *const HuffTable,
    dst: [*]u8,
    dst_cap: usize,
    out_pos: *usize,
) bool {
    while (true) {
        const sym = decodeSymbol(reader, lit_table) orelse return false;

        if (sym < 256) {
            // Literal byte
            if (out_pos.* >= dst_cap) return false;
            dst[out_pos.*] = @intCast(sym);
            out_pos.* += 1;
        } else if (sym == 256) {
            // End of block
            return true;
        } else {
            // Length/distance pair
            const len_idx = sym - 257;
            if (len_idx >= LEN_BASE.len) return false;

            var length: usize = LEN_BASE[len_idx];
            const extra_len = LEN_EXTRA[len_idx];
            if (extra_len > 0) {
                const extra = reader.readBits(@intCast(extra_len)) orelse return false;
                length += extra;
            }

            const dist_sym = decodeSymbol(reader, dist_table) orelse return false;
            if (dist_sym >= DIST_BASE.len) return false;

            var distance: usize = DIST_BASE[dist_sym];
            const extra_dist = DIST_EXTRA[dist_sym];
            if (extra_dist > 0) {
                const extra = reader.readBits(@intCast(extra_dist)) orelse return false;
                distance += extra;
            }

            // Copy from back-reference
            if (distance > out_pos.*) return false;
            if (out_pos.* + length > dst_cap) return false;

            const src_start = out_pos.* - distance;
            for (0..length) |i| {
                dst[out_pos.* + i] = dst[src_start + (i % distance)];
            }
            out_pos.* += length;
        }
    }
}

fn decodeDynamicTables(
    reader: *BitReader,
    lit_table: *HuffTable,
    dist_table: *HuffTable,
) bool {
    const hlit = (reader.readBits(5) orelse return false) + 257;
    const hdist = (reader.readBits(5) orelse return false) + 1;
    const hclen = (reader.readBits(4) orelse return false) + 4;

    // Read code length code lengths
    var cl_lens: [19]u8 = [_]u8{0} ** 19;
    for (0..hclen) |i| {
        cl_lens[CL_ORDER[i]] = @intCast(reader.readBits(3) orelse return false);
    }

    var cl_table: HuffTable = undefined;
    buildHuffTable(&cl_table, &cl_lens, 19);

    // Decode literal/length + distance code lengths
    var all_lens: [MAX_LIT + MAX_DIST]u8 = undefined;
    const total = hlit + hdist;
    var i: usize = 0;
    while (i < total) {
        const sym = decodeSymbol(reader, &cl_table) orelse return false;

        if (sym < 16) {
            all_lens[i] = @intCast(sym);
            i += 1;
        } else if (sym == 16) {
            // Copy previous 3-6 times
            if (i == 0) return false;
            const rep = (reader.readBits(2) orelse return false) + 3;
            const prev = all_lens[i - 1];
            for (0..rep) |_| {
                if (i >= total) break;
                all_lens[i] = prev;
                i += 1;
            }
        } else if (sym == 17) {
            // Repeat 0, 3-10 times
            const rep = (reader.readBits(3) orelse return false) + 3;
            for (0..rep) |_| {
                if (i >= total) break;
                all_lens[i] = 0;
                i += 1;
            }
        } else if (sym == 18) {
            // Repeat 0, 11-138 times
            const rep = (reader.readBits(7) orelse return false) + 11;
            for (0..rep) |_| {
                if (i >= total) break;
                all_lens[i] = 0;
                i += 1;
            }
        } else {
            return false;
        }
    }

    buildHuffTable(lit_table, all_lens[0..hlit], @intCast(hlit));
    buildHuffTable(dist_table, all_lens[hlit..total], @intCast(hdist));
    return true;
}

// ──────────────────────── Exported API ────────────────────────

/// Inflate raw DEFLATE data. Returns number of bytes written, or 0 on error.
export fn inflate_raw(src: [*]const u8, src_len: u32, dst: [*]u8, dst_cap: u32) u32 {
    var reader = BitReader.init(src, src_len);
    var out_pos: usize = 0;

    while (true) {
        const bfinal = reader.readBits(1) orelse return 0;
        const btype = reader.readBits(2) orelse return 0;

        if (btype == 0) {
            // Non-compressed block
            reader.alignToByte();
            const len = reader.readU16LE() orelse return 0;
            _ = reader.readU16LE() orelse return 0; // nlen (complement, skip)

            if (reader.pos + len > reader.src_len) return 0;
            if (out_pos + len > dst_cap) return 0;

            for (0..len) |i| {
                dst[out_pos + i] = reader.src[reader.pos + i];
            }
            reader.pos += len;
            out_pos += len;
        } else if (btype == 1) {
            // Fixed Huffman
            ensureFixedTables();
            if (!inflateBlock(&reader, &fixed_lit_table, &fixed_dist_table, dst, dst_cap, &out_pos)) return 0;
        } else if (btype == 2) {
            // Dynamic Huffman
            var lit_table: HuffTable = undefined;
            var dist_table: HuffTable = undefined;
            if (!decodeDynamicTables(&reader, &lit_table, &dist_table)) return 0;
            if (!inflateBlock(&reader, &lit_table, &dist_table, dst, dst_cap, &out_pos)) return 0;
        } else {
            return 0; // Reserved block type
        }

        if (bfinal == 1) break;
    }

    return @intCast(out_pos);
}

// ──────────────────────── CRC-32 (for ZIP validation) ────────────────────────

var crc_table: [256]u32 = undefined;
var crc_table_built = false;

fn ensureCrcTable() void {
    if (crc_table_built) return;
    for (0..256) |i| {
        var c: u32 = @intCast(i);
        for (0..8) |_| {
            c = if (c & 1 != 0) 0xEDB88320 ^ (c >> 1) else c >> 1;
        }
        crc_table[i] = c;
    }
    crc_table_built = true;
}

/// CRC-32 checksum of data[0..len].
export fn crc32(data: [*]const u8, len: u32) u32 {
    ensureCrcTable();
    var crc: u32 = 0xFFFFFFFF;
    for (0..len) |i| {
        crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

// ──────────────────────── Bit writer ────────────────────────

const BitWriter = struct {
    dst: [*]u8,
    dst_cap: usize,
    pos: usize,
    bit_buf: u32,
    bit_count: u5,

    fn init(dst: [*]u8, dst_cap: usize) BitWriter {
        return .{ .dst = dst, .dst_cap = dst_cap, .pos = 0, .bit_buf = 0, .bit_count = 0 };
    }

    fn writeBits(self: *BitWriter, value: u32, count: u5) bool {
        // DEFLATE writes bits LSB-first
        self.bit_buf |= value << self.bit_count;
        self.bit_count += count;
        while (self.bit_count >= 8) {
            if (self.pos >= self.dst_cap) return false;
            self.dst[self.pos] = @intCast(self.bit_buf & 0xFF);
            self.pos += 1;
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
        return true;
    }

    fn flush(self: *BitWriter) bool {
        if (self.bit_count > 0) {
            if (self.pos >= self.dst_cap) return false;
            self.dst[self.pos] = @intCast(self.bit_buf & 0xFF);
            self.pos += 1;
            self.bit_buf = 0;
            self.bit_count = 0;
        }
        return true;
    }
};

// ──────────────────────── Fixed Huffman encoder tables ────────────────────────

// Reverse `count` bits of `code` (Huffman codes are MSB-first, DEFLATE writes LSB-first)
fn reverseBits(code: u16, count: u5) u16 {
    var result: u16 = 0;
    var c = code;
    for (0..count) |_| {
        result = (result << 1) | (c & 1);
        c >>= 1;
    }
    return result;
}

// Get fixed Huffman code + length for a literal/length symbol (0..287)
fn fixedLitCode(sym: u16) struct { code: u16, len: u5 } {
    if (sym <= 143) {
        // 0-143: 8-bit codes starting at 00110000 (48)
        return .{ .code = reverseBits(sym + 0x30, 8), .len = 8 };
    } else if (sym <= 255) {
        // 144-255: 9-bit codes starting at 110010000 (400)
        return .{ .code = reverseBits(sym - 144 + 0x190, 9), .len = 9 };
    } else if (sym <= 279) {
        // 256-279: 7-bit codes starting at 0000000 (0)
        return .{ .code = reverseBits(sym - 256, 7), .len = 7 };
    } else {
        // 280-287: 8-bit codes starting at 11000000 (192)
        return .{ .code = reverseBits(sym - 280 + 0xC0, 8), .len = 8 };
    }
}

// Get fixed Huffman code for a distance symbol (0..29) — all 5-bit
fn fixedDistCode(sym: u16) u16 {
    return reverseBits(sym, 5);
}

// Find length symbol + extra bits for a given match length (3..258)
fn encodeLengthSym(length: usize) struct { sym: u16, extra: u32, extra_bits: u5 } {
    // Binary search through LEN_BASE
    var lo: usize = 0;
    var hi: usize = LEN_BASE.len - 1;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (LEN_BASE[mid] <= length) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const idx = lo;
    const extra: u32 = @intCast(length - LEN_BASE[idx]);
    const extra_bits: u5 = @intCast(@as(u8, LEN_EXTRA[idx]));
    return .{ .sym = @intCast(idx + 257), .extra = extra, .extra_bits = extra_bits };
}

// Find distance symbol + extra bits for a given distance (1..32768)
fn encodeDistSym(distance: usize) struct { sym: u16, extra: u32, extra_bits: u5 } {
    var lo: usize = 0;
    var hi: usize = DIST_BASE.len - 1;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (DIST_BASE[mid] <= distance) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const idx = lo;
    const extra: u32 = @intCast(distance - DIST_BASE[idx]);
    const extra_bits: u5 = @intCast(@as(u8, DIST_EXTRA[idx]));
    return .{ .sym = @intCast(idx), .extra = extra, .extra_bits = extra_bits };
}

// ──────────────────────── LZ77 hash chain ────────────────────────

const HASH_BITS = 15;
const HASH_SIZE = 1 << HASH_BITS;
const WINDOW_SIZE = 32768;
const MIN_MATCH = 3;
const MAX_MATCH = 258;
const MAX_CHAIN = 128; // max chain search depth

// Hash function for 3-byte sequences
fn hash3(src: [*]const u8, pos: usize) u32 {
    const a = @as(u32, src[pos]);
    const b = @as(u32, src[pos + 1]);
    const c = @as(u32, src[pos + 2]);
    return ((a << 10) ^ (b << 5) ^ c) & (HASH_SIZE - 1);
}

// ──────────────────────── Deflate engine ────────────────────────

// Scratch memory layout for hash tables (allocated in WASM linear memory)
const SCRATCH_SIZE = HASH_SIZE * 4 + WINDOW_SIZE * 4; // head[HASH_SIZE] + prev[WINDOW_SIZE]

/// Deflate raw data using LZ77 + fixed Huffman. Returns compressed bytes written, or 0 on error.
/// scratch must point to at least SCRATCH_SIZE bytes of writable memory.
export fn deflate_raw(src: [*]const u8, src_len: u32, dst: [*]u8, dst_cap: u32) u32 {
    return deflateImpl(src, src_len, dst, dst_cap, false);
}

/// Query required scratch size (not needed — scratch is internal now).
export fn deflate_scratch_size() u32 {
    return SCRATCH_SIZE;
}

// Internal memory for hash tables (avoids needing external scratch pointer)
var hash_head: [HASH_SIZE]u32 = undefined;
var hash_prev: [WINDOW_SIZE]u32 = undefined;

fn deflateImpl(src: [*]const u8, src_len_u32: u32, dst: [*]u8, dst_cap_u32: u32, store_only: bool) u32 {
    const src_len: usize = src_len_u32;
    const dst_cap: usize = dst_cap_u32;
    _ = store_only;

    var writer = BitWriter.init(dst, dst_cap);

    // Write single final block with fixed Huffman (BFINAL=1, BTYPE=01)
    if (!writer.writeBits(1, 1)) return 0; // BFINAL
    if (!writer.writeBits(1, 2)) return 0; // BTYPE=01 (fixed Huffman)

    // Initialize hash tables
    for (&hash_head) |*h| h.* = 0xFFFFFFFF; // sentinel = no entry
    // hash_prev doesn't need init (only valid entries are followed via head)

    var pos: usize = 0;

    while (pos < src_len) {
        var best_len: usize = 0;
        var best_dist: usize = 0;

        // Try to find a match if enough bytes remain
        if (pos + MIN_MATCH <= src_len) {
            const h = hash3(src, pos);
            var chain_idx = hash_head[h];
            var chain_count: u32 = 0;

            while (chain_idx != 0xFFFFFFFF and chain_count < MAX_CHAIN) : (chain_count += 1) {
                const match_pos: usize = chain_idx;
                if (pos - match_pos > WINDOW_SIZE) break;
                if (match_pos >= pos) break;

                // Check match length
                var mlen: usize = 0;
                const max_len = @min(MAX_MATCH, src_len - pos);
                while (mlen < max_len and src[match_pos + mlen] == src[pos + mlen]) {
                    mlen += 1;
                }

                if (mlen >= MIN_MATCH and mlen > best_len) {
                    best_len = mlen;
                    best_dist = pos - match_pos;
                    if (mlen == MAX_MATCH) break;
                }

                chain_idx = hash_prev[match_pos & (WINDOW_SIZE - 1)];
            }

            // Insert current position into hash chain
            hash_prev[pos & (WINDOW_SIZE - 1)] = hash_head[h];
            hash_head[h] = @intCast(pos);
        }

        if (best_len >= MIN_MATCH) {
            // Emit length/distance pair
            const le = encodeLengthSym(best_len);
            const lc = fixedLitCode(le.sym);
            if (!writer.writeBits(lc.code, lc.len)) return 0;
            if (le.extra_bits > 0) {
                if (!writer.writeBits(le.extra, le.extra_bits)) return 0;
            }

            const de = encodeDistSym(best_dist);
            if (!writer.writeBits(fixedDistCode(de.sym), 5)) return 0;
            if (de.extra_bits > 0) {
                if (!writer.writeBits(de.extra, de.extra_bits)) return 0;
            }

            // Insert skipped positions into hash chain for better future matches
            for (1..best_len) |i| {
                const ipos = pos + i;
                if (ipos + MIN_MATCH <= src_len) {
                    const ih = hash3(src, ipos);
                    hash_prev[ipos & (WINDOW_SIZE - 1)] = hash_head[ih];
                    hash_head[ih] = @intCast(ipos);
                }
            }

            pos += best_len;
        } else {
            // Emit literal
            const lc = fixedLitCode(src[pos]);
            if (!writer.writeBits(lc.code, lc.len)) return 0;
            pos += 1;
        }
    }

    // Emit end-of-block (symbol 256)
    const eob = fixedLitCode(256);
    if (!writer.writeBits(eob.code, eob.len)) return 0;

    // Flush remaining bits
    if (!writer.flush()) return 0;

    return @intCast(writer.pos);
}

// --- Tests ---

test "inflate fixed huffman" {
    const testing = @import("std").testing;
    // "Hello" compressed with fixed Huffman (raw DEFLATE, no zlib header)
    // Generated with: python3 -c "import zlib; print(list(zlib.compress(b'Hello', wbits=-15)))"
    const compressed = [_]u8{ 0xf3, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00 };
    var out: [32]u8 = undefined;
    const n = inflate_raw(&compressed, compressed.len, &out, 32);
    try testing.expectEqual(n, 5);
    try testing.expectEqualSlices(u8, "Hello", out[0..5]);
}

test "crc32 basic" {
    const testing = @import("std").testing;
    const data = "Hello";
    const crc = crc32(data, 5);
    // Known CRC-32 for "Hello"
    try testing.expectEqual(crc, 0xF7D18982);
}

test "deflate then inflate roundtrip" {
    const testing = @import("std").testing;
    const input = "Hello, World! Hello, World! Hello, World! This is a test of DEFLATE compression.";
    var compressed: [256]u8 = undefined;
    const comp_len = deflate_raw(input, input.len, &compressed, 256);
    try testing.expect(comp_len > 0);
    try testing.expect(comp_len < input.len); // should compress

    var decompressed: [256]u8 = undefined;
    const decomp_len = inflate_raw(&compressed, comp_len, &decompressed, 256);
    try testing.expectEqual(decomp_len, input.len);
    try testing.expectEqualSlices(u8, input, decompressed[0..decomp_len]);
}

test "deflate short string roundtrip" {
    const testing = @import("std").testing;
    const input = "AAAAAAAAAA"; // highly compressible
    var compressed: [64]u8 = undefined;
    const comp_len = deflate_raw(input, input.len, &compressed, 64);
    try testing.expect(comp_len > 0);

    var decompressed: [64]u8 = undefined;
    const decomp_len = inflate_raw(&compressed, comp_len, &decompressed, 64);
    try testing.expectEqual(decomp_len, input.len);
    try testing.expectEqualSlices(u8, input, decompressed[0..decomp_len]);
}
