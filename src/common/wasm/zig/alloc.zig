//! Segregated free-list allocator for WASM linear memory.
//!
//! Manages a persistent heap region within shared WASM memory. Used by the
//! TypeScript runtime to back ArrayStorage data directly in WASM memory,
//! eliminating copy-in/copy-out overhead for WASM kernel calls.
//!
//! Design: Power-of-2 size classes with per-class free lists.
//!   - Alloc: round up to next power-of-2, pop from that bucket's free list.
//!     If bucket is empty, carve from the bump region. O(1).
//!   - Free: push onto the appropriate bucket's free list. O(1).
//!   - No coalescing, no splitting, no searching.
//!
//! Each allocation is prefixed with a 8-byte header storing the size class index.
//! This allows free() to find the right bucket in O(1).
//!
//! Size classes: 32, 64, 128, 256, ..., 2^32 (4 GiB) = 28 buckets.
//! Covers the full WASM32 address space.

const NUM_BUCKETS: u32 = 28; // size classes: 2^5 (32) through 2^32 (4 GiB) — covers full WASM32 range
const MIN_SIZE_LOG2: u32 = 5; // smallest bucket = 32 bytes
const HEADER_SIZE: u32 = 8; // [size_class_idx: u32, next_free: u32]
const ALIGNMENT: u32 = 8;

// --- Header layout ---
// Offset 0: size_class_idx (u32) — which bucket this block belongs to
// Offset 4: next_free (u32) — next free block in the same bucket (0 = none)
//           Only meaningful when the block is on the free list.

const OFF_CLASS: u32 = 0;
const OFF_NEXT: u32 = 4;

// --- Heap state ---
var heap_start: u32 = 0;
var heap_end: u32 = 0;
var bump_ptr: u32 = 0; // next address for bump allocation (grows upward)
var total_free: u32 = 0;

// --- Segregated free lists ---
// Each bucket is a singly-linked list head (pointer to first free block, or 0).
// Stored as module globals. We use an array of u32 pointers.
// Index 0 = size class 32 bytes, index 1 = 64 bytes, ..., index 27 = 4 GiB.
var buckets: [NUM_BUCKETS]u32 = [_]u32{0} ** NUM_BUCKETS;

// --- Raw memory access helpers ---

fn readU32(addr: u32) u32 {
    const ptr: *align(1) const u32 = @ptrFromInt(addr);
    return ptr.*;
}

fn writeU32(addr: u32, val: u32) void {
    const ptr: *align(1) u32 = @ptrFromInt(addr);
    ptr.* = val;
}

// --- Size class helpers ---

/// Compute the size class index for a given byte size.
/// Returns the index into the buckets array (0..NUM_BUCKETS-1).
fn sizeClassIndex(size: u32) u32 {
    if (size <= (1 << MIN_SIZE_LOG2)) return 0;
    // ceil(log2(size)) - MIN_SIZE_LOG2
    // clz gives leading zeros; 32 - clz(size-1) = ceil(log2(size))
    const log2_ceil = 32 - @clz(size - 1);
    const idx = log2_ceil - MIN_SIZE_LOG2;
    return idx; // may exceed NUM_BUCKETS — caller must check
}

/// Get the block size (usable bytes) for a given size class index.
fn sizeClassBytes(idx: u32) u32 {
    return @as(u32, 1) << @intCast(idx + MIN_SIZE_LOG2);
}

/// Total block size including header, for a given size class.
fn totalBlockSize(idx: u32) u32 {
    return HEADER_SIZE + sizeClassBytes(idx);
}

// --- Exported API ---

/// Initialize the heap. Called once from JS after WASM memory is created.
export fn heap_init(base: u32, size: u32) void {
    heap_start = (base + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    heap_end = (base + size) & ~(ALIGNMENT - 1);
    bump_ptr = heap_start;
    total_free = 0;
    // Clear all bucket heads
    for (0..NUM_BUCKETS) |i| {
        buckets[i] = 0;
    }
}

/// Allocate `req_size` bytes. Returns pointer to usable region, or 0 on OOM.
/// All allocations are 8-byte aligned.
export fn heap_malloc(req_size: u32) u32 {
    if (req_size == 0) return 0;

    const idx = sizeClassIndex(req_size);
    if (idx >= NUM_BUCKETS) return 0; // too large

    // Try to pop from the free list for this size class
    const head = buckets[idx];
    if (head != 0) {
        // Pop from free list
        const next = readU32(head + OFF_NEXT);
        buckets[idx] = next;
        total_free -= sizeClassBytes(idx);
        // Return pointer past the header
        return head + HEADER_SIZE;
    }

    // Free list empty — bump-allocate
    const block_size = totalBlockSize(idx);
    const aligned_ptr = (bump_ptr + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    const new_bump = aligned_ptr + block_size;
    if (new_bump > heap_end) return 0; // OOM

    bump_ptr = new_bump;

    // Write header
    writeU32(aligned_ptr + OFF_CLASS, idx);
    writeU32(aligned_ptr + OFF_NEXT, 0);

    // Return pointer past the header
    return aligned_ptr + HEADER_SIZE;
}

/// Free a previously allocated block. O(1) — push onto the size class free list.
export fn heap_free(ptr: u32) void {
    if (ptr == 0) return;

    const block = ptr - HEADER_SIZE;
    const idx = readU32(block + OFF_CLASS);

    if (idx >= NUM_BUCKETS) return; // invalid

    // Push onto free list head
    writeU32(block + OFF_NEXT, buckets[idx]);
    buckets[idx] = block;
    total_free += sizeClassBytes(idx);
}

/// Reallocate: grow or shrink an existing allocation.
/// If the new size fits in the same size class, returns the same pointer.
/// Otherwise, allocates new, copies, frees old.
export fn heap_realloc(ptr: u32, new_size: u32) u32 {
    if (ptr == 0) return heap_malloc(new_size);
    if (new_size == 0) {
        heap_free(ptr);
        return 0;
    }

    const block = ptr - HEADER_SIZE;
    const old_idx = readU32(block + OFF_CLASS);
    const new_idx = sizeClassIndex(new_size);

    // Same size class — no-op
    if (new_idx == old_idx) return ptr;

    // Different size class — malloc, copy, free
    const new_ptr = heap_malloc(new_size);
    if (new_ptr == 0) return 0;

    const old_usable = sizeClassBytes(old_idx);
    const copy_size = if (old_usable < new_size) old_usable else new_size;
    const src: [*]const u8 = @ptrFromInt(ptr);
    const dst: [*]u8 = @ptrFromInt(new_ptr);
    @memcpy(dst[0..copy_size], src[0..copy_size]);

    heap_free(ptr);
    return new_ptr;
}

/// Return total free bytes across all free lists.
export fn heap_free_bytes() u32 {
    // Free list bytes + remaining bump space
    const bump_remaining = if (bump_ptr < heap_end) heap_end - bump_ptr else 0;
    return total_free + bump_remaining;
}

// --- Tests ---

test "sizeClassIndex basic" {
    const testing = @import("std").testing;
    try testing.expectEqual(sizeClassIndex(1), 0); // ≤32 → bucket 0
    try testing.expectEqual(sizeClassIndex(32), 0); // =32 → bucket 0
    try testing.expectEqual(sizeClassIndex(33), 1); // >32 → bucket 1 (64)
    try testing.expectEqual(sizeClassIndex(64), 1); // =64 → bucket 1
    try testing.expectEqual(sizeClassIndex(65), 2); // >64 → bucket 2 (128)
    try testing.expectEqual(sizeClassIndex(1024), 5); // =1024 → bucket 5
    try testing.expectEqual(sizeClassIndex(80000), 12); // 80000 → bucket 12 (131072)
}

test "sizeClassBytes round-trip" {
    const testing = @import("std").testing;
    try testing.expectEqual(sizeClassBytes(0), 32);
    try testing.expectEqual(sizeClassBytes(1), 64);
    try testing.expectEqual(sizeClassBytes(5), 1024);
    try testing.expectEqual(sizeClassBytes(19), 1 << 24); // 16 MiB
}
