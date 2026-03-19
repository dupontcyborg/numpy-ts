//! Shared sorting utilities for all sorting WASM kernels.
//!
//! Comparisons, swaps, heap sort, quickselect, radix sort, and complex
//! sort — parameterized by comptime type for full inlining.

// --- Comparison ---

/// NaN-safe less-than: returns true when a < b. NaN sorts to end (matching NumPy).
pub fn lessThan(comptime T: type, a: T, b: T) bool {
    const is_float = (T == f64 or T == f32);
    if (is_float) {
        if (a != a) return false;
        if (b != b) return true;
    }
    return a < b;
}

/// Stable indirect less-than: compares a[idx_a] vs a[idx_b], breaks ties by index.
pub fn stableLess(comptime T: type, a: [*]const T, idx_a: u32, idx_b: u32) bool {
    const va = a[idx_a];
    const vb = a[idx_b];
    const is_float = (T == f64 or T == f32);
    if (is_float) {
        const a_nan = va != va;
        const b_nan = vb != vb;
        if (a_nan and b_nan) return idx_a < idx_b;
        if (a_nan) return false;
        if (b_nan) return true;
    }
    if (va < vb) return true;
    if (va > vb) return false;
    return idx_a < idx_b;
}

/// Indirect less-than without stability (for quickselect where order doesn't matter).
pub fn indirectLess(comptime T: type, a: [*]const T, idx_a: u32, idx_b: u32) bool {
    const va = a[idx_a];
    const vb = a[idx_b];
    const is_float = (T == f64 or T == f32);
    if (is_float) {
        if (va != va) return false;
        if (vb != vb) return true;
    }
    return va < vb;
}

// --- Swap ---

/// Swap two elements of type T in array a.
pub fn swap(comptime T: type, a: [*]T, i: u32, j: u32) void {
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
}

/// Swap two u32 elements in an index array.
pub fn swapU32(a: [*]u32, i: u32, j: u32) void {
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
}

// --- Index initialisation ---

/// Fill out[0..N] with 0, 1, 2, ..., N-1.
pub fn initIndices(out: [*]u32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = i;
    }
}

// --- Median-of-three pivot selection ---

/// Select median-of-three pivot for value-based quickselect. Sorts lo/mid/hi in place.
pub fn medianOfThree(comptime T: type, a: [*]T, lo: u32, hi: u32) u32 {
    const mid = lo + (hi - lo) / 2;
    if (lessThan(T, a[hi], a[lo])) swap(T, a, lo, hi);
    if (lessThan(T, a[mid], a[lo])) swap(T, a, lo, mid);
    if (lessThan(T, a[hi], a[mid])) swap(T, a, mid, hi);
    return mid;
}

/// Select median-of-three pivot for indirect quickselect. Sorts indices at lo/mid/hi.
pub fn medianOfThreeIndirect(comptime T: type, a: [*]const T, out: [*]u32, lo: u32, hi: u32) u32 {
    const mid = lo + (hi - lo) / 2;
    if (indirectLess(T, a, out[hi], out[lo])) swapU32(out, lo, hi);
    if (indirectLess(T, a, out[mid], out[lo])) swapU32(out, lo, mid);
    if (indirectLess(T, a, out[hi], out[mid])) swapU32(out, mid, hi);
    return mid;
}

// --- Quickselect (value-based, for partition) ---

/// Lomuto partition with median-of-three pivot. Returns pivot's final position.
fn partitionImpl(comptime T: type, a: [*]T, lo: u32, hi: u32) u32 {
    const med = medianOfThree(T, a, lo, hi);
    swap(T, a, med, hi);
    const pivot = a[hi];
    var i: u32 = lo;
    var j: u32 = lo;
    while (j < hi) : (j += 1) {
        if (lessThan(T, a[j], pivot)) {
            swap(T, a, i, j);
            i += 1;
        }
    }
    swap(T, a, i, hi);
    return i;
}

/// Quickselect: rearranges a so that a[kth] is in its sorted position.
pub fn quickselect(comptime T: type, a: [*]T, lo_arg: u32, hi_arg: u32, kth: u32) void {
    var lo = lo_arg;
    var hi = hi_arg;
    while (lo < hi) {
        const pivot = partitionImpl(T, a, lo, hi);
        if (pivot == kth) return;
        if (kth < pivot) {
            hi = pivot - 1;
        } else {
            lo = pivot + 1;
        }
    }
}

// --- Quickselect (indirect, for argpartition) ---

/// Indirect Lomuto partition. Partitions the index array by comparing a[out[i]].
fn partitionImplIndirect(comptime T: type, a: [*]const T, out: [*]u32, lo: u32, hi: u32) u32 {
    const med = medianOfThreeIndirect(T, a, out, lo, hi);
    swapU32(out, med, hi);
    const pivot_idx = out[hi];
    var i: u32 = lo;
    var j: u32 = lo;
    while (j < hi) : (j += 1) {
        if (indirectLess(T, a, out[j], pivot_idx)) {
            swapU32(out, i, j);
            i += 1;
        }
    }
    swapU32(out, i, hi);
    return i;
}

/// Indirect quickselect: rearranges out so that a[out[kth]] is the kth-smallest value.
pub fn quickselectIndirect(comptime T: type, a: [*]const T, out: [*]u32, lo_arg: u32, hi_arg: u32, kth: u32) void {
    var lo = lo_arg;
    var hi = hi_arg;
    while (lo < hi) {
        const pivot = partitionImplIndirect(T, a, out, lo, hi);
        if (pivot == kth) return;
        if (kth < pivot) {
            hi = pivot - 1;
        } else {
            lo = pivot + 1;
        }
    }
}

// --- Heap sort (value-based, for sort) ---

/// Sift-down for a max-heap rooted at `start`, with elements up to `end`.
fn heapSiftDown(comptime T: type, a: [*]T, start: u32, end: u32) void {
    var root = start;
    while (true) {
        var child = 2 * root + 1;
        if (child > end) break;
        if (child + 1 <= end and lessThan(T, a[child], a[child + 1])) {
            child += 1;
        }
        if (lessThan(T, a[root], a[child])) {
            swap(T, a, root, child);
            root = child;
        } else {
            break;
        }
    }
}

/// In-place heap sort: sorts a[0..N] ascending. O(N log N) worst-case.
pub fn heapSort(comptime T: type, a: [*]T, N: u32) void {
    if (N <= 1) return;
    var start: u32 = (N - 2) / 2;
    while (true) {
        heapSiftDown(T, a, start, N - 1);
        if (start == 0) break;
        start -= 1;
    }
    var end: u32 = N - 1;
    while (end > 0) {
        swap(T, a, 0, end);
        end -= 1;
        heapSiftDown(T, a, 0, end);
    }
}

// --- Heap sort (indirect + stable, for argsort) ---

/// Stable sift-down: uses stableLess so equal elements preserve original index order.
fn heapSiftDownStable(comptime T: type, a: [*]const T, out: [*]u32, start: u32, end: u32) void {
    var root = start;
    while (true) {
        var child = 2 * root + 1;
        if (child > end) break;
        if (child + 1 <= end and stableLess(T, a, out[child], out[child + 1])) {
            child += 1;
        }
        if (stableLess(T, a, out[root], out[child])) {
            swapU32(out, root, child);
            root = child;
        } else {
            break;
        }
    }
}

/// Stable indirect heap sort: sorts out[0..N] so a[out[0]] <= a[out[1]] <= ...
/// Equal elements preserve original index order (matching NumPy stable sort).
pub fn heapSortStable(comptime T: type, a: [*]const T, out: [*]u32, N: u32) void {
    if (N <= 1) return;
    var start: u32 = (N - 2) / 2;
    while (true) {
        heapSiftDownStable(T, a, out, start, N - 1);
        if (start == 0) break;
        start -= 1;
    }
    var end: u32 = N - 1;
    while (end > 0) {
        swapU32(out, 0, end);
        end -= 1;
        heapSiftDownStable(T, a, out, 0, end);
    }
}

// --- Lexsort ---

/// Lexicographic comparison for multi-key sort. Last key is primary.
/// Returns true if element a should come before element b. Stable via index tiebreaker.
pub fn lexLess(comptime T: type, keys: [*]const T, num_keys: u32, N: u32, a: u32, b: u32) bool {
    const is_float = (T == f64 or T == f32);
    var k: u32 = num_keys;
    while (k > 0) {
        k -= 1;
        const va = keys[k * N + a];
        const vb = keys[k * N + b];
        if (is_float) {
            const a_nan = va != va;
            const b_nan = vb != vb;
            if (a_nan and b_nan) continue;
            if (a_nan) return false;
            if (b_nan) return true;
        }
        if (va < vb) return true;
        if (va > vb) return false;
    }
    return a < b;
}

/// Sift-down for lexsort max-heap.
fn lexSiftDown(comptime T: type, keys: [*]const T, num_keys: u32, N: u32, out: [*]u32, start: u32, end: u32) void {
    var root = start;
    while (true) {
        var child = 2 * root + 1;
        if (child >= end) break;
        if (child + 1 < end and lexLess(T, keys, num_keys, N, out[child], out[child + 1])) {
            child += 1;
        }
        if (!lexLess(T, keys, num_keys, N, out[root], out[child])) break;
        swapU32(out, root, child);
        root = child;
    }
}

/// Lexicographic heap sort over multi-key flat buffer. Stable.
pub fn lexHeapSort(comptime T: type, keys: [*]const T, num_keys: u32, N: u32, out: [*]u32) void {
    if (N <= 1) return;
    var start: u32 = N / 2;
    while (start > 0) {
        start -= 1;
        lexSiftDown(T, keys, num_keys, N, out, start, N);
    }
    var end: u32 = N;
    while (end > 1) {
        end -= 1;
        swapU32(out, 0, end);
        lexSiftDown(T, keys, num_keys, N, out, 0, end);
    }
}

// --- Radix sort ---

/// Map a value to a radix-sortable unsigned integer.
/// Signed ints: offset so that min maps to 0.
/// Floats: flip bits so IEEE-754 order matches unsigned integer order.
fn toRadixKey(comptime T: type, comptime U: type, v: T) U {
    if (T == f64) {
        const bits: u64 = @bitCast(v);
        return if (bits >> 63 != 0) ~bits else bits ^ (@as(u64, 1) << 63);
    } else if (T == f32) {
        const bits: u32 = @bitCast(v);
        return if (bits >> 31 != 0) ~bits else bits ^ (@as(u32, 1) << 31);
    } else if (T == i64) {
        return @bitCast(v +% @as(i64, @bitCast(@as(u64, 1) << 63)));
    } else if (T == i32) {
        return @bitCast(v +% @as(i32, @bitCast(@as(u32, 1) << 31)));
    } else if (T == i16) {
        return @bitCast(v +% @as(i16, @bitCast(@as(u16, 1) << 15)));
    } else if (T == i8) {
        return @bitCast(v +% @as(i8, @bitCast(@as(u8, 1) << 7)));
    } else {
        return v;
    }
}

/// Unsigned type used as radix key for a given element type.
fn RadixUnsigned(comptime T: type) type {
    return switch (T) {
        f64, u64, i64 => u64,
        f32, u32, i32 => u32,
        u16, i16 => u16,
        u8, i8 => u8,
        else => @compileError("unsupported radix sort type"),
    };
}

/// LSD radix sort: sorts a[0..N] ascending using a scratch buffer.
/// One pass per byte of the element type (1 for u8, 2 for u16, 4 for i32, etc.).
pub fn radixSort(comptime T: type, a: [*]T, N: u32, scratch: [*]T) void {
    if (N <= 1) return;
    const U = RadixUnsigned(T);
    const numBytes = @sizeOf(T);

    var src = a;
    var dst = scratch;

    var pass: u32 = 0;
    while (pass < numBytes) : (pass += 1) {
        var counts = [_]u32{0} ** 256;
        var i: u32 = 0;
        while (i < N) : (i += 1) {
            const key = toRadixKey(T, U, src[i]);
            const keyBytes: [numBytes]u8 = @bitCast(key);
            counts[keyBytes[pass]] += 1;
        }

        // Skip pass if every element has the same byte at this position
        var allSame = false;
        for (&counts) |c| {
            if (c == N) {
                allSame = true;
                break;
            }
        }
        if (allSame) continue;

        // Prefix sum → scatter offsets
        var total: u32 = 0;
        for (&counts) |*c| {
            const count = c.*;
            c.* = total;
            total += count;
        }

        // Scatter src → dst
        i = 0;
        while (i < N) : (i += 1) {
            const key = toRadixKey(T, U, src[i]);
            const keyBytes: [numBytes]u8 = @bitCast(key);
            dst[counts[keyBytes[pass]]] = src[i];
            counts[keyBytes[pass]] += 1;
        }

        const tmp = src;
        src = dst;
        dst = tmp;
    }

    // After an odd number of effective passes the result sits in scratch — copy back
    if (src != a) {
        var i: u32 = 0;
        while (i < N) : (i += 1) {
            a[i] = src[i];
        }
    }
}

// --- Batch slice operations ---

/// Sort numSlices contiguous slices of sliceSize elements each.
/// Automatically selects radix sort for ≤32-bit integer types, heap sort otherwise.
pub fn heapSortSlices(comptime T: type, a: [*]T, sliceSize: u32, numSlices: u32) void {
    // Radix sort wins for small integer types (1–4 byte keys, few passes).
    // For 8-byte types (f64/i64/u64) the 8-pass overhead exceeds heap sort at typical slice sizes.
    const useRadix = (T == u8 or T == i8 or T == u16 or T == i16 or
        T == u32 or T == i32) and sliceSize >= 16;

    if (useRadix) {
        var scratchBuf: [4096]T = undefined;
        if (sliceSize <= 4096) {
            const scratch = &scratchBuf;
            var i: u32 = 0;
            while (i < numSlices) : (i += 1) {
                radixSort(T, a + @as(usize, i) * @as(usize, sliceSize), sliceSize, scratch);
            }
            return;
        }
    }

    var i: u32 = 0;
    while (i < numSlices) : (i += 1) {
        heapSort(T, a + @as(usize, i) * @as(usize, sliceSize), sliceSize);
    }
}

/// Argsort numSlices contiguous slices. Initialises indices and sorts stably.
pub fn heapSortStableSlices(comptime T: type, a: [*]const T, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    var i: u32 = 0;
    while (i < numSlices) : (i += 1) {
        const off = @as(usize, i) * @as(usize, sliceSize);
        const sliceOut = out + off;
        initIndices(sliceOut, sliceSize);
        heapSortStable(T, a + off, sliceOut, sliceSize);
    }
}

/// Partition numSlices contiguous slices at kth position.
pub fn quickselectSlices(comptime T: type, a: [*]T, sliceSize: u32, numSlices: u32, kth: u32) void {
    if (sliceSize <= 1 or kth >= sliceSize) return;
    var i: u32 = 0;
    while (i < numSlices) : (i += 1) {
        quickselect(T, a + @as(usize, i) * @as(usize, sliceSize), 0, sliceSize - 1, kth);
    }
}

/// Argpartition numSlices contiguous slices at kth position.
pub fn quickselectIndirectSlices(comptime T: type, a: [*]const T, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    if (sliceSize <= 1 or kth >= sliceSize) {
        var j: u32 = 0;
        while (j < numSlices * sliceSize) : (j += 1) {
            out[j] = j % sliceSize;
        }
        return;
    }
    var i: u32 = 0;
    while (i < numSlices) : (i += 1) {
        const off = @as(usize, i) * @as(usize, sliceSize);
        const sliceOut = out + off;
        initIndices(sliceOut, sliceSize);
        quickselectIndirect(T, a + off, sliceOut, 0, sliceSize - 1, kth);
    }
}

// --- Complex comparison ---

/// Lexicographic less-than for complex numbers stored as interleaved [re, im] pairs.
/// NaN (in either component) sorts to end.
pub fn complexLess(comptime T: type, a: [*]const T, i: u32, j: u32) bool {
    const a_re = a[@as(usize, i) * 2];
    const a_im = a[@as(usize, i) * 2 + 1];
    const b_re = a[@as(usize, j) * 2];
    const b_im = a[@as(usize, j) * 2 + 1];
    const a_nan = (a_re != a_re) or (a_im != a_im);
    const b_nan = (b_re != b_re) or (b_im != b_im);
    if (a_nan and b_nan) return false;
    if (a_nan) return false;
    if (b_nan) return true;
    if (a_re < b_re) return true;
    if (a_re > b_re) return false;
    if (a_im < b_im) return true;
    if (a_im > b_im) return false;
    return false;
}

/// Stable lexicographic less-than for complex argsort. Breaks ties by index.
pub fn complexStableLess(comptime T: type, a: [*]const T, idx_a: u32, idx_b: u32) bool {
    const a_re = a[@as(usize, idx_a) * 2];
    const a_im = a[@as(usize, idx_a) * 2 + 1];
    const b_re = a[@as(usize, idx_b) * 2];
    const b_im = a[@as(usize, idx_b) * 2 + 1];
    const a_nan = (a_re != a_re) or (a_im != a_im);
    const b_nan = (b_re != b_re) or (b_im != b_im);
    if (a_nan and b_nan) return idx_a < idx_b;
    if (a_nan) return false;
    if (b_nan) return true;
    if (a_re < b_re) return true;
    if (a_re > b_re) return false;
    if (a_im < b_im) return true;
    if (a_im > b_im) return false;
    return idx_a < idx_b;
}

// --- Complex heap sort ---

/// Swap two complex elements (each stored as two consecutive floats).
fn complexSwap(comptime T: type, a: [*]T, p: u32, q: u32) void {
    const p2 = @as(usize, p) * 2;
    const q2 = @as(usize, q) * 2;
    var tmp = a[p2];
    a[p2] = a[q2];
    a[q2] = tmp;
    tmp = a[p2 + 1];
    a[p2 + 1] = a[q2 + 1];
    a[q2 + 1] = tmp;
}

/// Sift-down for complex max-heap.
fn complexSiftDown(comptime T: type, a: [*]T, start: u32, end: u32) void {
    var root = start;
    while (true) {
        var child = 2 * root + 1;
        if (child > end) break;
        if (child + 1 <= end and complexLess(T, a, child, child + 1)) {
            child += 1;
        }
        if (complexLess(T, a, root, child)) {
            complexSwap(T, a, root, child);
            root = child;
        } else {
            break;
        }
    }
}

/// Heap sort for complex arrays. N = number of complex elements (buffer has 2*N floats).
pub fn complexHeapSort(comptime T: type, a: [*]T, N: u32) void {
    if (N <= 1) return;
    var start: u32 = (N - 2) / 2;
    while (true) {
        complexSiftDown(T, a, start, N - 1);
        if (start == 0) break;
        start -= 1;
    }
    var end: u32 = N - 1;
    while (end > 0) {
        complexSwap(T, a, 0, end);
        end -= 1;
        complexSiftDown(T, a, 0, end);
    }
}

/// Stable sift-down for complex argsort.
fn complexStableSiftDown(comptime T: type, a: [*]const T, out: [*]u32, start: u32, end: u32) void {
    var root = start;
    while (true) {
        var child = 2 * root + 1;
        if (child > end) break;
        if (child + 1 <= end and complexStableLess(T, a, out[child], out[child + 1])) {
            child += 1;
        }
        if (complexStableLess(T, a, out[root], out[child])) {
            swapU32(out, root, child);
            root = child;
        } else {
            break;
        }
    }
}

/// Stable argsort for complex arrays. Equal elements preserve original index order.
pub fn complexHeapSortStable(comptime T: type, a: [*]const T, out: [*]u32, N: u32) void {
    if (N <= 1) return;
    var start: u32 = (N - 2) / 2;
    while (true) {
        complexStableSiftDown(T, a, out, start, N - 1);
        if (start == 0) break;
        start -= 1;
    }
    var end: u32 = N - 1;
    while (end > 0) {
        swapU32(out, 0, end);
        end -= 1;
        complexStableSiftDown(T, a, out, 0, end);
    }
}

/// Batch complex sort: sort numSlices contiguous slices of sliceSize complex elements.
pub fn complexHeapSortSlices(comptime T: type, a: [*]T, sliceSize: u32, numSlices: u32) void {
    var i: u32 = 0;
    while (i < numSlices) : (i += 1) {
        complexHeapSort(T, a + @as(usize, i) * @as(usize, sliceSize) * 2, sliceSize);
    }
}

/// Batch complex argsort: argsort numSlices contiguous slices.
pub fn complexHeapSortStableSlices(comptime T: type, a: [*]const T, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    var i: u32 = 0;
    while (i < numSlices) : (i += 1) {
        const off = @as(usize, i) * @as(usize, sliceSize);
        const sliceOut = out + off;
        initIndices(sliceOut, sliceSize);
        complexHeapSortStable(T, a + off * 2, sliceOut, sliceSize);
    }
}
