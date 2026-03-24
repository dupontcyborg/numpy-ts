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

// --- Introsort (quicksort + heapsort fallback + insertion sort base) ---

/// Insertion sort for small subarrays. Fast for N ≤ ~32 due to low overhead.
fn insertionSort(comptime T: type, a: [*]T, lo: u32, hi: u32) void {
    var i: u32 = lo + 1;
    while (i <= hi) : (i += 1) {
        const key = a[i];
        var j: u32 = i;
        while (j > lo and lessThan(T, key, a[j - 1])) {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = key;
    }
}

/// Hoare partition: more efficient than Lomuto (fewer swaps on average).
fn hoarePartition(comptime T: type, a: [*]T, lo: u32, hi: u32) u32 {
    const mid = medianOfThree(T, a, lo, hi);
    // After medianOfThree, a[mid] is the median. Use it as pivot.
    const pivot = a[mid];
    var i: u32 = lo;
    var j: u32 = hi;
    while (true) {
        while (lessThan(T, a[i], pivot)) i += 1;
        while (lessThan(T, pivot, a[j])) j -= 1;
        if (i >= j) return j;
        swap(T, a, i, j);
        i += 1;
        j -= 1;
    }
}

/// Floor of log2(n), used to compute introsort depth limit.
fn floorLog2(n: u32) u32 {
    if (n == 0) return 0;
    return 31 - @clz(n);
}

/// Introsort inner loop: quicksort with depth-limited heapsort fallback.
fn introSortImpl(comptime T: type, a: [*]T, lo_arg: u32, hi_arg: u32, depth: u32) void {
    var lo = lo_arg;
    var hi = hi_arg;
    var d = depth;

    while (hi > lo) {
        const size = hi - lo + 1;

        // Base case: insertion sort for small subarrays
        if (size <= 24) {
            insertionSort(T, a, lo, hi);
            return;
        }

        // Depth limit exceeded: fall back to heapsort to guarantee O(N log N)
        if (d == 0) {
            // Sort the subarray [lo..hi] using heapsort
            heapSort(T, a + lo, size);
            return;
        }
        d -= 1;

        const p = hoarePartition(T, a, lo, hi);

        // Recurse on the smaller partition, loop on the larger (tail-call optimization)
        if (p - lo < hi - p) {
            introSortImpl(T, a, lo, p, d);
            lo = p + 1;
        } else {
            introSortImpl(T, a, p + 1, hi, d);
            hi = p;
        }
    }
}

/// In-place introsort: quicksort + heapsort fallback + insertion sort base.
/// O(N log N) worst-case, cache-friendly, fast on real data.
pub fn introSort(comptime T: type, a: [*]T, N: u32) void {
    if (N <= 1) return;
    const depthLimit = 2 * floorLog2(N);
    introSortImpl(T, a, 0, N - 1, depthLimit);
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
/// Automatically selects radix sort for ≤32-bit types, introsort otherwise.
pub fn heapSortSlices(comptime T: type, a: [*]T, sliceSize: u32, numSlices: u32) void {
    // Radix sort wins for small integer types (1–4 byte keys, few passes).
    // For 8-byte types (f64/i64/u64) the 8-pass overhead exceeds introsort at typical slice sizes.
    // For floats, radix's toRadixKey overhead + 4-8 passes loses to introsort at small slice sizes.
    // Radix sort: 1 pass for u8/i8 (counting sort) always wins.
    // For 2-4 byte types, introsort wins at small slice sizes (<256).
    const useRadix = ((T == u8 or T == i8) and sliceSize >= 16) or
        ((T == u16 or T == i16 or T == u32 or T == i32) and sliceSize >= 256);

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
        introSort(T, a + @as(usize, i) * @as(usize, sliceSize), sliceSize);
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

// --- Tests ---

const testing = @import("std").testing;

fn isSorted(comptime T: type, a: []const T) bool {
    if (a.len <= 1) return true;
    var i: usize = 1;
    while (i < a.len) : (i += 1) {
        if (lessThan(T, a[i], a[i - 1])) return false;
    }
    return true;
}

fn isStablySorted(comptime T: type, a: []const T, out: []const u32) bool {
    if (out.len <= 1) return true;
    var i: usize = 1;
    while (i < out.len) : (i += 1) {
        if (!stableLess(T, @ptrCast(a.ptr), out[i - 1], out[i]) and
            stableLess(T, @ptrCast(a.ptr), out[i], out[i - 1])) return false;
    }
    return true;
}

// --- lessThan ---

test "lessThan: f64 normal values" {
    try testing.expect(lessThan(f64, 1.0, 2.0));
    try testing.expect(!lessThan(f64, 2.0, 1.0));
    try testing.expect(!lessThan(f64, 1.0, 1.0));
}

test "lessThan: f64 NaN sorts to end" {
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    try testing.expect(!lessThan(f64, nan, 1.0)); // NaN is NOT less than anything
    try testing.expect(lessThan(f64, 1.0, nan)); // anything IS less than NaN
    try testing.expect(!lessThan(f64, nan, nan)); // NaN vs NaN → false
}

test "lessThan: i32 signed" {
    try testing.expect(lessThan(i32, -5, 3));
    try testing.expect(!lessThan(i32, 3, -5));
    try testing.expect(!lessThan(i32, 0, 0));
}

test "lessThan: u8 unsigned" {
    try testing.expect(lessThan(u8, 0, 255));
    try testing.expect(!lessThan(u8, 128, 127));
}

// --- stableLess ---

test "stableLess: breaks ties by index" {
    const a = [_]f64{ 3.0, 1.0, 3.0, 1.0 };
    try testing.expect(stableLess(f64, &a, 1, 3)); // same value 1.0, idx 1 < 3
    try testing.expect(!stableLess(f64, &a, 3, 1)); // same value 1.0, idx 3 > 1
    try testing.expect(stableLess(f64, &a, 1, 0)); // 1.0 < 3.0
}

test "stableLess: NaN stable tiebreaker" {
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, nan };
    try testing.expect(stableLess(f64, &a, 0, 1)); // both NaN, idx 0 < 1
    try testing.expect(!stableLess(f64, &a, 1, 0));
}

// --- indirectLess ---

test "indirectLess: basic" {
    const a = [_]i32{ 10, 5, 20 };
    try testing.expect(indirectLess(i32, &a, 1, 0)); // a[1]=5 < a[0]=10
    try testing.expect(!indirectLess(i32, &a, 2, 0)); // a[2]=20 > a[0]=10
    try testing.expect(!indirectLess(i32, &a, 0, 0)); // equal → false (no tiebreaker)
}

// --- heapSort ---

test "heapSort: f64 with NaN" {
    var a = [_]f64{ 3.0, @as(f64, @bitCast(@as(u64, 0x7FF8000000000000))), 1.0, 2.0 };
    heapSort(f64, &a, 4);
    try testing.expectApproxEqAbs(a[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(a[2], 3.0, 1e-10);
    try testing.expect(a[3] != a[3]); // NaN at end
}

test "heapSort: i32 with negatives" {
    var a = [_]i32{ 5, -3, 0, -100, 42, 7 };
    heapSort(i32, &a, 6);
    try testing.expect(isSorted(i32, &a));
    try testing.expectEqual(a[0], -100);
    try testing.expectEqual(a[5], 42);
}

test "heapSort: u16 already sorted" {
    var a = [_]u16{ 1, 2, 3, 4, 5 };
    heapSort(u16, &a, 5);
    try testing.expect(isSorted(u16, &a));
}

test "heapSort: i8 reverse sorted" {
    var a = [_]i8{ 127, 50, 0, -50, -128 };
    heapSort(i8, &a, 5);
    try testing.expect(isSorted(i8, &a));
    try testing.expectEqual(a[0], -128);
    try testing.expectEqual(a[4], 127);
}

test "heapSort: single element" {
    var a = [_]f64{42.0};
    heapSort(f64, &a, 1);
    try testing.expectApproxEqAbs(a[0], 42.0, 1e-10);
}

test "heapSort: empty" {
    var a = [_]i32{};
    heapSort(i32, &a, 0);
}

test "heapSort: all equal" {
    var a = [_]u32{ 7, 7, 7, 7 };
    heapSort(u32, &a, 4);
    for (a) |v| try testing.expectEqual(v, 7);
}

// --- heapSortStable ---

test "heapSortStable: preserves order for equal f64" {
    const a = [_]f64{ 3.0, 1.0, 3.0, 1.0, 2.0 };
    var out: [5]u32 = undefined;
    initIndices(&out, 5);
    heapSortStable(f64, &a, &out, 5);
    // 1.0 at indices 1,3 → stable: 1 before 3
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    // 2.0 at index 4
    try testing.expectEqual(out[2], 4);
    // 3.0 at indices 0,2 → stable: 0 before 2
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 2);
}

test "heapSortStable: i32 basic" {
    const a = [_]i32{ 30, -10, 20, 0 };
    var out: [4]u32 = undefined;
    initIndices(&out, 4);
    heapSortStable(i32, &a, &out, 4);
    try testing.expectEqual(a[out[0]], -10);
    try testing.expectEqual(a[out[3]], 30);
}

// --- quickselect ---

test "quickselect: f64 kth=2" {
    var a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    quickselect(f64, &a, 0, 4, 2);
    try testing.expectApproxEqAbs(a[2], 2.0, 1e-10);
    // Elements before kth should be <= a[kth]
    for (0..2) |i| try testing.expect(a[i] <= a[2]);
    for (3..5) |i| try testing.expect(a[i] >= a[2]);
}

test "quickselect: i32 kth=0 (minimum)" {
    var a = [_]i32{ 5, 3, -1, 4, 2 };
    quickselect(i32, &a, 0, 4, 0);
    try testing.expectEqual(a[0], -1);
}

test "quickselect: u8 kth=N-1 (maximum)" {
    var a = [_]u8{ 50, 10, 40, 20, 30 };
    quickselect(u8, &a, 0, 4, 4);
    try testing.expectEqual(a[4], 50);
}

test "quickselect: already sorted data" {
    var a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    quickselect(i32, &a, 0, 7, 4);
    try testing.expectEqual(a[4], 5);
}

test "quickselect: reverse sorted data" {
    var a = [_]i32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    quickselect(i32, &a, 0, 7, 3);
    try testing.expectEqual(a[3], 4);
}

test "quickselect: f64 with NaN" {
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    var a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    quickselect(f64, &a, 0, 4, 2);
    try testing.expectApproxEqAbs(a[2], 3.0, 1e-10);
}

// --- quickselectIndirect ---

test "quickselectIndirect: f64 kth=2" {
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    var out = [_]u32{ 0, 1, 2, 3, 4 };
    quickselectIndirect(f64, &a, &out, 0, 4, 2);
    try testing.expectEqual(a[out[2]], 2.0);
    for (0..2) |i| try testing.expect(a[out[i]] <= a[out[2]]);
    for (3..5) |i| try testing.expect(a[out[i]] >= a[out[2]]);
}

test "quickselectIndirect: i32 kth=0" {
    const a = [_]i32{ 10, -5, 3, 7, -2 };
    var out = [_]u32{ 0, 1, 2, 3, 4 };
    quickselectIndirect(i32, &a, &out, 0, 4, 0);
    try testing.expectEqual(a[out[0]], -5);
}

// --- radixSort ---

test "radixSort: u8 basic" {
    var a = [_]u8{ 255, 0, 128, 1, 127, 64, 200, 50 };
    var scratch: [8]u8 = undefined;
    radixSort(u8, &a, 8, &scratch);
    try testing.expect(isSorted(u8, &a));
    try testing.expectEqual(a[0], 0);
    try testing.expectEqual(a[7], 255);
}

test "radixSort: i8 signed" {
    var a = [_]i8{ 0, -1, 1, -128, 127, -50, 50 };
    var scratch: [7]i8 = undefined;
    radixSort(i8, &a, 7, &scratch);
    try testing.expect(isSorted(i8, &a));
    try testing.expectEqual(a[0], -128);
    try testing.expectEqual(a[6], 127);
}

test "radixSort: u16 basic" {
    var a = [_]u16{ 65535, 0, 256, 255, 1, 32768, 100, 50000 };
    var scratch: [8]u16 = undefined;
    radixSort(u16, &a, 8, &scratch);
    try testing.expect(isSorted(u16, &a));
    try testing.expectEqual(a[0], 0);
    try testing.expectEqual(a[7], 65535);
}

test "radixSort: i16 signed" {
    var a = [_]i16{ 0, -1, 1, -32768, 32767, -100, 100, 0 };
    var scratch: [8]i16 = undefined;
    radixSort(i16, &a, 8, &scratch);
    try testing.expect(isSorted(i16, &a));
    try testing.expectEqual(a[0], -32768);
    try testing.expectEqual(a[7], 32767);
}

test "radixSort: i32 signed" {
    var a = [_]i32{ 1000, -500, 0, 2147483647, -2147483648, 42, -42, 100 };
    var scratch: [8]i32 = undefined;
    radixSort(i32, &a, 8, &scratch);
    try testing.expect(isSorted(i32, &a));
    try testing.expectEqual(a[0], -2147483648);
    try testing.expectEqual(a[7], 2147483647);
}

test "radixSort: u32 basic" {
    var a = [_]u32{ 4294967295, 0, 1, 1000000, 256, 65536 };
    var scratch: [6]u32 = undefined;
    radixSort(u32, &a, 6, &scratch);
    try testing.expect(isSorted(u32, &a));
}

test "radixSort: all equal" {
    var a = [_]u8{ 42, 42, 42, 42 };
    var scratch: [4]u8 = undefined;
    radixSort(u8, &a, 4, &scratch);
    for (a) |v| try testing.expectEqual(v, 42);
}

test "radixSort: single element" {
    var a = [_]i16{-7};
    var scratch: [1]i16 = undefined;
    radixSort(i16, &a, 1, &scratch);
    try testing.expectEqual(a[0], -7);
}

test "radixSort: already sorted" {
    var a = [_]u16{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var scratch: [8]u16 = undefined;
    radixSort(u16, &a, 8, &scratch);
    try testing.expect(isSorted(u16, &a));
}

test "radixSort: reverse sorted" {
    var a = [_]i8{ 100, 50, 0, -50, -100 };
    var scratch: [5]i8 = undefined;
    radixSort(i8, &a, 5, &scratch);
    try testing.expect(isSorted(i8, &a));
}

// --- heapSortSlices ---

test "heapSortSlices: f64 two slices" {
    var a = [_]f64{ 3.0, 1.0, 2.0, 6.0, 4.0, 5.0 };
    heapSortSlices(f64, &a, 3, 2);
    // First slice sorted
    try testing.expectApproxEqAbs(a[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[2], 3.0, 1e-10);
    // Second slice sorted
    try testing.expectApproxEqAbs(a[3], 4.0, 1e-10);
    try testing.expectApproxEqAbs(a[5], 6.0, 1e-10);
}

test "heapSortSlices: i16 uses radix sort path" {
    // 20 elements per slice triggers radix sort (threshold >= 16)
    var a = [_]i16{ 100, -50, 32, -128, 0, 500, -1, 7, 32, 100, -50, 200, 3, -200, 50, 1, -1, 0, 127, -128 };
    heapSortSlices(i16, &a, 20, 1);
    try testing.expect(isSorted(i16, &a));
    try testing.expectEqual(a[0], -200);
    try testing.expectEqual(a[19], 500);
}

test "heapSortSlices: u8 radix sort multiple slices" {
    var a = [_]u8{ 50, 10, 30, 20, 40, 255, 0, 128, 64, 200, 1, 99, 50, 75, 33, 44, 88, 12, 7, 250, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 100, 200, 150, 175, 125, 225, 25, 75, 50, 99 };
    heapSortSlices(u8, &a, 20, 2);
    // First 20 elements sorted
    try testing.expect(isSorted(u8, a[0..20]));
    // Second 20 elements sorted
    try testing.expect(isSorted(u8, a[20..40]));
}

// --- heapSortStableSlices ---

test "heapSortStableSlices: two slices stable" {
    const a = [_]f64{ 2.0, 1.0, 2.0, 5.0, 3.0, 5.0 };
    var out: [6]u32 = undefined;
    heapSortStableSlices(f64, &a, &out, 3, 2);
    // First slice: [2.0, 1.0, 2.0] → indices [1, 0, 2] (stable: 0 before 2 for equal 2.0)
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 2);
    // Second slice: [5.0, 3.0, 5.0] → indices [1, 0, 2]
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
    try testing.expectEqual(out[5], 2);
}

// --- quickselectSlices ---

test "quickselectSlices: two slices" {
    var a = [_]i32{ 5, 3, 1, 4, 2, 10, 6, 8, 7, 9 };
    quickselectSlices(i32, &a, 5, 2, 2);
    // kth=2 in first slice: a[2] should be 3
    try testing.expectEqual(a[2], 3);
    // kth=2 in second slice: a[7] should be 8
    try testing.expectEqual(a[7], 8);
}

// --- quickselectIndirectSlices ---

test "quickselectIndirectSlices: two slices" {
    const a = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0, 10.0, 6.0, 8.0, 7.0, 9.0 };
    var out: [10]u32 = undefined;
    quickselectIndirectSlices(f64, &a, &out, 5, 2, 2);
    // kth=2 in first slice: value at out[2] should be 3.0
    try testing.expectEqual(a[out[2]], 3.0);
    // kth=2 in second slice: value at out[7] should be 8.0
    try testing.expectEqual(a[5 + out[7]], 8.0);
}

// --- lexLess ---

test "lexLess: primary key decides" {
    // key0 (secondary): [1, 2], key1 (primary): [10, 5]
    const keys = [_]i32{ 1, 2, 10, 5 };
    // Element 1 has primary=5 < element 0 primary=10
    try testing.expect(lexLess(i32, &keys, 2, 2, 1, 0));
    try testing.expect(!lexLess(i32, &keys, 2, 2, 0, 1));
}

test "lexLess: tie in primary, secondary decides" {
    // key0 (secondary): [3, 1], key1 (primary): [5, 5]
    const keys = [_]i32{ 3, 1, 5, 5 };
    try testing.expect(lexLess(i32, &keys, 2, 2, 1, 0)); // secondary 1 < 3
}

test "lexLess: all equal, stable tiebreaker" {
    const keys = [_]i32{ 1, 1, 1, 1 };
    try testing.expect(lexLess(i32, &keys, 2, 2, 0, 1)); // idx 0 < 1
    try testing.expect(!lexLess(i32, &keys, 2, 2, 1, 0));
}

test "lexLess: f64 NaN in primary" {
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    // key0: [1.0, 2.0], key1 (primary): [nan, 3.0]
    const keys = [_]f64{ 1.0, 2.0, nan, 3.0 };
    try testing.expect(!lexLess(f64, &keys, 2, 2, 0, 1)); // NaN primary → not less
    try testing.expect(lexLess(f64, &keys, 2, 2, 1, 0)); // 3.0 < NaN
}

// --- complexLess ---

test "complexLess: real part decides" {
    const a = [_]f64{ 1.0, 5.0, 3.0, 2.0 }; // (1+5j), (3+2j)
    try testing.expect(complexLess(f64, &a, 0, 1)); // re 1 < 3
    try testing.expect(!complexLess(f64, &a, 1, 0));
}

test "complexLess: equal real, imag decides" {
    const a = [_]f64{ 3.0, 1.0, 3.0, 5.0 }; // (3+1j), (3+5j)
    try testing.expect(complexLess(f64, &a, 0, 1)); // im 1 < 5
}

test "complexLess: NaN sorts to end" {
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 0.0, 1.0, 2.0 }; // (NaN+0j), (1+2j)
    try testing.expect(!complexLess(f64, &a, 0, 1)); // NaN not less
    try testing.expect(complexLess(f64, &a, 1, 0)); // normal < NaN
}

// --- complexHeapSort ---

test "complexHeapSort: basic" {
    // (3+1j), (1+2j), (2+0j) → sorted: (1+2j), (2+0j), (3+1j)
    var a = [_]f64{ 3.0, 1.0, 1.0, 2.0, 2.0, 0.0 };
    complexHeapSort(f64, &a, 3);
    try testing.expectApproxEqAbs(a[0], 1.0, 1e-10); // re of first
    try testing.expectApproxEqAbs(a[1], 2.0, 1e-10); // im of first
    try testing.expectApproxEqAbs(a[2], 2.0, 1e-10); // re of second
    try testing.expectApproxEqAbs(a[4], 3.0, 1e-10); // re of third
}

test "complexHeapSort: equal real, sort by imag" {
    // (1+3j), (1+1j), (1+2j) → sorted: (1+1j), (1+2j), (1+3j)
    var a = [_]f64{ 1.0, 3.0, 1.0, 1.0, 1.0, 2.0 };
    complexHeapSort(f64, &a, 3);
    try testing.expectApproxEqAbs(a[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(a[5], 3.0, 1e-10);
}

// --- complexHeapSortStable ---

test "complexHeapSortStable: preserves order for equal" {
    // (1+1j), (2+2j), (1+1j) → stable: indices [0, 2, 1]
    const a = [_]f64{ 1.0, 1.0, 2.0, 2.0, 1.0, 1.0 };
    var out: [3]u32 = undefined;
    initIndices(&out, 3);
    complexHeapSortStable(f64, &a, &out, 3);
    try testing.expectEqual(out[0], 0); // first (1+1j)
    try testing.expectEqual(out[1], 2); // second (1+1j), idx 2 > 0
    try testing.expectEqual(out[2], 1); // (2+2j)
}
