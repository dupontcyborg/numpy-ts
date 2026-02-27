// WASM sort kernels: quicksort for f32/f64
// Median-of-three pivot + insertion sort for small partitions

const INSERTION_THRESHOLD = 16;

// ─── Generic quicksort ──────────────────────────────────────────────────────

fn insertionSort(comptime T: type, data: [*]T, lo: usize, hi: usize) void {
    if (hi <= lo) return;
    var i: usize = lo + 1;
    while (i <= hi) : (i += 1) {
        const key = data[i];
        var j: usize = i;
        while (j > lo and data[j - 1] > key) : (j -= 1) {
            data[j] = data[j - 1];
        }
        data[j] = key;
    }
}

fn medianOfThree(comptime T: type, data: [*]T, lo: usize, hi: usize) usize {
    const mid = lo + (hi - lo) / 2;
    // Sort lo, mid, hi and return mid as pivot index
    if (data[lo] > data[mid]) swap(T, data, lo, mid);
    if (data[lo] > data[hi]) swap(T, data, lo, hi);
    if (data[mid] > data[hi]) swap(T, data, mid, hi);
    return mid;
}

fn swap(comptime T: type, data: [*]T, a: usize, b: usize) void {
    const tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
}

fn partition(comptime T: type, data: [*]T, lo: usize, hi: usize) usize {
    const pivotIdx = medianOfThree(T, data, lo, hi);
    const pivot = data[pivotIdx];
    // Move pivot to hi-1
    swap(T, data, pivotIdx, hi);
    var i: usize = lo;
    var j: usize = if (hi > 0) hi - 1 else 0;
    if (hi == 0) {
        return lo;
    }
    while (true) {
        while (i <= j and data[i] < pivot) : (i += 1) {}
        while (j > i and data[j] > pivot) : (j -= 1) {}
        if (i >= j) break;
        swap(T, data, i, j);
        i += 1;
        if (j > 0) j -= 1;
    }
    // Move pivot to final position
    swap(T, data, i, hi);
    return i;
}

fn quicksort(comptime T: type, data: [*]T, lo: usize, hi: usize) void {
    if (hi <= lo) return;
    if (hi - lo + 1 <= INSERTION_THRESHOLD) {
        insertionSort(T, data, lo, hi);
        return;
    }
    const p = partition(T, data, lo, hi);
    if (p > 0) quicksort(T, data, lo, p -| 1);
    if (p < hi) quicksort(T, data, p + 1, hi);
}

// ─── Exports ────────────────────────────────────────────────────────────────

export fn sort_f64(ptr: [*]f64, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(f64, ptr, 0, len - 1);
}

export fn sort_f32(ptr: [*]f32, n: u32) void {
    const len = @as(usize, n);
    if (len <= 1) return;
    quicksort(f32, ptr, 0, len - 1);
}
