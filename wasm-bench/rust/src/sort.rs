// Sort kernels: quicksort, argsort, partition, argpartition for f32/f64

const INSERTION_THRESHOLD: usize = 16;

// ─── Value-based sort helpers ───────────────────────────────────────────────

unsafe fn insertion_sort<T: PartialOrd + Copy>(data: *mut T, lo: usize, hi: usize) {
    let mut i = lo + 1;
    while i <= hi {
        let key = *data.add(i);
        let mut j = i;
        while j > lo && *data.add(j - 1) > key {
            *data.add(j) = *data.add(j - 1);
            j -= 1;
        }
        *data.add(j) = key;
        i += 1;
    }
}

unsafe fn swap<T: Copy>(data: *mut T, a: usize, b: usize) {
    let tmp = *data.add(a);
    *data.add(a) = *data.add(b);
    *data.add(b) = tmp;
}

unsafe fn median_of_three<T: PartialOrd + Copy>(data: *mut T, lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    if *data.add(lo) > *data.add(mid) { swap(data, lo, mid); }
    if *data.add(lo) > *data.add(hi) { swap(data, lo, hi); }
    if *data.add(mid) > *data.add(hi) { swap(data, mid, hi); }
    mid
}

unsafe fn partition_vals<T: PartialOrd + Copy>(data: *mut T, lo: usize, hi: usize) -> usize {
    let pivot_idx = median_of_three(data, lo, hi);
    let pivot = *data.add(pivot_idx);
    swap(data, pivot_idx, hi);
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };
    loop {
        while i <= j && *data.add(i) < pivot { i += 1; }
        while j > i && *data.add(j) > pivot { j -= 1; }
        if i >= j { break; }
        swap(data, i, j);
        i += 1;
        if j > 0 { j -= 1; }
    }
    swap(data, i, hi);
    i
}

unsafe fn quicksort<T: PartialOrd + Copy>(data: *mut T, lo: usize, hi: usize) {
    if hi <= lo { return; }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort(data, lo, hi);
        return;
    }
    let p = partition_vals(data, lo, hi);
    if p > 0 { quicksort(data, lo, p.saturating_sub(1)); }
    if p < hi { quicksort(data, p + 1, hi); }
}

// ─── Index-based sort helpers (argsort / argpartition) ──────────────────────

unsafe fn swap_u32(data: *mut u32, a: usize, b: usize) {
    let tmp = *data.add(a);
    *data.add(a) = *data.add(b);
    *data.add(b) = tmp;
}

unsafe fn insertion_sort_idx<T: PartialOrd + Copy>(vals: *const T, idx: *mut u32, lo: usize, hi: usize) {
    let mut i = lo + 1;
    while i <= hi {
        let key_idx = *idx.add(i);
        let key_val = *vals.add(key_idx as usize);
        let mut j = i;
        while j > lo && *vals.add(*idx.add(j - 1) as usize) > key_val {
            *idx.add(j) = *idx.add(j - 1);
            j -= 1;
        }
        *idx.add(j) = key_idx;
        i += 1;
    }
}

unsafe fn median_of_three_idx<T: PartialOrd + Copy>(vals: *const T, idx: *mut u32, lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    if *vals.add(*idx.add(lo) as usize) > *vals.add(*idx.add(mid) as usize) { swap_u32(idx, lo, mid); }
    if *vals.add(*idx.add(lo) as usize) > *vals.add(*idx.add(hi) as usize) { swap_u32(idx, lo, hi); }
    if *vals.add(*idx.add(mid) as usize) > *vals.add(*idx.add(hi) as usize) { swap_u32(idx, mid, hi); }
    mid
}

unsafe fn partition_idx<T: PartialOrd + Copy>(vals: *const T, idx: *mut u32, lo: usize, hi: usize) -> usize {
    let pivot_pos = median_of_three_idx(vals, idx, lo, hi);
    let pivot = *vals.add(*idx.add(pivot_pos) as usize);
    swap_u32(idx, pivot_pos, hi);
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };
    loop {
        while i <= j && *vals.add(*idx.add(i) as usize) < pivot { i += 1; }
        while j > i && *vals.add(*idx.add(j) as usize) > pivot { j -= 1; }
        if i >= j { break; }
        swap_u32(idx, i, j);
        i += 1;
        if j > 0 { j -= 1; }
    }
    swap_u32(idx, i, hi);
    i
}

unsafe fn quicksort_idx<T: PartialOrd + Copy>(vals: *const T, idx: *mut u32, lo: usize, hi: usize) {
    if hi <= lo { return; }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort_idx(vals, idx, lo, hi);
        return;
    }
    let p = partition_idx(vals, idx, lo, hi);
    if p > 0 { quicksort_idx(vals, idx, lo, p.saturating_sub(1)); }
    if p < hi { quicksort_idx(vals, idx, p + 1, hi); }
}

unsafe fn quickselect<T: PartialOrd + Copy>(data: *mut T, mut lo: usize, mut hi: usize, kth: usize) {
    while lo < hi {
        if hi - lo + 1 <= INSERTION_THRESHOLD {
            insertion_sort(data, lo, hi);
            return;
        }
        let p = partition_vals(data, lo, hi);
        if p == kth { return; }
        if kth < p { hi = p.saturating_sub(1); } else { lo = p + 1; }
    }
}

unsafe fn quickselect_idx<T: PartialOrd + Copy>(vals: *const T, idx: *mut u32, mut lo: usize, mut hi: usize, kth: usize) {
    while lo < hi {
        if hi - lo + 1 <= INSERTION_THRESHOLD {
            insertion_sort_idx(vals, idx, lo, hi);
            return;
        }
        let p = partition_idx(vals, idx, lo, hi);
        if p == kth { return; }
        if kth < p { hi = p.saturating_sub(1); } else { lo = p + 1; }
    }
}

// ─── Exports ────────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sort_f64(ptr: *mut f64, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    quicksort(ptr, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn sort_f32(ptr: *mut f32, n: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    quicksort(ptr, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_f64(vals: *const f64, idx: *mut u32, n: u32) {
    let len = n as usize;
    for i in 0..len { *idx.add(i) = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(vals, idx, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_f32(vals: *const f32, idx: *mut u32, n: u32) {
    let len = n as usize;
    for i in 0..len { *idx.add(i) = i as u32; }
    if len <= 1 { return; }
    quicksort_idx(vals, idx, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn partition_f64(ptr: *mut f64, n: u32, kth: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    quickselect(ptr, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn partition_f32(ptr: *mut f32, n: u32, kth: u32) {
    let len = n as usize;
    if len <= 1 { return; }
    quickselect(ptr, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn argpartition_f64(vals: *const f64, idx: *mut u32, n: u32, kth: u32) {
    let len = n as usize;
    for i in 0..len { *idx.add(i) = i as u32; }
    if len <= 1 { return; }
    quickselect_idx(vals, idx, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn argpartition_f32(vals: *const f32, idx: *mut u32, n: u32, kth: u32) {
    let len = n as usize;
    for i in 0..len { *idx.add(i) = i as u32; }
    if len <= 1 { return; }
    quickselect_idx(vals, idx, 0, len - 1, kth as usize);
}
