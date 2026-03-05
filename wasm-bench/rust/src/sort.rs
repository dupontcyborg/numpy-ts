// Sort kernels: quicksort, argsort, partition, argpartition for f32/f64

const INSERTION_THRESHOLD: usize = 16;

// ─── Value-based sort helpers (safe, slice-based) ───────────────────────────

fn insertion_sort<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) {
    let mut i = lo + 1;
    while i <= hi {
        let key = data[i];
        let mut j = i;
        while j > lo && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
        i += 1;
    }
}

fn median_of_three<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    if data[lo] > data[mid] {
        data.swap(lo, mid);
    }
    if data[lo] > data[hi] {
        data.swap(lo, hi);
    }
    if data[mid] > data[hi] {
        data.swap(mid, hi);
    }
    mid
}

fn partition_vals<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) -> usize {
    let pivot_idx = median_of_three(data, lo, hi);
    let pivot = data[pivot_idx];
    // Move pivot to hi using direct copy (avoids bounds-checked swap)
    let tmp = data[pivot_idx];
    data[pivot_idx] = data[hi];
    data[hi] = tmp;
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };
    loop {
        while i <= j && data[i] < pivot {
            i += 1;
        }
        while j > i && data[j] > pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        // Direct swap via temp (avoids slice::swap bounds check)
        let t = data[i];
        data[i] = data[j];
        data[j] = t;
        i += 1;
        if j > 0 {
            j -= 1;
        }
    }
    let t = data[i];
    data[i] = data[hi];
    data[hi] = t;
    i
}

fn quicksort<T: PartialOrd + Copy>(data: &mut [T], lo: usize, hi: usize) {
    if hi <= lo {
        return;
    }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort(data, lo, hi);
        return;
    }
    let p = partition_vals(data, lo, hi);
    if p > 0 {
        quicksort(data, lo, p.saturating_sub(1));
    }
    if p < hi {
        quicksort(data, p + 1, hi);
    }
}

// ─── Index-based sort helpers (safe, slice-based) ───────────────────────────

fn insertion_sort_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) {
    let mut i = lo + 1;
    while i <= hi {
        let key_idx = idx[i];
        let key_val = vals[key_idx as usize];
        let mut j = i;
        while j > lo && vals[idx[j - 1] as usize] > key_val {
            idx[j] = idx[j - 1];
            j -= 1;
        }
        idx[j] = key_idx;
        i += 1;
    }
}

fn median_of_three_idx<T: PartialOrd + Copy>(
    vals: &[T],
    idx: &mut [u32],
    lo: usize,
    hi: usize,
) -> usize {
    let mid = lo + (hi - lo) / 2;
    if vals[idx[lo] as usize] > vals[idx[mid] as usize] {
        idx.swap(lo, mid);
    }
    if vals[idx[lo] as usize] > vals[idx[hi] as usize] {
        idx.swap(lo, hi);
    }
    if vals[idx[mid] as usize] > vals[idx[hi] as usize] {
        idx.swap(mid, hi);
    }
    mid
}

fn partition_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) -> usize {
    let pivot_pos = median_of_three_idx(vals, idx, lo, hi);
    let pivot = vals[idx[pivot_pos] as usize];
    let t = idx[pivot_pos];
    idx[pivot_pos] = idx[hi];
    idx[hi] = t;
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };
    loop {
        while i <= j && vals[idx[i] as usize] < pivot {
            i += 1;
        }
        while j > i && vals[idx[j] as usize] > pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        let t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
        i += 1;
        if j > 0 {
            j -= 1;
        }
    }
    let t = idx[i];
    idx[i] = idx[hi];
    idx[hi] = t;
    i
}

fn quicksort_idx<T: PartialOrd + Copy>(vals: &[T], idx: &mut [u32], lo: usize, hi: usize) {
    if hi <= lo {
        return;
    }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort_idx(vals, idx, lo, hi);
        return;
    }
    let p = partition_idx(vals, idx, lo, hi);
    if p > 0 {
        quicksort_idx(vals, idx, lo, p.saturating_sub(1));
    }
    if p < hi {
        quicksort_idx(vals, idx, p + 1, hi);
    }
}

fn quickselect<T: PartialOrd + Copy>(data: &mut [T], mut lo: usize, mut hi: usize, kth: usize) {
    while lo < hi {
        if hi - lo + 1 <= INSERTION_THRESHOLD {
            insertion_sort(data, lo, hi);
            return;
        }
        let p = partition_vals(data, lo, hi);
        if p == kth {
            return;
        }
        if kth < p {
            hi = p.saturating_sub(1);
        } else {
            lo = p + 1;
        }
    }
}

fn quickselect_idx<T: PartialOrd + Copy>(
    vals: &[T],
    idx: &mut [u32],
    mut lo: usize,
    mut hi: usize,
    kth: usize,
) {
    while lo < hi {
        if hi - lo + 1 <= INSERTION_THRESHOLD {
            insertion_sort_idx(vals, idx, lo, hi);
            return;
        }
        let p = partition_idx(vals, idx, lo, hi);
        if p == kth {
            return;
        }
        if kth < p {
            hi = p.saturating_sub(1);
        } else {
            lo = p + 1;
        }
    }
}

// ─── Statistics helpers ─────────────────────────────────────────────────

fn linear_interp_f64(data: &mut [f64], frac: f64) -> f64 {
    let n = data.len();
    let idx_f = frac * (n - 1) as f64;
    let lo = libm::floor(idx_f) as usize;
    let hi = if lo + 1 < n { lo + 1 } else { lo };
    let t = idx_f - libm::floor(idx_f);
    quickselect(data, 0, n - 1, lo);
    if hi != lo {
        quickselect(data, lo + 1, n - 1, hi);
    }
    data[lo] * (1.0 - t) + data[hi] * t
}

fn linear_interp_f32(data: &mut [f32], frac: f64) -> f32 {
    let n = data.len();
    let idx_f = frac * (n - 1) as f64;
    let lo = libm::floor(idx_f) as usize;
    let hi = if lo + 1 < n { lo + 1 } else { lo };
    let t = (idx_f - libm::floor(idx_f)) as f32;
    quickselect(data, 0, n - 1, lo);
    if hi != lo {
        quickselect(data, lo + 1, n - 1, hi);
    }
    data[lo] * (1.0 - t) + data[hi] * t
}

// ─── FFI Exports ─────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sort_f64(ptr: *mut f64, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn sort_f32(ptr: *mut f32, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quicksort(data, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_f64(vals: *const f64, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len {
        ix[i] = i as u32;
    }
    if len <= 1 {
        return;
    }
    quicksort_idx(v, ix, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_f32(vals: *const f32, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len {
        ix[i] = i as u32;
    }
    if len <= 1 {
        return;
    }
    quicksort_idx(v, ix, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn partition_f64(ptr: *mut f64, n: u32, kth: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quickselect(data, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn partition_f32(ptr: *mut f32, n: u32, kth: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    quickselect(data, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn argpartition_f64(vals: *const f64, idx: *mut u32, n: u32, kth: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len {
        ix[i] = i as u32;
    }
    if len <= 1 {
        return;
    }
    quickselect_idx(v, ix, 0, len - 1, kth as usize);
}

#[no_mangle]
pub unsafe extern "C" fn argpartition_f32(vals: *const f32, idx: *mut u32, n: u32, kth: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len {
        ix[i] = i as u32;
    }
    if len <= 1 {
        return;
    }
    quickselect_idx(v, ix, 0, len - 1, kth as usize);
}

// ─── Statistics: median, percentile, quantile ───────────────────────────

#[no_mangle]
pub unsafe extern "C" fn median_f64(ptr: *mut f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 {
        return 0.0;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 {
        return data[0];
    }
    linear_interp_f64(data, 0.5)
}
#[no_mangle]
pub unsafe extern "C" fn median_f32(ptr: *mut f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 {
        return 0.0;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 {
        return data[0];
    }
    linear_interp_f32(data, 0.5)
}
#[no_mangle]
pub unsafe extern "C" fn percentile_f64(ptr: *mut f64, n: u32, p: f64) -> f64 {
    let len = n as usize;
    if len == 0 {
        return 0.0;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 {
        return data[0];
    }
    linear_interp_f64(data, p / 100.0)
}
#[no_mangle]
pub unsafe extern "C" fn percentile_f32(ptr: *mut f32, n: u32, p: f64) -> f32 {
    let len = n as usize;
    if len == 0 {
        return 0.0;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 {
        return data[0];
    }
    linear_interp_f32(data, p / 100.0)
}
#[no_mangle]
pub unsafe extern "C" fn quantile_f64(ptr: *mut f64, n: u32, q: f64) -> f64 {
    let len = n as usize;
    if len == 0 {
        return 0.0;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 {
        return data[0];
    }
    linear_interp_f64(data, q)
}
#[no_mangle]
pub unsafe extern "C" fn quantile_f32(ptr: *mut f32, n: u32, q: f64) -> f32 {
    let len = n as usize;
    if len == 0 {
        return 0.0;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    if len == 1 {
        return data[0];
    }
    linear_interp_f32(data, q)
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER SORT — radix/counting sort for O(n) performance
// ═══════════════════════════════════════════════════════════════════════════

// ─── Counting sort for i8 (256 buckets, stack-only) ─────────────────────

unsafe fn counting_sort_i8(data: &mut [i8]) {
    let mut counts = [0u32; 256];
    for &v in data.iter() {
        counts[(v as u8).wrapping_add(128) as usize] += 1;
    }
    let mut pos = 0;
    for bucket in 0..256u32 {
        let mut c = counts[bucket as usize];
        while c > 0 {
            data[pos] = (bucket as u8).wrapping_sub(128) as i8;
            pos += 1;
            c -= 1;
        }
    }
}

unsafe fn counting_argsort_i8(vals: &[i8], idx: &mut [u32]) {
    let mut counts = [0u32; 256];
    for &v in vals.iter() {
        counts[(v as u8).wrapping_add(128) as usize] += 1;
    }
    let mut offsets = [0u32; 256];
    let mut total = 0u32;
    for bucket in 0..256 {
        offsets[bucket] = total;
        total += counts[bucket];
    }
    for i in 0..vals.len() {
        let key = (vals[i] as u8).wrapping_add(128) as usize;
        idx[offsets[key] as usize] = i as u32;
        offsets[key] += 1;
    }
}

// ─── Counting sort for i16 (65536 buckets, ~256KB stack) ────────────────

unsafe fn counting_sort_i16(data: &mut [i16]) {
    let mut counts = [0u32; 65536];
    for &v in data.iter() {
        counts[(v as u16).wrapping_add(32768) as usize] += 1;
    }
    let mut pos = 0;
    for bucket in 0..65536u32 {
        let mut c = counts[bucket as usize];
        while c > 0 {
            data[pos] = (bucket as u16).wrapping_sub(32768) as i16;
            pos += 1;
            c -= 1;
        }
    }
}

unsafe fn counting_argsort_i16(vals: &[i16], idx: &mut [u32]) {
    let mut counts = [0u32; 65536];
    for &v in vals.iter() {
        counts[(v as u16).wrapping_add(32768) as usize] += 1;
    }
    let mut offsets = [0u32; 65536];
    let mut total = 0u32;
    for bucket in 0..65536 {
        offsets[bucket] = total;
        total += counts[bucket];
    }
    for i in 0..vals.len() {
        let key = (vals[i] as u16).wrapping_add(32768) as usize;
        idx[offsets[key] as usize] = i as u32;
        offsets[key] += 1;
    }
}

// ─── LSD Radix sort for i32 (4 passes × 256 buckets) ───────────────────

unsafe fn wasm_alloc_scratch(bytes: usize) -> Option<*mut u8> {
    let pages_needed = (bytes + 65535) / 65536;
    let old_pages = core::arch::wasm32::memory_grow(0, pages_needed);
    if old_pages == usize::MAX {
        return None;
    }
    Some((old_pages * 65536) as *mut u8)
}

const RADIX_THRESHOLD: usize = 2048;

unsafe fn radix_sort_i32(data: *mut i32, len: usize) {
    // For small arrays, quicksort avoids radix overhead
    if len < RADIX_THRESHOLD {
        let s = core::slice::from_raw_parts_mut(data, len);
        quicksort(s, 0, len - 1);
        return;
    }

    let scratch_ptr = match wasm_alloc_scratch(len * 4) {
        Some(p) => p as *mut i32,
        None => {
            let s = core::slice::from_raw_parts_mut(data, len);
            quicksort(s, 0, len - 1);
            return;
        }
    };

    // Flip sign bit: signed → unsigned order
    for i in 0..len {
        *data.add(i) = (*data.add(i) as u32 ^ 0x80000000) as i32;
    }

    // 4-pass LSD radix sort using raw pointers
    let mut src = data;
    let mut dst = scratch_ptr;
    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 256];

        for i in 0..len {
            let key = ((*src.add(i) as u32) >> shift) & 0xFF;
            *counts.get_unchecked_mut(key as usize) += 1;
        }

        let mut offsets = [0u32; 256];
        let mut total = 0u32;
        for bucket in 0..256 {
            *offsets.get_unchecked_mut(bucket) = total;
            total += *counts.get_unchecked(bucket);
        }

        for i in 0..len {
            let v = *src.add(i);
            let key = ((v as u32) >> shift) & 0xFF;
            let off = offsets.get_unchecked_mut(key as usize);
            *dst.add(*off as usize) = v;
            *off += 1;
        }

        let t = src;
        src = dst;
        dst = t;
    }

    // After 4 passes (even), src == data → result is in data
    // Flip sign bit back
    for i in 0..len {
        *data.add(i) = (*data.add(i) as u32 ^ 0x80000000) as i32;
    }
}

unsafe fn radix_argsort_i32(vals: *const i32, idx: *mut u32, len: usize) {
    let scratch_ptr = match wasm_alloc_scratch(len * 4) {
        Some(p) => p as *mut u32,
        None => {
            let v = core::slice::from_raw_parts(vals, len);
            let ix = core::slice::from_raw_parts_mut(idx, len);
            quicksort_idx(v, ix, 0, len - 1);
            return;
        }
    };

    let mut src = idx;
    let mut dst = scratch_ptr;
    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut counts = [0u32; 256];

        for i in 0..len {
            let v = *vals.add(*src.add(i) as usize) as u32 ^ 0x80000000;
            let key = (v >> shift) & 0xFF;
            counts[key as usize] += 1;
        }

        let mut offsets = [0u32; 256];
        let mut total = 0u32;
        for bucket in 0..256 {
            offsets[bucket] = total;
            total += counts[bucket];
        }

        for i in 0..len {
            let idx_val = *src.add(i);
            let v = *vals.add(idx_val as usize) as u32 ^ 0x80000000;
            let key = (v >> shift) & 0xFF;
            *dst.add(offsets[key as usize] as usize) = idx_val;
            offsets[key as usize] += 1;
        }

        let t = src;
        src = dst;
        dst = t;
    }
    // After 4 passes (even), src == idx → result is in idx
}

// ─── Integer sort exports ───────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sort_i32(ptr: *mut i32, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    radix_sort_i32(ptr, len);
}
#[no_mangle]
pub unsafe extern "C" fn sort_i16(ptr: *mut i16, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    counting_sort_i16(data);
}
#[no_mangle]
pub unsafe extern "C" fn sort_i8(ptr: *mut i8, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    let data = core::slice::from_raw_parts_mut(ptr, len);
    counting_sort_i8(data);
}

#[no_mangle]
pub unsafe extern "C" fn argsort_i32(vals: *const i32, idx: *mut u32, n: u32) {
    let len = n as usize;
    for i in 0..len {
        *idx.add(i) = i as u32;
    }
    if len <= 1 {
        return;
    }
    radix_argsort_i32(vals, idx, len);
}
#[no_mangle]
pub unsafe extern "C" fn argsort_i16(vals: *const i16, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len {
        ix[i] = i as u32;
    }
    if len <= 1 {
        return;
    }
    counting_argsort_i16(v, ix);
}
#[no_mangle]
pub unsafe extern "C" fn argsort_i8(vals: *const i8, idx: *mut u32, n: u32) {
    let len = n as usize;
    let v = core::slice::from_raw_parts(vals, len);
    let ix = core::slice::from_raw_parts_mut(idx, len);
    for i in 0..len {
        ix[i] = i as u32;
    }
    if len <= 1 {
        return;
    }
    counting_argsort_i8(v, ix);
}
