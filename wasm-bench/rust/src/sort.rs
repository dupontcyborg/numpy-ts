// Sort kernels: quicksort for f32/f64
// Manual implementation since [T]::sort_unstable requires alloc

const INSERTION_THRESHOLD: usize = 16;

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
    if *data.add(lo) > *data.add(mid) {
        swap(data, lo, mid);
    }
    if *data.add(lo) > *data.add(hi) {
        swap(data, lo, hi);
    }
    if *data.add(mid) > *data.add(hi) {
        swap(data, mid, hi);
    }
    mid
}

unsafe fn partition<T: PartialOrd + Copy>(data: *mut T, lo: usize, hi: usize) -> usize {
    let pivot_idx = median_of_three(data, lo, hi);
    let pivot = *data.add(pivot_idx);
    swap(data, pivot_idx, hi);
    let mut i = lo;
    let mut j = if hi > 0 { hi - 1 } else { return lo };

    loop {
        while i <= j && *data.add(i) < pivot {
            i += 1;
        }
        while j > i && *data.add(j) > pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        swap(data, i, j);
        i += 1;
        if j > 0 {
            j -= 1;
        }
    }
    swap(data, i, hi);
    i
}

unsafe fn quicksort<T: PartialOrd + Copy>(data: *mut T, lo: usize, hi: usize) {
    if hi <= lo {
        return;
    }
    if hi - lo + 1 <= INSERTION_THRESHOLD {
        insertion_sort(data, lo, hi);
        return;
    }
    let p = partition(data, lo, hi);
    if p > 0 {
        quicksort(data, lo, p.saturating_sub(1));
    }
    if p < hi {
        quicksort(data, p + 1, hi);
    }
}

#[no_mangle]
pub unsafe extern "C" fn sort_f64(ptr: *mut f64, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    quicksort(ptr, 0, len - 1);
}

#[no_mangle]
pub unsafe extern "C" fn sort_f32(ptr: *mut f32, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    quicksort(ptr, 0, len - 1);
}
