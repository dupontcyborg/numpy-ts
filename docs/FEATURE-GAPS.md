# Feature Gaps Analysis: numpy-ts vs NumPy

This document tracks **features** (not API functions) that are missing or incomplete compared to NumPy. API function coverage is tracked separately.

## Overview

numpy-ts implements core NumPy functionality but lacks several deeper features. This document categorizes gaps by implementation scope and priority.

---

## Major Feature Gaps

### 1. Universal Functions (ufuncs) System
**Status: Not Implemented**
**Scope: MAJOR - Foundational NumPy concept**

NumPy's universal function (ufunc) system is **completely absent**. We have functions that perform similar operations, but none of the ufunc infrastructure.

**What's Missing:**

| Category | Missing Features |
|----------|-----------------|
| **ufunc objects** | No `np.ufunc` class, no ufunc instances |
| **ufunc methods** | `.reduce()`, `.accumulate()`, `.reduceat()`, `.outer()`, `.at()` |
| **ufunc creation** | `np.frompyfunc()`, `np.vectorize()` |
| **ufunc parameters** | `out`, `where`, `dtype`, `casting`, `order`, `subok` |
| **ufunc protocol** | `__array_ufunc__` for custom array types |

**Current State:**
- Operations like `add`, `multiply`, `sin`, etc. are plain functions
- No way to pre-allocate output arrays (`out` parameter)
- No conditional application (`where` parameter)
- No control over output dtype in operations
- No way to create custom vectorized functions
- No `.reduce()` on operations (must use separate `sum()`, `prod()` etc.)

**Impact:**
- Memory inefficiency: Cannot reuse arrays, must allocate new arrays for every operation
- Missing functionality: Cannot do `np.add.reduce()`, `np.multiply.outer()`, etc.
- No extensibility: Cannot create custom ufuncs
- No interoperability: Cannot implement `__array_ufunc__` protocol

**Implementation Notes:**
This would be a major architectural addition:
- New `Ufunc` class with methods
- Wrap all existing operations as ufunc instances
- Add parameter handling to all operations (50+ functions)
- Estimate: New subsystem, 1000+ lines, touches all operation files

---

### 2. Complex Number Data Types
**Status: Not Supported**
**Scope: MAJOR - Touches almost everything**

Missing:
- `complex64` and `complex128` dtypes
- `.real` and `.imag` properties
- Functions: `angle`, `conj`, `real`, `imag`, `iscomplex`, `isreal`, `iscomplexobj`, `isrealobj`
- Complex arithmetic in all operations
- Complex array I/O

**Implementation Notes:**
- No native JS complex array type - need interleaved Float32/64Array or custom wrapper
- Every arithmetic, comparison, trig, and reduction operation needs complex variants
- dtype.ts, storage.ts, all ops files, npy I/O affected
- Estimate: 15+ files, 50+ functions

**Blocked Features:** FFT module requires complex number support.

---

### 3. Structured Arrays / Record Arrays
**Status: Not Supported**
**Scope: MAJOR - New subsystem**

Missing:
- Compound dtypes (e.g., `dtype=[('x', 'f4'), ('y', 'f4')]`)
- Field access by name
- Structured data in .npy files
- `recarray` class

**Implementation Notes:**
- Requires new dtype representation for compound types
- Field offset calculations, named field access in storage
- Estimate: 5+ files, 20+ functions, architectural design needed

---

### 4. Advanced Indexing (Fancy Indexing)
**Status: ✅ Implemented**
**Scope: MODERATE - Concentrated in indexing**

Implemented via two new methods:
- `iindex(indices, axis?)` - Integer array indexing (NumPy's `arr[[0, 2, 4]]`)
- `bindex(mask, axis?)` - Boolean array indexing (NumPy's `arr[arr > 5]`)

Both methods support axis specification and work with existing `take()` and `compress()` internally.

---

## Moderate Feature Gaps

### 5. Tuple of Axes for Reductions
**Status: Not Supported**
**Scope: SMALL-MODERATE - Localized to reductions**

Reductions only accept single axis, not tuple (e.g., `axis=(0, 1)`).

Required Changes:
- reduction.ts: Modify `sum()`, `mean()`, `max()`, `min()`, `prod()`, `std()`, `var()`, `any()`, `all()` to accept `number | number[]`
- More complex keepdims handling

**Estimate:** 1 file, ~15 functions

---

### 6. Polynomial Module
**Status: Not Implemented**
**Scope: MODERATE - New module**

Missing: `polyval`, `polyfit`, `polyder`, `polyint`, `poly1d` class, `roots`

**Estimate:** New module, 300-500 lines, self-contained, no blockers

---

### 7. Error State Control
**Status: Not Implemented**
**Scope: SMALL - New utility**

Missing:
- `np.seterr()` / `np.geterr()`
- `np.errstate()` context manager
- Control over divide-by-zero, overflow, underflow, invalid handling

**Estimate:** 1 new file, ~100-200 lines

---

## Large New Subsystems

### 8. FFT Module
**Status: Not Implemented**
**Scope: LARGE - New module**
**Blocker: Requires complex number support**

Missing: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `fftfreq`, `fftshift`

**Estimate:** 500-1000 lines once complex numbers are available

---

### 9. Masked Arrays (`np.ma`)
**Status: Not Implemented**
**Scope: LARGE - New subsystem**

Missing:
- `MaskedArray` class
- Mask tracking alongside data
- All operations respecting masks
- `ma.array`, `ma.masked_where`, etc.

**Estimate:** New subsystem, significant architectural work

---

## Priority Rankings

### High Impact (Core Functionality Gaps)
1. **ufunc system** - Foundational NumPy concept, blocks memory efficiency and extensibility
2. **Complex numbers** - Blocks entire domains (signal processing, FFT)
3. ~~**Integer/boolean array indexing**~~ ✅ Implemented via `iindex()` and `bindex()`
4. **Tuple of axes** for reductions

### Medium Impact (Important Features)
5. **Polynomial module**
6. **Error state control**

### Lower Impact (Advanced/Specialized)
7. **Structured arrays**
8. **FFT module** (blocked by complex)
9. **Masked arrays**

---

## Implementation Scope Summary

| Feature | Scope | Files | Functions | Blockers |
|---------|-------|-------|-----------|----------|
| Tuple axes reductions | Small-Mod | 1 | 15 | None |
| Error state | Small | 1-2 | 5 | None |
| Polynomial module | Moderate | 1 new | 10-15 | None |
| ~~Fancy indexing~~ | ~~Moderate~~ | ~~2-3~~ | ~~10-15~~ | ✅ Done |
| Complex numbers | Major | 15+ | 50+ | None |
| Structured arrays | Major | 5+ | 20+ | None |
| **ufunc system** | **Major** | **All ops** | **50+** | **None** |
| FFT module | Large | 1 new | 15 | Complex |
| Masked arrays | Large | 3+ new | 30+ | None |

---

## Not Planned

The following NumPy features are not planned for numpy-ts:

- **F-order memory layout** (Fortran order) — exists in NumPy for Fortran/BLAS interop, which doesn't exist in JS
- **Datetime/Timedelta dtypes** (`datetime64`, `timedelta64`) — JS has native `Date` and better libraries for time math
- **String/Unicode dtypes** (`str_`, `bytes_`, `U`, `S`) — JS strings are first-class citizens
- **Object dtype** (heterogeneous data) — defeats typed array purpose; use regular JS arrays
- **Memory mapping** (`memmap`, `mmap_mode`) — browser security model doesn't support this
- **Matrix class** (deprecated in NumPy)
- **Chararray** (deprecated in NumPy)
