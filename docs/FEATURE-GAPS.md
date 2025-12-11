# Feature Gaps Analysis: numpy-ts vs NumPy

This document tracks **features** (not API functions) that are missing or incomplete compared to NumPy. API function coverage is tracked separately.

## Overview

numpy-ts implements core NumPy functionality but lacks several deeper features. This document categorizes gaps by implementation scope and priority.

---

## Major Feature Gaps

### 1. Complex Number Data Types
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

### 2. F-Order Memory Layout (Fortran Order)
**Status: Not Truly Supported**
**Scope: MAJOR - Fundamental architectural change**

Current State:
- Arrays are always stored in C-order (row-major)
- `asfortranarray()` exists but just returns a C-order copy
- `isCContiguous` / `isFContiguous` flags exist but F-contiguous arrays cannot be created
- No `order` parameter on array creation functions

Required Changes:
- storage.ts: Add `computeStridesF()`, modify `zeros()`, `ones()`, `copy()` to accept order
- ndarray.ts: Add `order` parameter to ~15 creation functions
- shape.ts: `reshape()`, `ravel()`, `flatten()` need order-aware logic
- All operations: Need to respect/preserve order in outputs

**Impact:** Cannot efficiently interface with column-major libraries, cannot preserve memory layout from external sources.

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
**Status: Partially Missing**
**Scope: MODERATE - Concentrated in indexing**

Current State:
- Boolean indexing: Partially supported via `where()`, `compress()`, `extract()`
- Integer array indexing: Not supported (cannot do `arr[[0, 2, 4]]`)
- Slicing only supports string-based syntax

Required Changes:
- slicing.ts: Extend to accept NDArray indices
- ndarray.ts: Modify `slice()` to detect array arguments
- New internal: `fancyIndex()` function

**Estimate:** 2-3 files, 10-15 functions

---

## Moderate Feature Gaps

### 5. ufunc `out` Parameter
**Status: Not Supported**
**Scope: MODERATE-LARGE - Touches all operations**

Operations cannot write to pre-allocated output arrays. This impacts memory efficiency for large array operations.

Required Changes:
- compute.ts: Modify `elementwiseBinaryOp()` to accept optional output storage
- All arithmetic ops (15+), comparison ops (10+), math ops (30+): Add `out?: NDArray` parameter
- Validation for shape/dtype compatibility

**Estimate:** Signature change to 50+ functions, logic change localized to compute.ts

---

### 6. ufunc `where` Parameter
**Status: Not Supported**
**Scope: MODERATE-LARGE - Touches all operations**

Cannot conditionally apply operations based on boolean mask.

**Estimate:** Similar scope to `out` parameter - 50+ function signatures

---

### 7. Tuple of Axes for Reductions
**Status: Not Supported**
**Scope: SMALL-MODERATE - Localized to reductions**

Reductions only accept single axis, not tuple (e.g., `axis=(0, 1)`).

Required Changes:
- reduction.ts: Modify `sum()`, `mean()`, `max()`, `min()`, `prod()`, `std()`, `var()`, `any()`, `all()` to accept `number | number[]`
- More complex keepdims handling

**Estimate:** 1 file, ~15 functions

---

### 8. Reduction Parameters (`dtype`, `initial`)
**Status: Not Supported**
**Scope: SMALL - Localized to reductions**

Missing:
- `dtype` parameter to control output type
- `initial` parameter as starting value

**Estimate:** 1 file, ~15-20 functions total

---

### 9. Polynomial Module
**Status: Not Implemented**
**Scope: MODERATE - New module**

Missing: `polyval`, `polyfit`, `polyder`, `polyint`, `poly1d` class, `roots`

**Estimate:** New module, 300-500 lines, self-contained, no blockers

---

### 10. Error State Control
**Status: Not Implemented**
**Scope: SMALL - New utility**

Missing:
- `np.seterr()` / `np.geterr()`
- `np.errstate()` context manager
- Control over divide-by-zero, overflow, underflow, invalid handling

**Estimate:** 1 new file, ~100-200 lines

---

## Large New Subsystems

### 11. FFT Module
**Status: Not Implemented**
**Scope: LARGE - New module**
**Blocker: Requires complex number support**

Missing: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `fftfreq`, `fftshift`

**Estimate:** 500-1000 lines once complex numbers are available

---

### 12. Masked Arrays (`np.ma`)
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
1. **Complex numbers** - Blocks entire domains (signal processing, FFT)
2. **F-order memory layout** - Interoperability with scientific libraries
3. **Integer/boolean array indexing** - Basic NumPy usage pattern
4. **ufunc `out` parameter** - Memory efficiency for large arrays
5. **Tuple of axes** for reductions

### Medium Impact (Important Features)
6. **ufunc `where` parameter**
7. **Reduction `dtype`/`initial` parameters**
8. **Polynomial module**
9. **Error state control**

### Lower Impact (Advanced/Specialized)
10. **Structured arrays**
11. **FFT module** (blocked by complex)
12. **Masked arrays**

---

## Implementation Scope Summary

| Feature | Scope | Files | Functions | Blockers |
|---------|-------|-------|-----------|----------|
| Tuple axes reductions | Small-Mod | 1 | 15 | None |
| `initial` param | Small | 1 | 10 | None |
| `dtype` param (reductions) | Small | 1 | 15 | None |
| Error state | Small | 1-2 | 5 | None |
| Polynomial module | Moderate | 1 new | 10-15 | None |
| Fancy indexing | Moderate | 2-3 | 10-15 | None |
| `out` parameter | Mod-Large | 10+ | 50+ | None |
| `where` parameter | Mod-Large | 10+ | 50+ | None |
| F-order | Major | 10+ | 30+ | None |
| Complex numbers | Major | 15+ | 50+ | None |
| Structured arrays | Major | 5+ | 20+ | None |
| FFT module | Large | 1 new | 15 | Complex |
| Masked arrays | Large | 3+ new | 30+ | None |

---

## Not Planned

The following NumPy features are not planned for numpy-ts:

- **Datetime/Timedelta dtypes** (`datetime64`, `timedelta64`)
- **String/Unicode dtypes** (`str_`, `bytes_`, `U`, `S`)
- **Object dtype** (heterogeneous data)
- **Memory mapping** (`memmap`, `mmap_mode`)
- **Matrix class** (deprecated in NumPy)
- **Chararray** (deprecated in NumPy)
