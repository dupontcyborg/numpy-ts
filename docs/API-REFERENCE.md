# API Reference

Complete NumPy 2.0+ API checklist. Check off as implemented.

---

## NDArray Properties

### Core Properties
- `shape: readonly number[]` - Array dimensions (e.g., `[2, 3]`)
- `ndim: number` - Number of dimensions
- `size: number` - Total number of elements
- `dtype: DType` - Data type (e.g., `'float64'`, `'int32'`)
- `data: TypedArray` - Underlying typed array buffer
- `strides: readonly number[]` - Stride for each dimension (in elements)

### View Tracking (Added 2025-10-20)
- `base: NDArray | null` - Base array for views, `null` if owns data
  ```typescript
  const arr = ones([4, 4]);
  const view = arr.slice('0:2', '0:2');
  console.log(view.base === arr);  // true - view tracks base
  console.log(arr.base === null);  // true - owns data
  ```

### Memory Layout Flags (Added 2025-10-20)
- `flags.C_CONTIGUOUS: boolean` - Array is C-contiguous (row-major)
- `flags.F_CONTIGUOUS: boolean` - Array is Fortran-contiguous (column-major)
- `flags.OWNDATA: boolean` - Array owns its data buffer (not a view)
  ```typescript
  const arr = ones([3, 3]);
  console.log(arr.flags.C_CONTIGUOUS);  // true - row-major layout
  console.log(arr.flags.OWNDATA);       // true - owns data

  const view = arr.slice(':', '0:2');
  console.log(view.flags.OWNDATA);      // false - doesn't own data
  ```

### Type Preservation (Guaranteed as of 2025-10-20)
All operations preserve dtype or follow NumPy promotion rules:
- **Preserving operations**: reshape, transpose, slice, copy, etc. keep same dtype
- **Scalar operations**: `arr.add(5)` preserves dtype
- **Mixed-dtype operations**: Follow NumPy promotion hierarchy
  - `float64 > float32 > int64 > int32 > int16 > int8 > uint64 > uint32 > uint16 > uint8 > bool`
  - Example: `int8 + float32 → float32`
- **Comparisons**: Always return `bool` dtype
- **Reductions**: Most preserve dtype, except `mean()` converts integers to `float64`

---

## Array Creation

### From Shape
- [x] `zeros(shape, dtype?)` - Array of zeros
- [x] `ones(shape, dtype?)` - Array of ones
- [x] `empty(shape, dtype?)` - Uninitialized array
- [x] `full(shape, fill_value, dtype?)` - Array filled with value
- [x] `eye(n, m?, k?, dtype?)` - Identity matrix
- [x] `identity(n, dtype?)` - Square identity matrix

### From Data
- [x] `array(object, dtype?)` - Create from nested arrays
- [x] `asarray(a, dtype?)` - Convert to array
- [x] `copy(a)` - Deep copy

### Numerical Ranges
- [x] `arange(start, stop, step?, dtype?)` - Evenly spaced values
- [x] `linspace(start, stop, num?)` - Evenly spaced over interval
- [x] `logspace(start, stop, num?, base?)` - Log-spaced values
- [x] `geomspace(start, stop, num?)` - Geometric progression

### Like Functions
- [x] `zeros_like(a, dtype?)` - Zeros with same shape
- [x] `ones_like(a, dtype?)` - Ones with same shape
- [x] `empty_like(a, dtype?)` - Empty with same shape
- [x] `full_like(a, fill_value, dtype?)` - Full with same shape

---

## Array Manipulation

### Shape
- [x] `reshape(a, shape)` - New shape _(view if C-contiguous, copy otherwise)_
- [x] `ravel(a)` - Flatten to 1D _(view if C-contiguous, copy otherwise)_
- [x] `flatten(a)` - Flatten (copy) _(always returns a copy, matching NumPy)_
- [x] `squeeze(a, axis?)` - Remove single-dimensional entries _(always returns a view)_
- [x] `expand_dims(a, axis)` - Add dimension _(always returns a view)_

### Transpose
- [x] `transpose(a, axes?)` - Permute dimensions _(implemented as NDArray.transpose() method)_
- [ ] `swapaxes(a, axis1, axis2)` - Swap two axes
- [ ] `moveaxis(a, source, destination)` - Move axes

### Joining
- [ ] `concatenate(arrays, axis?)` - Join arrays
- [ ] `stack(arrays, axis?)` - Stack along new axis
- [ ] `vstack(arrays)` - Stack vertically
- [ ] `hstack(arrays)` - Stack horizontally
- [ ] `dstack(arrays)` - Stack depth-wise

### Splitting
- [ ] `split(a, indices_or_sections, axis?)` - Split into sub-arrays
- [ ] `array_split(a, indices_or_sections, axis?)` - Split (unequal)
- [ ] `vsplit(a, indices_or_sections)` - Split vertically
- [ ] `hsplit(a, indices_or_sections)` - Split horizontally

### Tiling
- [ ] `tile(a, reps)` - Tile array
- [ ] `repeat(a, repeats, axis?)` - Repeat elements

---

## Mathematical Operations

**File organization:**
- Arithmetic operations → `src/ops/arithmetic.ts`
- Exponential/logarithmic/power → `src/ops/exponential.ts`
- Trigonometric → `src/ops/trig.ts` (future)
- Hyperbolic → `src/ops/hyperbolic.ts` (future)
- Rounding → `src/ops/rounding.ts` (future)

### Arithmetic
- [x] `add(x1, x2)` - Addition
- [x] `subtract(x1, x2)` - Subtraction
- [x] `multiply(x1, x2)` - Multiplication
- [x] `divide(x1, x2)` - Division
- [x] `mod(x1, x2)` - Modulo
- [x] `floor_divide(x1, x2)` - Floor division
- [x] `negative(x)` - Negate
- [x] `positive(x)` - Positive
- [x] `absolute(x)` - Absolute value
- [x] `sign(x)` - Sign
- [x] `reciprocal(x)` - Reciprocal

### Trigonometric
- [ ] `sin(x)` - Sine
- [ ] `cos(x)` - Cosine
- [ ] `tan(x)` - Tangent
- [ ] `arcsin(x)` - Inverse sine
- [ ] `arccos(x)` - Inverse cosine
- [ ] `arctan(x)` - Inverse tangent
- [ ] `arctan2(x1, x2)` - Four-quadrant inverse tangent
- [ ] `hypot(x1, x2)` - Hypotenuse
- [ ] `degrees(x)` - Radians to degrees
- [ ] `radians(x)` - Degrees to radians

### Hyperbolic
- [ ] `sinh(x)` - Hyperbolic sine
- [ ] `cosh(x)` - Hyperbolic cosine
- [ ] `tanh(x)` - Hyperbolic tangent
- [ ] `arcsinh(x)` - Inverse hyperbolic sine
- [ ] `arccosh(x)` - Inverse hyperbolic cosine
- [ ] `arctanh(x)` - Inverse hyperbolic tangent

### Exponential and Logarithmic
- [x] `sqrt(x)` - Square root
- [x] `power(x1, x2)` - Powe
- [ ] `square(x)` - Square
- [ ] `cbrt(x)` - Cube root
- [ ] `exp(x)` - Exponential
- [ ] `expm1(x)` - exp(x) - 1
- [ ] `exp2(x)` - 2^x
- [ ] `log(x)` - Natural logarithm
- [ ] `log10(x)` - Base-10 logarithm
- [ ] `log2(x)` - Base-2 logarithm
- [ ] `log1p(x)` - log(1 + x)
- [ ] `logaddexp(x1, x2)` - log(exp(x1) + exp(x2))

### Rounding
- [ ] `around(a, decimals?)` - Round
- [ ] `round(a, decimals?)` - Round (alias)
- [ ] `floor(x)` - Floor
- [ ] `ceil(x)` - Ceiling
- [ ] `trunc(x)` - Truncate
- [ ] `rint(x)` - Round to nearest integer

### Other Math
- [ ] `clip(a, min, max)` - Clip values

---

## Reductions

### Sum and Product
- [x] `sum(a, axis?, keepdims?)` - Sum
- [x] `prod(a, axis?, keepdims?)` - Product
- [ ] `cumsum(a, axis?)` - Cumulative sum
- [ ] `cumprod(a, axis?)` - Cumulative product

### Statistics
- [x] `mean(a, axis?, keepdims?)` - Mean
- [ ] `median(a, axis?, keepdims?)` - Median
- [x] `std(a, axis?, ddof?, keepdims?)` - Standard deviation
- [x] `var(a, axis?, keepdims?)` - Variance
- [ ] `percentile(a, q, axis?)` - Percentile
- [ ] `quantile(a, q, axis?)` - Quantile

### Min/Max
- [x] `min(a, axis?, keepdims?)` - Minimum
- [x] `max(a, axis?, keepdims?)` - Maximum
- [x] `argmin(a, axis?)` - Index of minimum
- [x] `argmax(a, axis?)` - Index of maximum
- [ ] `ptp(a, axis?)` - Peak-to-peak (max - min)

### Logic
- [x] `all(a, axis?, keepdims?)` - Test if all True
- [x] `any(a, axis?, keepdims?)` - Test if any True

---

## Comparison

- [x] `greater(x1, x2)` - Greater than _(implemented as NDArray.greater() method)_
- [x] `greater_equal(x1, x2)` - Greater or equal _(implemented as NDArray.greater_equal() method)_
- [x] `less(x1, x2)` - Less than _(implemented as NDArray.less() method)_
- [x] `less_equal(x1, x2)` - Less or equal _(implemented as NDArray.less_equal() method)_
- [x] `equal(x1, x2)` - Equal _(implemented as NDArray.equal() method)_
- [x] `not_equal(x1, x2)` - Not equal _(implemented as NDArray.not_equal() method)_
- [x] `allclose(a, b, rtol?, atol?)` - Close within tolerance _(implemented as NDArray.allclose() method)_
- [x] `isclose(a, b, rtol?, atol?)` - Element-wise close _(implemented as NDArray.isclose() method)_

---

## Logic

- [ ] `logical_and(x1, x2)` - Logical AND
- [ ] `logical_or(x1, x2)` - Logical OR
- [ ] `logical_not(x)` - Logical NOT
- [ ] `logical_xor(x1, x2)` - Logical XOR

---

## Linear Algebra (numpy.linalg)

### Matrix Products
- [x] `dot(a, b)` - Dot product _(fully NumPy-compatible: all 0D-ND combinations including scalars, vectors, matrices, and tensors)_
- [x] `matmul(a, b)` - Matrix product _(fully implemented with transpose detection via strides)_
- [x] `inner(a, b)` - Inner product _(contracts over last axes of both arrays)_
- [x] `outer(a, b)` - Outer product _(flattens inputs, computes all pairwise products)_
- [x] `tensordot(a, b, axes)` - Tensor dot product
- [ ] `einsum(subscripts, *operands)` - Einstein summation

### Decompositions
- [ ] `linalg.cholesky(a)` - Cholesky decomposition
- [ ] `linalg.qr(a)` - QR decomposition
- [ ] `linalg.svd(a, full_matrices?)` - Singular value decomposition
- [ ] `linalg.eig(a)` - Eigenvalues and eigenvectors
- [ ] `linalg.eigh(a)` - Eigenvalues (Hermitian)
- [ ] `linalg.eigvals(a)` - Eigenvalues only

### Solving
- [ ] `linalg.solve(a, b)` - Solve linear system
- [ ] `linalg.lstsq(a, b)` - Least-squares solution
- [ ] `linalg.inv(a)` - Matrix inverse
- [ ] `linalg.pinv(a, rcond?)` - Pseudo-inverse

### Norms and Numbers
- [ ] `linalg.norm(x, ord?, axis?)` - Norm
- [ ] `linalg.det(a)` - Determinant
- [ ] `linalg.matrix_rank(a, tol?)` - Matrix rank
- [x] `trace(a)` - Trace _(implemented as NDArray.trace() method; supports square and non-square matrices)_

---

## Random Sampling (numpy.random)

### Simple Random
- [ ] `random.rand(...shape)` - Uniform [0, 1)
- [ ] `random.randn(...shape)` - Standard normal
- [ ] `random.randint(low, high?, size?)` - Random integers
- [ ] `random.random(size?)` - Random floats [0, 1)

### Distributions
- [ ] `random.uniform(low, high, size?)` - Uniform distribution
- [ ] `random.normal(loc?, scale?, size?)` - Normal distribution
- [ ] `random.exponential(scale?, size?)` - Exponential
- [ ] `random.poisson(lam?, size?)` - Poisson
- [ ] `random.binomial(n, p, size?)` - Binomial

### Permutations
- [ ] `random.shuffle(x)` - Shuffle in-place
- [ ] `random.permutation(x)` - Permuted sequence
- [ ] `random.choice(a, size?, replace?)` - Random choice

### Generator (preferred API)
- [ ] `random.default_rng(seed?)` - Create generator
- [ ] `Generator.random(size?)` - Random floats
- [ ] `Generator.integers(low, high?, size?)` - Random integers
- [ ] `Generator.normal(loc?, scale?, size?)` - Normal distribution
- [ ] `Generator.standard_normal(size?)` - Standard normal

---

## Sorting and Searching

### Sorting
- [ ] `sort(a, axis?)` - Sort array
- [ ] `argsort(a, axis?)` - Indices that would sort
- [ ] `lexsort(keys)` - Indirect stable sort

### Searching
- [ ] `argmax(a, axis?)` - Index of maximum
- [ ] `argmin(a, axis?)` - Index of minimum
- [ ] `nonzero(a)` - Indices of non-zero elements
- [ ] `where(condition, x?, y?)` - Elements from x or y
- [ ] `searchsorted(a, v)` - Find indices to insert

### Counting
- [ ] `count_nonzero(a, axis?)` - Count non-zero elements
- [ ] `unique(a, return_index?, return_counts?)` - Unique elements

---

## Set Operations

- [ ] `unique(ar)` - Unique elements
- [ ] `in1d(ar1, ar2)` - Test membership
- [ ] `intersect1d(ar1, ar2)` - Intersection
- [ ] `union1d(ar1, ar2)` - Union
- [ ] `setdiff1d(ar1, ar2)` - Set difference
- [ ] `setxor1d(ar1, ar2)` - Symmetric difference

---

## I/O

### NumPy Files
- [x] `load(file)` - Load array from .npy/.npz file _(supports v1/v2/v3 format, all dtypes)_
- [x] `save(file, arr)` - Save array to .npy file _(writes v3 format)_
- [x] `savez(file, *arrays, **kwds)` - Save multiple arrays (.npz) _(supports positional and named arrays)_
- [x] `savez_compressed(file, *arrays, **kwds)` - Compressed .npz _(uses DEFLATE compression)_

### In-Memory Parsing/Serialization (Browser-compatible)
- [x] `parseNpy(buffer)` - Parse NPY bytes to NDArray
- [x] `serializeNpy(arr)` - Serialize NDArray to NPY bytes
- [x] `parseNpz(buffer)` / `parseNpzSync(buffer)` - Parse NPZ bytes
- [x] `serializeNpz(arrays)` / `serializeNpzSync(arrays)` - Serialize to NPZ bytes

### Text Files
- [ ] `loadtxt(fname, dtype?)` - Load from text
- [ ] `savetxt(fname, X)` - Save to text

---

## FFT

### Standard FFT
- [ ] `fft.fft(a, n?, axis?)` - 1-D FFT
- [ ] `fft.ifft(a, n?, axis?)` - 1-D inverse FFT
- [ ] `fft.fft2(a, s?, axes?)` - 2-D FFT
- [ ] `fft.ifft2(a, s?, axes?)` - 2-D inverse FFT
- [ ] `fft.fftn(a, s?, axes?)` - N-D FFT
- [ ] `fft.ifftn(a, s?, axes?)` - N-D inverse FFT

### Real FFT
- [ ] `fft.rfft(a, n?, axis?)` - Real input FFT
- [ ] `fft.irfft(a, n?, axis?)` - Inverse real FFT

### Helpers
- [ ] `fft.fftfreq(n, d?)` - FFT frequencies
- [ ] `fft.rfftfreq(n, d?)` - Real FFT frequencies
- [ ] `fft.fftshift(x, axes?)` - Shift zero-frequency to center
- [ ] `fft.ifftshift(x, axes?)` - Inverse of fftshift

---

## Polynomials

### Polynomial Class
- [ ] `polynomial.Polynomial(coef)` - Power series
- [ ] `Polynomial.fit(x, y, deg)` - Least-squares fit
- [ ] `Polynomial.roots()` - Roots
- [ ] `Polynomial.deriv()` - Derivative
- [ ] `Polynomial.integ()` - Integral

### Legacy (numpy.poly1d)
- [ ] `poly1d(coef)` - 1-D polynomial
- [ ] `polyval(p, x)` - Evaluate polynomial
- [ ] `polyfit(x, y, deg)` - Polynomial fit
- [ ] `polyder(p)` - Derivative
- [ ] `polyint(p)` - Integral
- [ ] `polyadd(a1, a2)` - Add polynomials
- [ ] `polymul(a1, a2)` - Multiply polynomials

---

## Advanced

### Broadcasting
- [ ] `broadcast_to(array, shape)` - Broadcast to shape
- [ ] `broadcast_arrays(*args)` - Broadcast multiple arrays

### Indexing
- [ ] `take(a, indices, axis?)` - Take elements
- [ ] `put(a, ind, v)` - Put values at indices
- [ ] `choose(a, choices)` - Construct from index array

### Testing
- [ ] `allclose(a, b, rtol?, atol?)` - Arrays close
- [ ] `isclose(a, b, rtol?, atol?)` - Element-wise close
- [ ] `array_equal(a1, a2)` - Arrays equal

---

**Last Updated**: 2025-11-29
