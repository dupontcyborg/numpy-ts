# API Reference

Complete NumPy 2.0+ API compatibility checklist.

**Last Updated**: 2025-12-09

## Progress Summary

Based on `npm run compare-api`:

- **Overall Coverage**: 323/333 (97.0%)
- **Top-level Functions**: 265/318 (83.3%)
- **NDArray Methods**: 33/53 (62.3%)

### Completed Categories (100%)
- Arithmetic (19/19)
- Array Creation (32/32)
- Array Manipulation (35/35)
- Bit Operations (9/9)
- Broadcasting (3/3)
- Comparison (10/10)
- Exponential (9/9)
- Gradient (4/4)
- Hyperbolic (6/6)
- I/O (8/8)
- Indexing (20/20)
- Linear Algebra (9/9)
- Linear Algebra (linalg) (19/19)
- Logic (12/12)
- Random (17/17)
- Reductions (30/30)
- Rounding (7/7)
- Searching (6/6)
- Set Operations (7/7)
- Sorting (6/6)
- Statistics (9/9)
- Trigonometric (12/12)

### Incomplete Categories
- FFT (0/18) - 0.0%
- Other Math (0/11) - 0.0%

---

## NDArray Properties

### Core Properties
- `shape: readonly number[]` - Array dimensions (e.g., `[2, 3]`)
- `ndim: number` - Number of dimensions
- `size: number` - Total number of elements
- `dtype: DType` - Data type (e.g., `'float64'`, `'int32'`)
- `data: TypedArray` - Underlying typed array buffer
- `strides: readonly number[]` - Stride for each dimension (in elements)

### View Tracking
- `base: NDArray | null` - Base array for views, `null` if owns data
  ```typescript
  const arr = ones([4, 4]);
  const view = arr.slice('0:2', '0:2');
  console.log(view.base === arr);  // true - view tracks base
  console.log(arr.base === null);  // true - owns data
  ```

### Memory Layout Flags
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

### Type Preservation
All operations preserve dtype or follow NumPy promotion rules:
- **Preserving operations**: reshape, transpose, slice, copy, etc. keep same dtype
- **Scalar operations**: `arr.add(5)` preserves dtype
- **Mixed-dtype operations**: Follow NumPy promotion hierarchy
  - `float64 > float32 > int64 > int32 > int16 > int8 > uint64 > uint32 > uint16 > uint8 > bool`
  - Example: `int8 + float32 → float32`
- **Comparisons**: Always return `bool` dtype
- **Reductions**: Most preserve dtype, except `mean()` converts integers to `float64`

---

## Arithmetic

- [x] `absolute(x)` - Absolute value
- [x] `add(x1, x2)` - Addition
- [x] `cbrt(x)` - Cube root
- [x] `divide(x1, x2)` - Division
- [x] `divmod(x1, x2)` - Quotient and remainder
- [x] `fabs(x)` - Absolute value (floating point)
- [x] `floor_divide(x1, x2)` - Floor division
- [x] `heaviside(x1, x2)` - Heaviside step function
- [x] `mod(x1, x2)` - Modulo
- [x] `multiply(x1, x2)` - Multiplication
- [x] `negative(x)` - Negate
- [x] `positive(x)` - Positive
- [x] `power(x1, x2)` - Power
- [x] `reciprocal(x)` - Reciprocal
- [x] `remainder(x1, x2)` - Remainder
- [x] `sign(x)` - Sign
- [x] `sqrt(x)` - Square root
- [x] `square(x)` - Square
- [x] `subtract(x1, x2)` - Subtraction

---

## Reductions

- [x] `all(a, axis?, keepdims?)` - Test if all True
- [x] `any(a, axis?, keepdims?)` - Test if any True
- [x] `argmax(a, axis?)` - Index of maximum
- [x] `argmin(a, axis?)` - Index of minimum
- [x] `average(a, axis?, weights?, keepdims?)` - Weighted average
- [x] `cumprod(a, axis?)` - Cumulative product
- [x] `cumsum(a, axis?)` - Cumulative sum
- [x] `max(a, axis?, keepdims?)` - Maximum
- [x] `mean(a, axis?, keepdims?)` - Mean
- [x] `median(a, axis?, keepdims?)` - Median
- [x] `min(a, axis?, keepdims?)` - Minimum
- [x] `nanargmax(a, axis?)` - Index of maximum (ignoring NaN)
- [x] `nanargmin(a, axis?)` - Index of minimum (ignoring NaN)
- [x] `nancumprod(a, axis?)` - Cumulative product (ignoring NaN)
- [x] `nancumsum(a, axis?)` - Cumulative sum (ignoring NaN)
- [x] `nanmax(a, axis?, keepdims?)` - Maximum (ignoring NaN)
- [x] `nanmean(a, axis?, keepdims?)` - Mean (ignoring NaN)
- [x] `nanmedian(a, axis?, keepdims?)` - Median (ignoring NaN)
- [x] `nanmin(a, axis?, keepdims?)` - Minimum (ignoring NaN)
- [x] `nanprod(a, axis?, keepdims?)` - Product (ignoring NaN)
- [x] `nanstd(a, axis?, ddof?, keepdims?)` - Standard deviation (ignoring NaN)
- [x] `nansum(a, axis?, keepdims?)` - Sum (ignoring NaN)
- [x] `nanvar(a, axis?, ddof?, keepdims?)` - Variance (ignoring NaN)
- [x] `percentile(a, q, axis?)` - Percentile
- [x] `prod(a, axis?, keepdims?)` - Product
- [x] `ptp(a, axis?)` - Peak-to-peak (max - min)
- [x] `quantile(a, q, axis?)` - Quantile
- [x] `std(a, axis?, ddof?, keepdims?)` - Standard deviation
- [x] `sum(a, axis?, keepdims?)` - Sum
- [x] `var(a, axis?, keepdims?)` - Variance

---

## Comparison

- [x] `allclose(a, b, rtol?, atol?)` - Arrays close within tolerance
- [x] `array_equal(a1, a2, equal_nan?)` - Arrays equal
- [x] `array_equiv(a1, a2)` - Arrays equivalent (broadcastable and equal)
- [x] `equal(x1, x2)` - Element-wise equal
- [x] `greater(x1, x2)` - Element-wise greater than
- [x] `greater_equal(x1, x2)` - Element-wise greater or equal
- [x] `isclose(a, b, rtol?, atol?)` - Element-wise close within tolerance
- [x] `less(x1, x2)` - Element-wise less than
- [x] `less_equal(x1, x2)` - Element-wise less or equal
- [x] `not_equal(x1, x2)` - Element-wise not equal

---

## Other Math

- [ ] `angle(z, deg?)` - Angle of complex argument
- [ ] `clip(a, min, max)` - Clip values to range
- [ ] `conj(x)` - Complex conjugate
- [ ] `conjugate(x)` - Complex conjugate
- [ ] `fmax(x1, x2)` - Element-wise maximum (propagates NaN)
- [ ] `fmin(x1, x2)` - Element-wise minimum (propagates NaN)
- [ ] `imag(val)` - Imaginary part
- [ ] `maximum(x1, x2)` - Element-wise maximum
- [ ] `minimum(x1, x2)` - Element-wise minimum
- [ ] `nan_to_num(x, copy?, nan?, posinf?, neginf?)` - Replace NaN/Inf with numbers
- [ ] `real(val)` - Real part

---

## Array Manipulation

- [x] `append(arr, values, axis?)` - Append values
- [x] `array_split(a, indices_or_sections, axis?)` - Split into sub-arrays (unequal)
- [x] `atleast_1d(...arrays)` - View with at least 1 dimension
- [x] `atleast_2d(...arrays)` - View with at least 2 dimensions
- [x] `atleast_3d(...arrays)` - View with at least 3 dimensions
- [x] `column_stack(tup)` - Stack 1-D arrays as columns
- [x] `concatenate(arrays, axis?)` - Join arrays
- [x] `delete(arr, obj, axis?)` - Delete sub-arrays
- [x] `dsplit(a, indices_or_sections)` - Split along 3rd axis
- [x] `dstack(arrays)` - Stack depth-wise (3rd axis)
- [x] `expand_dims(a, axis)` - Add dimension
- [x] `flip(m, axis?)` - Reverse array along axis
- [x] `fliplr(m)` - Flip left-right
- [x] `flipud(m)` - Flip up-down
- [x] `hsplit(a, indices_or_sections)` - Split horizontally
- [x] `hstack(arrays)` - Stack horizontally
- [x] `insert(arr, obj, values, axis?)` - Insert values
- [x] `moveaxis(a, source, destination)` - Move axes
- [x] `pad(array, pad_width, mode?, **kwargs)` - Pad array
- [x] `ravel(a)` - Flatten to 1D
- [x] `repeat(a, repeats, axis?)` - Repeat elements
- [x] `reshape(a, shape)` - New shape
- [x] `resize(a, new_shape)` - Resize array
- [x] `roll(a, shift, axis?)` - Roll array elements
- [x] `rollaxis(a, axis, start?)` - Roll axis backwards
- [x] `rot90(m, k?, axes?)` - Rotate 90 degrees
- [x] `row_stack(tup)` - Stack rows (alias for vstack)
- [x] `split(a, indices_or_sections, axis?)` - Split into sub-arrays
- [x] `squeeze(a, axis?)` - Remove single-dimensional entries
- [x] `stack(arrays, axis?)` - Stack along new axis
- [x] `swapaxes(a, axis1, axis2)` - Swap two axes
- [x] `tile(a, reps)` - Tile array
- [x] `transpose(a, axes?)` - Permute dimensions
- [x] `vsplit(a, indices_or_sections)` - Split vertically
- [x] `vstack(arrays)` - Stack vertically

---

## Array Creation

- [x] `arange(start, stop, step?, dtype?)` - Evenly spaced values
- [x] `array(object, dtype?)` - Create from nested arrays
- [x] `asanyarray(a, dtype?)` - Convert to array (preserves subclasses)
- [x] `asarray(a, dtype?)` - Convert to array
- [x] `ascontiguousarray(a, dtype?)` - C-contiguous array
- [x] `asfortranarray(a, dtype?)` - Fortran-contiguous array
- [x] `copy(a)` - Deep copy
- [x] `diag(v, k?)` - Extract diagonal or construct diagonal array
- [x] `diagflat(v, k?)` - Create 2-D array with flattened input on diagonal
- [x] `empty(shape, dtype?)` - Uninitialized array
- [x] `empty_like(a, dtype?)` - Empty with same shape
- [x] `eye(n, m?, k?, dtype?)` - Identity matrix
- [x] `frombuffer(buffer, dtype?, count?, offset?)` - Create from buffer
- [x] `fromfile(file, dtype?, count?, sep?)` - Create from file
- [x] `fromfunction(function, shape, dtype?)` - Create from function
- [x] `fromiter(iter, dtype, count?)` - Create from iterable
- [x] `fromstring(string, dtype?, count?, sep?)` - Create from string
- [x] `full(shape, fill_value, dtype?)` - Array filled with value
- [x] `full_like(a, fill_value, dtype?)` - Full with same shape
- [x] `geomspace(start, stop, num?)` - Geometric progression
- [x] `identity(n, dtype?)` - Square identity matrix
- [x] `linspace(start, stop, num?)` - Evenly spaced over interval
- [x] `logspace(start, stop, num?, base?)` - Log-spaced values
- [x] `meshgrid(*xi, indexing?)` - Coordinate matrices from vectors
- [x] `ones(shape, dtype?)` - Array of ones
- [x] `ones_like(a, dtype?)` - Ones with same shape
- [x] `tri(n, m?, k?, dtype?)` - Lower triangle
- [x] `tril(m, k?)` - Lower triangle of array
- [x] `triu(m, k?)` - Upper triangle of array
- [x] `vander(x, n?, increasing?)` - Vandermonde matrix
- [x] `zeros(shape, dtype?)` - Array of zeros
- [x] `zeros_like(a, dtype?)` - Zeros with same shape

---

## Trigonometric

- [x] `arccos(x)` - Inverse cosine
- [x] `arcsin(x)` - Inverse sine
- [x] `arctan(x)` - Inverse tangent
- [x] `arctan2(x1, x2)` - Four-quadrant inverse tangent
- [x] `cos(x)` - Cosine
- [x] `deg2rad(x)` - Degrees to radians
- [x] `degrees(x)` - Radians to degrees
- [x] `hypot(x1, x2)` - Hypotenuse
- [x] `rad2deg(x)` - Radians to degrees
- [x] `radians(x)` - Degrees to radians
- [x] `sin(x)` - Sine
- [x] `tan(x)` - Tangent

---

## Hyperbolic

- [x] `arccosh(x)` - Inverse hyperbolic cosine
- [x] `arcsinh(x)` - Inverse hyperbolic sine
- [x] `arctanh(x)` - Inverse hyperbolic tangent
- [x] `cosh(x)` - Hyperbolic cosine
- [x] `sinh(x)` - Hyperbolic sine
- [x] `tanh(x)` - Hyperbolic tangent

---

## Sorting

- [x] `argpartition(a, kth, axis?)` - Indices that would partially sort
- [x] `argsort(a, axis?)` - Indices that would sort
- [x] `lexsort(keys)` - Indirect stable sort
- [x] `partition(a, kth, axis?)` - Partial sort
- [x] `sort(a, axis?)` - Sort array
- [x] `sort_complex(a)` - Sort complex array

---

## Rounding

- [x] `around(a, decimals?)` - Round
- [x] `ceil(x)` - Ceiling
- [x] `fix(x)` - Round to nearest integer towards zero
- [x] `floor(x)` - Floor
- [x] `rint(x)` - Round to nearest integer
- [x] `round(a, decimals?)` - Round (alias for around)
- [x] `trunc(x)` - Truncate

---

## Statistics

- [x] `bincount(x, weights?, minlength?)` - Count occurrences of non-negative integers
- [x] `convolve(a, v, mode?)` - Discrete, linear convolution
- [x] `corrcoef(x, y?, rowvar?)` - Pearson correlation coefficients
- [x] `correlate(a, v, mode?)` - Cross-correlation of two 1-D sequences
- [x] `cov(m, y?, rowvar?, bias?, ddof?)` - Estimate covariance matrix
- [x] `digitize(x, bins, right?)` - Return indices of bins
- [x] `histogram(a, bins?, range?, density?, weights?)` - Compute histogram
- [x] `histogram2d(x, y, bins?, range?, density?, weights?)` - Compute 2D histogram
- [x] `histogramdd(sample, bins?, range?, density?, weights?)` - Compute multidimensional histogram

---

## Bit Operations

- [x] `bitwise_and(x1, x2)` - Bitwise AND
- [x] `bitwise_not(x)` - Bitwise NOT
- [x] `bitwise_or(x1, x2)` - Bitwise OR
- [x] `bitwise_xor(x1, x2)` - Bitwise XOR
- [x] `invert(x)` - Bitwise NOT (alias)
- [x] `left_shift(x1, x2)` - Left shift
- [x] `packbits(a, axis?, bitorder?)` - Pack binary values into bytes
- [x] `right_shift(x1, x2)` - Right shift
- [x] `unpackbits(a, axis?, count?, bitorder?)` - Unpack bytes into binary values

---

## Broadcasting

- [x] `broadcast_arrays(*args)` - Broadcast multiple arrays
- [x] `broadcast_shapes(*shapes)` - Broadcast shapes
- [x] `broadcast_to(array, shape)` - Broadcast to shape

---

## Indexing

- [x] `choose(a, choices)` - Construct from index array
- [x] `compress(condition, a, axis?)` - Select slices along axis
- [x] `diag_indices(n, ndim?)` - Diagonal indices
- [x] `diag_indices_from(arr)` - Diagonal indices from array
- [x] `indices(dimensions, dtype?)` - Array of grid indices
- [x] `ix_(*args)` - Open mesh from multiple sequences
- [x] `mask_indices(n, mask_func, k?)` - Indices for masked array
- [x] `place(arr, mask, vals)` - Place values into array
- [x] `put(a, ind, v)` - Put values at indices
- [x] `put_along_axis(arr, indices, values, axis)` - Put values along axis
- [x] `putmask(a, mask, values)` - Put values where mask is true
- [x] `ravel_multi_index(multi_index, dims, mode?, order?)` - Multi-index to flat index
- [x] `select(condlist, choicelist, default?)` - Select from choices based on conditions
- [x] `take(a, indices, axis?)` - Take elements
- [x] `take_along_axis(arr, indices, axis)` - Take values along axis
- [x] `tril_indices(n, k?, m?)` - Lower triangle indices
- [x] `tril_indices_from(arr, k?)` - Lower triangle indices from array
- [x] `triu_indices(n, k?, m?)` - Upper triangle indices
- [x] `triu_indices_from(arr, k?)` - Upper triangle indices from array
- [x] `unravel_index(indices, shape, order?)` - Flat index to multi-index

---

## Logic

- [x] `copysign(x1, x2)` - Copy sign of values
- [x] `isfinite(x)` - Test for finite values
- [x] `isinf(x)` - Test for infinity
- [x] `isnan(x)` - Test for NaN
- [x] `isnat(x)` - Test for NaT (not-a-time)
- [x] `logical_and(x1, x2)` - Logical AND
- [x] `logical_not(x)` - Logical NOT
- [x] `logical_or(x1, x2)` - Logical OR
- [x] `logical_xor(x1, x2)` - Logical XOR
- [x] `nextafter(x1, x2)` - Next floating-point value
- [x] `signbit(x)` - Sign bit
- [x] `spacing(x)` - Spacing to next value

---

## Searching

- [x] `count_nonzero(a, axis?)` - Count non-zero elements
- [x] `extract(condition, arr)` - Extract elements based on condition
- [x] `flatnonzero(a)` - Indices of non-zero elements (flattened)
- [x] `nonzero(a)` - Indices of non-zero elements
- [x] `searchsorted(a, v)` - Find indices to insert
- [x] `where(condition, x?, y?)` - Elements from x or y based on condition

---

## Gradient

- [x] `cross(a, b, axis?)` - Cross product
- [x] `diff(a, n?, axis?)` - Differences along axis
- [x] `ediff1d(ary, to_end?, to_begin?)` - 1-D differences
- [x] `gradient(f, *varargs, axis?, edge_order?)` - Gradient

---

## Linear Algebra

- [x] `diagonal(a, offset?, axis1?, axis2?)` - Extract diagonal
- [x] `dot(a, b)` - Dot product
- [x] `einsum(subscripts, *operands)` - Einstein summation
- [x] `inner(a, b)` - Inner product
- [x] `kron(a, b)` - Kronecker product
- [x] `matmul(a, b)` - Matrix product
- [x] `outer(a, b)` - Outer product
- [x] `tensordot(a, b, axes)` - Tensor dot product
- [x] `trace(a, offset?, axis1?, axis2?)` - Trace

---

## Exponential

- [x] `exp(x)` - Exponential
- [x] `exp2(x)` - 2^x
- [x] `expm1(x)` - exp(x) - 1
- [x] `log(x)` - Natural logarithm
- [x] `log10(x)` - Base-10 logarithm
- [x] `log1p(x)` - log(1 + x)
- [x] `log2(x)` - Base-2 logarithm
- [x] `logaddexp(x1, x2)` - log(exp(x1) + exp(x2))
- [x] `logaddexp2(x1, x2)` - log2(2^x1 + 2^x2)

---

## I/O

- [x] `fromregex(file, regexp, dtype)` - Create from regex matching
- [x] `genfromtxt(fname, dtype?, comments?, delimiter?, ...)` - Load from text with missing values
- [x] `load(file)` - Load array from .npy/.npz file
- [x] `loadtxt(fname, dtype?)` - Load from text
- [x] `save(file, arr)` - Save array to .npy file
- [x] `savetxt(fname, X)` - Save to text
- [x] `savez(file, *arrays, **kwds)` - Save multiple arrays (.npz)
- [x] `savez_compressed(file, *arrays, **kwds)` - Compressed .npz

### Browser-Compatible Parsing/Serialization

These work with bytes (ArrayBuffer/Uint8Array), not files:

- [x] `parseNpy(buffer)` - Parse NPY bytes to NDArray
- [x] `serializeNpy(arr)` - Serialize NDArray to NPY bytes
- [x] `parseNpz(buffer)` / `parseNpzSync(buffer)` - Parse NPZ bytes
- [x] `serializeNpz(arrays)` / `serializeNpzSync(arrays)` - Serialize to NPZ bytes

---

## Set Operations

- [x] `in1d(ar1, ar2)` - Test membership
- [x] `intersect1d(ar1, ar2)` - Intersection
- [x] `isin(element, test_elements)` - Element-wise membership
- [x] `setdiff1d(ar1, ar2)` - Set difference
- [x] `setxor1d(ar1, ar2)` - Symmetric difference
- [x] `union1d(ar1, ar2)` - Union
- [x] `unique(ar, return_index?, return_inverse?, return_counts?, axis?)` - Unique elements

---

## Linear Algebra (linalg)

- [x] `linalg.cholesky(a)` - Cholesky decomposition
- [x] `linalg.cond(x, p?)` - Condition number
- [x] `linalg.cross(a, b)` - Cross product
- [x] `linalg.det(a)` - Determinant
- [x] `linalg.eig(a)` - Eigenvalues and eigenvectors
- [x] `linalg.eigh(a, UPLO?)` - Eigenvalues (Hermitian)
- [x] `linalg.eigvals(a)` - Eigenvalues only
- [x] `linalg.eigvalsh(a, UPLO?)` - Eigenvalues (Hermitian) only
- [x] `linalg.inv(a)` - Matrix inverse
- [x] `linalg.lstsq(a, b, rcond?)` - Least-squares solution
- [x] `linalg.matrix_norm(x, ord?, axis?)` - Matrix norm
- [x] `linalg.matrix_power(a, n)` - Matrix power
- [x] `linalg.matrix_rank(a, tol?)` - Matrix rank
- [x] `linalg.norm(x, ord?, axis?, keepdims?)` - Norm
- [x] `linalg.pinv(a, rcond?)` - Pseudo-inverse
- [x] `linalg.qr(a, mode?)` - QR decomposition
- [x] `linalg.solve(a, b)` - Solve linear system
- [x] `linalg.svd(a, full_matrices?)` - Singular value decomposition
- [x] `linalg.vector_norm(x, ord?, axis?, keepdims?)` - Vector norm

---

## FFT

- [ ] `fft.fft(a, n?, axis?)` - 1-D FFT
- [ ] `fft.fft2(a, s?, axes?)` - 2-D FFT
- [ ] `fft.fftfreq(n, d?)` - FFT frequencies
- [ ] `fft.fftn(a, s?, axes?)` - N-D FFT
- [ ] `fft.fftshift(x, axes?)` - Shift zero-frequency to center
- [ ] `fft.hfft(a, n?, axis?)` - Hermitian FFT
- [ ] `fft.ifft(a, n?, axis?)` - 1-D inverse FFT
- [ ] `fft.ifft2(a, s?, axes?)` - 2-D inverse FFT
- [ ] `fft.ifftn(a, s?, axes?)` - N-D inverse FFT
- [ ] `fft.ifftshift(x, axes?)` - Inverse of fftshift
- [ ] `fft.ihfft(a, n?, axis?)` - Inverse Hermitian FFT
- [ ] `fft.irfft(a, n?, axis?)` - Inverse real FFT
- [ ] `fft.irfft2(a, s?, axes?)` - 2-D inverse real FFT
- [ ] `fft.irfftn(a, s?, axes?)` - N-D inverse real FFT
- [ ] `fft.rfft(a, n?, axis?)` - Real input FFT
- [ ] `fft.rfft2(a, s?, axes?)` - 2-D real FFT
- [ ] `fft.rfftfreq(n, d?)` - Real FFT frequencies
- [ ] `fft.rfftn(a, s?, axes?)` - N-D real FFT

---

## Random

- [x] `random.binomial(n, p, size?)` - Binomial distribution
- [x] `random.choice(a, size?, replace?, p?)` - Random choice
- [x] `random.default_rng(seed?)` - Create generator (PCG64)
- [x] `random.exponential(scale?, size?)` - Exponential distribution
- [x] `random.get_state()` - Get random state
- [x] `random.normal(loc?, scale?, size?)` - Normal distribution
- [x] `random.permutation(x)` - Permuted sequence
- [x] `random.poisson(lam?, size?)` - Poisson distribution
- [x] `random.rand(...shape)` - Uniform [0, 1)
- [x] `random.randint(low, high?, size?, dtype?)` - Random integers
- [x] `random.randn(...shape)` - Standard normal
- [x] `random.random(size?)` - Random floats [0, 1)
- [x] `random.seed(seed)` - Seed random generator (MT19937)
- [x] `random.set_state(state)` - Set random state
- [x] `random.shuffle(x)` - Shuffle in-place
- [x] `random.standard_normal(size?)` - Standard normal
- [x] `random.uniform(low?, high?, size?)` - Uniform distribution

### Random Module Compatibility

The random module implements both NumPy's legacy API (MT19937) and modern Generator API (PCG64):

| Functions | NumPy Match | Notes |
|-----------|-------------|-------|
| `random()`, `rand()`, `uniform()` | ✅ Exact | Same seed → identical output |
| `default_rng().random()`, `.uniform()` | ✅ Exact | PCG64 with SeedSequence |
| `randn()`, `normal()`, `standard_normal()` | Statistical | Uses Box-Muller (NumPy uses polar/Ziggurat) |
| `randint()`, `integers()` | Statistical | Correct range, different sequence |
| `exponential()`, `poisson()`, `binomial()` | Statistical | Correct distributions |
| `choice()`, `permutation()`, `shuffle()` | Statistical | Correct behavior |

**Exact match** means `np.random.seed(42)` in Python and `random.seed(42)` in numpy-ts produce identical sequences. **Statistical match** means correct distributions but different sequences.

---
