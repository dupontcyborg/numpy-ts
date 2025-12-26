# API Reference

Complete NumPy 2.0+ API compatibility checklist.

**Last Updated**: 2025-12-26

## Progress Summary

Based on `npm run compare-api`:

- **Overall Coverage**: 475/505 (94.1%)
- **Top-level Functions**: 472/505 (89.7%)
- **NDArray Methods**: 174/53 (88.7%)

### Completed Categories (100%)
- Arithmetic (29/29)
- Array Creation (35/35)
- Array Manipulation (46/46)
- Bit Operations (13/13)
- Broadcasting (3/3)
- Comparison (10/10)
- Exponential (9/9)
- FFT (18/18)
- Gradient (4/4)
- Hyperbolic (9/9)
- I/O (8/8)
- Indexing (21/21)
- Linear Algebra (16/16)
- Linear Algebra (linalg) (31/31)
- Logic (24/24)
- Other Math (15/15)
- Polynomials (10/10)
- Printing/Formatting (10/10)
- Random (53/53)
- Reductions (36/36)
- Rounding (7/7)
- Searching (7/7)
- Set Operations (11/11)
- Sorting (6/6)
- Statistics (11/11)
- Trigonometric (16/16)
- Type Checking (7/7)
- Utilities (10/10)

### Incomplete Categories
- NDArray Methods (47/53) - 88.7%
- Unplanned (0/30) - 0.0%

---

## Arithmetic

- [x] `abs` 
- [x] `absolute` 
- [x] `add` 
- [x] `cbrt` 
- [x] `divide` 
- [x] `divmod` 
- [x] `fabs` 
- [x] `float_power` 
- [x] `floor_divide` 
- [x] `fmod` 
- [x] `frexp` 
- [x] `gcd` 
- [x] `heaviside` 
- [x] `lcm` 
- [x] `ldexp` 
- [x] `mod` 
- [x] `modf` 
- [x] `multiply` 
- [x] `negative` 
- [x] `positive` 
- [x] `pow` 
- [x] `power` 
- [x] `reciprocal` 
- [x] `remainder` 
- [x] `sign` 
- [x] `sqrt` 
- [x] `square` 
- [x] `subtract` 
- [x] `true_divide` 

---

## Array Creation

- [x] `arange` 
- [x] `array` 
- [x] `asanyarray` 
- [x] `asarray` 
- [x] `asarray_chkfinite` 
- [x] `ascontiguousarray` 
- [x] `asfortranarray` 
- [x] `astype` 
- [x] `copy` 
- [x] `diag` 
- [x] `diagflat` 
- [x] `empty` 
- [x] `empty_like` 
- [x] `eye` 
- [x] `frombuffer` 
- [x] `fromfile` 
- [x] `fromfunction` 
- [x] `fromiter` 
- [x] `fromstring` 
- [x] `full` 
- [x] `full_like` 
- [x] `geomspace` 
- [x] `identity` 
- [x] `linspace` 
- [x] `logspace` 
- [x] `meshgrid` 
- [x] `ones` 
- [x] `ones_like` 
- [x] `require` 
- [x] `tri` 
- [x] `tril` 
- [x] `triu` 
- [x] `vander` 
- [x] `zeros` 
- [x] `zeros_like` 

---

## Array Manipulation

- [x] `append` 
- [x] `array_split` 
- [x] `atleast_1d` 
- [x] `atleast_2d` 
- [x] `atleast_3d` 
- [x] `block` 
- [x] `byteswap` 
- [x] `column_stack` 
- [x] `concat` 
- [x] `concatenate` 
- [x] `delete` 
- [x] `dsplit` 
- [x] `dstack` 
- [x] `expand_dims` 
- [x] `fill` 
- [x] `flatten` 
- [x] `flip` 
- [x] `fliplr` 
- [x] `flipud` 
- [x] `hsplit` 
- [x] `hstack` 
- [x] `insert` 
- [x] `item` 
- [x] `moveaxis` 
- [x] `pad` 
- [x] `ravel` 
- [x] `repeat` 
- [x] `reshape` 
- [x] `resize` 
- [x] `roll` 
- [x] `rollaxis` 
- [x] `rot90` 
- [x] `row_stack` 
- [x] `split` 
- [x] `squeeze` 
- [x] `stack` 
- [x] `swapaxes` 
- [x] `tile` 
- [x] `tobytes` 
- [x] `tofile` 
- [x] `tolist` 
- [x] `transpose` 
- [x] `unstack` 
- [x] `view` 
- [x] `vsplit` 
- [x] `vstack` 

---

## Bit Operations

- [x] `bitwise_and` 
- [x] `bitwise_count` 
- [x] `bitwise_invert` 
- [x] `bitwise_left_shift` 
- [x] `bitwise_not` 
- [x] `bitwise_or` 
- [x] `bitwise_right_shift` 
- [x] `bitwise_xor` 
- [x] `invert` 
- [x] `left_shift` 
- [x] `packbits` 
- [x] `right_shift` 
- [x] `unpackbits` 

---

## Broadcasting

- [x] `broadcast_arrays` 
- [x] `broadcast_shapes` 
- [x] `broadcast_to` 

---

## Comparison

- [x] `allclose` 
- [x] `array_equal` 
- [x] `array_equiv` 
- [x] `equal` 
- [x] `greater` 
- [x] `greater_equal` 
- [x] `isclose` 
- [x] `less` 
- [x] `less_equal` 
- [x] `not_equal` 

---

## Exponential

- [x] `exp` 
- [x] `exp2` 
- [x] `expm1` 
- [x] `log` 
- [x] `log10` 
- [x] `log1p` 
- [x] `log2` 
- [x] `logaddexp` 
- [x] `logaddexp2` 

---

## FFT

- [x] `fft.fft` 
- [x] `fft.fft2` 
- [x] `fft.fftfreq` 
- [x] `fft.fftn` 
- [x] `fft.fftshift` 
- [x] `fft.hfft` 
- [x] `fft.ifft` 
- [x] `fft.ifft2` 
- [x] `fft.ifftn` 
- [x] `fft.ifftshift` 
- [x] `fft.ihfft` 
- [x] `fft.irfft` 
- [x] `fft.irfft2` 
- [x] `fft.irfftn` 
- [x] `fft.rfft` 
- [x] `fft.rfft2` 
- [x] `fft.rfftfreq` 
- [x] `fft.rfftn` 

---

## Gradient

- [x] `cross` 
- [x] `diff` 
- [x] `ediff1d` 
- [x] `gradient` 

---

## Hyperbolic

- [x] `acosh` 
- [x] `arccosh` 
- [x] `arcsinh` 
- [x] `arctanh` 
- [x] `asinh` 
- [x] `atanh` 
- [x] `cosh` 
- [x] `sinh` 
- [x] `tanh` 

---

## I/O

- [x] `fromregex` 
- [x] `genfromtxt` 
- [x] `load` 
- [x] `loadtxt` 
- [x] `save` 
- [x] `savetxt` 
- [x] `savez` 
- [x] `savez_compressed` 

---

## Indexing

- [x] `choose` 
- [x] `compress` 
- [x] `diag_indices` 
- [x] `diag_indices_from` 
- [x] `fill_diagonal` 
- [x] `indices` 
- [x] `ix_` 
- [x] `mask_indices` 
- [x] `place` 
- [x] `put` 
- [x] `put_along_axis` 
- [x] `putmask` 
- [x] `ravel_multi_index` 
- [x] `select` 
- [x] `take` 
- [x] `take_along_axis` 
- [x] `tril_indices` 
- [x] `tril_indices_from` 
- [x] `triu_indices` 
- [x] `triu_indices_from` 
- [x] `unravel_index` 

---

## Linear Algebra

- [x] `diagonal` 
- [x] `dot` 
- [x] `einsum` 
- [x] `einsum_path` 
- [x] `inner` 
- [x] `kron` 
- [x] `matmul` 
- [x] `matrix_transpose` 
- [x] `matvec` 
- [x] `outer` 
- [x] `permute_dims` 
- [x] `tensordot` 
- [x] `trace` 
- [x] `vdot` 
- [x] `vecdot` 
- [x] `vecmat` 

---

## Linear Algebra (linalg)

- [x] `linalg.cholesky` 
- [x] `linalg.cond` 
- [x] `linalg.cross` 
- [x] `linalg.det` 
- [x] `linalg.diagonal` 
- [x] `linalg.eig` 
- [x] `linalg.eigh` 
- [x] `linalg.eigvals` 
- [x] `linalg.eigvalsh` 
- [x] `linalg.inv` 
- [x] `linalg.lstsq` 
- [x] `linalg.matmul` 
- [x] `linalg.matrix_norm` 
- [x] `linalg.matrix_power` 
- [x] `linalg.matrix_rank` 
- [x] `linalg.matrix_transpose` 
- [x] `linalg.multi_dot` 
- [x] `linalg.norm` 
- [x] `linalg.outer` 
- [x] `linalg.pinv` 
- [x] `linalg.qr` 
- [x] `linalg.slogdet` 
- [x] `linalg.solve` 
- [x] `linalg.svd` 
- [x] `linalg.svdvals` 
- [x] `linalg.tensordot` 
- [x] `linalg.tensorinv` 
- [x] `linalg.tensorsolve` 
- [x] `linalg.trace` 
- [x] `linalg.vecdot` 
- [x] `linalg.vector_norm` 

---

## Logic

- [x] `copysign` 
- [x] `iscomplex` 
- [x] `iscomplexobj` 
- [x] `isdtype` 
- [x] `isfinite` 
- [x] `isfortran` 
- [x] `isinf` 
- [x] `isnan` 
- [x] `isnat` 
- [x] `isneginf` 
- [x] `isposinf` 
- [x] `isreal` 
- [x] `isrealobj` 
- [x] `isscalar` 
- [x] `iterable` 
- [x] `logical_and` 
- [x] `logical_not` 
- [x] `logical_or` 
- [x] `logical_xor` 
- [x] `nextafter` 
- [x] `promote_types` 
- [x] `real_if_close` 
- [x] `signbit` 
- [x] `spacing` 

---

## NDArray Methods

- [x] `all` 
- [x] `any` 
- [x] `argmax` 
- [x] `argmin` 
- [x] `argpartition` 
- [x] `argsort` 
- [x] `astype` 
- [x] `byteswap` 
- [x] `choose` 
- [x] `clip` 
- [x] `compress` 
- [x] `conj` 
- [x] `conjugate` 
- [x] `copy` 
- [x] `cumprod` 
- [x] `cumsum` 
- [x] `diagonal` 
- [x] `dot` 
- [ ] `dump` 
- [ ] `dumps` 
- [x] `fill` 
- [x] `flatten` 
- [ ] `getfield` 
- [x] `item` 
- [x] `max` 
- [x] `mean` 
- [x] `min` 
- [x] `nonzero` 
- [x] `partition` 
- [x] `prod` 
- [x] `put` 
- [x] `ravel` 
- [x] `repeat` 
- [x] `reshape` 
- [x] `resize` 
- [x] `round` 
- [x] `searchsorted` 
- [ ] `setfield` 
- [ ] `setflags` 
- [x] `sort` 
- [x] `squeeze` 
- [x] `std` 
- [x] `sum` 
- [x] `swapaxes` 
- [x] `take` 
- [ ] `to_device` 
- [x] `tobytes` 
- [x] `tofile` 
- [x] `tolist` 
- [x] `trace` 
- [x] `transpose` 
- [x] `var` 
- [x] `view` 

---

## Other Math

- [x] `angle` 
- [x] `clip` 
- [x] `conj` 
- [x] `conjugate` 
- [x] `fmax` 
- [x] `fmin` 
- [x] `i0` 
- [x] `imag` 
- [x] `interp` 
- [x] `maximum` 
- [x] `minimum` 
- [x] `nan_to_num` 
- [x] `real` 
- [x] `sinc` 
- [x] `unwrap` 

---

## Polynomials

- [x] `poly` 
- [x] `polyadd` 
- [x] `polyder` 
- [x] `polydiv` 
- [x] `polyfit` 
- [x] `polyint` 
- [x] `polymul` 
- [x] `polysub` 
- [x] `polyval` 
- [x] `roots` 

---

## Printing/Formatting

- [x] `array2string` 
- [x] `array_repr` 
- [x] `array_str` 
- [x] `base_repr` 
- [x] `binary_repr` 
- [x] `format_float_positional` 
- [x] `format_float_scientific` 
- [x] `get_printoptions` 
- [x] `printoptions` 
- [x] `set_printoptions` 

---

## Random

- [x] `random.beta` 
- [x] `random.binomial` 
- [x] `random.bytes` 
- [x] `random.chisquare` 
- [x] `random.choice` 
- [x] `random.default_rng` 
- [x] `random.dirichlet` 
- [x] `random.exponential` 
- [x] `random.f` 
- [x] `random.gamma` 
- [x] `random.geometric` 
- [x] `random.get_bit_generator` 
- [x] `random.get_state` 
- [x] `random.gumbel` 
- [x] `random.hypergeometric` 
- [x] `random.laplace` 
- [x] `random.logistic` 
- [x] `random.lognormal` 
- [x] `random.logseries` 
- [x] `random.multinomial` 
- [x] `random.multivariate_normal` 
- [x] `random.negative_binomial` 
- [x] `random.noncentral_chisquare` 
- [x] `random.noncentral_f` 
- [x] `random.normal` 
- [x] `random.pareto` 
- [x] `random.permutation` 
- [x] `random.poisson` 
- [x] `random.power` 
- [x] `random.rand` 
- [x] `random.randint` 
- [x] `random.randn` 
- [x] `random.random` 
- [x] `random.random_integers` 
- [x] `random.random_sample` 
- [x] `random.ranf` 
- [x] `random.rayleigh` 
- [x] `random.sample` 
- [x] `random.seed` 
- [x] `random.set_bit_generator` 
- [x] `random.set_state` 
- [x] `random.shuffle` 
- [x] `random.standard_cauchy` 
- [x] `random.standard_exponential` 
- [x] `random.standard_gamma` 
- [x] `random.standard_normal` 
- [x] `random.standard_t` 
- [x] `random.triangular` 
- [x] `random.uniform` 
- [x] `random.vonmises` 
- [x] `random.wald` 
- [x] `random.weibull` 
- [x] `random.zipf` 

---

## Reductions

- [x] `all` 
- [x] `amax` 
- [x] `amin` 
- [x] `any` 
- [x] `argmax` 
- [x] `argmin` 
- [x] `average` 
- [x] `cumprod` 
- [x] `cumsum` 
- [x] `cumulative_prod` 
- [x] `cumulative_sum` 
- [x] `max` 
- [x] `mean` 
- [x] `median` 
- [x] `min` 
- [x] `nanargmax` 
- [x] `nanargmin` 
- [x] `nancumprod` 
- [x] `nancumsum` 
- [x] `nanmax` 
- [x] `nanmean` 
- [x] `nanmedian` 
- [x] `nanmin` 
- [x] `nanpercentile` 
- [x] `nanprod` 
- [x] `nanquantile` 
- [x] `nanstd` 
- [x] `nansum` 
- [x] `nanvar` 
- [x] `percentile` 
- [x] `prod` 
- [x] `ptp` 
- [x] `quantile` 
- [x] `std` 
- [x] `sum` 
- [x] `var` 

---

## Rounding

- [x] `around` 
- [x] `ceil` 
- [x] `fix` 
- [x] `floor` 
- [x] `rint` 
- [x] `round` 
- [x] `trunc` 

---

## Searching

- [x] `argwhere` 
- [x] `count_nonzero` 
- [x] `extract` 
- [x] `flatnonzero` 
- [x] `nonzero` 
- [x] `searchsorted` 
- [x] `where` 

---

## Set Operations

- [x] `intersect1d` 
- [x] `isin` 
- [x] `setdiff1d` 
- [x] `setxor1d` 
- [x] `trim_zeros` 
- [x] `union1d` 
- [x] `unique` 
- [x] `unique_all` 
- [x] `unique_counts` 
- [x] `unique_inverse` 
- [x] `unique_values` 

---

## Sorting

- [x] `argpartition` 
- [x] `argsort` 
- [x] `lexsort` 
- [x] `partition` 
- [x] `sort` 
- [x] `sort_complex` 

---

## Statistics

- [x] `bincount` 
- [x] `convolve` 
- [x] `corrcoef` 
- [x] `correlate` 
- [x] `cov` 
- [x] `digitize` 
- [x] `histogram` 
- [x] `histogram2d` 
- [x] `histogram_bin_edges` 
- [x] `histogramdd` 
- [x] `trapezoid` 

---

## Trigonometric

- [x] `acos` 
- [x] `arccos` 
- [x] `arcsin` 
- [x] `arctan` 
- [x] `arctan2` 
- [x] `asin` 
- [x] `atan` 
- [x] `atan2` 
- [x] `cos` 
- [x] `deg2rad` 
- [x] `degrees` 
- [x] `hypot` 
- [x] `rad2deg` 
- [x] `radians` 
- [x] `sin` 
- [x] `tan` 

---

## Type Checking

- [x] `can_cast` 
- [x] `common_type` 
- [x] `issubdtype` 
- [x] `min_scalar_type` 
- [x] `mintypecode` 
- [x] `result_type` 
- [x] `typename` 

---

## Unplanned

- [ ] `asmatrix` 
- [ ] `bartlett` 
- [ ] `blackman` 
- [ ] `bmat` 
- [ ] `busday_count` 
- [ ] `busday_offset` 
- [ ] `datetime_as_string` 
- [ ] `datetime_data` 
- [ ] `dump` 
- [ ] `dumps` 
- [ ] `from_dlpack` 
- [ ] `frompyfunc` 
- [ ] `get_include` 
- [ ] `getbufsize` 
- [ ] `geterrcall` 
- [ ] `getfield` 
- [ ] `hamming` 
- [ ] `hanning` 
- [ ] `info` 
- [ ] `is_busday` 
- [ ] `kaiser` 
- [ ] `nested_iters` 
- [ ] `piecewise` 
- [ ] `setbufsize` 
- [ ] `seterrcall` 
- [ ] `setfield` 
- [ ] `setflags` 
- [ ] `show_config` 
- [ ] `show_runtime` 
- [ ] `to_device` 

---

## Utilities

- [x] `apply_along_axis` 
- [x] `apply_over_axes` 
- [x] `copyto` 
- [x] `geterr` 
- [x] `may_share_memory` 
- [x] `ndim` 
- [x] `seterr` 
- [x] `shape` 
- [x] `shares_memory` 
- [x] `size` 

---

## Extra NDArray Methods

Methods in numpy-ts NDArray that don't exist in NumPy's ndarray.
These may be removed in future versions for strict NumPy compatibility:

- `T()` 
- `absolute()` 
- `add()` 
- `allclose()` 
- `arccos()` 
- `arccosh()` 
- `arcsin()` 
- `arcsinh()` 
- `arctan()` 
- `arctan2()` 
- `arctanh()` 
- `argwhere()` 
- `around()` 
- `average()` 
- `base()` 
- `bindex()` 
- `bitwise_and()` 
- `bitwise_not()` 
- `bitwise_or()` 
- `bitwise_xor()` 
- `cbrt()` 
- `ceil()` 
- `col()` 
- `cols()` 
- `copysign()` 
- `cos()` 
- `cosh()` 
- `data()` 
- `degrees()` 
- `diff()` 
- `divide()` 
- `divmod()` 
- `dtype()` 
- `equal()` 
- `exp()` 
- `exp2()` 
- `expand_dims()` 
- `expm1()` 
- `fabs()` 
- `fix()` 
- `flags()` 
- `floor()` 
- `floor_divide()` 
- `get()` 
- `greater()` 
- `greater_equal()` 
- `heaviside()` 
- `hypot()` 
- `iindex()` 
- `inner()` 
- `invert()` 
- `isclose()` 
- `isfinite()` 
- `isinf()` 
- `isnan()` 
- `isnat()` 
- `itemsize()` 
- `left_shift()` 
- `less()` 
- `less_equal()` 
- `log()` 
- `log10()` 
- `log1p()` 
- `log2()` 
- `logaddexp()` 
- `logaddexp2()` 
- `logical_and()` 
- `logical_not()` 
- `logical_or()` 
- `logical_xor()` 
- `matmul()` 
- `median()` 
- `mod()` 
- `moveaxis()` 
- `multiply()` 
- `nanargmax()` 
- `nanargmin()` 
- `nancumprod()` 
- `nancumsum()` 
- `nanmax()` 
- `nanmean()` 
- `nanmedian()` 
- `nanmin()` 
- `nanpercentile()` 
- `nanprod()` 
- `nanquantile()` 
- `nanstd()` 
- `nansum()` 
- `nanvar()` 
- `nbytes()` 
- `ndim()` 
- `negative()` 
- `nextafter()` 
- `not_equal()` 
- `outer()` 
- `percentile()` 
- `positive()` 
- `power()` 
- `ptp()` 
- `quantile()` 
- `radians()` 
- `reciprocal()` 
- `remainder()` 
- `right_shift()` 
- `rint()` 
- `row()` 
- `rows()` 
- `set()` 
- `shape()` 
- `sign()` 
- `signbit()` 
- `sin()` 
- `sinh()` 
- `size()` 
- `slice()` 
- `spacing()` 
- `sqrt()` 
- `square()` 
- `storage()` 
- `strides()` 
- `subtract()` 
- `tan()` 
- `tanh()` 
- `tensordot()` 
- `toArray()` 
- `toString()` 
- `trunc()` 

---

## Notes

### Core NDArray Properties
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
