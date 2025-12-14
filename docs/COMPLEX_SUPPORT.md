# Complex Number Support Tracking

This document tracks complex number (`complex64`, `complex128`) support across all numpy-ts operations.

## Status Legend

- ✅ **Implemented** - Full complex support with correct mathematical behavior
- ⏳ **TODO** - Should support complex, needs implementation
- ❌ **N/A** - Complex numbers not mathematically applicable (should throw error)
- 🔇 **Silent Fail** - Currently gives wrong results without error (CRITICAL to fix)

---

## Currently Implemented (✅)

These functions have proper complex support:

### Core Operations (`src/ops/arithmetic.ts`)
- `add` - Complex addition
- `subtract` - Complex subtraction
- `multiply` - Complex multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
- `divide` - Complex division
- `negative` - Negation: `-(a+bi) = -a-bi`
- `absolute` / `abs` - Magnitude: `|a+bi| = sqrt(a² + b²)`

### Power Operations (`src/ops/exponential.ts`)
- `sqrt` - Complex square root
- `power` - Complex exponentiation

### Comparison Operations (`src/ops/comparison.ts`)
- `equal` - Element-wise equality
- `notEqual` - Element-wise inequality
- `greater` - Lexicographic comparison (real, then imag)
- `greaterEqual` - Lexicographic comparison
- `less` - Lexicographic comparison
- `lessEqual` - Lexicographic comparison

### Reduction Operations (`src/ops/reduction.ts`)
- `sum` - Returns Complex for complex arrays
- `mean` - Returns Complex for complex arrays
- `prod` - Returns Complex for complex arrays

### Type Checking (`src/ops/logic.ts`)
- `iscomplex` - True for elements with non-zero imaginary part
- `iscomplexobj` - True if array dtype is complex
- `isreal` - True for elements with zero imaginary part
- `isrealobj` - True if array dtype is not complex

### Complex-Specific (`src/ops/complex.ts`)
- `real` - Extract real part
- `imag` - Extract imaginary part
- `conj` / `conjugate` - Complex conjugate
- `angle` - Phase angle (argument)

---

## TODO: Needs Complex Implementation (⏳)

### High Priority - Currently Giving Wrong Results (🔇 → ⏳)

These functions currently **silently return incorrect results** by ignoring the imaginary part:

#### Trigonometric (`src/ops/trig.ts`)
| Function | Formula | Current Behavior |
|----------|---------|------------------|
| `sin` | `sin(a+bi) = sin(a)cosh(b) + i·cos(a)sinh(b)` | 🔇 Returns `sin(a)` only |
| `cos` | `cos(a+bi) = cos(a)cosh(b) - i·sin(a)sinh(b)` | 🔇 Returns `cos(a)` only |
| `tan` | `tan(z) = sin(z)/cos(z)` | 🔇 Returns `tan(a)` only |
| `arcsin` | Complex inverse sine | 🔇 Returns `arcsin(a)` only |
| `arccos` | Complex inverse cosine | 🔇 Returns `arccos(a)` only |
| `arctan` | Complex inverse tangent | 🔇 Returns `arctan(a)` only |

#### Hyperbolic (`src/ops/hyperbolic.ts`)
| Function | Formula | Current Behavior |
|----------|---------|------------------|
| `sinh` | `sinh(a+bi) = sinh(a)cos(b) + i·cosh(a)sin(b)` | 🔇 Returns `sinh(a)` only |
| `cosh` | `cosh(a+bi) = cosh(a)cos(b) + i·sinh(a)sin(b)` | 🔇 Returns `cosh(a)` only |
| `tanh` | `tanh(z) = sinh(z)/cosh(z)` | 🔇 Returns `tanh(a)` only |
| `arcsinh` | Complex inverse hyperbolic sine | 🔇 Returns `arcsinh(a)` only |
| `arccosh` | Complex inverse hyperbolic cosine | 🔇 Returns `arccosh(a)` only |
| `arctanh` | Complex inverse hyperbolic tangent | 🔇 Returns `arctanh(a)` only |

#### Exponential/Logarithmic (`src/ops/exponential.ts`)
| Function | Formula | Current Behavior |
|----------|---------|------------------|
| `exp` | `exp(a+bi) = exp(a)(cos(b) + i·sin(b))` | 🔇 Returns `exp(a)` only |
| `exp2` | `2^(a+bi)` | 🔇 Returns `2^a` only |
| `expm1` | `exp(z) - 1` | 🔇 Returns `exp(a)-1` only |
| `log` | `log(a+bi) = log(|z|) + i·arg(z)` | 🔇 Returns `log(a)` only |
| `log2` | `log₂(z)` | 🔇 Returns `log₂(a)` only |
| `log10` | `log₁₀(z)` | 🔇 Returns `log₁₀(a)` only |
| `log1p` | `log(1+z)` | 🔇 Returns `log(1+a)` only |

### Medium Priority

#### Arithmetic (`src/ops/arithmetic.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `reciprocal` | ⏳ | `1/z` - needs complex division |
| `square` | ⏳ | `z²` - can use multiply |
| `cbrt` | ⏳ | Cube root of complex |
| `float_power` | ⏳ | Same as `power` |
| `positive` | ⏳ | Unary `+` (identity) |

#### Reduction (`src/ops/reduction.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `variance` | ⏳ | Use `|z - mean|²` |
| `std` | ⏳ | Square root of variance |
| `cumsum` | ⏳ | Cumulative sum |
| `cumprod` | ⏳ | Cumulative product |
| `average` | ⏳ | Weighted mean |
| `nansum` | ⏳ | Sum ignoring NaN |
| `nanmean` | ⏳ | Mean ignoring NaN |
| `nanprod` | ⏳ | Product ignoring NaN |

#### Linear Algebra (`src/ops/linalg.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `dot` | ⏳ | Matrix/vector dot product |
| `matmul` | ⏳ | Matrix multiplication |
| `inner` | ⏳ | Inner product (with conjugate) |
| `outer` | ⏳ | Outer product |
| `trace` | ⏳ | Sum of diagonal |
| `norm` | ⏳ | Vector/matrix norms |
| `det` | ⏳ | Determinant |
| `inv` | ⏳ | Matrix inverse |
| `solve` | ⏳ | Solve linear system |
| `eig` | ⏳ | Eigendecomposition |
| `svd` | ⏳ | Singular value decomposition |
| `qr` | ⏳ | QR decomposition |
| `cholesky` | ⏳ | Cholesky decomposition |

#### Gradient/Difference (`src/ops/gradient.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `diff` | ⏳ | Discrete difference |
| `gradient` | ⏳ | Numerical gradient |
| `ediff1d` | ⏳ | Differences between consecutive elements |

#### Set Operations (`src/ops/sets.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `unique` | ⏳ | Unique elements (equality works) |
| `in1d` | ⏳ | Test membership |
| `isin` | ⏳ | Element-wise membership |
| `intersect1d` | ⏳ | Set intersection |
| `union1d` | ⏳ | Set union |
| `setdiff1d` | ⏳ | Set difference |
| `setxor1d` | ⏳ | Symmetric difference |

#### Statistics (`src/ops/statistics.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `cov` | ⏳ | Covariance (uses conjugate) |
| `corrcoef` | ⏳ | Correlation coefficient |
| `correlate` | ⏳ | Cross-correlation |
| `convolve` | ⏳ | Convolution |

#### Logic (`src/ops/logic.ts`)
| Function | Status | Notes |
|----------|--------|-------|
| `isfinite` | ⏳ | Check both components |
| `isinf` | ⏳ | Check both components |
| `isnan` | ⏳ | Check both components |

---

## Not Applicable for Complex (❌)

These operations are not mathematically defined for complex numbers and should **throw an error** when called with complex input:

### Rounding (`src/ops/rounding.ts`)
| Function | Reason |
|----------|--------|
| `floor` | Rounding undefined for complex |
| `ceil` | Rounding undefined for complex |
| `round` | Rounding undefined for complex |
| `trunc` | Truncation undefined for complex |
| `fix` | Truncation undefined for complex |
| `rint` | Rounding undefined for complex |
| `around` | Rounding undefined for complex |

### Ordering/Comparison (`src/ops/reduction.ts`, `src/ops/sorting.ts`)
| Function | Reason |
|----------|--------|
| `max` | No total ordering for complex |
| `min` | No total ordering for complex |
| `argmax` | No total ordering for complex |
| `argmin` | No total ordering for complex |
| `ptp` | Requires max/min |
| `median` | Requires ordering |
| `percentile` | Requires ordering |
| `quantile` | Requires ordering |
| `sort` | No total ordering (NumPy uses lexicographic, but warns) |
| `argsort` | No total ordering |
| `searchsorted` | Requires ordering |

### Modulo/Division (`src/ops/arithmetic.ts`)
| Function | Reason |
|----------|--------|
| `mod` | Modulo undefined for complex |
| `fmod` | Modulo undefined for complex |
| `remainder` | Remainder undefined for complex |
| `floorDivide` | Floor division undefined |
| `divmod` | Division with modulo undefined |

### Sign (`src/ops/arithmetic.ts`)
| Function | Reason |
|----------|--------|
| `sign` | Sign function undefined for complex |
| `signbit` | Sign bit undefined for complex |
| `copysign` | Sign undefined for complex |

### Bitwise (`src/ops/bitwise.ts`)
| Function | Reason |
|----------|--------|
| `bitwise_and` | Integer operation only |
| `bitwise_or` | Integer operation only |
| `bitwise_xor` | Integer operation only |
| `bitwise_not` | Integer operation only |
| `invert` | Integer operation only |
| `left_shift` | Integer operation only |
| `right_shift` | Integer operation only |

### Integer-Only (`src/ops/arithmetic.ts`)
| Function | Reason |
|----------|--------|
| `gcd` | Greatest common divisor - integers only |
| `lcm` | Least common multiple - integers only |

### Float-Specific (`src/ops/arithmetic.ts`)
| Function | Reason |
|----------|--------|
| `frexp` | Float mantissa/exponent decomposition |
| `ldexp` | Float mantissa/exponent composition |
| `modf` | Integer/fractional decomposition |
| `nextafter` | Float ULP operations |
| `spacing` | Float ULP spacing |

### Other
| Function | Reason |
|----------|--------|
| `heaviside` | Step function - requires ordering |
| `bincount` | Counting - integer indices only |
| `digitize` | Binning - requires ordering |
| `histogram` | Binning - requires ordering |

---

## Implementation Notes

### Error Handling Pattern

For functions that don't support complex:

```typescript
import { isComplexDType } from '../core/dtype';

export function floor(storage: ArrayStorage): ArrayStorage {
  if (isComplexDType(storage.dtype)) {
    throw new TypeError(
      `floor is not supported for complex dtype '${storage.dtype}'. ` +
      `Complex numbers have no natural ordering for rounding operations.`
    );
  }
  // ... existing implementation
}
```

### Complex Math Formulas

Key formulas for implementing complex support:

```
exp(a+bi) = exp(a) * (cos(b) + i*sin(b))

log(a+bi) = log(sqrt(a² + b²)) + i*atan2(b, a)

sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)

sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)

sqrt(a+bi):
  r = sqrt(a² + b²)
  re = sqrt((r + a) / 2)
  im = sign(b) * sqrt((r - a) / 2)
```

---

## Priority Order for Implementation

1. **CRITICAL**: Add error throws for all N/A functions (prevent silent wrong results)
2. **HIGH**: Implement `exp`, `log` (foundation for trig/hyperbolic)
3. **HIGH**: Implement `sin`, `cos`, `tan` and inverses
4. **HIGH**: Implement `sinh`, `cosh`, `tanh` and inverses
5. **MEDIUM**: Implement remaining exponential functions
6. **MEDIUM**: Implement linear algebra operations
7. **MEDIUM**: Implement statistical operations
8. **LOW**: Implement set operations
