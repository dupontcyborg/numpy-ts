/**
 * NDArray - NumPy-compatible multidimensional array
 *
 * Core array class providing NumPy-like API
 */

import { parseSlice, normalizeSlice } from '../common/slicing';
import {
  type DType,
  type TypedArray,
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  getDTypeSize,
  isBigIntDType,
  isComplexDType,
  isComplexLike,
} from '../common/dtype';
import { Complex } from '../common/complex';
import { ArrayStorage } from '../common/storage';
import { computeBroadcastShape } from '../common/broadcasting';
import { NDArrayCore } from '../common/ndarray-core';
import * as arithmeticOps from '../common/ops/arithmetic';
import * as comparisonOps from '../common/ops/comparison';
import * as reductionOps from '../common/ops/reduction';
import * as shapeOps from '../common/ops/shape';
import * as linalgOps from '../common/ops/linalg';
import * as exponentialOps from '../common/ops/exponential';
import * as trigOps from '../common/ops/trig';
import * as hyperbolicOps from '../common/ops/hyperbolic';
import * as advancedOps from '../common/ops/advanced';
import * as bitwiseOps from '../common/ops/bitwise';
import * as logicOps from '../common/ops/logic';
import * as complexOps from '../common/ops/complex';
import * as sortingOps from '../common/ops/sorting';
import * as roundingOps from '../common/ops/rounding';
import * as setOps from '../common/ops/sets';
import * as gradientOps from '../common/ops/gradient';
import * as statisticsOps from '../common/ops/statistics';
import * as formattingOps from '../common/ops/formatting';

export class NDArray extends NDArrayCore {
  // Override _base with NDArray type
  protected override _base?: NDArray;

  constructor(storage: ArrayStorage, base?: NDArray) {
    super(storage, base);
    this._base = base;
  }

  /**
   * Create NDArray from storage (for ops modules)
   * @internal
   */
  static override _fromStorage(storage: ArrayStorage, base?: NDArray): NDArray {
    return new NDArray(storage, base);
  }

  /**
   * Base array if this is a view, null if this array owns its data
   * Similar to NumPy's base attribute
   */
  override get base(): NDArray | null {
    return this._base ?? null;
  }

  /**
   * Transpose of the array (shorthand for transpose())
   * Returns a view with axes reversed
   */
  get T(): NDArray {
    return this.transpose();
  }

  /**
   * Size of one array element in bytes
   */
  override get itemsize(): number {
    return getDTypeSize(this._storage.dtype);
  }

  /**
   * Total bytes consumed by the elements of the array
   */
  override get nbytes(): number {
    return this.size * this.itemsize;
  }

  /**
   * Fill the array with a scalar value (in-place)
   * @param value - Value to fill with
   */
  override fill(value: number | bigint): void {
    const dtype = this._storage.dtype;
    const size = this.size;

    if (isBigIntDType(dtype)) {
      const bigintValue = typeof value === 'bigint' ? value : BigInt(Math.round(Number(value)));
      for (let i = 0; i < size; i++) {
        this._storage.iset(i, bigintValue);
      }
    } else if (dtype === 'bool') {
      const boolValue = value ? 1 : 0;
      for (let i = 0; i < size; i++) {
        this._storage.iset(i, boolValue);
      }
    } else {
      const numValue = Number(value);
      for (let i = 0; i < size; i++) {
        this._storage.iset(i, numValue);
      }
    }
  }

  /**
   * Iterator protocol - iterate over the first axis
   * For 1D arrays, yields elements; for ND arrays, yields (N-1)D subarrays
   */
  override *[Symbol.iterator](): Iterator<NDArray | number | bigint | Complex> {
    if (this.ndim === 0) {
      // 0D array: yield the single element
      yield this._storage.iget(0);
    } else if (this.ndim === 1) {
      // 1D array: yield elements
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this._storage.iget(i);
      }
    } else {
      // ND array: yield slices along first axis
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this.slice(String(i));
      }
    }
  }

  /**
   * Get a single element from the array
   * @param indices - Array of indices, one per dimension (e.g., [0, 1] for 2D array)
   * @returns The element value (BigInt for int64/uint64, Complex for complex, number otherwise)
   */
  override get(indices: number[]): number | bigint | Complex {
    // Validate number of indices
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`
      );
    }

    // Normalize negative indices
    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      // Validate bounds
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`
        );
      }
      return normalized;
    });

    return this._storage.get(...normalizedIndices);
  }

  /**
   * Set a single element in the array
   * @param indices - Array of indices, one per dimension (e.g., [0, 1] for 2D array)
   * @param value - Value to set (will be converted to array's dtype)
   */
  override set(
    indices: number[],
    value: number | bigint | Complex | { re: number; im: number }
  ): void {
    // Validate number of indices
    if (indices.length !== this.ndim) {
      throw new Error(
        `Index has ${indices.length} dimensions, but array has ${this.ndim} dimensions`
      );
    }

    // Normalize negative indices
    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      // Validate bounds
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          `Index ${idx} is out of bounds for axis ${dim} with size ${this.shape[dim]}`
        );
      }
      return normalized;
    });

    // Convert value to appropriate type based on dtype
    const currentDtype = this.dtype as DType;

    if (isComplexDType(currentDtype)) {
      // For complex dtypes, pass the value directly to storage
      // (storage.set handles Complex, {re, im}, and number)
      this._storage.set(normalizedIndices, value);
    } else if (isBigIntDType(currentDtype)) {
      // Convert to BigInt for BigInt dtypes
      const numValue = value instanceof Complex ? value.re : Number(value);
      const convertedValue = typeof value === 'bigint' ? value : BigInt(Math.round(numValue));
      this._storage.set(normalizedIndices, convertedValue);
    } else if (currentDtype === 'bool') {
      // Convert to 0 or 1 for bool dtype
      const numValue = value instanceof Complex ? value.re : Number(value);
      const convertedValue = numValue ? 1 : 0;
      this._storage.set(normalizedIndices, convertedValue);
    } else {
      // Convert to number for all other dtypes
      const convertedValue = value instanceof Complex ? value.re : Number(value);
      this._storage.set(normalizedIndices, convertedValue);
    }
  }

  /**
   * Return a deep copy of the array
   */
  override copy(): NDArray {
    return new NDArray(this._storage.copy());
  }

  /**
   * Cast array to a different dtype
   * @param dtype - Target dtype
   * @param copy - If false and dtype matches, return self; otherwise create copy (default: true)
   * @returns Array with specified dtype
   */
  override astype(dtype: DType, copy: boolean = true): NDArray {
    const currentDtype = this.dtype as DType;

    // If dtype matches and copy=false, return self
    if (currentDtype === dtype && !copy) {
      return this;
    }

    // If dtype matches and copy=true, create a copy
    if (currentDtype === dtype && copy) {
      return this.copy();
    }

    // Need to convert dtype
    const shape = Array.from(this.shape);
    const size = this.size;

    // Get TypedArray constructor for conversion
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(`Cannot convert to dtype ${dtype}`);
    }
    const newData = new Constructor(size);
    const oldData = this.data;

    // Handle BigInt to other types
    if (isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      if (dtype === 'bool') {
        for (let i = 0; i < size; i++) {
          (newData as Uint8Array)[i] = typedOldData[i] !== BigInt(0) ? 1 : 0;
        }
      } else {
        for (let i = 0; i < size; i++) {
          (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(
            typedOldData[i]
          );
        }
      }
    }
    // Handle other types to BigInt
    else if (!isBigIntDType(currentDtype) && isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = BigInt(
          Math.round(Number(typedOldData[i]))
        );
      }
    }
    // Handle other types to bool
    else if (dtype === 'bool') {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Uint8Array)[i] = typedOldData[i] !== 0 ? 1 : 0;
      }
    }
    // Handle bool to other types
    else if (currentDtype === 'bool' && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Uint8Array;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    }
    // Handle regular numeric conversions
    else if (!isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    }
    // Handle BigInt to BigInt conversions (int64 <-> uint64)
    else {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = typedOldData[i]!;
      }
    }

    const storage = ArrayStorage.fromData(newData, shape, dtype);
    return new NDArray(storage);
  }

  // Arithmetic operations
  /**
   * Element-wise addition
   * @param other - Array or scalar to add
   * @returns Result of addition with broadcasting
   */
  add(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.add(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise subtraction
   * @param other - Array or scalar to subtract
   * @returns Result of subtraction with broadcasting
   */
  subtract(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.subtract(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise multiplication
   * @param other - Array or scalar to multiply
   * @returns Result of multiplication with broadcasting
   */
  multiply(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.multiply(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise division
   * @param other - Array or scalar to divide by
   * @returns Result of division with broadcasting
   */
  divide(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.divide(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise modulo operation
   * @param other - Array or scalar divisor
   * @returns Remainder after division
   */
  mod(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.mod(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise floor division
   * @param other - Array or scalar to divide by
   * @returns Floor of the quotient
   */
  floor_divide(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = arithmeticOps.floorDivide(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Numerical positive (element-wise +x)
   * @returns Copy of the array
   */
  positive(): NDArray {
    const resultStorage = arithmeticOps.positive(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise reciprocal (1/x)
   * @returns New array with reciprocals
   */
  reciprocal(): NDArray {
    const resultStorage = arithmeticOps.reciprocal(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Mathematical operations
  /**
   * Square root of each element
   * Promotes integer types to float64
   * @returns New array with square roots
   */
  sqrt(): NDArray {
    const resultStorage = exponentialOps.sqrt(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Raise elements to power
   * @param exponent - Power to raise to (array or scalar)
   * @returns New array with powered values
   */
  power(exponent: NDArray | number): NDArray {
    const exponentStorage = typeof exponent === 'number' ? exponent : exponent._storage;
    const resultStorage = exponentialOps.power(this._storage, exponentStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Natural exponential (e^x) of each element
   * Promotes integer types to float64
   * @returns New array with exp values
   */
  exp(): NDArray {
    const resultStorage = exponentialOps.exp(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Base-2 exponential (2^x) of each element
   * Promotes integer types to float64
   * @returns New array with exp2 values
   */
  exp2(): NDArray {
    const resultStorage = exponentialOps.exp2(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Exponential minus one (e^x - 1) of each element
   * More accurate than exp(x) - 1 for small x
   * Promotes integer types to float64
   * @returns New array with expm1 values
   */
  expm1(): NDArray {
    const resultStorage = exponentialOps.expm1(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Natural logarithm (ln) of each element
   * Promotes integer types to float64
   * @returns New array with log values
   */
  log(): NDArray {
    const resultStorage = exponentialOps.log(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Base-2 logarithm of each element
   * Promotes integer types to float64
   * @returns New array with log2 values
   */
  log2(): NDArray {
    const resultStorage = exponentialOps.log2(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Base-10 logarithm of each element
   * Promotes integer types to float64
   * @returns New array with log10 values
   */
  log10(): NDArray {
    const resultStorage = exponentialOps.log10(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Natural logarithm of (1 + x) of each element
   * More accurate than log(1 + x) for small x
   * Promotes integer types to float64
   * @returns New array with log1p values
   */
  log1p(): NDArray {
    const resultStorage = exponentialOps.log1p(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Logarithm of the sum of exponentials: log(exp(x1) + exp(x2))
   * More numerically stable than computing the expression directly
   * Promotes integer types to float64
   * @param x2 - Second operand (array or scalar)
   * @returns New array with logaddexp values
   */
  logaddexp(x2: NDArray | number): NDArray {
    const x2Storage = typeof x2 === 'number' ? x2 : x2._storage;
    const resultStorage = exponentialOps.logaddexp(this._storage, x2Storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Logarithm base 2 of the sum of exponentials: log2(2^x1 + 2^x2)
   * More numerically stable than computing the expression directly
   * Promotes integer types to float64
   * @param x2 - Second operand (array or scalar)
   * @returns New array with logaddexp2 values
   */
  logaddexp2(x2: NDArray | number): NDArray {
    const x2Storage = typeof x2 === 'number' ? x2 : x2._storage;
    const resultStorage = exponentialOps.logaddexp2(this._storage, x2Storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Absolute value of each element
   * @returns New array with absolute values
   */
  absolute(): NDArray {
    const resultStorage = arithmeticOps.absolute(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Numerical negative (element-wise negation)
   * @returns New array with negated values
   */
  negative(): NDArray {
    const resultStorage = arithmeticOps.negative(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Sign of each element (-1, 0, or 1)
   * @returns New array with signs
   */
  sign(): NDArray {
    const resultStorage = arithmeticOps.sign(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Rounding operations
  /**
   * Round an array to the given number of decimals
   * @param decimals - Number of decimal places to round to (default: 0)
   * @returns New array with rounded values
   */
  around(decimals: number = 0): NDArray {
    const resultStorage = roundingOps.around(this._storage, decimals);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Round an array to the given number of decimals (alias for around)
   * @param decimals - Number of decimal places to round to (default: 0)
   * @returns New array with rounded values
   */
  round(decimals: number = 0): NDArray {
    return this.around(decimals);
  }

  /**
   * Return the ceiling of the input, element-wise
   * @returns New array with ceiling values
   */
  ceil(): NDArray {
    const resultStorage = roundingOps.ceil(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Round to nearest integer towards zero
   * @returns New array with values truncated towards zero
   */
  fix(): NDArray {
    const resultStorage = roundingOps.fix(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return the floor of the input, element-wise
   * @returns New array with floor values
   */
  floor(): NDArray {
    const resultStorage = roundingOps.floor(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Round elements to the nearest integer
   * @returns New array with rounded integer values
   */
  rint(): NDArray {
    const resultStorage = roundingOps.rint(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return the truncated value of the input, element-wise
   * @returns New array with truncated values
   */
  trunc(): NDArray {
    const resultStorage = roundingOps.trunc(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Trigonometric operations
  /**
   * Sine of each element (in radians)
   * Promotes integer types to float64
   * @returns New array with sine values
   */
  sin(): NDArray {
    const resultStorage = trigOps.sin(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Cosine of each element (in radians)
   * Promotes integer types to float64
   * @returns New array with cosine values
   */
  cos(): NDArray {
    const resultStorage = trigOps.cos(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Tangent of each element (in radians)
   * Promotes integer types to float64
   * @returns New array with tangent values
   */
  tan(): NDArray {
    const resultStorage = trigOps.tan(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Inverse sine of each element
   * Promotes integer types to float64
   * @returns New array with arcsin values (radians)
   */
  arcsin(): NDArray {
    const resultStorage = trigOps.arcsin(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Inverse cosine of each element
   * Promotes integer types to float64
   * @returns New array with arccos values (radians)
   */
  arccos(): NDArray {
    const resultStorage = trigOps.arccos(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Inverse tangent of each element
   * Promotes integer types to float64
   * @returns New array with arctan values (radians)
   */
  arctan(): NDArray {
    const resultStorage = trigOps.arctan(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise arc tangent of this/other choosing the quadrant correctly
   * @param other - x-coordinates (array or scalar)
   * @returns Angle in radians between -π and π
   */
  arctan2(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = trigOps.arctan2(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Given the "legs" of a right triangle, return its hypotenuse
   * Equivalent to sqrt(this**2 + other**2), element-wise
   * @param other - Second leg (array or scalar)
   * @returns Hypotenuse values
   */
  hypot(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = trigOps.hypot(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Convert angles from radians to degrees
   * @returns New array with angles in degrees
   */
  degrees(): NDArray {
    const resultStorage = trigOps.degrees(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Convert angles from degrees to radians
   * @returns New array with angles in radians
   */
  radians(): NDArray {
    const resultStorage = trigOps.radians(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Hyperbolic operations
  /**
   * Hyperbolic sine of each element
   * Promotes integer types to float64
   * @returns New array with sinh values
   */
  sinh(): NDArray {
    const resultStorage = hyperbolicOps.sinh(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Hyperbolic cosine of each element
   * Promotes integer types to float64
   * @returns New array with cosh values
   */
  cosh(): NDArray {
    const resultStorage = hyperbolicOps.cosh(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Hyperbolic tangent of each element
   * Promotes integer types to float64
   * @returns New array with tanh values
   */
  tanh(): NDArray {
    const resultStorage = hyperbolicOps.tanh(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Inverse hyperbolic sine of each element
   * Promotes integer types to float64
   * @returns New array with arcsinh values
   */
  arcsinh(): NDArray {
    const resultStorage = hyperbolicOps.arcsinh(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Inverse hyperbolic cosine of each element
   * Promotes integer types to float64
   * @returns New array with arccosh values
   */
  arccosh(): NDArray {
    const resultStorage = hyperbolicOps.arccosh(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Inverse hyperbolic tangent of each element
   * Promotes integer types to float64
   * @returns New array with arctanh values
   */
  arctanh(): NDArray {
    const resultStorage = hyperbolicOps.arctanh(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Comparison operations
  /**
   * Element-wise greater than comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  greater(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.greater(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise greater than or equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  greater_equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.greaterEqual(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise less than comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  less(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.less(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise less than or equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  less_equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.lessEqual(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise equality comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.equal(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise not equal comparison
   * @param other - Value or array to compare with
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  not_equal(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.notEqual(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise comparison with tolerance
   * Returns True where |a - b| <= (atol + rtol * |b|)
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  isclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = comparisonOps.isclose(this._storage, otherStorage, rtol, atol);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise comparison with tolerance
   * Returns True where |a - b| <= (atol + rtol * |b|)
   * @param other - Value or array to compare with
   * @param rtol - Relative tolerance (default: 1e-5)
   * @param atol - Absolute tolerance (default: 1e-8)
   * @returns Boolean array (represented as uint8: 1=true, 0=false)
   */
  allclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    return comparisonOps.allclose(this._storage, otherStorage, rtol, atol);
  }

  // Bitwise operations
  /**
   * Bitwise AND element-wise
   * @param other - Array or scalar for AND operation (must be integer type)
   * @returns Result of bitwise AND
   */
  bitwise_and(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = bitwiseOps.bitwise_and(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Bitwise OR element-wise
   * @param other - Array or scalar for OR operation (must be integer type)
   * @returns Result of bitwise OR
   */
  bitwise_or(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = bitwiseOps.bitwise_or(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Bitwise XOR element-wise
   * @param other - Array or scalar for XOR operation (must be integer type)
   * @returns Result of bitwise XOR
   */
  bitwise_xor(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = bitwiseOps.bitwise_xor(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Bitwise NOT (inversion) element-wise
   * @returns Result of bitwise NOT
   */
  bitwise_not(): NDArray {
    const resultStorage = bitwiseOps.bitwise_not(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Invert (bitwise NOT) element-wise - alias for bitwise_not
   * @returns Result of bitwise inversion
   */
  invert(): NDArray {
    const resultStorage = bitwiseOps.invert(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Left shift elements by positions
   * @param shift - Shift amount (array or scalar)
   * @returns Result of left shift
   */
  left_shift(shift: NDArray | number): NDArray {
    const shiftStorage = typeof shift === 'number' ? shift : shift._storage;
    const resultStorage = bitwiseOps.left_shift(this._storage, shiftStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Right shift elements by positions
   * @param shift - Shift amount (array or scalar)
   * @returns Result of right shift
   */
  right_shift(shift: NDArray | number): NDArray {
    const shiftStorage = typeof shift === 'number' ? shift : shift._storage;
    const resultStorage = bitwiseOps.right_shift(this._storage, shiftStorage);
    return NDArray._fromStorage(resultStorage);
  }

  // Logic operations
  /**
   * Logical AND element-wise
   * @param other - Array or scalar for AND operation
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_and(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = logicOps.logical_and(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Logical OR element-wise
   * @param other - Array or scalar for OR operation
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_or(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = logicOps.logical_or(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Logical NOT element-wise
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_not(): NDArray {
    const resultStorage = logicOps.logical_not(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Logical XOR element-wise
   * @param other - Array or scalar for XOR operation
   * @returns Boolean array (1 = true, 0 = false)
   */
  logical_xor(other: NDArray | number): NDArray {
    const otherStorage = typeof other === 'number' ? other : other._storage;
    const resultStorage = logicOps.logical_xor(this._storage, otherStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Test element-wise for finiteness (not infinity and not NaN)
   * @returns Boolean array
   */
  isfinite(): NDArray {
    const resultStorage = logicOps.isfinite(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Test element-wise for positive or negative infinity
   * @returns Boolean array
   */
  isinf(): NDArray {
    const resultStorage = logicOps.isinf(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Test element-wise for NaN (Not a Number)
   * @returns Boolean array
   */
  isnan(): NDArray {
    const resultStorage = logicOps.isnan(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Test element-wise for NaT (Not a Time)
   * @returns Boolean array (always false without datetime support)
   */
  isnat(): NDArray {
    const resultStorage = logicOps.isnat(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Change the sign of x1 to that of x2, element-wise
   * @param x2 - Values whose sign is used
   * @returns Array with magnitude from this and sign from x2
   */
  copysign(x2: NDArray | number): NDArray {
    const x2Storage = typeof x2 === 'number' ? x2 : x2._storage;
    const resultStorage = logicOps.copysign(this._storage, x2Storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Returns element-wise True where signbit is set (less than zero)
   * @returns Boolean array
   */
  signbit(): NDArray {
    const resultStorage = logicOps.signbit(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return the next floating-point value after x1 towards x2, element-wise
   * @param x2 - Direction to look for the next representable value
   * @returns Array of next representable values
   */
  nextafter(x2: NDArray | number): NDArray {
    const x2Storage = typeof x2 === 'number' ? x2 : x2._storage;
    const resultStorage = logicOps.nextafter(this._storage, x2Storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return the distance between x and the nearest adjacent number
   * @returns Array of spacing values
   */
  spacing(): NDArray {
    const resultStorage = logicOps.spacing(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Reductions
  /**
   * Sum array elements over a given axis
   * @param axis - Axis along which to sum. If undefined, sum all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Sum of array elements, or array of sums along axis
   */
  sum(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.sum(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Compute the arithmetic mean along the specified axis
   * @param axis - Axis along which to compute mean. If undefined, compute mean of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Mean of array elements, or array of means along axis
   *
   * Note: mean() returns float64 for integer dtypes, matching NumPy behavior
   */
  mean(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.mean(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Return the maximum along a given axis
   * @param axis - Axis along which to compute maximum. If undefined, compute maximum of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Maximum of array elements, or array of maximums along axis
   */
  max(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.max(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Return the minimum along a given axis
   * @param axis - Axis along which to compute minimum. If undefined, compute minimum of all elements
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Minimum of array elements, or array of minimums along axis
   */
  min(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.min(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Product of array elements over a given axis
   * @param axis - Axis along which to compute the product. If undefined, product of all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Product of array elements, or array of products along axis
   */
  prod(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.prod(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Indices of the minimum values along an axis
   * @param axis - Axis along which to find minimum indices. If undefined, index of global minimum.
   * @returns Indices of minimum values
   */
  argmin(axis?: number): NDArray | number {
    const result = reductionOps.argmin(this._storage, axis);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Indices of the maximum values along an axis
   * @param axis - Axis along which to find maximum indices. If undefined, index of global maximum.
   * @returns Indices of maximum values
   */
  argmax(axis?: number): NDArray | number {
    const result = reductionOps.argmax(this._storage, axis);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute variance along the specified axis
   * @param axis - Axis along which to compute variance. If undefined, variance of all elements.
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Variance of array elements
   */
  var(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.variance(this._storage, axis, ddof, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute standard deviation along the specified axis
   * @param axis - Axis along which to compute std. If undefined, std of all elements.
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Standard deviation of array elements
   */
  std(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.std(this._storage, axis, ddof, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Test whether all array elements along a given axis evaluate to True
   * @param axis - Axis along which to perform logical AND. If undefined, test all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Boolean or array of booleans
   */
  all(axis?: number, keepdims: boolean = false): NDArray | boolean {
    const result = reductionOps.all(this._storage, axis, keepdims);
    return typeof result === 'boolean' ? result : NDArray._fromStorage(result);
  }

  /**
   * Test whether any array elements along a given axis evaluate to True
   * @param axis - Axis along which to perform logical OR. If undefined, test all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Boolean or array of booleans
   */
  any(axis?: number, keepdims: boolean = false): NDArray | boolean {
    const result = reductionOps.any(this._storage, axis, keepdims);
    return typeof result === 'boolean' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return the cumulative sum of elements along a given axis
   * @param axis - Axis along which to compute cumsum. If undefined, compute over flattened array.
   * @returns Array with cumulative sums
   */
  cumsum(axis?: number): NDArray {
    return NDArray._fromStorage(reductionOps.cumsum(this._storage, axis));
  }

  /**
   * Return the cumulative product of elements along a given axis
   * @param axis - Axis along which to compute cumprod. If undefined, compute over flattened array.
   * @returns Array with cumulative products
   */
  cumprod(axis?: number): NDArray {
    return NDArray._fromStorage(reductionOps.cumprod(this._storage, axis));
  }

  /**
   * Peak to peak (maximum - minimum) value along a given axis
   * @param axis - Axis along which to compute ptp. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Range of values
   */
  ptp(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.ptp(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Compute the median along the specified axis
   * @param axis - Axis along which to compute median. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Median of array elements
   */
  median(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.median(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute the q-th percentile of the data along the specified axis
   * @param q - Percentile to compute (0-100)
   * @param axis - Axis along which to compute percentile. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Percentile of array elements
   */
  percentile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.percentile(this._storage, q, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute the q-th quantile of the data along the specified axis
   * @param q - Quantile to compute (0-1)
   * @param axis - Axis along which to compute quantile. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Quantile of array elements
   */
  quantile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.quantile(this._storage, q, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute the weighted average along the specified axis
   * @param weights - Array of weights (optional)
   * @param axis - Axis along which to compute average. If undefined, compute over all elements.
   * @returns Weighted average of array elements
   */
  average(weights?: NDArray, axis?: number): NDArray | number | Complex {
    const result = reductionOps.average(this._storage, axis, weights?.storage);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Return the sum of array elements, treating NaNs as zero
   * @param axis - Axis along which to compute sum. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Sum of array elements ignoring NaNs
   */
  nansum(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.nansum(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Return the product of array elements, treating NaNs as ones
   * @param axis - Axis along which to compute product. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Product of array elements ignoring NaNs
   */
  nanprod(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.nanprod(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Compute the arithmetic mean, ignoring NaNs
   * @param axis - Axis along which to compute mean. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Mean of array elements ignoring NaNs
   */
  nanmean(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.nanmean(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Compute the variance, ignoring NaNs
   * @param axis - Axis along which to compute variance. If undefined, compute over all elements.
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Variance of array elements ignoring NaNs
   */
  nanvar(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.nanvar(this._storage, axis, ddof, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute the standard deviation, ignoring NaNs
   * @param axis - Axis along which to compute std. If undefined, compute over all elements.
   * @param ddof - Delta degrees of freedom (default: 0)
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Standard deviation of array elements ignoring NaNs
   */
  nanstd(axis?: number, ddof: number = 0, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.nanstd(this._storage, axis, ddof, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return minimum of an array or minimum along an axis, ignoring NaNs
   * @param axis - Axis along which to compute minimum. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Minimum of array elements ignoring NaNs
   */
  nanmin(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.nanmin(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Return maximum of an array or maximum along an axis, ignoring NaNs
   * @param axis - Axis along which to compute maximum. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Maximum of array elements ignoring NaNs
   */
  nanmax(axis?: number, keepdims: boolean = false): NDArray | number | Complex {
    const result = reductionOps.nanmax(this._storage, axis, keepdims);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Compute the q-th quantile of the data along the specified axis, ignoring NaNs
   * @param q - Quantile to compute (0-1)
   * @param axis - Axis along which to compute quantile. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Quantile of array elements ignoring NaNs
   */
  nanquantile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.nanquantile(this._storage, q, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Compute the q-th percentile of the data along the specified axis, ignoring NaNs
   * @param q - Percentile to compute (0-100)
   * @param axis - Axis along which to compute percentile. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Percentile of array elements ignoring NaNs
   */
  nanpercentile(q: number, axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.nanpercentile(this._storage, q, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return the indices of the minimum values, ignoring NaNs
   * @param axis - Axis along which to find minimum indices. If undefined, index of global minimum.
   * @returns Indices of minimum values ignoring NaNs
   */
  nanargmin(axis?: number): NDArray | number {
    const result = reductionOps.nanargmin(this._storage, axis);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return the indices of the maximum values, ignoring NaNs
   * @param axis - Axis along which to find maximum indices. If undefined, index of global maximum.
   * @returns Indices of maximum values ignoring NaNs
   */
  nanargmax(axis?: number): NDArray | number {
    const result = reductionOps.nanargmax(this._storage, axis);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  /**
   * Return the cumulative sum of elements, treating NaNs as zero
   * @param axis - Axis along which to compute cumsum. If undefined, compute over flattened array.
   * @returns Array with cumulative sums ignoring NaNs
   */
  nancumsum(axis?: number): NDArray {
    return NDArray._fromStorage(reductionOps.nancumsum(this._storage, axis));
  }

  /**
   * Return the cumulative product of elements, treating NaNs as one
   * @param axis - Axis along which to compute cumprod. If undefined, compute over flattened array.
   * @returns Array with cumulative products ignoring NaNs
   */
  nancumprod(axis?: number): NDArray {
    return NDArray._fromStorage(reductionOps.nancumprod(this._storage, axis));
  }

  /**
   * Compute the median, ignoring NaNs
   * @param axis - Axis along which to compute median. If undefined, compute over all elements.
   * @param keepdims - If true, reduced axes are left as dimensions with size 1
   * @returns Median of array elements ignoring NaNs
   */
  nanmedian(axis?: number, keepdims: boolean = false): NDArray | number {
    const result = reductionOps.nanmedian(this._storage, axis, keepdims);
    return typeof result === 'number' ? result : NDArray._fromStorage(result);
  }

  // ========================================
  // Sorting and Searching
  // ========================================

  /**
   * Return a sorted copy of the array
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   * @returns Sorted array
   */
  sort(axis: number = -1): NDArray {
    return NDArray._fromStorage(sortingOps.sort(this._storage, axis));
  }

  /**
   * Returns the indices that would sort this array
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   * @returns Array of indices that sort the array
   */
  argsort(axis: number = -1): NDArray {
    return NDArray._fromStorage(sortingOps.argsort(this._storage, axis));
  }

  /**
   * Partially sort the array
   * @param kth - Element index to partition by
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   * @returns Partitioned array
   */
  partition(kth: number, axis: number = -1): NDArray {
    return NDArray._fromStorage(sortingOps.partition(this._storage, kth, axis));
  }

  /**
   * Returns indices that would partition the array
   * @param kth - Element index to partition by
   * @param axis - Axis along which to sort. Default is -1 (last axis)
   * @returns Array of indices
   */
  argpartition(kth: number, axis: number = -1): NDArray {
    return NDArray._fromStorage(sortingOps.argpartition(this._storage, kth, axis));
  }

  /**
   * Return the indices of non-zero elements
   * @returns Tuple of arrays, one for each dimension
   */
  nonzero(): NDArray[] {
    const storages = sortingOps.nonzero(this._storage);
    return storages.map((s) => NDArray._fromStorage(s));
  }

  /**
   * Find the indices of array elements that are non-zero, grouped by element
   * Returns a 2D array where each row is the index of a non-zero element.
   * @returns 2D array of shape (N, ndim) where N is number of non-zero elements
   */
  argwhere(): NDArray {
    return NDArray._fromStorage(sortingOps.argwhere(this._storage));
  }

  /**
   * Find indices where elements should be inserted to maintain order
   * @param v - Values to insert
   * @param side - 'left' or 'right' side to insert
   * @returns Indices where values should be inserted
   */
  searchsorted(v: NDArray, side: 'left' | 'right' = 'left'): NDArray {
    return NDArray._fromStorage(sortingOps.searchsorted(this._storage, v._storage, side));
  }

  // Gradient and difference operations
  /**
   * Calculate the n-th discrete difference along the given axis
   * @param n - Number of times values are differenced (default: 1)
   * @param axis - Axis along which to compute difference (default: -1)
   * @returns Array of differences
   */
  diff(n: number = 1, axis: number = -1): NDArray {
    return NDArray._fromStorage(gradientOps.diff(this._storage, n, axis));
  }

  // Shape manipulation
  /**
   * Reshape array to a new shape
   * Returns a new array with the specified shape
   * @param shape - New shape (must be compatible with current size)
   * @returns Reshaped array
   */
  reshape(...shape: number[]): NDArray {
    const newShape = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    const resultStorage = shapeOps.reshape(this._storage, newShape);
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Return a flattened copy of the array
   * @returns 1D array containing all elements
   */
  flatten(): NDArray {
    const resultStorage = shapeOps.flatten(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Return a flattened array (view when possible, otherwise copy)
   * @returns 1D array containing all elements
   */
  ravel(): NDArray {
    const resultStorage = shapeOps.ravel(this._storage);
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Transpose array (permute dimensions)
   * @param axes - Permutation of axes. If undefined, reverse the dimensions
   * @returns Transposed array (always a view)
   */
  transpose(axes?: number[]): NDArray {
    const resultStorage = shapeOps.transpose(this._storage, axes);
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Remove axes of length 1
   * @param axis - Axis to squeeze. If undefined, squeeze all axes of length 1
   * @returns Array with specified dimensions removed (always a view)
   */
  squeeze(axis?: number): NDArray {
    const resultStorage = shapeOps.squeeze(this._storage, axis);
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Expand the shape by inserting a new axis of length 1
   * @param axis - Position where new axis is placed
   * @returns Array with additional dimension (always a view)
   */
  expand_dims(axis: number): NDArray {
    const resultStorage = shapeOps.expandDims(this._storage, axis);
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Swap two axes of an array
   * @param axis1 - First axis
   * @param axis2 - Second axis
   * @returns Array with swapped axes (always a view)
   */
  swapaxes(axis1: number, axis2: number): NDArray {
    const resultStorage = shapeOps.swapaxes(this._storage, axis1, axis2);
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Move axes to new positions
   * @param source - Original positions of axes to move
   * @param destination - New positions for axes
   * @returns Array with moved axes (always a view)
   */
  moveaxis(source: number | number[], destination: number | number[]): NDArray {
    const resultStorage = shapeOps.moveaxis(this._storage, source, destination);
    const base = this._base ?? this;
    return NDArray._fromStorage(resultStorage, base);
  }

  /**
   * Repeat elements of an array
   * @param repeats - Number of repetitions for each element
   * @param axis - Axis along which to repeat (if undefined, flattens first)
   * @returns New array with repeated elements
   */
  repeat(repeats: number | number[], axis?: number): NDArray {
    const resultStorage = shapeOps.repeat(this._storage, repeats, axis);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Take elements from array along an axis
   * @param indices - Indices of elements to take
   * @param axis - Axis along which to take (if undefined, flattens first)
   * @returns New array with selected elements
   */
  take(indices: number[], axis?: number): NDArray {
    const resultStorage = advancedOps.take(this._storage, indices, axis);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Put values at specified indices (modifies array in-place)
   * @param indices - Indices at which to place values
   * @param values - Values to put
   */
  put(indices: number[], values: NDArray | number | bigint): void {
    const valuesStorage = values instanceof NDArray ? values._storage : values;
    advancedOps.put(this._storage, indices, valuesStorage);
  }

  // Fancy indexing operations
  /**
   * Integer array indexing (fancy indexing)
   *
   * Select elements using an array of indices. This is NumPy's "fancy indexing"
   * feature: `arr[[0, 2, 4]]` becomes `arr.iindex([0, 2, 4])` or `arr.iindex(indices)`.
   *
   * @param indices - Array of integer indices (as number[], NDArray, or nested arrays)
   * @param axis - Axis along which to index (default: 0, or flattens if undefined with flat indices)
   * @returns New array with selected elements
   *
   * @example
   * ```typescript
   * const arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
   *
   * // Select rows 0 and 2
   * arr.iindex([0, 2]);           // [[1, 2, 3], [7, 8, 9]]
   *
   * // Select along axis 1 (columns)
   * arr.iindex([0, 2], 1);        // [[1, 3], [4, 6], [7, 9]]
   *
   * // With NDArray indices
   * const idx = np.array([0, 2]);
   * arr.iindex(idx);              // [[1, 2, 3], [7, 8, 9]]
   * ```
   */
  iindex(indices: NDArray | number[] | number[][], axis: number = 0): NDArray {
    // Convert NDArray to number[]
    let indexArray: number[];
    if (indices instanceof NDArray) {
      // Flatten NDArray indices to 1D array of numbers
      indexArray = [];
      for (let i = 0; i < indices.size; i++) {
        const val = indices.storage.iget(i);
        // Handle bigint, Complex, or number types
        const numVal =
          typeof val === 'bigint' ? Number(val) : val instanceof Complex ? val.re : val;
        indexArray.push(numVal);
      }
    } else if (Array.isArray(indices) && indices.length > 0 && Array.isArray(indices[0])) {
      // Flatten nested arrays
      indexArray = (indices as number[][]).flat();
    } else {
      indexArray = indices as number[];
    }

    return this.take(indexArray, axis);
  }

  /**
   * Boolean array indexing (fancy indexing with mask)
   *
   * Select elements where a boolean mask is true. This is NumPy's boolean
   * indexing: `arr[arr > 5]` becomes `arr.bindex(arr.greater(5))`.
   *
   * @param mask - Boolean NDArray mask
   * @param axis - Axis along which to apply the mask (default: flattens array)
   * @returns New 1D array with selected elements (or along axis if specified)
   *
   * @example
   * ```typescript
   * const arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
   *
   * // Select all elements > 5
   * const mask = arr.greater(5);
   * arr.bindex(mask);             // [6, 7, 8, 9]
   *
   * // Select rows where first column > 3
   * const rowMask = np.array([false, true, true]);
   * arr.bindex(rowMask, 0);       // [[4, 5, 6], [7, 8, 9]]
   * ```
   */
  bindex(mask: NDArray, axis?: number): NDArray {
    return NDArray._fromStorage(advancedOps.compress(mask._storage, this._storage, axis));
  }

  // Linear algebra operations
  /**
   * Matrix multiplication
   * @param other - Array to multiply with
   * @returns Result of matrix multiplication
   */
  matmul(other: NDArray): NDArray {
    const resultStorage = linalgOps.matmul(this._storage, other._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Dot product (matching NumPy behavior)
   * @param other - Array to dot with
   * @returns Result of dot product (scalar or array depending on dimensions, Complex for complex arrays)
   */
  dot(other: NDArray): NDArray | number | bigint | Complex {
    const result = linalgOps.dot(this._storage, other._storage);
    if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Sum of diagonal elements (trace)
   * @returns Sum of diagonal elements (Complex for complex arrays)
   */
  trace(): number | bigint | Complex {
    return linalgOps.trace(this._storage);
  }

  /**
   * Inner product (contracts over last axes of both arrays)
   * @param other - Array to compute inner product with
   * @returns Inner product result (Complex for complex arrays)
   */
  inner(other: NDArray): NDArray | number | bigint | Complex {
    const result = linalgOps.inner(this._storage, other._storage);
    if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  /**
   * Outer product (flattens inputs then computes a[i]*b[j])
   * @param other - Array to compute outer product with
   * @returns 2D outer product matrix
   */
  outer(other: NDArray): NDArray {
    const result = linalgOps.outer(this._storage, other._storage);
    return NDArray._fromStorage(result);
  }

  /**
   * Tensor dot product along specified axes
   * @param other - Array to contract with
   * @param axes - Axes to contract (integer or [a_axes, b_axes])
   * @returns Tensor dot product result
   */
  tensordot(
    other: NDArray,
    axes: number | [number[], number[]] = 2
  ): NDArray | number | bigint | Complex {
    const result = linalgOps.tensordot(this._storage, other._storage, axes);
    if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  }

  // Additional arithmetic operations

  /**
   * Element-wise cube root
   * Promotes integer types to float64
   * @returns New array with cube root values
   */
  cbrt(): NDArray {
    const resultStorage = arithmeticOps.cbrt(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise absolute value (always returns float)
   * @returns New array with absolute values as float
   */
  fabs(): NDArray {
    const resultStorage = arithmeticOps.fabs(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Returns both quotient and remainder (floor divide and modulo)
   * @param divisor - Array or scalar divisor
   * @returns Tuple of [quotient, remainder] arrays
   */
  divmod(divisor: NDArray | number): [NDArray, NDArray] {
    const divisorStorage = typeof divisor === 'number' ? divisor : divisor._storage;
    const [quotientStorage, remainderStorage] = arithmeticOps.divmod(this._storage, divisorStorage);
    return [NDArray._fromStorage(quotientStorage), NDArray._fromStorage(remainderStorage)];
  }

  /**
   * Element-wise square (x**2)
   * @returns New array with squared values
   */
  square(): NDArray {
    const resultStorage = arithmeticOps.square(this._storage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Element-wise remainder (same as mod)
   * @param divisor - Array or scalar divisor
   * @returns New array with remainder values
   */
  remainder(divisor: NDArray | number): NDArray {
    const divisorStorage = typeof divisor === 'number' ? divisor : divisor._storage;
    const resultStorage = arithmeticOps.remainder(this._storage, divisorStorage);
    return NDArray._fromStorage(resultStorage);
  }

  /**
   * Heaviside step function
   * @param x2 - Value to use when this array element is 0
   * @returns New array with heaviside values
   */
  heaviside(x2: NDArray | number): NDArray {
    const x2Storage = typeof x2 === 'number' ? x2 : x2._storage;
    const resultStorage = arithmeticOps.heaviside(this._storage, x2Storage);
    return NDArray._fromStorage(resultStorage);
  }

  // Slicing
  /**
   * Slice the array using NumPy-style string syntax
   *
   * @param sliceStrs - Slice specifications, one per dimension
   * @returns Sliced view of the array
   */
  override slice(...sliceStrs: string[]): NDArray {
    if (sliceStrs.length === 0) {
      return this;
    }

    if (sliceStrs.length > this.ndim) {
      throw new Error(
        `Too many indices for array: array is ${this.ndim}-dimensional, but ${sliceStrs.length} were indexed`
      );
    }

    // Parse slice strings and normalize them
    const sliceSpecs = sliceStrs.map((str, i) => {
      const spec = parseSlice(str);
      const normalized = normalizeSlice(spec, this.shape[i]!);
      return normalized;
    });

    // Pad with full slices for remaining dimensions
    while (sliceSpecs.length < this.ndim) {
      sliceSpecs.push({
        start: 0,
        stop: this.shape[sliceSpecs.length]!,
        step: 1,
        isIndex: false,
      });
    }

    // Calculate new shape and strides
    const newShape: number[] = [];
    const newStrides: number[] = [];
    let newOffset = this._storage.offset;

    for (let i = 0; i < sliceSpecs.length; i++) {
      const spec = sliceSpecs[i]!;
      const stride = this._storage.strides[i]!;

      // Update offset based on start position
      newOffset += spec.start * stride;

      if (!spec.isIndex) {
        // Calculate size of this dimension
        // For positive step: (stop - start) / step
        // For negative step: (start - stop) / |step| (since we go from high to low)
        let dimSize: number;
        if (spec.step > 0) {
          dimSize = Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
        } else {
          // Negative step: iterate from start down to (but not including) stop
          dimSize = Math.max(0, Math.ceil((spec.start - spec.stop) / Math.abs(spec.step)));
        }
        newShape.push(dimSize);
        newStrides.push(stride * spec.step);
      }
      // If isIndex is true, this dimension is removed (scalar indexing)
    }

    // Create sliced view
    const slicedStorage = ArrayStorage.fromData(
      this._storage.data,
      newShape,
      this._storage.dtype,
      newStrides,
      newOffset
    );

    const base = this._base ?? this;
    return new NDArray(slicedStorage, base);
  }

  // Convenience methods
  /**
   * Get a single row (convenience method)
   * @param i - Row index
   * @returns Row as 1D or (n-1)D array
   */

  row(i: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('row() requires at least 2 dimensions');
    }
    return this.slice(String(i), ':');
  }

  /**
   * Get a single column (convenience method)
   * @param j - Column index
   * @returns Column as 1D or (n-1)D array
   */
  col(j: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('col() requires at least 2 dimensions');
    }
    return this.slice(':', String(j));
  }

  /**
   * Get a range of rows (convenience method)
   * @param start - Start row index
   * @param stop - Stop row index (exclusive)
   * @returns Rows as array
   */
  rows(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('rows() requires at least 2 dimensions');
    }
    return this.slice(`${start}:${stop}`, ':');
  }

  /**
   * Get a range of columns (convenience method)
   * @param start - Start column index
   * @param stop - Stop column index (exclusive)
   * @returns Columns as array
   */
  cols(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('cols() requires at least 2 dimensions');
    }
    return this.slice(':', `${start}:${stop}`);
  }

  // String representation
  /**
   * String representation of the array
   * @returns String describing the array shape and dtype
   */
  override toString(): string {
    return `NDArray(shape=${JSON.stringify(this.shape)}, dtype=${this.dtype})`;
  }

  /**
   * Convert to nested JavaScript array
   * @returns Nested JavaScript array representation
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override toArray(): any {
    // Handle 0-dimensional arrays (scalars)
    if (this.ndim === 0) {
      return this._storage.iget(0);
    }

    const shape = this.shape;
    const ndim = shape.length;

    // Recursive function to build nested array
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const buildNestedArray = (indices: number[], dim: number): any => {
      if (dim === ndim) {
        return this._storage.get(...indices);
      }

      const arr = [];
      for (let i = 0; i < shape[dim]!; i++) {
        indices[dim] = i;
        arr.push(buildNestedArray(indices, dim + 1));
      }
      return arr;
    };

    return buildNestedArray(new Array(ndim), 0);
  }

  /**
   * Return the array as a nested list (same as toArray)
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override tolist(): any {
    return this.toArray();
  }

  /**
   * Return the raw bytes of the array data
   */
  override tobytes(): ArrayBuffer {
    if (this._storage.isCContiguous) {
      const data = this._storage.data;
      const bytesPerElement = data.BYTES_PER_ELEMENT;
      const offset = this._storage.offset * bytesPerElement;
      const length = this.size * bytesPerElement;
      return data.buffer.slice(offset, offset + length) as ArrayBuffer;
    }
    const copy = this.copy();
    const data = copy._storage.data;
    return data.buffer.slice(0, this.size * data.BYTES_PER_ELEMENT) as ArrayBuffer;
  }

  /**
   * Copy an element of an array to a standard scalar and return it
   */
  override item(...args: number[]): number | bigint | Complex {
    if (args.length === 0) {
      if (this.size !== 1) {
        throw new Error('can only convert an array of size 1 to a Python scalar');
      }
      return this._storage.iget(0);
    }
    if (args.length === 1) {
      const flatIdx = args[0]!;
      if (flatIdx < 0 || flatIdx >= this.size) {
        throw new Error(`index ${flatIdx} is out of bounds for size ${this.size}`);
      }
      return this._storage.iget(flatIdx);
    }
    return this.get(args);
  }

  /**
   * Swap the bytes of the array elements
   */
  byteswap(inplace: boolean = false): NDArray {
    const target = inplace ? this : this.copy();
    const data = target._storage.data;
    const bytesPerElement = data.BYTES_PER_ELEMENT;
    if (bytesPerElement === 1) return target;

    const buffer = data.buffer;
    const view = new DataView(buffer);

    for (let i = 0; i < data.length; i++) {
      const byteOffset = i * bytesPerElement;
      if (bytesPerElement === 2) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        view.setUint8(byteOffset, b1);
        view.setUint8(byteOffset + 1, b0);
      } else if (bytesPerElement === 4) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        const b2 = view.getUint8(byteOffset + 2);
        const b3 = view.getUint8(byteOffset + 3);
        view.setUint8(byteOffset, b3);
        view.setUint8(byteOffset + 1, b2);
        view.setUint8(byteOffset + 2, b1);
        view.setUint8(byteOffset + 3, b0);
      } else if (bytesPerElement === 8) {
        const b0 = view.getUint8(byteOffset);
        const b1 = view.getUint8(byteOffset + 1);
        const b2 = view.getUint8(byteOffset + 2);
        const b3 = view.getUint8(byteOffset + 3);
        const b4 = view.getUint8(byteOffset + 4);
        const b5 = view.getUint8(byteOffset + 5);
        const b6 = view.getUint8(byteOffset + 6);
        const b7 = view.getUint8(byteOffset + 7);
        view.setUint8(byteOffset, b7);
        view.setUint8(byteOffset + 1, b6);
        view.setUint8(byteOffset + 2, b5);
        view.setUint8(byteOffset + 3, b4);
        view.setUint8(byteOffset + 4, b3);
        view.setUint8(byteOffset + 5, b2);
        view.setUint8(byteOffset + 6, b1);
        view.setUint8(byteOffset + 7, b0);
      }
    }
    return target;
  }

  /**
   * Return a view of the array with a different dtype
   */
  view(dtype?: DType): NDArray {
    if (!dtype || dtype === this.dtype) {
      return NDArray._fromStorage(this._storage, this._base ?? this);
    }
    const oldSize = getDTypeSize(this.dtype as DType);
    const newSize = getDTypeSize(dtype);
    if (oldSize !== newSize) {
      throw new Error(
        'When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.'
      );
    }
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) throw new Error(`Unsupported dtype: ${dtype}`);
    const data = this._storage.data;
    const byteOffset = data.byteOffset + this._storage.offset * oldSize;
    const newData = new Constructor(data.buffer as ArrayBuffer, byteOffset, this.size);
    const storage = ArrayStorage.fromData(
      newData as TypedArray,
      [...this.shape],
      dtype,
      [...this._storage.strides],
      0
    );
    return NDArray._fromStorage(storage, this._base ?? this);
  }

  /**
   * Construct an array from an index array and a list of arrays to choose from
   * @param choices - Array of NDArrays to choose from
   * @returns New array with selected elements
   */
  choose(choices: NDArray[]): NDArray {
    const choiceStorages = choices.map((c) => c._storage);
    const result = advancedOps.choose(this._storage, choiceStorages);
    return NDArray._fromStorage(result);
  }

  /**
   * Clip (limit) the values in an array
   * @param a_min - Minimum value (null for no minimum)
   * @param a_max - Maximum value (null for no maximum)
   * @returns Array with values clipped to [a_min, a_max]
   */
  clip(a_min: number | NDArray | null, a_max: number | NDArray | null): NDArray {
    const minStorage = a_min instanceof NDArray ? a_min._storage : a_min;
    const maxStorage = a_max instanceof NDArray ? a_max._storage : a_max;
    const result = arithmeticOps.clip(this._storage, minStorage, maxStorage);
    return NDArray._fromStorage(result);
  }

  /**
   * Return selected slices of this array along given axis
   * @param condition - Boolean array that selects which entries to return
   * @param axis - Axis along which to take slices (if undefined, works on flattened array)
   * @returns Array with selected entries
   */
  compress(condition: NDArray | boolean[], axis?: number): NDArray {
    const condStorage =
      condition instanceof NDArray
        ? condition._storage
        : ArrayStorage.fromData(
            new Uint8Array(condition.map((b) => (b ? 1 : 0))),
            [condition.length],
            'bool'
          );
    const result = advancedOps.compress(condStorage, this._storage, axis);
    return NDArray._fromStorage(result);
  }

  /**
   * Return the complex conjugate, element-wise
   * @returns Complex conjugate of the array
   */
  conj(): NDArray {
    const result = complexOps.conj(this._storage);
    return NDArray._fromStorage(result);
  }

  /**
   * Return the complex conjugate, element-wise (alias for conj)
   * @returns Complex conjugate of the array
   */
  conjugate(): NDArray {
    return this.conj();
  }

  /**
   * Return specified diagonals
   * @param offset - Offset of the diagonal from the main diagonal (default: 0)
   * @param axis1 - First axis of the 2-D sub-arrays (default: 0)
   * @param axis2 - Second axis of the 2-D sub-arrays (default: 1)
   * @returns Array of diagonals
   */
  diagonal(offset: number = 0, axis1: number = 0, axis2: number = 1): NDArray {
    const result = linalgOps.diagonal(this._storage, offset, axis1, axis2);
    return NDArray._fromStorage(result);
  }

  /**
   * Return a new array with the specified shape
   * If the new array is larger, it will be filled with repeated copies of the original data
   * @param newShape - Shape of the resized array
   * @returns New array with the specified shape
   */
  resize(newShape: number[]): NDArray {
    const result = shapeOps.resize(this._storage, newShape);
    return NDArray._fromStorage(result);
  }

  /**
   * Write array to a file (stub - use node.ts module for file operations)
   */

  tofile(_file: string, _sep: string = '', _format: string = ''): void {
    throw new Error(
      'tofile() requires file system access. Use the node module: import { save } from "numpy-ts/node"'
    );
  }
}

// Creation functions

/**
 * Create array of zeros
 * @param shape - Shape of the array
 * @param dtype - Data type (default: float64)
 * @returns Array filled with zeros
 */
export function zeros(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const storage = ArrayStorage.zeros(shape, dtype);
  return new NDArray(storage);
}

/**
 * Create array of ones
 * @param shape - Shape of the array
 * @param dtype - Data type (default: float64)
 * @returns Array filled with ones
 */
export function ones(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  const storage = ArrayStorage.ones(shape, dtype);
  return new NDArray(storage);
}

/**
 * Helper to infer shape from nested arrays
 */
function inferShape(data: unknown): number[] {
  const shape: number[] = [];
  let current = data;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0];
  }
  return shape;
}

/**
 * Helper to check if data contains BigInt values
 */
function containsBigInt(data: unknown): boolean {
  if (typeof data === 'bigint') return true;
  if (Array.isArray(data)) {
    return data.some((item) => containsBigInt(item));
  }
  return false;
}

/**
 * Helper to check if data contains Complex values
 */
function containsComplex(data: unknown): boolean {
  if (isComplexLike(data)) return true;
  if (Array.isArray(data)) {
    return data.some((item) => containsComplex(item));
  }
  return false;
}

/**
 * Helper to flatten nested arrays keeping BigInt values
 */
function flattenKeepBigInt(data: unknown): unknown[] {
  const result: unknown[] = [];
  function flatten(arr: unknown): void {
    if (Array.isArray(arr)) {
      arr.forEach((item) => flatten(item));
    } else {
      result.push(arr);
    }
  }
  flatten(data);
  return result;
}

/**
 * Create array from nested JavaScript arrays
 * @param data - Nested arrays or existing NDArray
 * @param dtype - Data type (optional, will be inferred if not provided)
 * @returns New NDArray
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function array(data: any, dtype?: DType): NDArray {
  // If data is already an NDArray, optionally convert dtype
  if (data instanceof NDArray) {
    if (!dtype || data.dtype === dtype) {
      return data.copy();
    }
    return data.astype(dtype);
  }

  const hasBigInt = containsBigInt(data);
  const hasComplex = containsComplex(data);

  // Infer shape from nested arrays
  const shape = inferShape(data);
  const size = shape.reduce((a: number, b: number) => a * b, 1);

  // Determine dtype
  let actualDtype = dtype;
  if (!actualDtype) {
    if (hasComplex) {
      actualDtype = 'complex128';
    } else if (hasBigInt) {
      actualDtype = 'int64';
    } else {
      actualDtype = DEFAULT_DTYPE;
    }
  }

  const isComplex = isComplexDType(actualDtype);

  // Get TypedArray constructor
  const Constructor = getTypedArrayConstructor(actualDtype);
  if (!Constructor) {
    throw new Error(`Cannot create array with dtype ${actualDtype}`);
  }

  // For complex types, physical size is 2x logical size
  const physicalSize = isComplex ? size * 2 : size;
  const typedData = new Constructor(physicalSize);
  const flatData = flattenKeepBigInt(data);

  // Fill the typed array
  if (isBigIntDType(actualDtype)) {
    const bigintData = typedData as BigInt64Array | BigUint64Array;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      bigintData[i] = typeof val === 'bigint' ? val : BigInt(Math.round(Number(val)));
    }
  } else if (actualDtype === 'bool') {
    const boolData = typedData as Uint8Array;
    for (let i = 0; i < size; i++) {
      boolData[i] = flatData[i] ? 1 : 0;
    }
  } else if (isComplex) {
    // Complex: store as interleaved [re, im, re, im, ...]
    const complexData = typedData as Float64Array | Float32Array;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      let re: number, im: number;

      if (val instanceof Complex) {
        re = val.re;
        im = val.im;
      } else if (typeof val === 'object' && val !== null && 're' in val) {
        re = (val as { re: number; im?: number }).re;
        im = (val as { re: number; im?: number }).im ?? 0;
      } else {
        // Scalar number - treat as real with 0 imaginary
        re = Number(val);
        im = 0;
      }

      complexData[i * 2] = re;
      complexData[i * 2 + 1] = im;
    }
  } else {
    const numData = typedData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < size; i++) {
      const val = flatData[i];
      numData[i] = typeof val === 'bigint' ? Number(val) : Number(val);
    }
  }

  const storage = ArrayStorage.fromData(typedData, shape, actualDtype);
  return new NDArray(storage);
}

/**
 * Create array with evenly spaced values within a given interval
 * Similar to Python's range() but returns array
 * @param start - Start value (or stop if only one argument)
 * @param stop - Stop value (exclusive)
 * @param step - Step between values (default: 1)
 * @param dtype - Data type (default: float64)
 * @returns Array of evenly spaced values
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  let actualStart = start;
  let actualStop = stop;

  if (stop === undefined) {
    actualStart = 0;
    actualStop = start;
  }

  if (actualStop === undefined) {
    throw new Error('stop is required');
  }

  const length = Math.max(0, Math.ceil((actualStop - actualStart) / step));

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create arange array with dtype ${dtype}`);
  }

  const data = new Constructor(length);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < length; i++) {
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(actualStart + i * step));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < length; i++) {
      (data as Uint8Array)[i] = actualStart + i * step !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < length; i++) {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = actualStart + i * step;
    }
  }

  const storage = ArrayStorage.fromData(data, [length], dtype);
  return new NDArray(storage);
}

/**
 * Create array with evenly spaced values over a specified interval
 * @param start - Starting value
 * @param stop - Ending value (inclusive)
 * @param num - Number of samples (default: 50)
 * @param dtype - Data type (default: float64)
 * @returns Array of evenly spaced values
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([start], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create linspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const step = (stop - start) / (num - 1);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(start + i * step));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      (data as Uint8Array)[i] = start + i * step !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = start + i * step;
    }
  }

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArray(storage);
}

/**
 * Create array with logarithmically spaced values
 * Returns num samples, equally spaced on a log scale from base^start to base^stop
 * @param start - base^start is the starting value
 * @param stop - base^stop is the ending value
 * @param num - Number of samples (default: 50)
 * @param base - Base of the log space (default: 10.0)
 * @param dtype - Data type (default: float64)
 * @returns Array of logarithmically spaced values
 */
export function logspace(
  start: number,
  stop: number,
  num: number = 50,
  base: number = 10.0,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([Math.pow(base, start)], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create logspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const step = (stop - start) / (num - 1);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(Math.pow(base, exponent)));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as Uint8Array)[i] = Math.pow(base, exponent) !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      const exponent = start + i * step;
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Math.pow(base, exponent);
    }
  }

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArray(storage);
}

/**
 * Create array with geometrically spaced values
 * Returns num samples, equally spaced on a log scale (geometric progression)
 * @param start - Starting value
 * @param stop - Ending value
 * @param num - Number of samples (default: 50)
 * @param dtype - Data type (default: float64)
 * @returns Array of geometrically spaced values
 */
export function geomspace(
  start: number,
  stop: number,
  num: number = 50,
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  if (num < 0) {
    throw new Error('num must be non-negative');
  }

  if (start === 0 || stop === 0) {
    throw new Error('Geometric sequence cannot include zero');
  }

  if (num === 0) {
    return array([], dtype);
  }

  if (num === 1) {
    return array([start], dtype);
  }

  const signStart = Math.sign(start);
  const signStop = Math.sign(stop);

  if (signStart !== signStop) {
    throw new Error('Geometric sequence cannot contain both positive and negative values');
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create geomspace array with dtype ${dtype}`);
  }

  const data = new Constructor(num);
  const logStart = Math.log(Math.abs(start));
  const logStop = Math.log(Math.abs(stop));
  const step = (logStop - logStart) / (num - 1);

  if (isBigIntDType(dtype)) {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as BigInt64Array | BigUint64Array)[i] = BigInt(Math.round(value));
    }
  } else if (dtype === 'bool') {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as Uint8Array)[i] = value !== 0 ? 1 : 0;
    }
  } else {
    for (let i = 0; i < num; i++) {
      const value = signStart * Math.exp(logStart + i * step);
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = value;
    }
  }

  const storage = ArrayStorage.fromData(data, [num], dtype);
  return new NDArray(storage);
}

/**
 * Create identity matrix
 * @param n - Number of rows
 * @param m - Number of columns (default: n)
 * @param k - Index of diagonal (0 for main diagonal, positive for upper, negative for lower)
 * @param dtype - Data type (default: float64)
 * @returns Identity matrix
 */
export function eye(n: number, m?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArray {
  const cols = m ?? n;
  const result = zeros([n, cols], dtype);
  const data = result.data;

  if (isBigIntDType(dtype)) {
    const typedData = data as unknown as BigInt64Array | BigUint64Array;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < cols) {
        typedData[i * cols + j] = BigInt(1);
      }
    }
  } else {
    const typedData = data as unknown as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j >= 0 && j < cols) {
        typedData[i * cols + j] = 1;
      }
    }
  }

  return result;
}

/**
 * Create an uninitialized array
 * Note: TypedArrays are zero-initialized by default in JavaScript
 * @param shape - Shape of the array
 * @param dtype - Data type (default: float64)
 * @returns Uninitialized array
 */
export function empty(shape: number[], dtype: DType = DEFAULT_DTYPE): NDArray {
  return zeros(shape, dtype);
}

/**
 * Create array filled with a constant value
 * @param shape - Shape of the array
 * @param fill_value - Value to fill the array with
 * @param dtype - Data type (optional, inferred from fill_value if not provided)
 * @returns Array filled with the constant value
 */
export function full(
  shape: number[],
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  let actualDtype = dtype;
  if (!actualDtype) {
    if (typeof fill_value === 'bigint') {
      actualDtype = 'int64';
    } else if (typeof fill_value === 'boolean') {
      actualDtype = 'bool';
    } else if (Number.isInteger(fill_value)) {
      actualDtype = 'int32';
    } else {
      actualDtype = DEFAULT_DTYPE;
    }
  }

  const Constructor = getTypedArrayConstructor(actualDtype);
  if (!Constructor) {
    throw new Error(`Cannot create full array with dtype ${actualDtype}`);
  }
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Constructor(size);

  if (isBigIntDType(actualDtype)) {
    const bigintValue =
      typeof fill_value === 'bigint' ? fill_value : BigInt(Math.round(Number(fill_value)));
    (data as BigInt64Array | BigUint64Array).fill(bigintValue);
  } else if (actualDtype === 'bool') {
    (data as Uint8Array).fill(fill_value ? 1 : 0);
  } else {
    (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(Number(fill_value));
  }

  const storage = ArrayStorage.fromData(data, shape, actualDtype);
  return new NDArray(storage);
}

/**
 * Create a square identity matrix
 * @param n - Size of the square matrix
 * @param dtype - Data type (default: float64)
 * @returns n×n identity matrix
 */
export function identity(n: number, dtype: DType = DEFAULT_DTYPE): NDArray {
  return eye(n, n, 0, dtype);
}

/**
 * Convert input to an ndarray
 * @param a - Input data (array-like or NDArray)
 * @param dtype - Data type (optional)
 * @returns NDArray representation of the input
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function asarray(a: NDArray | any, dtype?: DType): NDArray {
  if (a instanceof NDArray) {
    if (!dtype || a.dtype === dtype) {
      return a;
    }
    return a.astype(dtype);
  }
  return array(a, dtype);
}

/**
 * Convert input to an array, checking for NaN and Inf values.
 * @param a - Input data
 * @param dtype - Data type (optional)
 * @returns NDArray
 * @throws If array contains NaN or Inf values
 */
export function asarray_chkfinite(a: NDArray | ArrayLike<number | bigint>, dtype?: DType): NDArray {
  const arr = asarray(a, dtype);
  for (let i = 0; i < arr.size; i++) {
    if (!isFinite(Number(arr.data[i]))) {
      throw new Error('array must not contain infs or NaNs');
    }
  }
  return arr;
}

/**
 * Return array satisfying requirements.
 * @param a - Input array
 * @param dtype - Data type (optional)
 * @param requirements - Requirements ('C', 'F', 'A', 'W', 'O', 'E') (optional)
 * @returns Array satisfying requirements
 */
export function require(a: NDArray, dtype?: DType, requirements?: string | string[]): NDArray {
  // Parse requirements
  const reqs = Array.isArray(requirements) ? requirements : requirements ? [requirements] : [];

  // Normalize requirement strings
  const normalizedReqs = new Set(
    reqs.map((r) => {
      const upper = r.toUpperCase();
      // Map common requirement aliases
      if (upper === 'C_CONTIGUOUS' || upper === 'CONTIGUOUS') return 'C';
      if (upper === 'F_CONTIGUOUS' || upper === 'FORTRAN') return 'F';
      if (upper === 'WRITEABLE') return 'W';
      if (upper === 'ENSUREARRAY') return 'E';
      if (upper === 'OWNDATA') return 'O';
      return upper;
    })
  );

  // Handle dtype conversion if needed
  let result = a;
  if (dtype && a.dtype !== dtype) {
    result = a.astype(dtype);
  }

  // For most requirements, we return a copy (which satisfies C_CONTIGUOUS, WRITEABLE, OWNDATA)
  if (normalizedReqs.has('C') || normalizedReqs.has('W') || normalizedReqs.has('O')) {
    if (result === a) {
      result = a.copy();
    }
  }

  // F_CONTIGUOUS would require a special layout - for now just return the copy
  // (JavaScript arrays are inherently row-major like C)

  return result;
}

/**
 * Create a deep copy of an array
 * @param a - Array to copy
 * @returns Deep copy of the array
 */
export function copy(a: NDArray): NDArray {
  return a.copy();
}

/**
 * Create array of zeros with the same shape as another array
 * @param a - Array to match shape from
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Array of zeros
 */
export function zeros_like(a: NDArray, dtype?: DType): NDArray {
  return zeros(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create array of ones with the same shape as another array
 * @param a - Array to match shape from
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Array of ones
 */
export function ones_like(a: NDArray, dtype?: DType): NDArray {
  return ones(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create uninitialized array with the same shape as another array
 * @param a - Array to match shape from
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Uninitialized array
 */
export function empty_like(a: NDArray, dtype?: DType): NDArray {
  return empty(Array.from(a.shape), dtype ?? (a.dtype as DType));
}

/**
 * Create array filled with a constant value, same shape as another array
 * @param a - Array to match shape from
 * @param fill_value - Value to fill with
 * @param dtype - Data type (optional, uses a's dtype if not provided)
 * @returns Filled array
 */
export function full_like(
  a: NDArray,
  fill_value: number | bigint | boolean,
  dtype?: DType
): NDArray {
  return full(Array.from(a.shape), fill_value, dtype ?? (a.dtype as DType));
}

/**
 * Convert input to an ndarray (alias for asarray for compatibility)
 * In numpy-ts, this behaves the same as asarray since we don't have subclasses
 * @param a - Input data
 * @param dtype - Data type (optional)
 * @returns NDArray
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function asanyarray(a: NDArray | any, dtype?: DType): NDArray {
  return asarray(a, dtype);
}

/**
 * Return a contiguous array (ndim >= 1) in memory (C order)
 * Since our arrays are already C-contiguous in memory, this either
 * returns the input unchanged or creates a contiguous copy
 * @param a - Input data
 * @param dtype - Data type (optional)
 * @returns Contiguous array in C order
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function ascontiguousarray(a: NDArray | any, dtype?: DType): NDArray {
  const arr = asarray(a, dtype);
  if (arr.flags.C_CONTIGUOUS) {
    return arr;
  }
  return arr.copy();
}

/**
 * Return an array laid out in Fortran order in memory
 * Note: numpy-ts uses C-order internally, so this creates a copy
 * that is equivalent to the Fortran-ordered layout
 * @param a - Input data
 * @param dtype - Data type (optional)
 * @returns Array (copy in C order, as Fortran order is not supported)
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function asfortranarray(a: NDArray | any, dtype?: DType): NDArray {
  const arr = asarray(a, dtype);
  // We always return C-contiguous arrays, so just return a copy
  return arr.copy();
}

/**
 * Extract a diagonal or construct a diagonal array
 * @param v - Input array (if 2D, extract diagonal; if 1D, construct diagonal matrix)
 * @param k - Diagonal offset (default 0 is main diagonal, positive above, negative below)
 * @returns Diagonal elements as 1D array, or 2D diagonal matrix
 */
export function diag(v: NDArray, k: number = 0): NDArray {
  if (v.ndim === 1) {
    // Construct diagonal matrix from 1D array
    const n = v.size;
    const size = n + Math.abs(k);
    const result = zeros([size, size], v.dtype as DType);

    for (let i = 0; i < n; i++) {
      const row = k >= 0 ? i : i - k;
      const col = k >= 0 ? i + k : i;
      result.set([row, col], v.get([i]) as number);
    }
    return result;
  } else if (v.ndim === 2) {
    // Extract diagonal from 2D array
    const [rows, cols] = v.shape;
    let startRow: number, startCol: number, diagLength: number;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
      diagLength = Math.min(rows!, cols! - k);
    } else {
      startRow = -k;
      startCol = 0;
      diagLength = Math.min(rows! + k, cols!);
    }

    if (diagLength <= 0) {
      return zeros([0], v.dtype as DType);
    }

    const Constructor = getTypedArrayConstructor(v.dtype as DType);
    const data = new Constructor!(diagLength);

    for (let i = 0; i < diagLength; i++) {
      const val = v.get([startRow + i, startCol + i]);
      if (isBigIntDType(v.dtype as DType)) {
        (data as BigInt64Array | BigUint64Array)[i] =
          typeof val === 'bigint' ? val : BigInt(val as number);
      } else {
        (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = val as number;
      }
    }

    const storage = ArrayStorage.fromData(data, [diagLength], v.dtype as DType);
    return new NDArray(storage);
  } else {
    throw new Error('Input must be 1-D or 2-D');
  }
}

/**
 * Create a 2-D array with the flattened input as a diagonal
 * @param v - Input array (will be flattened)
 * @param k - Diagonal offset (default 0)
 * @returns 2D diagonal matrix
 */
export function diagflat(v: NDArray, k: number = 0): NDArray {
  const flat = v.flatten();
  return diag(flat, k);
}

/**
 * Construct an array by executing a function over each coordinate
 * @param fn - Function that takes coordinate indices and returns value
 * @param shape - Shape of output array
 * @param dtype - Data type (default: float64)
 * @returns Array with values computed from function
 */
export function fromfunction(
  fn: (...indices: number[]) => number | bigint | boolean,
  shape: number[],
  dtype: DType = DEFAULT_DTYPE
): NDArray {
  const size = shape.reduce((a, b) => a * b, 1);
  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Cannot create array with dtype ${dtype}`);
  }
  const data = new Constructor(size);
  const ndim = shape.length;
  const indices = new Array(ndim).fill(0);

  for (let i = 0; i < size; i++) {
    const value = fn(...indices);

    if (isBigIntDType(dtype)) {
      (data as BigInt64Array | BigUint64Array)[i] =
        typeof value === 'bigint' ? value : BigInt(Number(value));
    } else if (dtype === 'bool') {
      (data as Uint8Array)[i] = value ? 1 : 0;
    } else {
      (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(value);
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d]! < shape[d]!) {
        break;
      }
      indices[d] = 0;
    }
  }

  const storage = ArrayStorage.fromData(data, shape, dtype);
  return new NDArray(storage);
}

/**
 * Return coordinate matrices from coordinate vectors
 * @param arrays - 1D coordinate arrays
 * @param indexing - 'xy' (Cartesian, default) or 'ij' (matrix indexing)
 * @returns Array of coordinate grids
 */
export function meshgrid(...args: (NDArray | { indexing?: 'xy' | 'ij' })[]): NDArray[] {
  // Parse arguments - last arg might be options
  let arrays: NDArray[] = [];
  let indexing: 'xy' | 'ij' = 'xy';

  for (const arg of args) {
    if (arg instanceof NDArray) {
      arrays.push(arg);
    } else if (typeof arg === 'object' && 'indexing' in arg) {
      indexing = arg.indexing || 'xy';
    }
  }

  if (arrays.length === 0) {
    return [];
  }

  if (arrays.length === 1) {
    return [arrays[0]!.copy()];
  }

  // Get sizes
  const sizes = arrays.map((a) => a.size);

  // For 'xy' indexing, swap first two dimensions
  if (indexing === 'xy' && arrays.length >= 2) {
    arrays = [arrays[1]!, arrays[0]!, ...arrays.slice(2)];
    [sizes[0], sizes[1]] = [sizes[1]!, sizes[0]!];
  }

  // Output shape is the combination of all input sizes
  const outputShape = sizes;
  const ndim = outputShape.length;

  const results: NDArray[] = [];

  for (let i = 0; i < arrays.length; i++) {
    const inputArr = arrays[i]!;
    const inputSize = inputArr.size;

    // Build the shape for broadcasting this array
    const broadcastShape: number[] = new Array(ndim).fill(1);
    broadcastShape[i] = inputSize;

    // Reshape and broadcast
    const reshaped = inputArr.reshape(...broadcastShape);
    const resultStorage = advancedOps.broadcast_to(reshaped.storage, outputShape);
    const result = NDArray._fromStorage(resultStorage.copy()); // copy to make contiguous
    results.push(result);
  }

  // For 'xy' indexing, swap back the first two results
  if (indexing === 'xy' && results.length >= 2) {
    [results[0], results[1]] = [results[1]!, results[0]!];
  }

  return results;
}

/**
 * An array with ones at and below the given diagonal and zeros elsewhere
 * @param N - Number of rows
 * @param M - Number of columns (default: N)
 * @param k - Diagonal offset (default 0)
 * @param dtype - Data type (default: float64)
 * @returns Triangular array
 */
export function tri(N: number, M?: number, k: number = 0, dtype: DType = DEFAULT_DTYPE): NDArray {
  const cols = M ?? N;
  const result = zeros([N, cols], dtype);

  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i + k && j < cols; j++) {
      if (j >= 0) {
        result.set([i, j], 1);
      }
    }
  }

  return result;
}

/**
 * Lower triangle of an array
 * @param m - Input array
 * @param k - Diagonal above which to zero elements (default 0)
 * @returns Copy with upper triangle zeroed
 */
export function tril(m: NDArray, k: number = 0): NDArray {
  if (m.ndim < 2) {
    throw new Error('Input must have at least 2 dimensions');
  }

  const result = m.copy();
  const shape = result.shape;
  const rows = shape[shape.length - 2]!;
  const cols = shape[shape.length - 1]!;

  // Handle multi-dimensional arrays
  const outerSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (j > i + k) {
          // Build the indices array
          const indices: number[] = [];
          let temp = outer;
          for (let d = shape.length - 3; d >= 0; d--) {
            indices.unshift(temp % shape[d]!);
            temp = Math.floor(temp / shape[d]!);
          }
          indices.push(i, j);
          result.set(indices, 0);
        }
      }
    }
  }

  return result;
}

/**
 * Upper triangle of an array
 * @param m - Input array
 * @param k - Diagonal below which to zero elements (default 0)
 * @returns Copy with lower triangle zeroed
 */
export function triu(m: NDArray, k: number = 0): NDArray {
  if (m.ndim < 2) {
    throw new Error('Input must have at least 2 dimensions');
  }

  const result = m.copy();
  const shape = result.shape;
  const rows = shape[shape.length - 2]!;
  const cols = shape[shape.length - 1]!;

  // Handle multi-dimensional arrays
  const outerSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);

  for (let outer = 0; outer < outerSize; outer++) {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (j < i + k) {
          // Build the indices array
          const indices: number[] = [];
          let temp = outer;
          for (let d = shape.length - 3; d >= 0; d--) {
            indices.unshift(temp % shape[d]!);
            temp = Math.floor(temp / shape[d]!);
          }
          indices.push(i, j);
          result.set(indices, 0);
        }
      }
    }
  }

  return result;
}

/**
 * Generate a Vandermonde matrix
 * @param x - Input 1D array
 * @param N - Number of columns (default: length of x)
 * @param increasing - Order of powers (default: false, highest powers first)
 * @returns Vandermonde matrix
 */
export function vander(x: NDArray, N?: number, increasing: boolean = false): NDArray {
  if (x.ndim !== 1) {
    throw new Error('Input must be 1-D');
  }

  const len = x.size;
  const cols = N ?? len;

  if (cols < 0) {
    throw new Error('N must be non-negative');
  }

  const result = zeros([len, cols], x.dtype as DType);

  for (let i = 0; i < len; i++) {
    const val = x.get([i]) as number;
    for (let j = 0; j < cols; j++) {
      const power = increasing ? j : cols - 1 - j;
      result.set([i, j], Math.pow(val, power));
    }
  }

  return result;
}

/**
 * Interpret a buffer as a 1-dimensional array
 * @param buffer - Buffer-like object (ArrayBuffer, TypedArray, or DataView)
 * @param dtype - Data type (default: float64)
 * @param count - Number of items to read (-1 means all)
 * @param offset - Start reading from this byte offset
 * @returns NDArray from buffer data
 */
export function frombuffer(
  buffer: ArrayBuffer | ArrayBufferView,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  offset: number = 0
): NDArray {
  let arrayBuffer: ArrayBufferLike;
  let byteOffset = offset;

  if (buffer instanceof ArrayBuffer) {
    arrayBuffer = buffer;
  } else {
    // It's a TypedArray or DataView
    arrayBuffer = buffer.buffer;
    byteOffset += buffer.byteOffset;
  }

  const bytesPerElement = getBytesPerElement(dtype);
  const availableBytes = arrayBuffer.byteLength - byteOffset;
  const maxElements = Math.floor(availableBytes / bytesPerElement);
  const numElements = count < 0 ? maxElements : Math.min(count, maxElements);

  if (numElements <= 0) {
    return array([], dtype);
  }

  const Constructor = getTypedArrayConstructor(dtype);
  if (!Constructor) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }

  // Create a view into the buffer
  const data = new Constructor(arrayBuffer as ArrayBuffer, byteOffset, numElements);
  const storage = ArrayStorage.fromData(data as TypedArray, [numElements], dtype);
  return new NDArray(storage);
}

/**
 * Construct an array by executing a function over each coordinate.
 * Note: This is a JS implementation - fromfile for actual files isn't directly applicable in browser JS.
 * This function creates an array from an iterable or callable.
 * @param file - In JS context, this is an iterable yielding values
 * @param dtype - Data type
 * @param count - Number of items to read (-1 means all)
 * @returns NDArray from the iterable
 */
export function fromfile(
  file: Iterable<number | bigint>,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1
): NDArray {
  // In JavaScript, we interpret this as reading from an iterable
  const values: Array<number | bigint> = [];
  let i = 0;

  for (const val of file) {
    if (count >= 0 && i >= count) break;
    values.push(val);
    i++;
  }

  return array(values, dtype);
}

/**
 * Create a new 1-dimensional array from an iterable object
 * @param iter - Iterable object
 * @param dtype - Data type
 * @param count - Number of items to read (-1 means all)
 * @returns NDArray from the iterable
 */
export function fromiter(
  iter: Iterable<number | bigint>,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1
): NDArray {
  const values: Array<number | bigint> = [];
  let i = 0;

  for (const val of iter) {
    if (count >= 0 && i >= count) break;
    values.push(val);
    i++;
  }

  return array(values, dtype);
}

/**
 * Create a new 1-dimensional array from text string
 * @param string - Input string containing numbers separated by whitespace or separator
 * @param dtype - Data type (default: float64)
 * @param count - Number of items to read (-1 means all)
 * @param sep - Separator between values (default: any whitespace)
 * @returns NDArray from parsed string
 */
export function fromstring(
  string: string,
  dtype: DType = DEFAULT_DTYPE,
  count: number = -1,
  sep: string = ''
): NDArray {
  // Split the string by separator (or whitespace if sep is empty)
  let parts: string[];
  if (sep === '') {
    parts = string.trim().split(/\s+/);
  } else {
    parts = string.split(sep);
  }

  // Parse values
  const values: Array<number | bigint> = [];
  let i = 0;
  for (const part of parts) {
    if (count >= 0 && i >= count) break;
    const trimmed = part.trim();
    if (trimmed === '') continue;

    if (isBigIntDType(dtype)) {
      values.push(BigInt(trimmed));
    } else {
      values.push(parseFloat(trimmed));
    }
    i++;
  }

  return array(values, dtype);
}

/**
 * Helper to get bytes per element for a dtype
 */
function getBytesPerElement(dtype: DType): number {
  switch (dtype) {
    case 'int8':
    case 'uint8':
    case 'bool':
      return 1;
    case 'int16':
    case 'uint16':
      return 2;
    case 'int32':
    case 'uint32':
    case 'float32':
      return 4;
    case 'int64':
    case 'uint64':
    case 'float64':
      return 8;
    default:
      return 8;
  }
}

// Mathematical functions (standalone)

/**
 * Element-wise square root
 * @param x - Input array
 * @returns Array of square roots
 */
export function sqrt(x: NDArray): NDArray {
  return x.sqrt();
}

/**
 * Element-wise power
 * @param x - Base array
 * @param exponent - Exponent (array or scalar)
 * @returns Array of x raised to exponent
 */
export function power(x: NDArray, exponent: NDArray | number): NDArray {
  return x.power(exponent);
}

// Alias for power
export { power as pow };

/**
 * Element-wise natural exponential (e^x)
 * @param x - Input array
 * @returns Array of e^x values
 */
export function exp(x: NDArray): NDArray {
  return x.exp();
}

/**
 * Element-wise base-2 exponential (2^x)
 * @param x - Input array
 * @returns Array of 2^x values
 */
export function exp2(x: NDArray): NDArray {
  return x.exp2();
}

/**
 * Element-wise exponential minus one (e^x - 1)
 * More accurate than exp(x) - 1 for small x
 * @param x - Input array
 * @returns Array of expm1 values
 */
export function expm1(x: NDArray): NDArray {
  return x.expm1();
}

/**
 * Element-wise natural logarithm (ln)
 * @param x - Input array
 * @returns Array of log values
 */
export function log(x: NDArray): NDArray {
  return x.log();
}

/**
 * Element-wise base-2 logarithm
 * @param x - Input array
 * @returns Array of log2 values
 */
export function log2(x: NDArray): NDArray {
  return x.log2();
}

/**
 * Element-wise base-10 logarithm
 * @param x - Input array
 * @returns Array of log10 values
 */
export function log10(x: NDArray): NDArray {
  return x.log10();
}

/**
 * Element-wise natural logarithm of (1 + x)
 * More accurate than log(1 + x) for small x
 * @param x - Input array
 * @returns Array of log1p values
 */
export function log1p(x: NDArray): NDArray {
  return x.log1p();
}

/**
 * Logarithm of the sum of exponentials: log(exp(x1) + exp(x2))
 * More numerically stable than computing the expression directly
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Array of logaddexp values
 */
export function logaddexp(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.logaddexp(x2);
}

/**
 * Logarithm base 2 of the sum of exponentials: log2(2^x1 + 2^x2)
 * More numerically stable than computing the expression directly
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Array of logaddexp2 values
 */
export function logaddexp2(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.logaddexp2(x2);
}

/**
 * Element-wise absolute value
 * @param x - Input array
 * @returns Array of absolute values
 */
export function absolute(x: NDArray): NDArray {
  return x.absolute();
}

// Alias for absolute
export { absolute as abs };

/**
 * Element-wise negation
 * @param x - Input array
 * @returns Array of negated values
 */
export function negative(x: NDArray): NDArray {
  return x.negative();
}

/**
 * Element-wise sign (-1, 0, or 1)
 * @param x - Input array
 * @returns Array of signs
 */
export function sign(x: NDArray): NDArray {
  return x.sign();
}

/**
 * Add arguments element-wise
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns Element-wise sum
 */
export function add(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.add(x2);
}

/**
 * Subtract arguments element-wise
 * @param x1 - First array
 * @param x2 - Second array or scalar to subtract
 * @returns Element-wise difference
 */
export function subtract(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.subtract(x2);
}

/**
 * Multiply arguments element-wise
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns Element-wise product
 */
export function multiply(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.multiply(x2);
}

/**
 * Element-wise modulo
 * @param x - Dividend array
 * @param divisor - Divisor (array or scalar)
 * @returns Remainder after division
 */
export function mod(x: NDArray, divisor: NDArray | number): NDArray {
  return x.mod(divisor);
}

/**
 * Element-wise division
 * @param x - Dividend array
 * @param divisor - Divisor (array or scalar)
 * @returns Array of quotients
 */
export function divide(x: NDArray, divisor: NDArray | number): NDArray {
  return x.divide(divisor);
}

// Alias for divide
export { divide as true_divide };

/**
 * Element-wise floor division
 * @param x - Dividend array
 * @param divisor - Divisor (array or scalar)
 * @returns Floor of the quotient
 */
export function floor_divide(x: NDArray, divisor: NDArray | number): NDArray {
  return x.floor_divide(divisor);
}

/**
 * Element-wise positive (unary +)
 * @param x - Input array
 * @returns Copy of the array
 */
export function positive(x: NDArray): NDArray {
  return x.positive();
}

/**
 * Element-wise reciprocal (1/x)
 * @param x - Input array
 * @returns Array of reciprocals
 */
export function reciprocal(x: NDArray): NDArray {
  return x.reciprocal();
}

/**
 * Dot product of two arrays
 *
 * Fully NumPy-compatible. Behavior depends on input dimensions:
 * - 0D · 0D: Multiply scalars → scalar
 * - 0D · ND or ND · 0D: Element-wise multiply → ND
 * - 1D · 1D: Inner product → scalar
 * - 2D · 2D: Matrix multiplication → 2D
 * - 2D · 1D: Matrix-vector product → 1D
 * - 1D · 2D: Vector-matrix product → 1D
 * - ND · 1D (N>2): Sum over last axis → (N-1)D
 * - 1D · ND (N>2): Sum over first axis → (N-1)D
 * - ND · MD (N,M≥2): Tensor contraction → (N+M-2)D
 *
 * @param a - First array
 * @param b - Second array
 * @returns Result of dot product (Complex for complex arrays)
 */
export function dot(a: NDArray, b: NDArray): NDArray | number | bigint | Complex {
  return a.dot(b);
}

/**
 * Sum of diagonal elements
 *
 * @param a - Input 2D array
 * @returns Sum of diagonal elements (Complex for complex arrays)
 */
export function trace(a: NDArray): number | bigint | Complex {
  return a.trace();
}

/**
 * Extract a diagonal from a matrix or N-D array
 *
 * @param a - Input array (must be at least 2D)
 * @param offset - Offset of the diagonal from the main diagonal (default: 0)
 * @param axis1 - First axis (default: 0)
 * @param axis2 - Second axis (default: 1)
 * @returns Array containing the diagonal elements
 */
export function diagonal(
  a: NDArray,
  offset: number = 0,
  axis1: number = 0,
  axis2: number = 1
): NDArray {
  const resultStorage = linalgOps.diagonal(a.storage, offset, axis1, axis2);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Kronecker product of two arrays
 *
 * @param a - First input array
 * @param b - Second input array
 * @returns Kronecker product of a and b
 */
export function kron(a: NDArray, b: NDArray): NDArray {
  const resultStorage = linalgOps.kron(a.storage, b.storage);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Permute array dimensions
 *
 * @param a - Input array
 * @param axes - Optional permutation of axes (defaults to reverse order)
 * @returns Transposed view
 */
export function transpose(a: NDArray, axes?: number[]): NDArray {
  return a.transpose(axes);
}

/**
 * Inner product of two arrays
 *
 * Contracts over last axes of both arrays.
 * Result shape: (*a.shape[:-1], *b.shape[:-1])
 *
 * @param a - First array
 * @param b - Second array
 * @returns Inner product result (Complex for complex arrays)
 */
export function inner(a: NDArray, b: NDArray): NDArray | number | bigint | Complex {
  return a.inner(b);
}

/**
 * Outer product of two arrays
 *
 * Flattens inputs then computes result[i,j] = a[i] * b[j]
 *
 * @param a - First array
 * @param b - Second array
 * @returns 2D outer product matrix
 */
export function outer(a: NDArray, b: NDArray): NDArray {
  return a.outer(b);
}

/**
 * Tensor dot product along specified axes
 *
 * @param a - First array
 * @param b - Second array
 * @param axes - Axes to contract (integer or [a_axes, b_axes])
 * @returns Tensor dot product
 */
export function tensordot(
  a: NDArray,
  b: NDArray,
  axes: number | [number[], number[]] = 2
): NDArray | number | bigint | Complex {
  return a.tensordot(b, axes);
}

// Trigonometric functions (standalone)

/**
 * Element-wise sine
 * @param x - Input array (angles in radians)
 * @returns Array of sine values
 */
export function sin(x: NDArray): NDArray {
  return x.sin();
}

/**
 * Element-wise cosine
 * @param x - Input array (angles in radians)
 * @returns Array of cosine values
 */
export function cos(x: NDArray): NDArray {
  return x.cos();
}

/**
 * Element-wise tangent
 * @param x - Input array (angles in radians)
 * @returns Array of tangent values
 */
export function tan(x: NDArray): NDArray {
  return x.tan();
}

/**
 * Element-wise inverse sine
 * @param x - Input array (values in range [-1, 1])
 * @returns Array of angles in radians
 */
export function arcsin(x: NDArray): NDArray {
  return x.arcsin();
}

// Alias for arcsin
export { arcsin as asin };

/**
 * Element-wise inverse cosine
 * @param x - Input array (values in range [-1, 1])
 * @returns Array of angles in radians
 */
export function arccos(x: NDArray): NDArray {
  return x.arccos();
}

// Alias for arccos
export { arccos as acos };

/**
 * Element-wise inverse tangent
 * @param x - Input array
 * @returns Array of angles in radians
 */
export function arctan(x: NDArray): NDArray {
  return x.arctan();
}

// Alias for arctan
export { arctan as atan };

/**
 * Element-wise arc tangent of x1/x2 choosing the quadrant correctly
 * @param x1 - y-coordinates
 * @param x2 - x-coordinates (array or scalar)
 * @returns Angles in radians between -π and π
 */
export function arctan2(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.arctan2(x2);
}

// Alias for arctan2
export { arctan2 as atan2 };

/**
 * Given the "legs" of a right triangle, return its hypotenuse
 * Equivalent to sqrt(x1**2 + x2**2), element-wise
 * @param x1 - First leg
 * @param x2 - Second leg (array or scalar)
 * @returns Hypotenuse values
 */
export function hypot(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.hypot(x2);
}

/**
 * Convert angles from radians to degrees
 * @param x - Input array (angles in radians)
 * @returns Angles in degrees
 */
export function degrees(x: NDArray): NDArray {
  return x.degrees();
}

/**
 * Convert angles from degrees to radians
 * @param x - Input array (angles in degrees)
 * @returns Angles in radians
 */
export function radians(x: NDArray): NDArray {
  return x.radians();
}

/**
 * Convert angles from degrees to radians (alias for radians)
 * @param x - Input array (angles in degrees)
 * @returns Angles in radians
 */
export function deg2rad(x: NDArray): NDArray {
  return x.radians();
}

/**
 * Convert angles from radians to degrees (alias for degrees)
 * @param x - Input array (angles in radians)
 * @returns Angles in degrees
 */
export function rad2deg(x: NDArray): NDArray {
  return x.degrees();
}

// Hyperbolic functions (standalone)

/**
 * Element-wise hyperbolic sine
 * @param x - Input array
 * @returns Array of sinh values
 */
export function sinh(x: NDArray): NDArray {
  return x.sinh();
}

/**
 * Element-wise hyperbolic cosine
 * @param x - Input array
 * @returns Array of cosh values
 */
export function cosh(x: NDArray): NDArray {
  return x.cosh();
}

/**
 * Element-wise hyperbolic tangent
 * @param x - Input array
 * @returns Array of tanh values
 */
export function tanh(x: NDArray): NDArray {
  return x.tanh();
}

/**
 * Element-wise inverse hyperbolic sine
 * @param x - Input array
 * @returns Array of arcsinh values
 */
export function arcsinh(x: NDArray): NDArray {
  return x.arcsinh();
}

// Alias for arcsinh
export { arcsinh as asinh };

/**
 * Element-wise inverse hyperbolic cosine
 * @param x - Input array (values >= 1)
 * @returns Array of arccosh values
 */
export function arccosh(x: NDArray): NDArray {
  return x.arccosh();
}

// Alias for arccosh
export { arccosh as acosh };

/**
 * Element-wise inverse hyperbolic tangent
 * @param x - Input array (values in range (-1, 1))
 * @returns Array of arctanh values
 */
export function arctanh(x: NDArray): NDArray {
  return x.arctanh();
}

// Alias for arctanh
export { arctanh as atanh };

// ========================================
// Array Manipulation Functions
// ========================================

/**
 * Swap two axes of an array
 *
 * @param a - Input array
 * @param axis1 - First axis
 * @param axis2 - Second axis
 * @returns View with axes swapped
 */
export function swapaxes(a: NDArray, axis1: number, axis2: number): NDArray {
  return a.swapaxes(axis1, axis2);
}

/**
 * Move axes to new positions
 *
 * @param a - Input array
 * @param source - Original positions of axes to move
 * @param destination - New positions for axes
 * @returns View with axes moved
 */
export function moveaxis(
  a: NDArray,
  source: number | number[],
  destination: number | number[]
): NDArray {
  return a.moveaxis(source, destination);
}

/**
 * Concatenate arrays along an existing axis
 *
 * @param arrays - Arrays to concatenate
 * @param axis - Axis along which to concatenate (default: 0)
 * @returns Concatenated array
 */
export function concatenate(arrays: NDArray[], axis: number = 0): NDArray {
  if (arrays.length === 0) {
    throw new Error('need at least one array to concatenate');
  }
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.concatenate(storages, axis);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Stack arrays along a new axis
 *
 * @param arrays - Arrays to stack (must have same shape)
 * @param axis - Axis in the result array along which to stack (default: 0)
 * @returns Stacked array
 */
export function stack(arrays: NDArray[], axis: number = 0): NDArray {
  if (arrays.length === 0) {
    throw new Error('need at least one array to stack');
  }
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.stack(storages, axis);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Stack arrays vertically (row-wise)
 *
 * @param arrays - Arrays to stack
 * @returns Vertically stacked array
 */
export function vstack(arrays: NDArray[]): NDArray {
  if (arrays.length === 0) {
    throw new Error('need at least one array to stack');
  }
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.vstack(storages);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Stack arrays horizontally (column-wise)
 *
 * @param arrays - Arrays to stack
 * @returns Horizontally stacked array
 */
export function hstack(arrays: NDArray[]): NDArray {
  if (arrays.length === 0) {
    throw new Error('need at least one array to stack');
  }
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.hstack(storages);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Stack arrays depth-wise (along third axis)
 *
 * @param arrays - Arrays to stack
 * @returns Depth-stacked array
 */
export function dstack(arrays: NDArray[]): NDArray {
  if (arrays.length === 0) {
    throw new Error('need at least one array to stack');
  }
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.dstack(storages);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Join a sequence of arrays along an existing axis (alias for concatenate)
 */
export function concat(arrays: NDArray[], axis: number = 0): NDArray {
  return concatenate(arrays, axis);
}

/**
 * Split an array into a sequence of sub-arrays along an axis (inverse of stack)
 */
export function unstack(a: NDArray, axis: number = 0): NDArray[] {
  const storages = shapeOps.unstack(a.storage, axis);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Assemble an nd-array from nested lists of blocks
 */
export function block(arrays: NDArray[]): NDArray {
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.block(storages);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Split array into multiple sub-arrays
 *
 * @param a - Array to split
 * @param indicesOrSections - Number of equal sections or indices where to split
 * @param axis - Axis along which to split (default: 0)
 * @returns List of sub-arrays
 */
export function split(
  a: NDArray,
  indicesOrSections: number | number[],
  axis: number = 0
): NDArray[] {
  const storages = shapeOps.split(a.storage, indicesOrSections, axis);
  return storages.map((s) => NDArray._fromStorage(s, a.base ?? a));
}

/**
 * Split array into multiple sub-arrays (allows unequal splits)
 *
 * @param a - Array to split
 * @param indicesOrSections - Number of sections or indices where to split
 * @param axis - Axis along which to split (default: 0)
 * @returns List of sub-arrays
 */
export function array_split(
  a: NDArray,
  indicesOrSections: number | number[],
  axis: number = 0
): NDArray[] {
  const storages = shapeOps.arraySplit(a.storage, indicesOrSections, axis);
  return storages.map((s) => NDArray._fromStorage(s, a.base ?? a));
}

/**
 * Split array vertically (row-wise)
 *
 * @param a - Array to split
 * @param indicesOrSections - Number of sections or indices where to split
 * @returns List of sub-arrays
 */
export function vsplit(a: NDArray, indicesOrSections: number | number[]): NDArray[] {
  const storages = shapeOps.vsplit(a.storage, indicesOrSections);
  return storages.map((s) => NDArray._fromStorage(s, a.base ?? a));
}

/**
 * Split array horizontally (column-wise)
 *
 * @param a - Array to split
 * @param indicesOrSections - Number of sections or indices where to split
 * @returns List of sub-arrays
 */
export function hsplit(a: NDArray, indicesOrSections: number | number[]): NDArray[] {
  const storages = shapeOps.hsplit(a.storage, indicesOrSections);
  return storages.map((s) => NDArray._fromStorage(s, a.base ?? a));
}

/**
 * Tile array by repeating along each axis
 *
 * @param a - Input array
 * @param reps - Number of repetitions along each axis
 * @returns Tiled array
 */
export function tile(a: NDArray, reps: number | number[]): NDArray {
  const resultStorage = shapeOps.tile(a.storage, reps);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Repeat elements of an array
 *
 * @param a - Input array
 * @param repeats - Number of repetitions for each element
 * @param axis - Axis along which to repeat (if undefined, flattens first)
 * @returns Array with repeated elements
 */
export function repeat(a: NDArray, repeats: number | number[], axis?: number): NDArray {
  return a.repeat(repeats, axis);
}

/**
 * Return a contiguous flattened array
 *
 * @param a - Input array
 * @returns Flattened 1-D array (view if possible)
 */
export function ravel(a: NDArray): NDArray {
  return a.ravel();
}

/**
 * Return a copy of the array collapsed into one dimension
 */
export function flatten(a: NDArray): NDArray {
  return a.flatten();
}

/**
 * Fill the array with a scalar value (in-place)
 */
export function fill(a: NDArray, value: number | bigint): void {
  a.fill(value);
}

/**
 * Copy an element of an array to a standard scalar and return it
 */
export function item(a: NDArray, ...args: number[]): number | bigint | Complex {
  return a.item(...args);
}

/**
 * Return the array as a nested list
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function tolist(a: NDArray): any {
  return a.tolist();
}

/**
 * Return the raw bytes of the array data
 */
export function tobytes(a: NDArray): ArrayBuffer {
  return a.tobytes();
}

/**
 * Swap the bytes of the array elements
 */
export function byteswap(a: NDArray, inplace: boolean = false): NDArray {
  return a.byteswap(inplace);
}

/**
 * New view of array with the same data
 */
export function view(a: NDArray, dtype?: DType): NDArray {
  return a.view(dtype);
}

/**
 * Write array to a file as text or binary
 */
export function tofile(a: NDArray, file: string, sep: string = '', format: string = ''): void {
  a.tofile(file, sep, format);
}

/**
 * Reshape array to new shape
 *
 * @param a - Input array
 * @param newShape - New shape
 * @returns Reshaped array (view if possible)
 */
export function reshape(a: NDArray, newShape: number[]): NDArray {
  return a.reshape(...newShape);
}

/**
 * Remove axes of length 1
 *
 * @param a - Input array
 * @param axis - Axis to squeeze (optional, squeezes all if not specified)
 * @returns Squeezed array (view)
 */
export function squeeze(a: NDArray, axis?: number): NDArray {
  return a.squeeze(axis);
}

/**
 * Expand the shape of an array by inserting a new axis
 *
 * @param a - Input array
 * @param axis - Position where new axis should be inserted
 * @returns Array with expanded shape (view)
 */
export function expand_dims(a: NDArray, axis: number): NDArray {
  return a.expand_dims(axis);
}

/**
 * Reverse the order of elements along the given axis
 *
 * @param m - Input array
 * @param axis - Axis or axes to flip (flips all if undefined)
 * @returns Flipped array
 */
export function flip(m: NDArray, axis?: number | number[]): NDArray {
  const resultStorage = shapeOps.flip(m.storage, axis);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Flip array in the left/right direction (reverse along axis 1)
 *
 * @param m - Input array (must be at least 2-D)
 * @returns Flipped array
 */
export function fliplr(m: NDArray): NDArray {
  if (m.ndim < 2) {
    throw new Error('Input must be at least 2-D');
  }
  return flip(m, 1);
}

/**
 * Flip array in the up/down direction (reverse along axis 0)
 *
 * @param m - Input array (must be at least 2-D)
 * @returns Flipped array
 */
export function flipud(m: NDArray): NDArray {
  if (m.ndim < 2) {
    throw new Error('Input must be at least 2-D');
  }
  return flip(m, 0);
}

/**
 * Rotate array by 90 degrees
 *
 * @param m - Input array
 * @param k - Number of times to rotate (default 1)
 * @param axes - The axes to rotate in (default [0, 1])
 * @returns Rotated array
 */
export function rot90(m: NDArray, k: number = 1, axes: [number, number] = [0, 1]): NDArray {
  const resultStorage = shapeOps.rot90(m.storage, k, axes);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Roll array elements along a given axis
 *
 * @param a - Input array
 * @param shift - Number of positions to shift
 * @param axis - Axis along which to roll (rolls flattened array if undefined)
 * @returns Rolled array
 */
export function roll(a: NDArray, shift: number | number[], axis?: number | number[]): NDArray {
  const resultStorage = shapeOps.roll(a.storage, shift, axis);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Roll the specified axis backwards until it lies in a given position
 *
 * @param a - Input array
 * @param axis - The axis to roll backwards
 * @param start - Position to roll to (default 0)
 * @returns Array with rolled axis (view)
 */
export function rollaxis(a: NDArray, axis: number, start: number = 0): NDArray {
  const resultStorage = shapeOps.rollaxis(a.storage, axis, start);
  return NDArray._fromStorage(resultStorage, a.base ?? a);
}

/**
 * Convert inputs to arrays with at least 1 dimension
 *
 * @param arrays - Input arrays
 * @returns Arrays with at least 1 dimension
 */
export function atleast_1d(...arrays: NDArray[]): NDArray | NDArray[] {
  const storages = arrays.map((a) => a.storage);
  const resultStorages = shapeOps.atleast1d(storages);
  const results = resultStorages.map((s, i) => {
    if (s === storages[i]) {
      return arrays[i]!;
    }
    return NDArray._fromStorage(s);
  });
  return results.length === 1 ? results[0]! : results;
}

/**
 * Convert inputs to arrays with at least 2 dimensions
 *
 * @param arrays - Input arrays
 * @returns Arrays with at least 2 dimensions
 */
export function atleast_2d(...arrays: NDArray[]): NDArray | NDArray[] {
  const storages = arrays.map((a) => a.storage);
  const resultStorages = shapeOps.atleast2d(storages);
  const results = resultStorages.map((s, i) => {
    if (s === storages[i]) {
      return arrays[i]!;
    }
    return NDArray._fromStorage(s);
  });
  return results.length === 1 ? results[0]! : results;
}

/**
 * Convert inputs to arrays with at least 3 dimensions
 *
 * @param arrays - Input arrays
 * @returns Arrays with at least 3 dimensions
 */
export function atleast_3d(...arrays: NDArray[]): NDArray | NDArray[] {
  const storages = arrays.map((a) => a.storage);
  const resultStorages = shapeOps.atleast3d(storages);
  const results = resultStorages.map((s, i) => {
    if (s === storages[i]) {
      return arrays[i]!;
    }
    return NDArray._fromStorage(s);
  });
  return results.length === 1 ? results[0]! : results;
}

/**
 * Split array along third axis (depth)
 *
 * @param ary - Input array (must be at least 3-D)
 * @param indices_or_sections - Number of sections or indices where to split
 * @returns List of sub-arrays
 */
export function dsplit(ary: NDArray, indices_or_sections: number | number[]): NDArray[] {
  const storages = shapeOps.dsplit(ary.storage, indices_or_sections);
  return storages.map((s) => NDArray._fromStorage(s, ary.base ?? ary));
}

/**
 * Stack 1-D arrays as columns into a 2-D array
 *
 * @param arrays - 1-D arrays to stack
 * @returns 2-D array with inputs as columns
 */
export function column_stack(arrays: NDArray[]): NDArray {
  if (arrays.length === 0) {
    throw new Error('need at least one array to stack');
  }
  const storages = arrays.map((a) => a.storage);
  const resultStorage = shapeOps.columnStack(storages);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Stack arrays in sequence vertically (alias for vstack)
 *
 * @param arrays - Arrays to stack
 * @returns Vertically stacked array
 */
export function row_stack(arrays: NDArray[]): NDArray {
  return vstack(arrays);
}

/**
 * Return a new array with the given shape (repeating data if needed)
 *
 * @param a - Input array
 * @param new_shape - New shape
 * @returns Resized array
 */
export function resize(a: NDArray, new_shape: number[]): NDArray {
  const resultStorage = shapeOps.resize(a.storage, new_shape);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Append values to the end of an array
 *
 * @param arr - Input array
 * @param values - Values to append
 * @param axis - Axis along which to append (flattens if undefined)
 * @returns Array with values appended
 */
export function append(
  arr: NDArray,
  values: NDArray | ArrayLike<number | bigint> | number,
  axis?: number
): NDArray {
  // Convert values to NDArray if needed
  const valArray =
    values instanceof NDArray
      ? values
      : array(values as ArrayLike<number | bigint> | number, arr.dtype as DType);

  if (axis === undefined) {
    // Flatten both and concatenate
    const flatArr = arr.flatten();
    const flatValues = valArray.flatten();
    return concatenate([flatArr, flatValues]);
  }

  // Concatenate along specified axis
  return concatenate([arr, valArray], axis);
}

/**
 * Return a new array with sub-arrays along an axis deleted
 *
 * @param arr - Input array
 * @param obj - Indices to delete
 * @param axis - Axis along which to delete (flattens if undefined)
 * @returns Array with elements deleted
 */

export function delete_(arr: NDArray, obj: number | number[], axis?: number): NDArray {
  const dtype = arr.dtype as DType;

  if (axis === undefined) {
    // Delete from flattened array
    const flat = arr.flatten();
    const indices = Array.isArray(obj) ? obj : [obj];
    const normalizedIndices = indices.map((i) => (i < 0 ? flat.size + i : i));
    const keepIndices: number[] = [];

    for (let i = 0; i < flat.size; i++) {
      if (!normalizedIndices.includes(i)) {
        keepIndices.push(i);
      }
    }

    const Constructor = getTypedArrayConstructor(dtype);
    const data = new Constructor!(keepIndices.length);

    for (let i = 0; i < keepIndices.length; i++) {
      const val = flat.get([keepIndices[i]!]);
      if (isBigIntDType(dtype)) {
        (data as BigInt64Array | BigUint64Array)[i] =
          typeof val === 'bigint' ? val : BigInt(val as number);
      } else {
        (data as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = val as number;
      }
    }

    const storage = ArrayStorage.fromData(data, [keepIndices.length], dtype);
    return new NDArray(storage);
  }

  // Delete along specified axis
  const shape = arr.shape;
  const ndim = shape.length;
  const normalizedAxis = axis < 0 ? ndim + axis : axis;

  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const indices = Array.isArray(obj) ? obj : [obj];
  const normalizedIndices = new Set(indices.map((i) => (i < 0 ? axisSize + i : i)));

  // Build slices to keep
  const keepRanges: [number, number][] = [];
  let start = 0;

  for (let i = 0; i <= axisSize; i++) {
    if (normalizedIndices.has(i) || i === axisSize) {
      if (i > start) {
        keepRanges.push([start, i]);
      }
      start = i + 1;
    }
  }

  if (keepRanges.length === 0) {
    // Delete all elements along this axis
    const newShape = [...shape];
    newShape[normalizedAxis] = 0;
    return zeros(newShape, dtype);
  }

  // Split and concatenate the kept parts
  const parts: NDArray[] = [];
  for (const [rangeStart, rangeEnd] of keepRanges) {
    // Create a slice for this range
    const slices: string[] = shape.map(() => ':');
    slices[normalizedAxis] = `${rangeStart}:${rangeEnd}`;
    parts.push(arr.slice(...slices));
  }

  return concatenate(parts, normalizedAxis);
}

/**
 * Insert values along the given axis before the given indices
 *
 * @param arr - Input array
 * @param obj - Index before which to insert
 * @param values - Values to insert
 * @param axis - Axis along which to insert (flattens if undefined)
 * @returns Array with values inserted
 */
export function insert(
  arr: NDArray,
  obj: number,
  values: NDArray | ArrayLike<number | bigint> | number,
  axis?: number
): NDArray {
  // Convert values to NDArray if needed
  const valArray =
    values instanceof NDArray
      ? values
      : array(values as ArrayLike<number | bigint> | number, arr.dtype as DType);

  if (axis === undefined) {
    // Insert into flattened array
    const flat = arr.flatten();
    const flatValues = valArray.flatten();
    const idx = obj < 0 ? flat.size + obj : obj;

    if (idx < 0 || idx > flat.size) {
      throw new Error(`index ${obj} is out of bounds for array of size ${flat.size}`);
    }

    const before = idx > 0 ? flat.slice(`0:${idx}`) : null;
    const after = idx < flat.size ? flat.slice(`${idx}:`) : null;

    const parts: NDArray[] = [];
    if (before) parts.push(before);
    parts.push(flatValues);
    if (after) parts.push(after);

    return concatenate(parts);
  }

  // Insert along specified axis
  const shape = arr.shape;
  const ndim = shape.length;
  const normalizedAxis = axis < 0 ? ndim + axis : axis;

  if (normalizedAxis < 0 || normalizedAxis >= ndim) {
    throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }

  const axisSize = shape[normalizedAxis]!;
  const idx = obj < 0 ? axisSize + obj : obj;

  if (idx < 0 || idx > axisSize) {
    throw new Error(`index ${obj} is out of bounds for axis ${axis} with size ${axisSize}`);
  }

  const parts: NDArray[] = [];

  if (idx > 0) {
    const slices: string[] = shape.map(() => ':');
    slices[normalizedAxis] = `0:${idx}`;
    parts.push(arr.slice(...slices));
  }

  parts.push(valArray);

  if (idx < axisSize) {
    const slices: string[] = shape.map(() => ':');
    slices[normalizedAxis] = `${idx}:`;
    parts.push(arr.slice(...slices));
  }

  return concatenate(parts, normalizedAxis);
}

/**
 * Pad an array
 *
 * @param array - Input array
 * @param pad_width - Number of values padded to edges of each axis
 * @param mode - Padding mode ('constant', 'edge', 'reflect', 'symmetric', 'wrap')
 * @param constant_values - Value for constant padding (default 0)
 * @returns Padded array
 */
export function pad(
  arr: NDArray,
  pad_width: number | [number, number] | Array<[number, number]>,
  mode: 'constant' | 'edge' | 'reflect' | 'symmetric' | 'wrap' = 'constant',
  constant_values: number = 0
): NDArray {
  const shape = arr.shape;
  const ndim = shape.length;
  const dtype = arr.dtype as DType;

  // Normalize pad_width to [[before, after], ...] for each axis
  let padWidths: Array<[number, number]>;
  if (typeof pad_width === 'number') {
    padWidths = shape.map(() => [pad_width, pad_width] as [number, number]);
  } else if (Array.isArray(pad_width) && typeof pad_width[0] === 'number') {
    // Single [before, after] pair for all axes
    padWidths = shape.map(() => pad_width as [number, number]);
  } else {
    padWidths = pad_width as Array<[number, number]>;
  }

  if (padWidths.length !== ndim) {
    throw new Error(`pad_width must have ${ndim} elements`);
  }

  // Calculate new shape
  const newShape = shape.map((s, i) => s + padWidths[i]![0] + padWidths[i]![1]);
  const newSize = newShape.reduce((a, b) => a * b, 1);

  const Constructor = getTypedArrayConstructor(dtype);
  const outputData = new Constructor!(newSize);
  const isBigInt = isBigIntDType(dtype);

  // Initialize with constant value for constant mode
  if (mode === 'constant') {
    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array).fill(BigInt(constant_values));
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>).fill(constant_values);
    }
  }

  // Copy original data to center
  const outputIndices = new Array(ndim).fill(0);

  for (let i = 0; i < newSize; i++) {
    // Check if this position is in the original data region
    let inOriginal = true;
    const sourceIndices: number[] = [];

    for (let d = 0; d < ndim; d++) {
      const [padBefore] = padWidths[d]!;
      const srcIdx = outputIndices[d]! - padBefore;
      if (srcIdx < 0 || srcIdx >= shape[d]!) {
        inOriginal = false;
        break;
      }
      sourceIndices.push(srcIdx);
    }

    let value: number | bigint;

    if (inOriginal) {
      // Get from original array
      value = arr.get(sourceIndices) as number | bigint;
    } else if (mode === 'constant') {
      // Already filled with constant
      // Increment indices and continue
      for (let d = ndim - 1; d >= 0; d--) {
        outputIndices[d]++;
        if (outputIndices[d]! < newShape[d]!) break;
        outputIndices[d] = 0;
      }
      continue;
    } else {
      // Calculate source index based on mode
      const mappedIndices: number[] = [];
      for (let d = 0; d < ndim; d++) {
        const [padBefore] = padWidths[d]!;
        let srcIdx = outputIndices[d]! - padBefore;
        const axisSize = shape[d]!;

        if (srcIdx < 0) {
          if (mode === 'edge') {
            srcIdx = 0;
          } else if (mode === 'reflect') {
            srcIdx = -srcIdx;
            if (srcIdx >= axisSize) srcIdx = axisSize - 1;
          } else if (mode === 'symmetric') {
            srcIdx = -srcIdx - 1;
            if (srcIdx >= axisSize) srcIdx = axisSize - 1;
            if (srcIdx < 0) srcIdx = 0;
          } else if (mode === 'wrap') {
            srcIdx = ((srcIdx % axisSize) + axisSize) % axisSize;
          }
        } else if (srcIdx >= axisSize) {
          if (mode === 'edge') {
            srcIdx = axisSize - 1;
          } else if (mode === 'reflect') {
            srcIdx = 2 * axisSize - srcIdx - 2;
            if (srcIdx < 0) srcIdx = 0;
          } else if (mode === 'symmetric') {
            srcIdx = 2 * axisSize - srcIdx - 1;
            if (srcIdx < 0) srcIdx = 0;
          } else if (mode === 'wrap') {
            srcIdx = srcIdx % axisSize;
          }
        }

        mappedIndices.push(Math.max(0, Math.min(axisSize - 1, srcIdx)));
      }
      value = arr.get(mappedIndices) as number | bigint;
    }

    // Write to output
    if (isBigInt) {
      (outputData as BigInt64Array | BigUint64Array)[i] =
        typeof value === 'bigint' ? value : BigInt(Number(value));
    } else {
      (outputData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = Number(value);
    }

    // Increment indices
    for (let d = ndim - 1; d >= 0; d--) {
      outputIndices[d]++;
      if (outputIndices[d]! < newShape[d]!) break;
      outputIndices[d] = 0;
    }
  }

  const storage = ArrayStorage.fromData(outputData, newShape, dtype);
  return new NDArray(storage);
}

// ========================================
// Advanced Functions
// ========================================

/**
 * Broadcast an array to a given shape
 *
 * @param a - Input array
 * @param shape - Target shape
 * @returns View broadcast to target shape
 */
export function broadcast_to(a: NDArray, shape: number[]): NDArray {
  const resultStorage = advancedOps.broadcast_to(a.storage, shape);
  return NDArray._fromStorage(resultStorage, a.base ?? a);
}

/**
 * Broadcast arrays to a common shape
 *
 * @param arrays - Arrays to broadcast
 * @returns Arrays broadcast to common shape
 */
export function broadcast_arrays(...arrays: NDArray[]): NDArray[] {
  const storages = arrays.map((a) => a.storage);
  const resultStorages = advancedOps.broadcast_arrays(storages);
  return resultStorages.map((s, i) => NDArray._fromStorage(s, arrays[i]!.base ?? arrays[i]!));
}

/**
 * Compute the broadcast shape for multiple shapes
 *
 * Returns the resulting shape if all shapes are broadcast-compatible.
 * Throws an error if shapes are not broadcast-compatible.
 *
 * @param shapes - Variable number of shapes to broadcast
 * @returns The broadcast output shape
 * @throws Error if shapes are not broadcast-compatible
 */
export function broadcast_shapes(...shapes: number[][]): number[] {
  return advancedOps.broadcast_shapes(...shapes);
}

/**
 * Take elements from an array along an axis
 *
 * @param a - Input array
 * @param indices - Indices of elements to take
 * @param axis - Axis along which to take (if undefined, flattens first)
 * @returns Array with selected elements
 */
export function take(a: NDArray, indices: number[], axis?: number): NDArray {
  return a.take(indices, axis);
}

/**
 * Put values at specified indices (modifies array in-place)
 *
 * @param a - Target array
 * @param indices - Indices at which to place values
 * @param values - Values to put
 */
export function put(a: NDArray, indices: number[], values: NDArray | number | bigint): void {
  a.put(indices, values);
}

/**
 * Integer array indexing (fancy indexing)
 *
 * Select elements from an array using an array of indices.
 * NumPy equivalent: `arr[[0, 2, 4]]`
 *
 * @param a - Input array
 * @param indices - Array of integer indices (as number[], NDArray, or nested arrays)
 * @param axis - Axis along which to index (default: 0)
 * @returns New array with selected elements
 *
 * @example
 * ```typescript
 * const arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
 * np.iindex(arr, [0, 2]);      // [[1, 2, 3], [7, 8, 9]]
 * np.iindex(arr, [0, 2], 1);   // [[1, 3], [4, 6], [7, 9]]
 * ```
 */
export function iindex(
  a: NDArray,
  indices: NDArray | number[] | number[][],
  axis: number = 0
): NDArray {
  return a.iindex(indices, axis);
}

/**
 * Boolean array indexing (fancy indexing with mask)
 *
 * Select elements from an array where a boolean mask is true.
 * NumPy equivalent: `arr[arr > 5]`
 *
 * @param a - Input array
 * @param mask - Boolean NDArray mask
 * @param axis - Axis along which to apply the mask (default: flattens array)
 * @returns New array with selected elements
 *
 * @example
 * ```typescript
 * const arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
 * const mask = arr.greater(5);
 * np.bindex(arr, mask);        // [6, 7, 8, 9]
 * ```
 */
export function bindex(a: NDArray, mask: NDArray, axis?: number): NDArray {
  return a.bindex(mask, axis);
}

/**
 * Copy values from one array to another, broadcasting as necessary.
 *
 * @param dst - Destination array (modified in-place)
 * @param src - Source array or scalar
 * @param where - Optional boolean array. Only copy where True (not yet implemented)
 * @throws Error if shapes are not broadcastable
 *
 * @example
 * ```typescript
 * const dst = np.zeros([3, 3]);
 * const src = np.array([1, 2, 3]);
 * np.copyto(dst, src);  // Each row of dst becomes [1, 2, 3]
 * ```
 */
export function copyto(dst: NDArray, src: NDArray | number | bigint, where?: NDArray): void {
  if (where !== undefined) {
    throw new Error('copyto with where parameter is not yet implemented');
  }

  const dstStorage = dst.storage;
  const dstShape = dst.shape;
  const dstSize = dst.size;
  const dstDtype = dst.dtype as DType;

  // Handle scalar source
  if (typeof src === 'number' || typeof src === 'bigint') {
    dst.fill(src);
    return;
  }

  const srcStorage = src.storage;
  const srcShape = src.shape;

  // Check if shapes are broadcastable
  // dst shape must be broadcastable FROM src shape
  const broadcastShape = computeBroadcastShape([srcShape as number[], dstShape as number[]]);
  if (!broadcastShape) {
    throw new Error(
      `could not broadcast input array from shape (${srcShape.join(',')}) into shape (${dstShape.join(',')})`
    );
  }

  // Verify broadcast shape matches dst shape
  if (
    broadcastShape.length !== dstShape.length ||
    !broadcastShape.every((d, i) => d === dstShape[i])
  ) {
    throw new Error(
      `could not broadcast input array from shape (${srcShape.join(',')}) into shape (${dstShape.join(',')})`
    );
  }

  // Broadcast src to dst shape
  const broadcastedSrc = advancedOps.broadcast_to(srcStorage, dstShape as number[]);

  // Copy values
  if (isBigIntDType(dstDtype)) {
    for (let i = 0; i < dstSize; i++) {
      const val = broadcastedSrc.iget(i);
      const bigintVal = typeof val === 'bigint' ? val : BigInt(Math.round(Number(val)));
      dstStorage.iset(i, bigintVal);
    }
  } else if (dstDtype === 'bool') {
    for (let i = 0; i < dstSize; i++) {
      const val = broadcastedSrc.iget(i);
      dstStorage.iset(i, val ? 1 : 0);
    }
  } else {
    for (let i = 0; i < dstSize; i++) {
      const val = broadcastedSrc.iget(i);
      dstStorage.iset(i, Number(val));
    }
  }
}

/**
 * Construct array from index array and choices
 *
 * @param a - Index array (integer indices into choices)
 * @param choices - Arrays to choose from
 * @returns Array constructed from choices
 */
export function choose(a: NDArray, choices: NDArray[]): NDArray {
  const choiceStorages = choices.map((c) => c.storage);
  const resultStorage = advancedOps.choose(a.storage, choiceStorages);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Check if two arrays are element-wise equal
 *
 * @param a - First array
 * @param b - Second array
 * @param equal_nan - Whether to consider NaN equal to NaN (default: false)
 * @returns True if arrays are equal element-wise
 */
export function array_equal(a: NDArray, b: NDArray, equal_nan: boolean = false): boolean {
  return advancedOps.array_equal(a.storage, b.storage, equal_nan);
}

/**
 * Returns True if two arrays are element-wise equal within a tolerance.
 * Unlike array_equal, this function broadcasts the arrays before comparison.
 *
 * @param a1 - First input array
 * @param a2 - Second input array
 * @returns True if arrays are equivalent (after broadcasting)
 */
export function array_equiv(a1: NDArray, a2: NDArray): boolean {
  return comparisonOps.arrayEquiv(a1.storage, a2.storage);
}

// ============================================================================
// Top-level Reduction Functions
// ============================================================================

/**
 * Sum of array elements over a given axis.
 * @param a - Input array
 * @param axis - Axis along which to sum. If undefined, sum all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Sum value(s)
 */
export function sum(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  return a.sum(axis, keepdims);
}

/**
 * Compute the arithmetic mean along the specified axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Mean value(s)
 */
export function mean(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  return a.mean(axis, keepdims);
}

/**
 * Return the product of array elements over a given axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Product value(s)
 */
export function prod(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  return a.prod(axis, keepdims);
}

/**
 * Compute the standard deviation along the specified axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Standard deviation value(s)
 */
export function std(
  a: NDArray,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): NDArray | number {
  return a.std(axis, ddof, keepdims);
}

/**
 * Compute the variance along the specified axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param ddof - Delta degrees of freedom (default: 0)
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Variance value(s)
 */
export function variance(
  a: NDArray,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): NDArray | number {
  return a.var(axis, ddof, keepdims);
}

/**
 * Returns the indices of the minimum values along an axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use flattened array.
 * @returns Indices of minimum values
 */
export function argmin(a: NDArray, axis?: number): NDArray | number {
  return a.argmin(axis);
}

/**
 * Returns the indices of the maximum values along an axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use flattened array.
 * @returns Indices of maximum values
 */
export function argmax(a: NDArray, axis?: number): NDArray | number {
  return a.argmax(axis);
}

/**
 * Test whether all array elements along a given axis evaluate to True.
 * @param a - Input array
 * @param axis - Axis along which to test. If undefined, test all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Boolean result(s)
 */
export function all(a: NDArray, axis?: number, keepdims: boolean = false): NDArray | boolean {
  return a.all(axis, keepdims);
}

/**
 * Test whether any array element along a given axis evaluates to True.
 * @param a - Input array
 * @param axis - Axis along which to test. If undefined, test all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Boolean result(s)
 */
export function any(a: NDArray, axis?: number, keepdims: boolean = false): NDArray | boolean {
  return a.any(axis, keepdims);
}

/**
 * Return the cumulative sum of the elements along a given axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, flattened array is used.
 * @returns Array with cumulative sums
 */
export function cumsum(a: NDArray, axis?: number): NDArray {
  return NDArray._fromStorage(reductionOps.cumsum(a.storage, axis));
}

// Alias for cumsum
export { cumsum as cumulative_sum };

/**
 * Return the cumulative product of the elements along a given axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, flattened array is used.
 * @returns Array with cumulative products
 */
export function cumprod(a: NDArray, axis?: number): NDArray {
  return NDArray._fromStorage(reductionOps.cumprod(a.storage, axis));
}

// Alias for cumprod
export { cumprod as cumulative_prod };

/**
 * Return the maximum along a given axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Maximum value(s)
 */
export function max(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  return a.max(axis, keepdims);
}

// Alias for max
export { max as amax };

/**
 * Return the minimum along a given axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Minimum value(s)
 */
export function min(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  return a.min(axis, keepdims);
}

// Alias for min
export { min as amin };

/**
 * Peak to peak (maximum - minimum) value along a given axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Peak to peak value(s)
 */
export function ptp(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  const result = reductionOps.ptp(a.storage, axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Compute the median along the specified axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Median value(s)
 */
export function median(a: NDArray, axis?: number, keepdims: boolean = false): NDArray | number {
  const result = reductionOps.median(a.storage, axis, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Compute the q-th percentile of the data along the specified axis.
 * @param a - Input array
 * @param q - Percentile (0-100)
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Percentile value(s)
 */
export function percentile(
  a: NDArray,
  q: number,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  const result = reductionOps.percentile(a.storage, q, axis, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Compute the q-th quantile of the data along the specified axis.
 * @param a - Input array
 * @param q - Quantile (0-1)
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Quantile value(s)
 */
export function quantile(
  a: NDArray,
  q: number,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  const result = reductionOps.quantile(a.storage, q, axis, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Compute the weighted average along the specified axis.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param weights - Array of weights (must be same shape as array along specified axis)
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Weighted average value(s)
 */
export function average(
  a: NDArray,
  axis?: number,
  weights?: NDArray,
  keepdims: boolean = false
): NDArray | number | Complex {
  const weightsStorage = weights ? weights.storage : undefined;
  const result = reductionOps.average(a.storage, axis, weightsStorage, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

// ============================================================================
// NaN-aware Reduction Functions
// ============================================================================

/**
 * Return the sum of array elements over a given axis, treating NaNs as zero.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Sum value(s)
 */
export function nansum(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  const result = reductionOps.nansum(a.storage, axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Return the product of array elements over a given axis, treating NaNs as one.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Product value(s)
 */
export function nanprod(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  const result = reductionOps.nanprod(a.storage, axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Compute the arithmetic mean along the specified axis, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Mean value(s)
 */
export function nanmean(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  const result = reductionOps.nanmean(a.storage, axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Compute the variance along the specified axis, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param ddof - Delta degrees of freedom (default 0)
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Variance value(s)
 */
export function nanvar(
  a: NDArray,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): NDArray | number {
  const result = reductionOps.nanvar(a.storage, axis, ddof, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Compute the standard deviation along the specified axis, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param ddof - Delta degrees of freedom (default 0)
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Standard deviation value(s)
 */
export function nanstd(
  a: NDArray,
  axis?: number,
  ddof: number = 0,
  keepdims: boolean = false
): NDArray | number {
  const result = reductionOps.nanstd(a.storage, axis, ddof, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Return minimum of an array, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Minimum value(s)
 */
export function nanmin(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  const result = reductionOps.nanmin(a.storage, axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Return maximum of an array, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Maximum value(s)
 */
export function nanmax(
  a: NDArray,
  axis?: number,
  keepdims: boolean = false
): NDArray | number | Complex {
  const result = reductionOps.nanmax(a.storage, axis, keepdims);
  if (typeof result === 'number' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Return indices of the minimum value, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use flattened array.
 * @returns Index/indices of minimum value(s)
 */
export function nanargmin(a: NDArray, axis?: number): NDArray | number {
  const result = reductionOps.nanargmin(a.storage, axis);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Return indices of the maximum value, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use flattened array.
 * @returns Index/indices of maximum value(s)
 */
export function nanargmax(a: NDArray, axis?: number): NDArray | number {
  const result = reductionOps.nanargmax(a.storage, axis);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Return cumulative sum of elements, treating NaNs as zero.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use flattened array.
 * @returns Array with cumulative sums
 */
export function nancumsum(a: NDArray, axis?: number): NDArray {
  return NDArray._fromStorage(reductionOps.nancumsum(a.storage, axis));
}

/**
 * Return cumulative product of elements, treating NaNs as one.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use flattened array.
 * @returns Array with cumulative products
 */
export function nancumprod(a: NDArray, axis?: number): NDArray {
  return NDArray._fromStorage(reductionOps.nancumprod(a.storage, axis));
}

/**
 * Compute the median, ignoring NaNs.
 * @param a - Input array
 * @param axis - Axis along which to compute. If undefined, use all elements.
 * @param keepdims - If true, reduced axes are left as dimensions with size 1
 * @returns Median value(s)
 */
export function nanmedian(a: NDArray, axis?: number, keepdims: boolean = false): NDArray | number {
  const result = reductionOps.nanmedian(a.storage, axis, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Compute the q-th quantile of data along specified axis, ignoring NaNs
 */
export function nanquantile(
  a: NDArray,
  q: number,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  const result = reductionOps.nanquantile(a.storage, q, axis, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

/**
 * Compute the q-th percentile of data along specified axis, ignoring NaNs
 */
export function nanpercentile(
  a: NDArray,
  q: number,
  axis?: number,
  keepdims: boolean = false
): NDArray | number {
  const result = reductionOps.nanpercentile(a.storage, q, axis, keepdims);
  return typeof result === 'number' ? result : NDArray._fromStorage(result);
}

// ========================================
// Arithmetic Functions (Additional)
// ========================================

/**
 * Element-wise cube root
 *
 * @param x - Input array
 * @returns Array with cube root of each element
 */
export function cbrt(x: NDArray): NDArray {
  return x.cbrt();
}

/**
 * Element-wise absolute value (always returns float)
 *
 * @param x - Input array
 * @returns Array with absolute values as float
 */
export function fabs(x: NDArray): NDArray {
  return x.fabs();
}

/**
 * Returns both quotient and remainder (floor divide and modulo)
 *
 * @param x - Dividend array
 * @param y - Divisor (array or scalar)
 * @returns Tuple of [quotient, remainder] arrays
 */
export function divmod(x: NDArray, y: NDArray | number): [NDArray, NDArray] {
  return x.divmod(y);
}

/**
 * Element-wise square (x**2)
 *
 * @param x - Input array
 * @returns Array with squared values
 */
export function square(x: NDArray): NDArray {
  return x.square();
}

/**
 * Element-wise remainder (same as mod)
 *
 * @param x - Dividend array
 * @param y - Divisor (array or scalar)
 * @returns Array with remainder values
 */
export function remainder(x: NDArray, y: NDArray | number): NDArray {
  return x.remainder(y);
}

/**
 * Heaviside step function
 *
 * @param x1 - Input array
 * @param x2 - Value to use when x1 is 0
 * @returns Array with heaviside values (0 if x1 < 0, x2 if x1 == 0, 1 if x1 > 0)
 */
export function heaviside(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.heaviside(x2);
}

/**
 * First array raised to power of second, always promoting to float
 * @param x1 - Base values
 * @param x2 - Exponent values
 * @returns Result in float64
 */
export function float_power(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(arithmeticOps.float_power(x1.storage, x2Storage));
}

/**
 * Element-wise remainder of division (fmod)
 * Unlike mod/remainder, fmod matches C fmod behavior
 * @param x1 - Dividend
 * @param x2 - Divisor
 * @returns Remainder
 */
export function fmod(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(arithmeticOps.fmod(x1.storage, x2Storage));
}

/**
 * Decompose floating point numbers into mantissa and exponent
 * Returns [mantissa, exponent] where x = mantissa * 2^exponent
 * @param x - Input array
 * @returns Tuple of [mantissa, exponent] arrays
 */
export function frexp(x: NDArray): [NDArray, NDArray] {
  const [mantissa, exponent] = arithmeticOps.frexp(x.storage);
  return [NDArray._fromStorage(mantissa), NDArray._fromStorage(exponent)];
}

/**
 * Greatest common divisor
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns GCD
 */
export function gcd(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(arithmeticOps.gcd(x1.storage, x2Storage));
}

/**
 * Least common multiple
 * @param x1 - First array
 * @param x2 - Second array or scalar
 * @returns LCM
 */
export function lcm(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(arithmeticOps.lcm(x1.storage, x2Storage));
}

/**
 * Returns x1 * 2^x2, element-wise
 * @param x1 - Mantissa
 * @param x2 - Exponent
 * @returns Result
 */
export function ldexp(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(arithmeticOps.ldexp(x1.storage, x2Storage));
}

/**
 * Return fractional and integral parts of array
 * @param x - Input array
 * @returns Tuple of [fractional, integral] arrays
 */
export function modf(x: NDArray): [NDArray, NDArray] {
  const [fractional, integral] = arithmeticOps.modf(x.storage);
  return [NDArray._fromStorage(fractional), NDArray._fromStorage(integral)];
}

// ========================================
// Other Math Functions
// ========================================

/**
 * Clip (limit) the values in an array.
 *
 * Given an interval, values outside the interval are clipped to the interval edges.
 *
 * @param a - Input array
 * @param a_min - Minimum value (null to not clip minimum)
 * @param a_max - Maximum value (null to not clip maximum)
 * @returns Clipped array
 */
export function clip(
  a: NDArray,
  a_min: NDArray | number | null,
  a_max: NDArray | number | null
): NDArray {
  const minStorage = a_min instanceof NDArray ? a_min.storage : a_min;
  const maxStorage = a_max instanceof NDArray ? a_max.storage : a_max;
  return NDArray._fromStorage(arithmeticOps.clip(a.storage, minStorage, maxStorage));
}

/**
 * Element-wise maximum of array elements.
 *
 * Compare two arrays and return a new array containing the element-wise maxima.
 * If one of the elements being compared is a NaN, then that element is returned.
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Element-wise maximum
 */
export function maximum(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = x2 instanceof NDArray ? x2.storage : x2;
  return NDArray._fromStorage(arithmeticOps.maximum(x1.storage, x2Storage));
}

/**
 * Element-wise minimum of array elements.
 *
 * Compare two arrays and return a new array containing the element-wise minima.
 * If one of the elements being compared is a NaN, then that element is returned.
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Element-wise minimum
 */
export function minimum(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = x2 instanceof NDArray ? x2.storage : x2;
  return NDArray._fromStorage(arithmeticOps.minimum(x1.storage, x2Storage));
}

/**
 * Element-wise maximum of array elements, ignoring NaNs.
 *
 * Compare two arrays and return a new array containing the element-wise maxima.
 * If one of the values being compared is a NaN, the other is returned.
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Element-wise maximum, NaN-aware
 */
export function fmax(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = x2 instanceof NDArray ? x2.storage : x2;
  return NDArray._fromStorage(arithmeticOps.fmax(x1.storage, x2Storage));
}

/**
 * Element-wise minimum of array elements, ignoring NaNs.
 *
 * Compare two arrays and return a new array containing the element-wise minima.
 * If one of the values being compared is a NaN, the other is returned.
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Element-wise minimum, NaN-aware
 */
export function fmin(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = x2 instanceof NDArray ? x2.storage : x2;
  return NDArray._fromStorage(arithmeticOps.fmin(x1.storage, x2Storage));
}

/**
 * Replace NaN with zero and Inf with large finite numbers.
 *
 * @param x - Input array
 * @param nan - Value to replace NaN (default: 0.0)
 * @param posinf - Value to replace positive infinity (default: largest finite)
 * @param neginf - Value to replace negative infinity (default: most negative finite)
 * @returns Array with replacements
 */
export function nan_to_num(
  x: NDArray,
  nan: number = 0.0,
  posinf?: number,
  neginf?: number
): NDArray {
  return NDArray._fromStorage(arithmeticOps.nan_to_num(x.storage, nan, posinf, neginf));
}

/**
 * One-dimensional linear interpolation.
 *
 * Returns the one-dimensional piecewise linear interpolant to a function
 * with given discrete data points (xp, fp), evaluated at x.
 *
 * @param x - The x-coordinates at which to evaluate the interpolated values
 * @param xp - The x-coordinates of the data points (must be increasing)
 * @param fp - The y-coordinates of the data points
 * @param left - Value for x < xp[0] (default: fp[0])
 * @param right - Value for x > xp[-1] (default: fp[-1])
 * @returns Interpolated values
 */
export function interp(
  x: NDArray,
  xp: NDArray,
  fp: NDArray,
  left?: number,
  right?: number
): NDArray {
  return NDArray._fromStorage(arithmeticOps.interp(x.storage, xp.storage, fp.storage, left, right));
}

/**
 * Unwrap by changing deltas between values to 2*pi complement.
 *
 * Unwrap radian phase p by changing absolute jumps greater than
 * discont to their 2*pi complement along the given axis.
 *
 * @param p - Input array of phase angles in radians
 * @param discont - Maximum discontinuity between values (default: pi)
 * @param axis - Axis along which to unwrap (default: -1, last axis)
 * @param period - Size of the range over which the input wraps (default: 2*pi)
 * @returns Unwrapped array
 */
export function unwrap(
  p: NDArray,
  discont: number = Math.PI,
  axis: number = -1,
  period: number = 2 * Math.PI
): NDArray {
  return NDArray._fromStorage(arithmeticOps.unwrap(p.storage, discont, axis, period));
}

/**
 * Return the normalized sinc function.
 *
 * sinc(x) = sin(pi*x) / (pi*x)
 *
 * The sinc function is 1 at x = 0, and sin(pi*x)/(pi*x) otherwise.
 *
 * @param x - Input array
 * @returns Array of sinc values
 */
export function sinc(x: NDArray): NDArray {
  return NDArray._fromStorage(arithmeticOps.sinc(x.storage));
}

/**
 * Modified Bessel function of the first kind, order 0.
 *
 * @param x - Input array
 * @returns Array of I0 values
 */
export function i0(x: NDArray): NDArray {
  return NDArray._fromStorage(arithmeticOps.i0(x.storage));
}

// ========================================
// Bitwise Functions
// ========================================

/**
 * Bitwise AND element-wise
 *
 * @param x1 - First input array (must be integer type)
 * @param x2 - Second input array or scalar (must be integer type)
 * @returns Result of bitwise AND
 */
export function bitwise_and(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.bitwise_and(x2);
}

/**
 * Bitwise OR element-wise
 *
 * @param x1 - First input array (must be integer type)
 * @param x2 - Second input array or scalar (must be integer type)
 * @returns Result of bitwise OR
 */
export function bitwise_or(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.bitwise_or(x2);
}

/**
 * Bitwise XOR element-wise
 *
 * @param x1 - First input array (must be integer type)
 * @param x2 - Second input array or scalar (must be integer type)
 * @returns Result of bitwise XOR
 */
export function bitwise_xor(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.bitwise_xor(x2);
}

/**
 * Bitwise NOT (inversion) element-wise
 *
 * @param x - Input array (must be integer type)
 * @returns Result of bitwise NOT
 */
export function bitwise_not(x: NDArray): NDArray {
  return x.bitwise_not();
}

/**
 * Invert (bitwise NOT) element-wise
 * Alias for bitwise_not
 *
 * @param x - Input array (must be integer type)
 * @returns Result of bitwise inversion
 */
export function invert(x: NDArray): NDArray {
  return x.invert();
}

/**
 * Left shift elements by positions
 *
 * @param x1 - Input array (must be integer type)
 * @param x2 - Shift amount (array or scalar)
 * @returns Result of left shift
 */
export function left_shift(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.left_shift(x2);
}

/**
 * Right shift elements by positions
 *
 * @param x1 - Input array (must be integer type)
 * @param x2 - Shift amount (array or scalar)
 * @returns Result of right shift
 */
export function right_shift(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.right_shift(x2);
}

/**
 * Pack binary values into uint8 array
 *
 * Packs the elements of a binary-valued array into bits in a uint8 array.
 *
 * @param a - Input array (values are interpreted as binary: 0 or non-zero)
 * @param axis - The dimension over which bit-packing is done (default: -1)
 * @param bitorder - Order of bits: 'big' or 'little' (default: 'big')
 * @returns Packed uint8 array
 */
export function packbits(
  a: NDArray,
  axis: number = -1,
  bitorder: 'big' | 'little' = 'big'
): NDArray {
  const resultStorage = bitwiseOps.packbits(a.storage, axis, bitorder);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Unpack uint8 array into binary values
 *
 * Unpacks elements of a uint8 array into a binary-valued output array.
 *
 * @param a - Input uint8 array
 * @param axis - The dimension over which bit-unpacking is done (default: -1)
 * @param count - Number of elements to unpack, or -1 for all (default: -1)
 * @param bitorder - Order of bits: 'big' or 'little' (default: 'big')
 * @returns Unpacked uint8 array of 0s and 1s
 */
export function unpackbits(
  a: NDArray,
  axis: number = -1,
  count: number = -1,
  bitorder: 'big' | 'little' = 'big'
): NDArray {
  const resultStorage = bitwiseOps.unpackbits(a.storage, axis, count, bitorder);
  return NDArray._fromStorage(resultStorage);
}

/**
 * Count the number of 1-bits (population count) in each element.
 *
 * @param x - Input array (must be integer type)
 * @returns Array with population count for each element
 */
export function bitwise_count(x: NDArray): NDArray {
  return NDArray._fromStorage(bitwiseOps.bitwise_count(x.storage));
}

/**
 * Bitwise invert (alias for bitwise_not).
 *
 * @param x - Input array (must be integer type)
 * @returns Result with bitwise NOT values
 */
export function bitwise_invert(x: NDArray): NDArray {
  return NDArray._fromStorage(bitwiseOps.bitwise_invert(x.storage));
}

/**
 * Bitwise left shift (alias for left_shift).
 *
 * @param x1 - Input array (must be integer type)
 * @param x2 - Shift amount (array or scalar)
 * @returns Result with left-shifted values
 */
export function bitwise_left_shift(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(bitwiseOps.bitwise_left_shift(x1.storage, x2Storage));
}

/**
 * Bitwise right shift (alias for right_shift).
 *
 * @param x1 - Input array (must be integer type)
 * @param x2 - Shift amount (array or scalar)
 * @returns Result with right-shifted values
 */
export function bitwise_right_shift(x1: NDArray, x2: NDArray | number): NDArray {
  const x2Storage = typeof x2 === 'number' ? x2 : x2.storage;
  return NDArray._fromStorage(bitwiseOps.bitwise_right_shift(x1.storage, x2Storage));
}

// ========================================
// Logic Functions
// ========================================

/**
 * Logical AND element-wise
 *
 * Returns a boolean array where each element is the logical AND
 * of corresponding elements (non-zero = true, zero = false).
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Boolean result array
 */
export function logical_and(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.logical_and(x2);
}

/**
 * Logical OR element-wise
 *
 * Returns a boolean array where each element is the logical OR
 * of corresponding elements (non-zero = true, zero = false).
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Boolean result array
 */
export function logical_or(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.logical_or(x2);
}

/**
 * Logical NOT element-wise
 *
 * Returns a boolean array where each element is the logical NOT
 * of the input (non-zero = false, zero = true).
 *
 * @param x - Input array
 * @returns Boolean result array
 */
export function logical_not(x: NDArray): NDArray {
  return x.logical_not();
}

/**
 * Logical XOR element-wise
 *
 * Returns a boolean array where each element is the logical XOR
 * of corresponding elements (non-zero = true, zero = false).
 *
 * @param x1 - First input array
 * @param x2 - Second input array or scalar
 * @returns Boolean result array
 */
export function logical_xor(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.logical_xor(x2);
}

/**
 * Test element-wise for finiteness (not infinity and not NaN)
 *
 * @param x - Input array
 * @returns Boolean array where True means finite
 */
export function isfinite(x: NDArray): NDArray {
  return x.isfinite();
}

/**
 * Test element-wise for positive or negative infinity
 *
 * @param x - Input array
 * @returns Boolean array where True means infinite
 */
export function isinf(x: NDArray): NDArray {
  return x.isinf();
}

/**
 * Test element-wise for NaN (Not a Number)
 *
 * @param x - Input array
 * @returns Boolean array where True means NaN
 */
export function isnan(x: NDArray): NDArray {
  return x.isnan();
}

/**
 * Test element-wise for NaT (Not a Time)
 *
 * @param x - Input array
 * @returns Boolean array (always false without datetime support)
 */
export function isnat(x: NDArray): NDArray {
  return x.isnat();
}

/**
 * Change the sign of x1 to that of x2, element-wise
 *
 * Returns a value with the magnitude of x1 and the sign of x2.
 *
 * @param x1 - Values to change sign of (magnitude source)
 * @param x2 - Values whose sign is used (sign source)
 * @returns Array with magnitude from x1 and sign from x2
 */
export function copysign(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.copysign(x2);
}

/**
 * Returns element-wise True where signbit is set (less than zero)
 *
 * @param x - Input array
 * @returns Boolean array where True means signbit is set
 */
export function signbit(x: NDArray): NDArray {
  return x.signbit();
}

/**
 * Return the next floating-point value after x1 towards x2, element-wise
 *
 * @param x1 - Values to find the next representable value of
 * @param x2 - Direction to look for the next representable value
 * @returns Array of next representable values
 */
export function nextafter(x1: NDArray, x2: NDArray | number): NDArray {
  return x1.nextafter(x2);
}

/**
 * Return the distance between x and the nearest adjacent number
 *
 * @param x - Input array
 * @returns Array of spacing values
 */
export function spacing(x: NDArray): NDArray {
  return x.spacing();
}

/**
 * Test element-wise for complex number.
 *
 * For complex arrays, returns true for elements with non-zero imaginary part.
 * For real arrays, always returns false.
 *
 * @param x - Input array
 * @returns Boolean array
 */
export function iscomplex(x: NDArray): NDArray {
  return NDArray._fromStorage(logicOps.iscomplex(x.storage));
}

/**
 * Check whether array is complex type.
 *
 * @param x - Input array
 * @returns true if dtype is complex64 or complex128
 */
export function iscomplexobj(x: NDArray): boolean {
  return logicOps.iscomplexobj(x.storage);
}

/**
 * Test element-wise for real number (not complex).
 *
 * For complex arrays, returns true for elements with zero imaginary part.
 * For real arrays, always returns true.
 *
 * @param x - Input array
 * @returns Boolean array
 */
export function isreal(x: NDArray): NDArray {
  return NDArray._fromStorage(logicOps.isreal(x.storage));
}

/**
 * Check whether array is real type (not complex).
 *
 * @param x - Input array
 * @returns true if dtype is NOT complex64 or complex128
 */
export function isrealobj(x: NDArray): boolean {
  return logicOps.isrealobj(x.storage);
}

/**
 * Return the real part of complex argument.
 *
 * For complex arrays, returns the real components.
 * For real arrays, returns a copy of the input.
 *
 * @param x - Input array
 * @returns Array with real parts
 */
export function real(x: NDArray): NDArray {
  return NDArray._fromStorage(complexOps.real(x.storage));
}

/**
 * Return the imaginary part of complex argument.
 *
 * For complex arrays, returns the imaginary components.
 * For real arrays, returns zeros.
 *
 * @param x - Input array
 * @returns Array with imaginary parts
 */
export function imag(x: NDArray): NDArray {
  return NDArray._fromStorage(complexOps.imag(x.storage));
}

/**
 * Return the complex conjugate.
 *
 * For complex arrays, negates the imaginary part: (a + bi) -> (a - bi)
 * For real arrays, returns a copy of the input.
 *
 * @param x - Input array
 * @returns Complex conjugate array
 */
export function conj(x: NDArray): NDArray {
  return NDArray._fromStorage(complexOps.conj(x.storage));
}

// Alias for conj
export const conjugate = conj;

/**
 * Return the angle (phase) of complex argument.
 *
 * angle(z) = arctan2(imag(z), real(z))
 *
 * For real arrays, returns 0 for positive, pi for negative.
 *
 * @param x - Input array
 * @param deg - Return angle in degrees if true (default: false, returns radians)
 * @returns Array with angles in radians (or degrees)
 */
export function angle(x: NDArray, deg: boolean = false): NDArray {
  return NDArray._fromStorage(complexOps.angle(x.storage, deg));
}

/**
 * Test element-wise for negative infinity
 * @param x - Input array
 * @returns Boolean array
 */
export function isneginf(x: NDArray): NDArray {
  return NDArray._fromStorage(logicOps.isneginf(x.storage));
}

/**
 * Test element-wise for positive infinity
 * @param x - Input array
 * @returns Boolean array
 */
export function isposinf(x: NDArray): NDArray {
  return NDArray._fromStorage(logicOps.isposinf(x.storage));
}

/**
 * Check if array is Fortran contiguous (column-major order)
 * @param x - Input array
 * @returns true if F-contiguous
 */
export function isfortran(x: NDArray): boolean {
  return logicOps.isfortran(x.storage);
}

/**
 * Returns array with complex parts close to zero set to real
 * Since numpy-ts doesn't support complex numbers, returns copy
 * @param x - Input array
 * @param tol - Tolerance
 * @returns Copy of input array
 */
export function real_if_close(x: NDArray, tol: number = 100): NDArray {
  return NDArray._fromStorage(logicOps.real_if_close(x.storage, tol));
}

/**
 * Check if element is a scalar type
 * @param val - Value to check
 * @returns true if scalar
 */
export function isscalar(val: unknown): boolean {
  return logicOps.isscalar(val);
}

/**
 * Check if object is iterable
 * @param obj - Object to check
 * @returns true if iterable
 */
export function iterable(obj: unknown): boolean {
  return logicOps.iterable(obj);
}

/**
 * Check if dtype meets specified criteria
 * @param dtype - Dtype to check
 * @param kind - Kind of dtype ('b' bool, 'i' int, 'u' uint, 'f' float)
 * @returns true if dtype matches kind
 */
export function isdtype(dtype: DType, kind: string): boolean {
  return logicOps.isdtype(dtype, kind);
}

/**
 * Find the dtype that can represent both input dtypes
 * @param dtype1 - First dtype
 * @param dtype2 - Second dtype
 * @returns Promoted dtype
 */
export function promote_types(dtype1: DType, dtype2: DType): DType {
  return logicOps.promote_types(dtype1, dtype2);
}

// ========================================
// Linear Algebra Functions (Additional)
// ========================================

/**
 * Einstein summation convention
 *
 * Performs tensor contractions and reductions using Einstein notation.
 *
 * @param subscripts - Einstein summation subscripts (e.g., 'ij,jk->ik')
 * @param operands - Input arrays
 * @returns Result of the Einstein summation
 *
 * @example
 * // Matrix multiplication
 * einsum('ij,jk->ik', a, b)
 *
 * @example
 * // Inner product
 * einsum('i,i->', a, b)
 *
 * @example
 * // Trace
 * einsum('ii->', a)
 */
export function einsum(
  subscripts: string,
  ...operands: NDArray[]
): NDArray | number | bigint | Complex {
  const storages = operands.map((op) => op.storage);
  const result = linalgOps.einsum(subscripts, ...storages);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Evaluate the lowest cost contraction order for an einsum expression.
 *
 * @param subscripts - Einsum subscripts (e.g., 'ij,jk,kl->il')
 * @param operands - Input arrays
 * @returns A tuple of [path, info_string]
 *
 * @example
 * const [path, info] = einsum_path('ij,jk,kl->il', a, b, c);
 */
export function einsum_path(
  subscripts: string,
  ...operands: (NDArray | number[])[]
): [Array<[number, number] | number[]>, string] {
  // Convert operands to storages or shapes
  const operandsOrShapes = operands.map((op) => {
    if (Array.isArray(op)) {
      // Plain array represents a shape
      return op;
    }
    return op.storage;
  });
  return linalgOps.einsum_path(subscripts, ...operandsOrShapes);
}

/**
 * Return the dot product of two vectors (flattened).
 *
 * Unlike dot(), vdot flattens both inputs before computing the dot product.
 * For complex numbers, vdot uses the complex conjugate of the first argument.
 *
 * @param a - First input array (will be flattened)
 * @param b - Second input array (will be flattened)
 * @returns Scalar dot product
 */
export function vdot(a: NDArray, b: NDArray): number | bigint | Complex {
  return linalgOps.vdot(a.storage, b.storage);
}

/**
 * Vector dot product along the last axis.
 *
 * Computes the dot product of vectors along the last axis of both inputs.
 * The last dimensions of a and b must match.
 *
 * @param a - First input array
 * @param b - Second input array
 * @param axis - Axis along which to compute (default: -1, meaning last axis)
 * @returns Result with last dimension removed
 */
export function vecdot(
  a: NDArray,
  b: NDArray,
  axis: number = -1
): NDArray | number | bigint | Complex {
  const result = linalgOps.vecdot(a.storage, b.storage, axis);
  if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
    return result;
  }
  return NDArray._fromStorage(result);
}

/**
 * Transpose the last two axes of an array.
 *
 * Equivalent to swapaxes(a, -2, -1) or transpose with axes that swap the last two.
 * For a 2D array, this is the same as transpose.
 *
 * @param a - Input array with at least 2 dimensions
 * @returns Array with last two axes transposed
 */
export function matrix_transpose(a: NDArray): NDArray {
  return NDArray._fromStorage(linalgOps.matrix_transpose(a.storage));
}

/**
 * Permute the dimensions of an array.
 *
 * This is an alias for transpose to match the Array API standard.
 *
 * @param a - Input array
 * @param axes - Permutation of axes. If not specified, reverses the axes.
 * @returns Transposed array
 */
export function permute_dims(a: NDArray, axes?: number[]): NDArray {
  return NDArray._fromStorage(linalgOps.permute_dims(a.storage, axes));
}

/**
 * Matrix-vector multiplication.
 *
 * Computes the matrix-vector product over the last two axes of x1 and
 * the last axis of x2.
 *
 * @param x1 - First input array (matrix) with shape (..., M, N)
 * @param x2 - Second input array (vector) with shape (..., N)
 * @returns Result with shape (..., M)
 */
export function matvec(x1: NDArray, x2: NDArray): NDArray {
  return NDArray._fromStorage(linalgOps.matvec(x1.storage, x2.storage));
}

/**
 * Vector-matrix multiplication.
 *
 * Computes the vector-matrix product over the last axis of x1 and
 * the second-to-last axis of x2.
 *
 * @param x1 - First input array (vector) with shape (..., M)
 * @param x2 - Second input array (matrix) with shape (..., M, N)
 * @returns Result with shape (..., N)
 */
export function vecmat(x1: NDArray, x2: NDArray): NDArray {
  return NDArray._fromStorage(linalgOps.vecmat(x1.storage, x2.storage));
}

// ============================================================================
// numpy.linalg Module
// ============================================================================

/**
 * numpy.linalg module - Linear algebra functions
 */
export const linalg = {
  /**
   * Cross product of two vectors.
   */
  cross: (
    a: NDArray,
    b: NDArray,
    axisa: number = -1,
    axisb: number = -1,
    axisc: number = -1,
    axis?: number
  ): NDArray | number | Complex => {
    const result = linalgOps.cross(a.storage, b.storage, axisa, axisb, axisc, axis);
    if (typeof result === 'number' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  },

  /**
   * Compute the norm of a vector or matrix.
   */
  norm: (
    x: NDArray,
    ord: number | 'fro' | 'nuc' | null = null,
    axis: number | [number, number] | null = null,
    keepdims: boolean = false
  ): NDArray | number => {
    const result = linalgOps.norm(x.storage, ord, axis, keepdims);
    if (typeof result === 'number') {
      return result;
    }
    return NDArray._fromStorage(result);
  },

  /**
   * Compute the vector norm.
   */
  vector_norm: (
    x: NDArray,
    ord: number = 2,
    axis?: number | null,
    keepdims: boolean = false
  ): NDArray | number => {
    const result = linalgOps.vector_norm(x.storage, ord, axis, keepdims);
    if (typeof result === 'number') {
      return result;
    }
    return NDArray._fromStorage(result);
  },

  /**
   * Compute the matrix norm.
   */
  matrix_norm: (
    x: NDArray,
    ord: number | 'fro' | 'nuc' = 'fro',
    keepdims: boolean = false
  ): NDArray | number => {
    const result = linalgOps.matrix_norm(x.storage, ord, keepdims);
    if (typeof result === 'number') {
      return result;
    }
    return NDArray._fromStorage(result);
  },

  /**
   * QR decomposition.
   */
  qr: (
    a: NDArray,
    mode: 'reduced' | 'complete' | 'r' | 'raw' = 'reduced'
  ): { q: NDArray; r: NDArray } | NDArray | { h: NDArray; tau: NDArray } => {
    const result = linalgOps.qr(a.storage, mode);
    if (result instanceof ArrayStorage) {
      // 'r' mode returns just R
      return NDArray._fromStorage(result);
    } else if ('q' in result && 'r' in result) {
      return {
        q: NDArray._fromStorage(result.q),
        r: NDArray._fromStorage(result.r),
      };
    } else {
      // 'raw' mode returns h and tau
      return {
        h: NDArray._fromStorage(result.h),
        tau: NDArray._fromStorage(result.tau),
      };
    }
  },

  /**
   * Cholesky decomposition.
   */
  cholesky: (a: NDArray, upper: boolean = false): NDArray => {
    return NDArray._fromStorage(linalgOps.cholesky(a.storage, upper));
  },

  /**
   * Singular Value Decomposition.
   */
  svd: (
    a: NDArray,
    full_matrices: boolean = true,
    compute_uv: boolean = true
  ): { u: NDArray; s: NDArray; vt: NDArray } | NDArray => {
    const result = linalgOps.svd(a.storage, full_matrices, compute_uv);
    if ('u' in result) {
      return {
        u: NDArray._fromStorage(result.u),
        s: NDArray._fromStorage(result.s),
        vt: NDArray._fromStorage(result.vt),
      };
    }
    return NDArray._fromStorage(result);
  },

  /**
   * Compute the determinant of a matrix.
   */
  det: (a: NDArray): number => {
    return linalgOps.det(a.storage);
  },

  /**
   * Compute the matrix inverse.
   */
  inv: (a: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.inv(a.storage));
  },

  /**
   * Solve a linear system.
   */
  solve: (a: NDArray, b: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.solve(a.storage, b.storage));
  },

  /**
   * Least-squares solution to a linear matrix equation.
   */
  lstsq: (
    a: NDArray,
    b: NDArray,
    rcond: number | null = null
  ): { x: NDArray; residuals: NDArray; rank: number; s: NDArray } => {
    const result = linalgOps.lstsq(a.storage, b.storage, rcond);
    return {
      x: NDArray._fromStorage(result.x),
      residuals: NDArray._fromStorage(result.residuals),
      rank: result.rank,
      s: NDArray._fromStorage(result.s),
    };
  },

  /**
   * Compute the condition number.
   */
  cond: (a: NDArray, p: number | 'fro' | 'nuc' = 2): number => {
    return linalgOps.cond(a.storage, p);
  },

  /**
   * Compute the matrix rank.
   */
  matrix_rank: (a: NDArray, tol?: number): number => {
    return linalgOps.matrix_rank(a.storage, tol);
  },

  /**
   * Raise a square matrix to an integer power.
   */
  matrix_power: (a: NDArray, n: number): NDArray => {
    return NDArray._fromStorage(linalgOps.matrix_power(a.storage, n));
  },

  /**
   * Compute the Moore-Penrose pseudo-inverse.
   */
  pinv: (a: NDArray, rcond: number = 1e-15): NDArray => {
    return NDArray._fromStorage(linalgOps.pinv(a.storage, rcond));
  },

  /**
   * Compute eigenvalues and eigenvectors.
   */
  eig: (a: NDArray): { w: NDArray; v: NDArray } => {
    const result = linalgOps.eig(a.storage);
    return {
      w: NDArray._fromStorage(result.w),
      v: NDArray._fromStorage(result.v),
    };
  },

  /**
   * Compute eigenvalues and eigenvectors of a Hermitian matrix.
   */
  eigh: (a: NDArray, UPLO: 'L' | 'U' = 'L'): { w: NDArray; v: NDArray } => {
    const result = linalgOps.eigh(a.storage, UPLO);
    return {
      w: NDArray._fromStorage(result.w),
      v: NDArray._fromStorage(result.v),
    };
  },

  /**
   * Compute eigenvalues of a matrix.
   */
  eigvals: (a: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.eigvals(a.storage));
  },

  /**
   * Compute eigenvalues of a Hermitian matrix.
   */
  eigvalsh: (a: NDArray, UPLO: 'L' | 'U' = 'L'): NDArray => {
    return NDArray._fromStorage(linalgOps.eigvalsh(a.storage, UPLO));
  },

  /**
   * Return specified diagonals.
   */
  diagonal: (a: NDArray, offset: number = 0, axis1: number = 0, axis2: number = 1): NDArray => {
    return NDArray._fromStorage(linalgOps.diagonal(a.storage, offset, axis1, axis2));
  },

  /**
   * Matrix multiplication.
   */
  matmul: (a: NDArray, b: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.matmul(a.storage, b.storage));
  },

  /**
   * Transpose the last two axes of an array.
   */
  matrix_transpose: (a: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.matrix_transpose(a.storage));
  },

  /**
   * Compute the dot product of two or more arrays.
   */
  multi_dot: (arrays: NDArray[]): NDArray => {
    const storages = arrays.map((arr) => arr.storage);
    return NDArray._fromStorage(linalgOps.multi_dot(storages));
  },

  /**
   * Outer product of two vectors.
   */
  outer: (a: NDArray, b: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.outer(a.storage, b.storage));
  },

  /**
   * Compute sign and (natural) logarithm of the determinant.
   */
  slogdet: (a: NDArray): { sign: number; logabsdet: number } => {
    return linalgOps.slogdet(a.storage);
  },

  /**
   * Compute singular values of a matrix.
   */
  svdvals: (a: NDArray): NDArray => {
    return NDArray._fromStorage(linalgOps.svdvals(a.storage));
  },

  /**
   * Tensor dot product along specified axes.
   */
  tensordot: (
    a: NDArray,
    b: NDArray,
    axes: number | [number[], number[]] = 2
  ): NDArray | number | bigint | Complex => {
    const result = linalgOps.tensordot(a.storage, b.storage, axes);
    if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  },

  /**
   * Compute the tensor inverse.
   */
  tensorinv: (a: NDArray, ind: number = 2): NDArray => {
    return NDArray._fromStorage(linalgOps.tensorinv(a.storage, ind));
  },

  /**
   * Solve the tensor equation a x = b for x.
   */
  tensorsolve: (a: NDArray, b: NDArray, axes?: number[] | null): NDArray => {
    return NDArray._fromStorage(linalgOps.tensorsolve(a.storage, b.storage, axes));
  },

  /**
   * Sum along diagonals.
   */
  trace: (a: NDArray): number | bigint | Complex => {
    return linalgOps.trace(a.storage);
  },

  /**
   * Vector dot product.
   */
  vecdot: (a: NDArray, b: NDArray, axis: number = -1): NDArray | number | bigint | Complex => {
    const result = linalgOps.vecdot(a.storage, b.storage, axis);
    if (typeof result === 'number' || typeof result === 'bigint' || result instanceof Complex) {
      return result;
    }
    return NDArray._fromStorage(result);
  },
};

// ============================================================================
// Indexing Functions
// ============================================================================

/**
 * Take values from the input array by matching 1d index and data slices along axis.
 *
 * @param arr - Input array
 * @param indices - Index array with same ndim as arr
 * @param axis - The axis along which to select values
 * @returns Array of values taken along the axis
 */
export function take_along_axis(arr: NDArray, indices: NDArray, axis: number): NDArray {
  return NDArray._fromStorage(advancedOps.take_along_axis(arr.storage, indices.storage, axis));
}

/**
 * Put values into the destination array using 1d index and data slices along axis.
 *
 * @param arr - Destination array (modified in-place)
 * @param indices - Index array with same ndim as arr
 * @param values - Values to put
 * @param axis - The axis along which to put values
 */
export function put_along_axis(
  arr: NDArray,
  indices: NDArray,
  values: NDArray,
  axis: number
): void {
  advancedOps.put_along_axis(arr.storage, indices.storage, values.storage, axis);
}

/**
 * Change elements of array based on conditional mask.
 *
 * @param a - Array to modify (in-place)
 * @param mask - Boolean mask array
 * @param values - Values to put where mask is True
 */
export function putmask(a: NDArray, mask: NDArray, values: NDArray | number | bigint): void {
  const valuesArg = values instanceof NDArray ? values.storage : values;
  advancedOps.putmask(a.storage, mask.storage, valuesArg);
}

/**
 * Return selected slices of array along given axis.
 *
 * @param condition - Boolean array for selecting
 * @param a - Array from which to select
 * @param axis - Axis along which to select (if undefined, works on flattened array)
 * @returns Compressed array
 */
export function compress(condition: NDArray, a: NDArray, axis?: number): NDArray {
  return NDArray._fromStorage(advancedOps.compress(condition.storage, a.storage, axis));
}

/**
 * Return an array drawn from elements in choicelist, depending on conditions.
 *
 * @param condlist - List of boolean arrays (conditions)
 * @param choicelist - List of arrays to choose from
 * @param defaultVal - Default value when no condition is met (default 0)
 * @returns Array with selected values
 */
export function select(
  condlist: NDArray[],
  choicelist: NDArray[],
  defaultVal: number | bigint = 0
): NDArray {
  const condStorages = condlist.map((c) => c.storage);
  const choiceStorages = choicelist.map((c) => c.storage);
  return NDArray._fromStorage(advancedOps.select(condStorages, choiceStorages, defaultVal));
}

/**
 * Change elements of an array based on conditional and input values.
 *
 * @param arr - Array to modify (in-place)
 * @param mask - Boolean mask array
 * @param vals - Values to place where mask is True (cycles if shorter)
 */
export function place(arr: NDArray, mask: NDArray, vals: NDArray): void {
  advancedOps.place(arr.storage, mask.storage, vals.storage);
}

/**
 * Fill the main diagonal of a given array (modifies in-place)
 * @param a - Array (at least 2D)
 * @param val - Value or array of values to fill diagonal with
 * @param wrap - Whether to wrap for tall matrices
 */
export function fill_diagonal(a: NDArray, val: NDArray | number, wrap: boolean = false): void {
  const valStorage = typeof val === 'number' ? val : val.storage;
  advancedOps.fill_diagonal(a.storage, valStorage, wrap);
}

/**
 * Return the indices to access the main diagonal of an array.
 *
 * @param n - Size of arrays for which indices are returned
 * @param ndim - Number of dimensions (default 2)
 * @returns Tuple of index arrays
 */
export function diag_indices(n: number, ndim: number = 2): NDArray[] {
  const storages = advancedOps.diag_indices(n, ndim);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return the indices to access the main diagonal of an n-dimensional array.
 *
 * @param arr - Input array (must have all equal dimensions)
 * @returns Tuple of index arrays
 */
export function diag_indices_from(arr: NDArray): NDArray[] {
  const storages = advancedOps.diag_indices_from(arr.storage);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return the indices for the lower-triangle of an (n, m) array.
 *
 * @param n - Number of rows
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @param m - Number of columns (default n)
 * @returns Tuple of row and column index arrays
 */
export function tril_indices(n: number, k: number = 0, m?: number): NDArray[] {
  const storages = advancedOps.tril_indices(n, k, m);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return the indices for the lower-triangle of arr.
 *
 * @param arr - Input 2-D array
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @returns Tuple of row and column index arrays
 */
export function tril_indices_from(arr: NDArray, k: number = 0): NDArray[] {
  const storages = advancedOps.tril_indices_from(arr.storage, k);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return the indices for the upper-triangle of an (n, m) array.
 *
 * @param n - Number of rows
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @param m - Number of columns (default n)
 * @returns Tuple of row and column index arrays
 */
export function triu_indices(n: number, k: number = 0, m?: number): NDArray[] {
  const storages = advancedOps.triu_indices(n, k, m);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return the indices for the upper-triangle of arr.
 *
 * @param arr - Input 2-D array
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @returns Tuple of row and column index arrays
 */
export function triu_indices_from(arr: NDArray, k: number = 0): NDArray[] {
  const storages = advancedOps.triu_indices_from(arr.storage, k);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return the indices to access (n, n) arrays, given a masking function.
 *
 * @param n - The returned indices will be valid to access arrays of shape (n, n)
 * @param mask_func - A function that generates an (n, n) boolean mask
 * @param k - Optional diagonal offset passed to mask_func
 * @returns Tuple of row and column index arrays
 */
export function mask_indices(
  n: number,
  mask_func: (n: number, k: number) => NDArray,
  k: number = 0
): NDArray[] {
  // Wrap the function to work with storage
  const storageMaskFunc = (n: number, k: number) => mask_func(n, k).storage;
  const storages = advancedOps.mask_indices(n, storageMaskFunc, k);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Return an array representing the indices of a grid.
 *
 * @param dimensions - The shape of the grid
 * @param dtype - Data type of result (default 'int32')
 * @returns Array of shape (len(dimensions), *dimensions)
 */
export function indices(
  dimensions: number[],
  dtype: 'int32' | 'int64' | 'float64' = 'int32'
): NDArray {
  return NDArray._fromStorage(advancedOps.indices(dimensions, dtype));
}

/**
 * Construct an open mesh from multiple sequences.
 *
 * This function returns a list of arrays with shapes suitable for broadcasting.
 *
 * @param args - 1-D sequences
 * @returns Tuple of arrays for open mesh
 */
export function ix_(...args: NDArray[]): NDArray[] {
  const storages = advancedOps.ix_(...args.map((a) => a.storage));
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Convert a tuple of index arrays into an array of flat indices.
 *
 * @param multi_index - Tuple of index arrays
 * @param dims - Shape of array into which indices apply
 * @param mode - How to handle out-of-bounds indices ('raise', 'wrap', 'clip')
 * @returns Flattened indices
 */
export function ravel_multi_index(
  multi_index: NDArray[],
  dims: number[],
  mode: 'raise' | 'wrap' | 'clip' = 'raise'
): NDArray {
  const storages = multi_index.map((a) => a.storage);
  return NDArray._fromStorage(advancedOps.ravel_multi_index(storages, dims, mode));
}

/**
 * Convert a flat index or array of flat indices into a tuple of coordinate arrays.
 *
 * @param indices - Array of indices or single index
 * @param shape - Shape of the array to index into
 * @param order - Row-major ('C') or column-major ('F') order
 * @returns Tuple of coordinate arrays
 */
export function unravel_index(
  indices: NDArray | number,
  shape: number[],
  order: 'C' | 'F' = 'C'
): NDArray[] {
  const indicesArg = indices instanceof NDArray ? indices.storage : indices;
  const storages = advancedOps.unravel_index(indicesArg, shape, order);
  return storages.map((s) => NDArray._fromStorage(s));
}

// ========================================
// Sorting and Searching Functions
// ========================================

/**
 * Return a sorted copy of an array
 * @param a - Input array
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Sorted array
 */
export function sort(a: NDArray, axis: number = -1): NDArray {
  return NDArray._fromStorage(sortingOps.sort(a.storage, axis));
}

/**
 * Returns the indices that would sort an array
 * @param a - Input array
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Array of indices that sort the input array
 */
export function argsort(a: NDArray, axis: number = -1): NDArray {
  return NDArray._fromStorage(sortingOps.argsort(a.storage, axis));
}

/**
 * Perform an indirect stable sort using a sequence of keys
 * @param keys - Array of NDArrays, the last key is the primary sort key
 * @returns Array of indices that would sort the keys
 */
export function lexsort(keys: NDArray[]): NDArray {
  const storages = keys.map((k) => k.storage);
  return NDArray._fromStorage(sortingOps.lexsort(storages));
}

/**
 * Partially sort an array
 * @param a - Input array
 * @param kth - Element index to partition by
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Partitioned array
 */
export function partition(a: NDArray, kth: number, axis: number = -1): NDArray {
  return NDArray._fromStorage(sortingOps.partition(a.storage, kth, axis));
}

/**
 * Returns indices that would partition an array
 * @param a - Input array
 * @param kth - Element index to partition by
 * @param axis - Axis along which to sort. Default is -1 (last axis)
 * @returns Array of indices
 */
export function argpartition(a: NDArray, kth: number, axis: number = -1): NDArray {
  return NDArray._fromStorage(sortingOps.argpartition(a.storage, kth, axis));
}

/**
 * Sort a complex array using the real part first, then the imaginary part
 * For real arrays, returns a sorted 1D array
 * @param a - Input array
 * @returns Sorted 1D array
 */
export function sort_complex(a: NDArray): NDArray {
  return NDArray._fromStorage(sortingOps.sort_complex(a.storage));
}

/**
 * Return the indices of the elements that are non-zero
 * @param a - Input array
 * @returns Tuple of arrays, one for each dimension
 */
export function nonzero(a: NDArray): NDArray[] {
  const storages = sortingOps.nonzero(a.storage);
  return storages.map((s) => NDArray._fromStorage(s));
}

/**
 * Find the indices of array elements that are non-zero, grouped by element
 * Returns a 2D array where each row is the index of a non-zero element.
 * This is equivalent to transpose(nonzero(a)).
 * @param a - Input array
 * @returns 2D array of shape (N, ndim) where N is number of non-zero elements
 */
export function argwhere(a: NDArray): NDArray {
  return NDArray._fromStorage(sortingOps.argwhere(a.storage));
}

/**
 * Return indices of non-zero elements in flattened array
 * @param a - Input array
 * @returns Array of indices
 */
export function flatnonzero(a: NDArray): NDArray {
  return NDArray._fromStorage(sortingOps.flatnonzero(a.storage));
}

/**
 * Return elements from x or y depending on condition
 * If only condition is given, returns indices where condition is true (like nonzero)
 * @param condition - Boolean array or condition
 * @param x - Values where condition is true (optional)
 * @param y - Values where condition is false (optional)
 * @returns Array with elements chosen from x or y, or indices if only condition given
 */
export function where(condition: NDArray, x?: NDArray, y?: NDArray): NDArray | NDArray[] {
  const result = sortingOps.where(condition.storage, x?.storage, y?.storage);
  if (Array.isArray(result)) {
    return result.map((s) => NDArray._fromStorage(s));
  }
  return NDArray._fromStorage(result);
}

/**
 * Find indices where elements should be inserted to maintain order
 * @param a - Input array (must be sorted in ascending order)
 * @param v - Values to insert
 * @param side - 'left' or 'right' side to insert
 * @returns Indices where values should be inserted
 */
export function searchsorted(a: NDArray, v: NDArray, side: 'left' | 'right' = 'left'): NDArray {
  return NDArray._fromStorage(sortingOps.searchsorted(a.storage, v.storage, side));
}

/**
 * Return the elements of an array that satisfy some condition
 * @param condition - Boolean array
 * @param a - Input array
 * @returns 1D array of elements where condition is true
 */
export function extract(condition: NDArray, a: NDArray): NDArray {
  return NDArray._fromStorage(sortingOps.extract(condition.storage, a.storage));
}

/**
 * Count number of non-zero values in the array
 * @param a - Input array
 * @param axis - Axis along which to count (optional)
 * @returns Count of non-zero values
 */
export function count_nonzero(a: NDArray, axis?: number): NDArray | number {
  const result = sortingOps.count_nonzero(a.storage, axis);
  if (typeof result === 'number') {
    return result;
  }
  return NDArray._fromStorage(result);
}

// ============================================================================
// Rounding Functions
// ============================================================================

/**
 * Round an array to the given number of decimals
 * @param a - Input array
 * @param decimals - Number of decimal places to round to (default: 0)
 * @returns Rounded array
 */
export function around(a: NDArray, decimals: number = 0): NDArray {
  return NDArray._fromStorage(roundingOps.around(a.storage, decimals));
}

// Alias for around
export { around as round_ };

/**
 * Return the ceiling of the input, element-wise
 * @param x - Input array
 * @returns Element-wise ceiling
 */
export function ceil(x: NDArray): NDArray {
  return NDArray._fromStorage(roundingOps.ceil(x.storage));
}

/**
 * Round to nearest integer towards zero
 * @param x - Input array
 * @returns Array with values truncated towards zero
 */
export function fix(x: NDArray): NDArray {
  return NDArray._fromStorage(roundingOps.fix(x.storage));
}

/**
 * Return the floor of the input, element-wise
 * @param x - Input array
 * @returns Element-wise floor
 */
export function floor(x: NDArray): NDArray {
  return NDArray._fromStorage(roundingOps.floor(x.storage));
}

/**
 * Round elements of the array to the nearest integer
 * @param x - Input array
 * @returns Array with rounded integer values
 */
export function rint(x: NDArray): NDArray {
  return NDArray._fromStorage(roundingOps.rint(x.storage));
}

/**
 * Evenly round to the given number of decimals (alias for around)
 * @param a - Input array
 * @param decimals - Number of decimal places to round to (default: 0)
 * @returns Rounded array
 */
export { around as round };

/**
 * Return the truncated value of the input, element-wise
 * @param x - Input array
 * @returns Element-wise truncated values
 */
export function trunc(x: NDArray): NDArray {
  return NDArray._fromStorage(roundingOps.trunc(x.storage));
}

// ============================================================================
// Set Operations
// ============================================================================

/**
 * Find the unique elements of an array
 * @param ar - Input array
 * @param returnIndex - If True, also return the indices of the first occurrences
 * @param returnInverse - If True, also return the indices to reconstruct the original array
 * @param returnCounts - If True, also return the number of times each unique value appears
 * @returns Unique sorted values, and optionally indices/inverse/counts
 */
export function unique(
  ar: NDArray,
  returnIndex: boolean = false,
  returnInverse: boolean = false,
  returnCounts: boolean = false
): NDArray | { values: NDArray; indices?: NDArray; inverse?: NDArray; counts?: NDArray } {
  const result = setOps.unique(ar.storage, returnIndex, returnInverse, returnCounts);
  if (result instanceof ArrayStorage) {
    return NDArray._fromStorage(result);
  }
  const out: { values: NDArray; indices?: NDArray; inverse?: NDArray; counts?: NDArray } = {
    values: NDArray._fromStorage(result.values),
  };
  if (result.indices) {
    out.indices = NDArray._fromStorage(result.indices);
  }
  if (result.inverse) {
    out.inverse = NDArray._fromStorage(result.inverse);
  }
  if (result.counts) {
    out.counts = NDArray._fromStorage(result.counts);
  }
  return out;
}

/**
 * Test whether each element of a 1-D array is also present in a second array
 * @param ar1 - Input array
 * @param ar2 - Test values
 * @returns Boolean array indicating membership
 */
export function in1d(ar1: NDArray, ar2: NDArray): NDArray {
  return NDArray._fromStorage(setOps.in1d(ar1.storage, ar2.storage));
}

/**
 * Find the intersection of two arrays
 * @param ar1 - First input array
 * @param ar2 - Second input array
 * @returns Sorted 1D array of common and unique elements
 */
export function intersect1d(ar1: NDArray, ar2: NDArray): NDArray {
  return NDArray._fromStorage(setOps.intersect1d(ar1.storage, ar2.storage));
}

/**
 * Test whether each element of an ND array is also present in a second array
 * @param element - Input array
 * @param testElements - Test values
 * @returns Boolean array indicating membership (same shape as element)
 */
export function isin(element: NDArray, testElements: NDArray): NDArray {
  return NDArray._fromStorage(setOps.isin(element.storage, testElements.storage));
}

/**
 * Find the set difference of two arrays
 * @param ar1 - First input array
 * @param ar2 - Second input array
 * @returns Sorted 1D array of values in ar1 that are not in ar2
 */
export function setdiff1d(ar1: NDArray, ar2: NDArray): NDArray {
  return NDArray._fromStorage(setOps.setdiff1d(ar1.storage, ar2.storage));
}

/**
 * Find the set exclusive-or of two arrays
 * @param ar1 - First input array
 * @param ar2 - Second input array
 * @returns Sorted 1D array of values that are in only one array
 */
export function setxor1d(ar1: NDArray, ar2: NDArray): NDArray {
  return NDArray._fromStorage(setOps.setxor1d(ar1.storage, ar2.storage));
}

/**
 * Find the union of two arrays
 * @param ar1 - First input array
 * @param ar2 - Second input array
 * @returns Sorted 1D array of unique values from both arrays
 */
export function union1d(ar1: NDArray, ar2: NDArray): NDArray {
  return NDArray._fromStorage(setOps.union1d(ar1.storage, ar2.storage));
}

/**
 * Trim leading and/or trailing zeros from a 1-D array.
 *
 * @param filt - Input 1-D array
 * @param trim - 'fb' to trim front and back, 'f' for front only, 'b' for back only (default: 'fb')
 * @returns Trimmed array
 */
export function trim_zeros(filt: NDArray, trim: 'f' | 'b' | 'fb' = 'fb'): NDArray {
  return NDArray._fromStorage(setOps.trim_zeros(filt.storage, trim));
}

/**
 * Find the unique elements of an array, returning all optional outputs.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Object with values, indices, inverse_indices, and counts (all as NDArray)
 */
export function unique_all(x: NDArray): {
  values: NDArray;
  indices: NDArray;
  inverse_indices: NDArray;
  counts: NDArray;
} {
  const result = setOps.unique_all(x.storage);
  return {
    values: NDArray._fromStorage(result.values),
    indices: NDArray._fromStorage(result.indices),
    inverse_indices: NDArray._fromStorage(result.inverse_indices),
    counts: NDArray._fromStorage(result.counts),
  };
}

/**
 * Find the unique elements of an array and their counts.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Object with values and counts (both as NDArray)
 */
export function unique_counts(x: NDArray): {
  values: NDArray;
  counts: NDArray;
} {
  const result = setOps.unique_counts(x.storage);
  return {
    values: NDArray._fromStorage(result.values),
    counts: NDArray._fromStorage(result.counts),
  };
}

/**
 * Find the unique elements of an array and their inverse indices.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Object with values and inverse_indices (both as NDArray)
 */
export function unique_inverse(x: NDArray): {
  values: NDArray;
  inverse_indices: NDArray;
} {
  const result = setOps.unique_inverse(x.storage);
  return {
    values: NDArray._fromStorage(result.values),
    inverse_indices: NDArray._fromStorage(result.inverse_indices),
  };
}

/**
 * Find the unique elements of an array (values only).
 *
 * This is equivalent to unique(x) but with a clearer name for the Array API.
 *
 * @param x - Input array (flattened for uniqueness)
 * @returns Array of unique values, sorted
 */
export function unique_values(x: NDArray): NDArray {
  return NDArray._fromStorage(setOps.unique_values(x.storage));
}

// Gradient and difference functions

/**
 * Calculate the n-th discrete difference along the given axis
 * @param a - Input array
 * @param n - Number of times values are differenced (default: 1)
 * @param axis - Axis along which to compute difference (default: -1)
 * @returns Array of differences
 */
export function diff(a: NDArray, n: number = 1, axis: number = -1): NDArray {
  return NDArray._fromStorage(gradientOps.diff(a.storage, n, axis));
}

/**
 * The differences between consecutive elements of a flattened array
 * @param ary - Input array
 * @param to_end - Number(s) to append at the end
 * @param to_begin - Number(s) to prepend at the beginning
 * @returns Array of differences
 */
export function ediff1d(
  ary: NDArray,
  to_end: number[] | null = null,
  to_begin: number[] | null = null
): NDArray {
  return NDArray._fromStorage(gradientOps.ediff1d(ary.storage, to_end, to_begin));
}

/**
 * Return the gradient of an N-dimensional array
 * The gradient is computed using second order accurate central differences
 * in the interior and first order accurate one-sided differences at the boundaries.
 * @param f - Input array
 * @param varargs - Spacing between values (scalar or array per dimension)
 * @param axis - Axis or axes along which to compute gradient
 * @returns Array of gradients (one per axis) or single gradient
 */
export function gradient(
  f: NDArray,
  varargs: number | number[] = 1,
  axis: number | number[] | null = null
): NDArray | NDArray[] {
  const result = gradientOps.gradient(f.storage, varargs, axis);
  if (Array.isArray(result)) {
    return result.map((s) => NDArray._fromStorage(s));
  }
  return NDArray._fromStorage(result);
}

/**
 * Return the cross product of two (arrays of) vectors
 * @param a - First input array
 * @param b - Second input array
 * @param axisa - Axis of a that defines the vector(s) (default: -1)
 * @param axisb - Axis of b that defines the vector(s) (default: -1)
 * @param axisc - Axis of c containing the cross product (default: -1)
 * @returns Cross product array
 */
export function cross(
  a: NDArray,
  b: NDArray,
  axisa: number = -1,
  axisb: number = -1,
  axisc: number = -1
): NDArray {
  return NDArray._fromStorage(gradientOps.cross(a.storage, b.storage, axisa, axisb, axisc));
}

// ============================================================================
// Statistics functions
// ============================================================================

/**
 * Count number of occurrences of each value in array of non-negative ints.
 *
 * @param x - Input array (must contain non-negative integers)
 * @param weights - Optional weights, same shape as x
 * @param minlength - Minimum number of bins for output (default: 0)
 * @returns Array of bin counts
 */
export function bincount(x: NDArray, weights?: NDArray, minlength: number = 0): NDArray {
  return NDArray._fromStorage(statisticsOps.bincount(x.storage, weights?.storage, minlength));
}

/**
 * Return the indices of the bins to which each value in input array belongs.
 *
 * @param x - Input array to be binned
 * @param bins - Array of bins (monotonically increasing or decreasing)
 * @param right - If true, intervals are closed on the right (default: false)
 * @returns Array of bin indices
 */
export function digitize(x: NDArray, bins: NDArray, right: boolean = false): NDArray {
  return NDArray._fromStorage(statisticsOps.digitize(x.storage, bins.storage, right));
}

/**
 * Compute the histogram of a set of data.
 *
 * @param a - Input data (flattened if not 1D)
 * @param bins - Number of bins (default: 10) or array of bin edges
 * @param range - Lower and upper range of bins
 * @param density - If true, return probability density function (default: false)
 * @param weights - Optional weights for each data point
 * @returns Tuple of [hist, bin_edges]
 */
export function histogram(
  a: NDArray,
  bins: number | NDArray = 10,
  range?: [number, number],
  density: boolean = false,
  weights?: NDArray
): [NDArray, NDArray] {
  const result = statisticsOps.histogram(
    a.storage,
    typeof bins === 'number' ? bins : bins.storage,
    range,
    density,
    weights?.storage
  );
  return [NDArray._fromStorage(result.hist), NDArray._fromStorage(result.bin_edges)];
}

/**
 * Compute the bi-dimensional histogram of two data samples.
 *
 * @param x - Array of x coordinates
 * @param y - Array of y coordinates (must have same length as x)
 * @param bins - Number of bins or [nx, ny] or [x_edges, y_edges]
 * @param range - [[xmin, xmax], [ymin, ymax]]
 * @param density - If true, return probability density function
 * @param weights - Optional weights for each data point
 * @returns Tuple of [hist, x_edges, y_edges]
 */
export function histogram2d(
  x: NDArray,
  y: NDArray,
  bins: number | [number, number] | [NDArray, NDArray] = 10,
  range?: [[number, number], [number, number]],
  density: boolean = false,
  weights?: NDArray
): [NDArray, NDArray, NDArray] {
  let binsArg: number | [number, number] | [ArrayStorage, ArrayStorage];
  if (typeof bins === 'number') {
    binsArg = bins;
  } else if (Array.isArray(bins) && bins.length === 2) {
    if (typeof bins[0] === 'number') {
      binsArg = bins as [number, number];
    } else {
      binsArg = [(bins[0] as NDArray).storage, (bins[1] as NDArray).storage];
    }
  } else {
    binsArg = 10;
  }

  const result = statisticsOps.histogram2d(
    x.storage,
    y.storage,
    binsArg,
    range,
    density,
    weights?.storage
  );
  return [
    NDArray._fromStorage(result.hist),
    NDArray._fromStorage(result.x_edges),
    NDArray._fromStorage(result.y_edges),
  ];
}

/**
 * Compute the multidimensional histogram of some data.
 *
 * @param sample - Array of shape (N, D) where N is number of samples and D is number of dimensions
 * @param bins - Number of bins for all axes, or array of bin counts per axis
 * @param range - Array of [min, max] for each dimension
 * @param density - If true, return probability density function
 * @param weights - Optional weights for each sample
 * @returns Tuple of [hist, edges (array of edge arrays)]
 */
export function histogramdd(
  sample: NDArray,
  bins: number | number[] = 10,
  range?: [number, number][],
  density: boolean = false,
  weights?: NDArray
): [NDArray, NDArray[]] {
  const result = statisticsOps.histogramdd(sample.storage, bins, range, density, weights?.storage);
  return [NDArray._fromStorage(result.hist), result.edges.map((e) => NDArray._fromStorage(e))];
}

/**
 * Cross-correlation of two 1-dimensional sequences.
 *
 * @param a - First input sequence
 * @param v - Second input sequence
 * @param mode - 'full', 'same', or 'valid' (default: 'full')
 * @returns Cross-correlation of a and v
 */
export function correlate(
  a: NDArray,
  v: NDArray,
  mode: 'full' | 'same' | 'valid' = 'full'
): NDArray {
  return NDArray._fromStorage(statisticsOps.correlate(a.storage, v.storage, mode));
}

/**
 * Discrete, linear convolution of two one-dimensional sequences.
 *
 * @param a - First input sequence
 * @param v - Second input sequence
 * @param mode - 'full', 'same', or 'valid' (default: 'full')
 * @returns Convolution of a and v
 */
export function convolve(
  a: NDArray,
  v: NDArray,
  mode: 'full' | 'same' | 'valid' = 'full'
): NDArray {
  return NDArray._fromStorage(statisticsOps.convolve(a.storage, v.storage, mode));
}

/**
 * Estimate a covariance matrix.
 *
 * @param m - Input array (1D or 2D). Each row represents a variable, columns are observations.
 * @param y - Optional second array (for 2 variable case)
 * @param rowvar - If true, each row is a variable (default: true)
 * @param bias - If true, use N for normalization; if false, use N-1 (default: false)
 * @param ddof - Delta degrees of freedom (overrides bias if provided)
 * @returns Covariance matrix
 */
export function cov(
  m: NDArray,
  y?: NDArray,
  rowvar: boolean = true,
  bias: boolean = false,
  ddof?: number
): NDArray {
  return NDArray._fromStorage(statisticsOps.cov(m.storage, y?.storage, rowvar, bias, ddof));
}

/**
 * Return Pearson product-moment correlation coefficients.
 *
 * @param x - Input array (1D or 2D)
 * @param y - Optional second array (for 2 variable case)
 * @param rowvar - If true, each row is a variable (default: true)
 * @returns Correlation coefficient matrix
 */
export function corrcoef(x: NDArray, y?: NDArray, rowvar: boolean = true): NDArray {
  return NDArray._fromStorage(statisticsOps.corrcoef(x.storage, y?.storage, rowvar));
}

/**
 * Compute the edges of the bins for histogram.
 *
 * This function computes the bin edges without computing the histogram itself.
 *
 * @param a - Input data (flattened if not 1D)
 * @param bins - Number of bins (default: 10) or a string specifying the bin algorithm
 * @param range - Lower and upper range of bins. If not provided, uses [a.min(), a.max()]
 * @param weights - Optional weights for each data point (used for some algorithms)
 * @returns Array of bin edges (length = bins + 1)
 */
export function histogram_bin_edges(
  a: NDArray,
  bins: number | 'auto' | 'fd' | 'doane' | 'scott' | 'stone' | 'rice' | 'sturges' | 'sqrt' = 10,
  range?: [number, number],
  weights?: NDArray
): NDArray {
  return NDArray._fromStorage(
    statisticsOps.histogram_bin_edges(a.storage, bins, range, weights?.storage)
  );
}

/**
 * Integrate along the given axis using the composite trapezoidal rule.
 *
 * @param y - Input array to integrate
 * @param x - Optional sample points corresponding to y values. If not provided, spacing is assumed to be 1.
 * @param dx - Spacing between sample points when x is not given (default: 1.0)
 * @param axis - The axis along which to integrate (default: -1, meaning last axis)
 * @returns Definite integral approximated using the composite trapezoidal rule
 */
export function trapezoid(
  y: NDArray,
  x?: NDArray,
  dx: number = 1.0,
  axis: number = -1
): NDArray | number {
  const result = statisticsOps.trapezoid(y.storage, x?.storage, dx, axis);
  if (typeof result === 'number') {
    return result;
  }
  return NDArray._fromStorage(result);
}

// ========================================
// Utility Functions
// ========================================

/**
 * Apply a function along a given axis.
 *
 * @param func1d - Function that takes a 1D array and returns a scalar or array
 * @param axis - Axis along which to apply the function
 * @param arr - Input array
 * @returns Result array
 */
export function apply_along_axis(
  func1d: (arr: NDArray) => NDArray | number,
  axis: number,
  arr: NDArray
): NDArray {
  // Wrapper to convert NDArray function to ArrayStorage function
  const storageFunc = (storage: ArrayStorage): ArrayStorage | number => {
    const result = func1d(NDArray._fromStorage(storage));
    if (result instanceof NDArray) {
      return result.storage;
    }
    return result;
  };
  return NDArray._fromStorage(advancedOps.apply_along_axis(arr.storage, axis, storageFunc));
}

/**
 * Apply a function over multiple axes.
 *
 * @param func - Function that operates on an array along an axis
 * @param a - Input array
 * @param axes - Axes over which to apply the function
 * @returns Result array
 */
export function apply_over_axes(
  func: (a: NDArray, axis: number) => NDArray,
  a: NDArray,
  axes: number[]
): NDArray {
  // Wrapper to convert NDArray function to ArrayStorage function
  const storageFunc = (storage: ArrayStorage, axis: number): ArrayStorage => {
    const result = func(NDArray._fromStorage(storage), axis);
    return result.storage;
  };
  return NDArray._fromStorage(advancedOps.apply_over_axes(a.storage, storageFunc, axes));
}

/**
 * Check if two arrays may share memory.
 *
 * @param a - First array
 * @param b - Second array
 * @returns True if arrays may share memory
 */
export function may_share_memory(a: NDArray, b: NDArray): boolean {
  return advancedOps.may_share_memory(a.storage, b.storage);
}

/**
 * Check if two arrays share memory.
 *
 * @param a - First array
 * @param b - Second array
 * @returns True if arrays share memory
 */
export function shares_memory(a: NDArray, b: NDArray): boolean {
  return advancedOps.shares_memory(a.storage, b.storage);
}

/**
 * Return the number of dimensions of an array.
 *
 * @param a - Input array
 * @returns Number of dimensions
 */
export function ndim(a: NDArray | number | unknown[] | unknown): number {
  if (typeof a === 'number') {
    return 0;
  }
  if (a instanceof NDArray) {
    return a.ndim;
  }
  // Handle nested arrays
  let dim = 0;
  let current = a;
  while (Array.isArray(current)) {
    dim++;
    current = current[0];
  }
  return dim;
}

/**
 * Return the shape of an array.
 *
 * @param a - Input array
 * @returns Shape tuple
 */
export function shape(a: NDArray | number | unknown[] | unknown): number[] {
  if (typeof a === 'number') {
    return [];
  }
  if (a instanceof NDArray) {
    return Array.from(a.shape);
  }
  // Handle nested arrays
  const result: number[] = [];
  let current = a;
  while (Array.isArray(current)) {
    result.push(current.length);
    current = current[0];
  }
  return result;
}

/**
 * Return the number of elements in an array.
 *
 * @param a - Input array
 * @returns Number of elements
 */
export function size(a: NDArray | number | unknown[] | unknown): number {
  if (typeof a === 'number') {
    return 1;
  }
  if (a instanceof NDArray) {
    return a.size;
  }
  // Handle nested arrays
  const s = shape(a);
  return s.reduce((acc, dim) => acc * dim, 1);
}

// Re-export error handling functions
export { geterr, seterr, type FloatErrorState } from '../common/ops/advanced';

// ========================================
// Type Checking Functions
// ========================================

/**
 * Casting rules for safe type conversion
 * In 'safe' mode: casting is allowed only if the value can be represented without loss
 * In 'same_kind': same kind of types allowed (e.g., float to float, int to int)
 * In 'unsafe': any conversion is allowed
 */
type CastingRule = 'no' | 'equiv' | 'safe' | 'same_kind' | 'unsafe';

// Type hierarchy for safe casting
const TYPE_HIERARCHY: Record<DType, number> = {
  bool: 0,
  uint8: 1,
  int8: 2,
  uint16: 3,
  int16: 4,
  uint32: 5,
  int32: 6,
  uint64: 7,
  int64: 8,
  float32: 9,
  float64: 10,
  complex64: 11,
  complex128: 12,
};

// Type kinds
const TYPE_KINDS: Record<DType, string> = {
  bool: 'b',
  uint8: 'u',
  uint16: 'u',
  uint32: 'u',
  uint64: 'u',
  int8: 'i',
  int16: 'i',
  int32: 'i',
  int64: 'i',
  float32: 'f',
  float64: 'f',
  complex64: 'c',
  complex128: 'c',
};

/**
 * Returns true if cast between data types can occur according to the casting rule.
 *
 * @param from_dtype - Data type to cast from
 * @param to_dtype - Data type to cast to
 * @param casting - Casting rule: 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'
 * @returns Whether the cast is allowed
 *
 * @example
 * ```typescript
 * can_cast('int32', 'float64')  // true - safe upcast
 * can_cast('float64', 'int32')  // false - would lose precision
 * can_cast('float64', 'int32', 'unsafe')  // true - forced cast
 * ```
 */
export function can_cast(
  from_dtype: DType | NDArray,
  to_dtype: DType,
  casting: CastingRule = 'safe'
): boolean {
  // Handle NDArray input
  const fromDtype: DType = from_dtype instanceof NDArray ? (from_dtype.dtype as DType) : from_dtype;

  // Exact same type - always allowed
  if (fromDtype === to_dtype) return true;

  switch (casting) {
    case 'no':
      // No casting allowed
      return false;

    case 'equiv':
      // Only equivalent types (same dtype or same kind and size)
      return fromDtype === to_dtype;

    case 'safe':
      // Safe casting: no loss of precision
      // Bool can go to anything
      if (fromDtype === 'bool') return true;

      // Complex can only go to larger complex
      if (isComplexDType(fromDtype)) {
        if (!isComplexDType(to_dtype)) return false;
        return TYPE_HIERARCHY[fromDtype] <= TYPE_HIERARCHY[to_dtype];
      }

      // Can always cast to complex
      if (isComplexDType(to_dtype)) return true;

      // Integer to float is safe if float is large enough
      if (TYPE_KINDS[fromDtype] === 'i' || TYPE_KINDS[fromDtype] === 'u') {
        if (TYPE_KINDS[to_dtype] === 'f') {
          // int8, int16, uint8, uint16 can safely go to float32
          if (to_dtype === 'float32') {
            return ['int8', 'int16', 'uint8', 'uint16'].includes(fromDtype);
          }
          // All integers can go to float64
          return to_dtype === 'float64';
        }
      }

      // Float can only go to larger float
      if (TYPE_KINDS[fromDtype] === 'f' && TYPE_KINDS[to_dtype] === 'f') {
        return TYPE_HIERARCHY[fromDtype] <= TYPE_HIERARCHY[to_dtype];
      }

      // Integer to integer
      if (
        (TYPE_KINDS[fromDtype] === 'i' || TYPE_KINDS[fromDtype] === 'u') &&
        (TYPE_KINDS[to_dtype] === 'i' || TYPE_KINDS[to_dtype] === 'u')
      ) {
        // Can only go to same or larger size
        return TYPE_HIERARCHY[fromDtype] <= TYPE_HIERARCHY[to_dtype];
      }

      return false;

    case 'same_kind': {
      // Same kind casting (int to int, float to float, etc.)
      const fromKind = TYPE_KINDS[fromDtype];
      const toKind = TYPE_KINDS[to_dtype];

      // Bool is compatible with integers
      if (fromDtype === 'bool' && (toKind === 'i' || toKind === 'u' || toKind === 'b')) {
        return true;
      }

      // Same kind or sub-kinds
      if (fromKind === toKind) return true;

      // Integers are same kind (signed/unsigned)
      if ((fromKind === 'i' || fromKind === 'u') && (toKind === 'i' || toKind === 'u')) {
        return true;
      }

      // Float to complex is same kind
      if (fromKind === 'f' && toKind === 'c') return true;

      return false;
    }

    case 'unsafe':
      // Any casting is allowed
      return true;

    default:
      return false;
  }
}

/**
 * Return a scalar type which is common to the input arrays.
 *
 * The return type will always be an inexact (floating point) type, even if
 * all the input arrays are integer types.
 *
 * @param arrays - Input arrays
 * @returns Common scalar type ('float32' or 'float64', or complex variants)
 *
 * @example
 * ```typescript
 * const a = array([1, 2], { dtype: 'int32' });
 * const b = array([3.0, 4.0], { dtype: 'float32' });
 * common_type(a, b)  // 'float32'
 * ```
 */
export function common_type(...arrays: NDArray[]): DType {
  if (arrays.length === 0) {
    return 'float64';
  }

  let hasComplex = false;
  let maxFloatSize = 32; // Start with float32

  for (const arr of arrays) {
    const dtype = arr.dtype as DType;

    if (isComplexDType(dtype)) {
      hasComplex = true;
      if (dtype === 'complex128') {
        maxFloatSize = 64;
      }
    } else if (dtype === 'float64') {
      maxFloatSize = 64;
    } else if (dtype === 'int64' || dtype === 'uint64') {
      // 64-bit integers need float64 for precision
      maxFloatSize = 64;
    }
  }

  if (hasComplex) {
    return maxFloatSize === 64 ? 'complex128' : 'complex64';
  }
  return maxFloatSize === 64 ? 'float64' : 'float32';
}

/**
 * Returns the type that results from applying the NumPy type promotion rules
 * to the arguments.
 *
 * @param arrays_and_dtypes - A mix of arrays or dtype strings
 * @returns The result dtype
 *
 * @example
 * ```typescript
 * result_type('int32', 'float32')  // 'float64'
 * result_type(array([1, 2], { dtype: 'int8' }), 'float32')  // 'float32'
 * ```
 */
export function result_type(...arrays_and_dtypes: (NDArray | DType)[]): DType {
  if (arrays_and_dtypes.length === 0) {
    return 'float64';
  }

  // Extract dtypes
  const dtypes: DType[] = arrays_and_dtypes.map((item) => {
    if (item instanceof NDArray) {
      return item.dtype as DType;
    }
    return item;
  });

  // Use promoteDTypes from dtype module for proper promotion
  let result = dtypes[0]!;
  for (let i = 1; i < dtypes.length; i++) {
    result = promoteDTypesInternal(result, dtypes[i]!);
  }

  return result;
}

// Internal helper that mirrors the dtype.ts promoteDTypes logic
function promoteDTypesInternal(dtype1: DType, dtype2: DType): DType {
  if (dtype1 === dtype2) return dtype1;

  // Boolean - promote to the other type
  if (dtype1 === 'bool') return dtype2;
  if (dtype2 === 'bool') return dtype1;

  // Complex types
  if (isComplexDType(dtype1) || isComplexDType(dtype2)) {
    if (isComplexDType(dtype1) && isComplexDType(dtype2)) {
      return dtype1 === 'complex128' || dtype2 === 'complex128' ? 'complex128' : 'complex64';
    }
    const complexDtype = isComplexDType(dtype1) ? dtype1 : dtype2;
    const realDtype = isComplexDType(dtype1) ? dtype2 : dtype1;

    if (complexDtype === 'complex128') return 'complex128';
    if (realDtype === 'float64' || realDtype === 'int64' || realDtype === 'uint64') {
      return 'complex128';
    }
    return 'complex64';
  }

  // Float types
  const isFloat1 = TYPE_KINDS[dtype1] === 'f';
  const isFloat2 = TYPE_KINDS[dtype2] === 'f';

  if (isFloat1 || isFloat2) {
    if (dtype1 === 'float64' || dtype2 === 'float64') return 'float64';

    // float32 with large integers promotes to float64
    if (dtype1 === 'float32' || dtype2 === 'float32') {
      const intDtype = isFloat1 ? dtype2 : dtype1;
      if (['int32', 'int64', 'uint32', 'uint64'].includes(intDtype)) {
        return 'float64';
      }
      return 'float32';
    }
  }

  // Integer promotion
  const kind1 = TYPE_KINDS[dtype1];
  const kind2 = TYPE_KINDS[dtype2];

  // Mixing signed and unsigned requires promoting to a type that can hold all values
  if ((kind1 === 'i' && kind2 === 'u') || (kind1 === 'u' && kind2 === 'i')) {
    const unsignedDtype = kind1 === 'u' ? dtype1 : dtype2;

    // Map unsigned to next larger signed type to hold all values
    // uint8 + signed -> int16 (can hold 0-255 and negatives)
    // uint16 + signed -> int32
    // uint32 + signed -> int64
    // uint64 + signed -> float64 (no int128)
    if (unsignedDtype === 'uint64') return 'float64';
    if (unsignedDtype === 'uint32') return 'int64';
    if (unsignedDtype === 'uint16') return 'int32';
    if (unsignedDtype === 'uint8') return 'int16';
  }

  // Otherwise, return higher hierarchy
  return TYPE_HIERARCHY[dtype1] >= TYPE_HIERARCHY[dtype2] ? dtype1 : dtype2;
}

/**
 * For scalar val, returns the data type with the smallest size and smallest
 * scalar kind which can hold its value.
 *
 * @param val - Scalar value
 * @returns Minimum dtype to represent the value
 *
 * @example
 * ```typescript
 * min_scalar_type(10)   // 'uint8'
 * min_scalar_type(1000) // 'uint16'
 * min_scalar_type(-1)   // 'int8'
 * min_scalar_type(3.14) // 'float64'
 * ```
 */
export function min_scalar_type(val: number | bigint | boolean): DType {
  if (typeof val === 'boolean') {
    return 'bool';
  }

  if (typeof val === 'bigint') {
    if (val >= 0n) {
      if (val <= 255n) return 'uint8';
      if (val <= 65535n) return 'uint16';
      if (val <= 4294967295n) return 'uint32';
      return 'uint64';
    } else {
      if (val >= -128n && val <= 127n) return 'int8';
      if (val >= -32768n && val <= 32767n) return 'int16';
      if (val >= -2147483648n && val <= 2147483647n) return 'int32';
      return 'int64';
    }
  }

  // Number type
  if (!Number.isFinite(val) || !Number.isInteger(val)) {
    // Floats always use float64
    return 'float64';
  }

  if (val >= 0) {
    if (val <= 255) return 'uint8';
    if (val <= 65535) return 'uint16';
    if (val <= 4294967295) return 'uint32';
    return 'float64'; // Too large for uint64 as number
  } else {
    if (val >= -128 && val <= 127) return 'int8';
    if (val >= -32768 && val <= 32767) return 'int16';
    if (val >= -2147483648 && val <= 2147483647) return 'int32';
    return 'float64'; // Too large for int64 as number
  }
}

/**
 * Returns true if first argument is a typecode lower/equal in type hierarchy.
 *
 * In NumPy, issubdtype checks if dtype1 is a subtype of dtype2.
 * For specific types (not categories), types are only subtypes of themselves.
 *
 * @param dtype1 - First dtype to check
 * @param dtype2 - Second dtype (or type category string)
 * @returns Whether dtype1 is a subtype of dtype2
 *
 * @example
 * ```typescript
 * issubdtype('int32', 'int32')      // true - same type
 * issubdtype('int32', 'int64')      // false - different types
 * issubdtype('int32', 'integer')    // true - int32 is an integer type
 * issubdtype('float32', 'floating') // true - float32 is a floating type
 * ```
 */
export function issubdtype(dtype1: DType | NDArray, dtype2: DType | string): boolean {
  // Handle NDArray input
  const dt1: DType = dtype1 instanceof NDArray ? (dtype1.dtype as DType) : dtype1;

  // Handle category strings
  if (dtype2 === 'integer') {
    return ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'].includes(dt1);
  }
  if (dtype2 === 'signedinteger') {
    return ['int8', 'int16', 'int32', 'int64'].includes(dt1);
  }
  if (dtype2 === 'unsignedinteger') {
    return ['uint8', 'uint16', 'uint32', 'uint64'].includes(dt1);
  }
  if (dtype2 === 'floating') {
    return ['float32', 'float64'].includes(dt1);
  }
  if (dtype2 === 'complexfloating') {
    return ['complex64', 'complex128'].includes(dt1);
  }
  if (dtype2 === 'number' || dtype2 === 'numeric') {
    return !['bool'].includes(dt1);
  }
  if (dtype2 === 'inexact') {
    return ['float32', 'float64', 'complex64', 'complex128'].includes(dt1);
  }

  // For specific dtypes, only exact matches are subtypes
  return dt1 === dtype2;
}

/**
 * Return a description for the given data type code.
 *
 * @param dtype - Data type to describe
 * @returns Human-readable description
 *
 * @example
 * ```typescript
 * typename('float64')  // 'float64'
 * typename('int32')    // 'int32'
 * ```
 */
export function typename(dtype: DType): string {
  const descriptions: Record<DType, string> = {
    bool: 'bool',
    uint8: 'uint8',
    uint16: 'uint16',
    uint32: 'uint32',
    uint64: 'uint64',
    int8: 'int8',
    int16: 'int16',
    int32: 'int32',
    int64: 'int64',
    float32: 'float32',
    float64: 'float64',
    complex64: 'complex64',
    complex128: 'complex128',
  };

  return descriptions[dtype] || dtype;
}

/**
 * Return the character for the minimum-size type to which given types can be
 * safely cast.
 *
 * @param typechars - String of type characters
 * @param typeset - Which set of types to consider: 'GDFgdf' (default)
 * @param default_dtype - Default type if no valid types found
 * @returns Type character for minimum safe type
 *
 * @example
 * ```typescript
 * mintypecode('if')  // 'd' (float64)
 * mintypecode('ff')  // 'f' (float32)
 * ```
 */
export function mintypecode(
  typechars: string,
  typeset: string = 'GDFgdf',
  default_dtype: string = 'd'
): string {
  // Type character mappings (NumPy style)
  const charToType: Record<string, DType> = {
    b: 'int8',
    B: 'uint8',
    h: 'int16',
    H: 'uint16',
    i: 'int32',
    I: 'uint32',
    l: 'int64',
    L: 'uint64',
    f: 'float32',
    d: 'float64',
    F: 'complex64',
    D: 'complex128',
    g: 'float64', // long double maps to float64
    G: 'complex128', // long complex maps to complex128
    '?': 'bool',
  };

  const typeToChar: Record<DType, string> = {
    int8: 'b',
    uint8: 'B',
    int16: 'h',
    uint16: 'H',
    int32: 'i',
    uint32: 'I',
    int64: 'l',
    uint64: 'L',
    float32: 'f',
    float64: 'd',
    complex64: 'F',
    complex128: 'D',
    bool: '?',
  };

  // Convert input chars to dtypes
  const dtypes: DType[] = [];
  for (const char of typechars) {
    if (charToType[char]) {
      dtypes.push(charToType[char]);
    }
  }

  if (dtypes.length === 0) {
    return default_dtype;
  }

  // Find minimum type that can hold all
  let result = dtypes[0]!;
  for (let i = 1; i < dtypes.length; i++) {
    result = promoteDTypesInternal(result, dtypes[i]!);
  }

  // Filter by typeset if provided
  const allowedChars = new Set(typeset);
  const resultChar = typeToChar[result] || default_dtype;

  if (allowedChars.has(resultChar) || allowedChars.size === 0) {
    return resultChar;
  }

  // Find minimum allowed type
  const orderedChars = 'gfdGFD';
  for (const char of orderedChars) {
    if (allowedChars.has(char)) {
      return char;
    }
  }

  return default_dtype;
}

// ========================================
// Polynomial Functions
// ========================================

/**
 * Find the coefficients of a polynomial with the given sequence of roots.
 *
 * Returns the coefficients of the polynomial whose leading coefficient is one
 * for the given sequence of zeros (multiple roots must be included in the
 * sequence as many times as their multiplicity).
 *
 * @param seq_of_zeros - Sequence of polynomial roots
 * @returns Array of polynomial coefficients, highest power first
 *
 * @example
 * ```typescript
 * poly([1, 2, 3])  // array([1, -6, 11, -6]) = (x-1)(x-2)(x-3)
 * poly([0, 0, 0])  // array([1, 0, 0, 0]) = x^3
 * ```
 */
export function poly(seq_of_zeros: NDArray | number[]): NDArray {
  const roots = seq_of_zeros instanceof NDArray ? seq_of_zeros.toArray().flat() : seq_of_zeros;

  if (roots.length === 0) {
    return array([1]);
  }

  // Start with [1] (leading coefficient)
  let coeffs = [1];

  // Multiply by (x - root) for each root
  for (const root of roots) {
    const newCoeffs = new Array(coeffs.length + 1).fill(0);
    // Multiply current polynomial by (x - root)
    for (let i = 0; i < coeffs.length; i++) {
      newCoeffs[i] += coeffs[i]!; // x term
      newCoeffs[i + 1] -= coeffs[i]! * (root as number); // -root term
    }
    coeffs = newCoeffs;
  }

  return array(coeffs);
}

/**
 * Find the sum of two polynomials.
 *
 * @param a1 - First polynomial coefficients (highest power first)
 * @param a2 - Second polynomial coefficients (highest power first)
 * @returns Sum polynomial coefficients
 *
 * @example
 * ```typescript
 * polyadd([1, 2], [3, 4, 5])  // array([3, 5, 7])
 * ```
 */
export function polyadd(a1: NDArray | number[], a2: NDArray | number[]): NDArray {
  const c1 = a1 instanceof NDArray ? (a1.toArray().flat() as number[]) : a1;
  const c2 = a2 instanceof NDArray ? (a2.toArray().flat() as number[]) : a2;

  const maxLen = Math.max(c1.length, c2.length);
  const result = new Array(maxLen).fill(0);

  // Add c1 (aligned to the right/low powers)
  for (let i = 0; i < c1.length; i++) {
    result[maxLen - c1.length + i] += c1[i]!;
  }

  // Add c2 (aligned to the right/low powers)
  for (let i = 0; i < c2.length; i++) {
    result[maxLen - c2.length + i] += c2[i]!;
  }

  return array(result);
}

/**
 * Return the derivative of the specified order of a polynomial.
 *
 * @param p - Polynomial coefficients (highest power first)
 * @param m - Order of the derivative (default: 1)
 * @returns Derivative polynomial coefficients
 *
 * @example
 * ```typescript
 * polyder([3, 2, 1])  // array([6, 2]) - derivative of 3x^2 + 2x + 1
 * polyder([4, 3, 2, 1], 2)  // array([24, 6]) - second derivative
 * ```
 */
export function polyder(p: NDArray | number[], m: number = 1): NDArray {
  let coeffs = p instanceof NDArray ? (p.toArray().flat() as number[]) : [...p];

  for (let order = 0; order < m; order++) {
    if (coeffs.length <= 1) {
      return array([0]);
    }

    const newCoeffs = new Array(coeffs.length - 1);
    const degree = coeffs.length - 1;

    for (let i = 0; i < degree; i++) {
      newCoeffs[i] = coeffs[i]! * (degree - i);
    }

    coeffs = newCoeffs;
  }

  return array(coeffs);
}

/**
 * Returns the quotient and remainder of polynomial division.
 *
 * @param u - Dividend polynomial coefficients (highest power first)
 * @param v - Divisor polynomial coefficients (highest power first)
 * @returns Tuple [quotient, remainder]
 *
 * @example
 * ```typescript
 * polydiv([1, -3, 2], [1, -1])  // [array([1, -2]), array([0])]
 * // (x^2 - 3x + 2) / (x - 1) = (x - 2) with remainder 0
 * ```
 */
export function polydiv(u: NDArray | number[], v: NDArray | number[]): [NDArray, NDArray] {
  const dividend = u instanceof NDArray ? (u.toArray().flat() as number[]) : [...u];
  const divisor = v instanceof NDArray ? (v.toArray().flat() as number[]) : [...v];

  if (divisor.length === 0 || (divisor.length === 1 && divisor[0] === 0)) {
    throw new Error('Division by zero polynomial');
  }

  // Remove leading zeros
  while (dividend.length > 1 && dividend[0] === 0) dividend.shift();
  while (divisor.length > 1 && divisor[0] === 0) divisor.shift();

  if (dividend.length < divisor.length) {
    return [array([0]), array(dividend)];
  }

  const quotient: number[] = [];
  const remainder = [...dividend];

  while (remainder.length >= divisor.length) {
    const coeff = remainder[0]! / divisor[0]!;
    quotient.push(coeff);

    for (let i = 0; i < divisor.length; i++) {
      remainder[i] = remainder[i]! - coeff * divisor[i]!;
    }

    remainder.shift();
  }

  // Handle zero remainder
  if (remainder.length === 0 || remainder.every((x) => Math.abs(x) < 1e-14)) {
    return [array(quotient.length > 0 ? quotient : [0]), array([0])];
  }

  return [array(quotient.length > 0 ? quotient : [0]), array(remainder)];
}

/**
 * Least squares polynomial fit.
 *
 * Fit a polynomial p(x) = p[0] * x^deg + ... + p[deg] of degree deg to points (x, y).
 *
 * @param x - x-coordinates of the sample points
 * @param y - y-coordinates of the sample points
 * @param deg - Degree of the fitting polynomial
 * @returns Polynomial coefficients, highest power first
 *
 * @example
 * ```typescript
 * const x = array([0, 1, 2, 3, 4]);
 * const y = array([0, 1, 4, 9, 16]);
 * polyfit(x, y, 2)  // approximately array([1, 0, 0]) for y = x^2
 * ```
 */
export function polyfit(x: NDArray, y: NDArray, deg: number): NDArray {
  const xData = x.toArray().flat() as number[];
  const yData = y.toArray().flat() as number[];
  const n = xData.length;

  if (n !== yData.length) {
    throw new Error('x and y must have the same length');
  }
  if (deg < 0) {
    throw new Error('Degree must be non-negative');
  }
  if (n <= deg) {
    throw new Error('Need more data points than degree');
  }

  // Build Vandermonde matrix
  const vandermonde: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = deg; j >= 0; j--) {
      row.push(Math.pow(xData[i]!, j));
    }
    vandermonde.push(row);
  }

  // Solve using normal equations: (V^T V) c = V^T y
  const vt = transpose2D(vandermonde);
  const vtv = matmul2D(vt, vandermonde);
  const vty = matvec2D(vt, yData);

  // Solve the linear system using Gaussian elimination with partial pivoting
  const coeffs = solveLinear(vtv, vty);

  return array(coeffs);
}

// Helper functions for polyfit
function transpose2D(m: number[][]): number[][] {
  const rows = m.length;
  const cols = m[0]!.length;
  const result: number[][] = [];
  for (let j = 0; j < cols; j++) {
    const row: number[] = [];
    for (let i = 0; i < rows; i++) {
      row.push(m[i]![j]!);
    }
    result.push(row);
  }
  return result;
}

function matmul2D(a: number[][], b: number[][]): number[][] {
  const aRows = a.length;
  const aCols = a[0]!.length;
  const bCols = b[0]!.length;
  const result: number[][] = [];
  for (let i = 0; i < aRows; i++) {
    const row: number[] = [];
    for (let j = 0; j < bCols; j++) {
      let sum = 0;
      for (let k = 0; k < aCols; k++) {
        sum += a[i]![k]! * b[k]![j]!;
      }
      row.push(sum);
    }
    result.push(row);
  }
  return result;
}

function matvec2D(m: number[][], v: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < m.length; i++) {
    let sum = 0;
    for (let j = 0; j < m[i]!.length; j++) {
      sum += m[i]![j]! * v[j]!;
    }
    result.push(sum);
  }
  return result;
}

function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const augmented = A.map((row, i) => [...row, b[i]!]);

  // Gaussian elimination with partial pivoting
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(augmented[k]![i]!) > Math.abs(augmented[maxRow]![i]!)) {
        maxRow = k;
      }
    }

    // Swap rows
    [augmented[i], augmented[maxRow]] = [augmented[maxRow]!, augmented[i]!];

    // Eliminate
    for (let k = i + 1; k < n; k++) {
      const factor = augmented[k]![i]! / augmented[i]![i]!;
      for (let j = i; j <= n; j++) {
        augmented[k]![j] = augmented[k]![j]! - factor * augmented[i]![j]!;
      }
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = augmented[i]![n]!;
    for (let j = i + 1; j < n; j++) {
      x[i] -= augmented[i]![j]! * x[j]!;
    }
    x[i] /= augmented[i]![i]!;
  }

  return x;
}

/**
 * Return an antiderivative (indefinite integral) of a polynomial.
 *
 * @param p - Polynomial coefficients (highest power first)
 * @param m - Order of the antiderivative (default: 1)
 * @param k - Integration constants (default: 0)
 * @returns Antiderivative polynomial coefficients
 *
 * @example
 * ```typescript
 * polyint([3, 2, 1])  // array([1, 1, 1, 0]) - integral of 3x^2 + 2x + 1
 * polyint([6, 2], 1, 5)  // array([2, 1, 5]) - with constant 5
 * ```
 */
export function polyint(p: NDArray | number[], m: number = 1, k: number | number[] = 0): NDArray {
  let coeffs = p instanceof NDArray ? (p.toArray().flat() as number[]) : [...p];

  const constants = Array.isArray(k) ? k : [k];

  for (let order = 0; order < m; order++) {
    const newDegree = coeffs.length;
    const newCoeffs = new Array(newDegree + 1);

    for (let i = 0; i < newDegree; i++) {
      newCoeffs[i] = coeffs[i]! / (newDegree - i);
    }

    // Add integration constant
    newCoeffs[newDegree] = constants[order] !== undefined ? constants[order]! : 0;

    coeffs = newCoeffs;
  }

  return array(coeffs);
}

/**
 * Find the product of two polynomials.
 *
 * @param a1 - First polynomial coefficients (highest power first)
 * @param a2 - Second polynomial coefficients (highest power first)
 * @returns Product polynomial coefficients
 *
 * @example
 * ```typescript
 * polymul([1, 2], [1, 3])  // array([1, 5, 6]) = (x+2)(x+3)
 * ```
 */
export function polymul(a1: NDArray | number[], a2: NDArray | number[]): NDArray {
  const c1 = a1 instanceof NDArray ? (a1.toArray().flat() as number[]) : a1;
  const c2 = a2 instanceof NDArray ? (a2.toArray().flat() as number[]) : a2;

  const resultLen = c1.length + c2.length - 1;
  const result = new Array(resultLen).fill(0);

  for (let i = 0; i < c1.length; i++) {
    for (let j = 0; j < c2.length; j++) {
      result[i + j] += c1[i]! * c2[j]!;
    }
  }

  return array(result);
}

/**
 * Difference (subtraction) of two polynomials.
 *
 * @param a1 - First polynomial coefficients (highest power first)
 * @param a2 - Second polynomial coefficients (highest power first)
 * @returns Difference polynomial coefficients
 *
 * @example
 * ```typescript
 * polysub([1, 2, 3], [0, 1, 1])  // array([1, 1, 2])
 * ```
 */
export function polysub(a1: NDArray | number[], a2: NDArray | number[]): NDArray {
  const c1 = a1 instanceof NDArray ? (a1.toArray().flat() as number[]) : a1;
  const c2 = a2 instanceof NDArray ? (a2.toArray().flat() as number[]) : a2;

  const maxLen = Math.max(c1.length, c2.length);
  const result = new Array(maxLen).fill(0);

  // Add c1 (aligned to the right/low powers)
  for (let i = 0; i < c1.length; i++) {
    result[maxLen - c1.length + i] += c1[i]!;
  }

  // Subtract c2 (aligned to the right/low powers)
  for (let i = 0; i < c2.length; i++) {
    result[maxLen - c2.length + i] -= c2[i]!;
  }

  return array(result);
}

/**
 * Evaluate a polynomial at specific values.
 *
 * @param p - Polynomial coefficients (highest power first)
 * @param x - Values at which to evaluate the polynomial
 * @returns Polynomial values at x
 *
 * @example
 * ```typescript
 * polyval([1, -2, 1], 3)  // 4 = 3^2 - 2*3 + 1
 * polyval([1, 0, 0], array([1, 2, 3]))  // array([1, 4, 9])
 * ```
 */
export function polyval(p: NDArray | number[], x: NDArray | number | number[]): NDArray | number {
  const coeffs = p instanceof NDArray ? (p.toArray().flat() as number[]) : p;

  const evaluateAt = (val: number): number => {
    // Horner's method for polynomial evaluation
    let result = 0;
    for (const coeff of coeffs) {
      result = result * val + coeff;
    }
    return result;
  };

  if (typeof x === 'number') {
    return evaluateAt(x);
  }

  const xArray = x instanceof NDArray ? (x.toArray().flat() as number[]) : x;
  const results = xArray.map(evaluateAt);
  return array(results);
}

/**
 * Return the roots of a polynomial with coefficients given in p.
 *
 * The values in the rank-1 array p are coefficients of a polynomial.
 * If the length of p is n+1 then the polynomial is:
 * p[0]*x^n + p[1]*x^(n-1) + ... + p[n-1]*x + p[n]
 *
 * @param p - Polynomial coefficients (highest power first)
 * @returns Array of roots (may be complex, returned as numbers for real roots)
 *
 * @example
 * ```typescript
 * roots([1, -3, 2])  // array([2, 1]) - roots of x^2 - 3x + 2
 * roots([1, 0, -1])  // array([1, -1]) - roots of x^2 - 1
 * ```
 */
export function roots(p: NDArray | number[]): NDArray {
  const coeffs = p instanceof NDArray ? (p.toArray().flat() as number[]) : [...p];

  // Remove leading zeros
  while (coeffs.length > 1 && coeffs[0] === 0) {
    coeffs.shift();
  }

  const degree = coeffs.length - 1;

  if (degree <= 0) {
    return array([]);
  }

  // Normalize by leading coefficient
  const lead = coeffs[0]!;
  const normalized = coeffs.map((c) => c / lead);

  if (degree === 1) {
    // Linear: ax + b = 0 -> x = -b/a
    return array([-normalized[1]!]);
  }

  if (degree === 2) {
    // Quadratic formula
    const a = 1;
    const b = normalized[1]!;
    const c = normalized[2]!;
    const discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
      const sqrtD = Math.sqrt(discriminant);
      return array([(-b + sqrtD) / 2, (-b - sqrtD) / 2]);
    } else {
      // Complex roots - return real parts as approximation
      // (Full complex support would need complex array)
      const realPart = -b / 2;
      return array([realPart, realPart]);
    }
  }

  // For higher degrees, use companion matrix eigenvalue method
  // Build companion matrix
  const n = degree;
  const companion: number[][] = [];

  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    if (i < n - 1) {
      row[i + 1] = 1;
    }
    companion.push(row);
  }

  // Fill first row with -coefficients (normalized)
  for (let i = 0; i < n; i++) {
    companion[n - 1]![i] = -normalized[n - i]!;
  }

  // Find eigenvalues using QR algorithm (simplified power iteration for roots)
  const eigenvalues = qrAlgorithm(companion);

  return array(eigenvalues);
}

// Simplified QR algorithm for finding eigenvalues
function qrAlgorithm(A: number[][]): number[] {
  const n = A.length;
  let H = [...A.map((row) => [...row])];
  const maxIter = 100;
  const tol = 1e-10;

  // Simple QR iteration
  for (let iter = 0; iter < maxIter; iter++) {
    // QR decomposition using Gram-Schmidt
    const { Q, R } = qrDecompose(H);

    // H = R * Q
    H = matmul2D(R, Q);

    // Check for convergence (off-diagonal elements near zero)
    let maxOff = 0;
    for (let i = 1; i < n; i++) {
      maxOff = Math.max(maxOff, Math.abs(H[i]![i - 1]!));
    }
    if (maxOff < tol) break;
  }

  // Extract diagonal as eigenvalues
  const eigenvalues: number[] = [];
  for (let i = 0; i < n; i++) {
    eigenvalues.push(H[i]![i]!);
  }

  return eigenvalues;
}

function qrDecompose(A: number[][]): { Q: number[][]; R: number[][] } {
  const n = A.length;
  const Q: number[][] = A.map((row) => [...row]);
  const R: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let j = 0; j < n; j++) {
    // Orthogonalize column j against previous columns
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let k = 0; k < n; k++) {
        dot += Q[k]![i]! * A[k]![j]!;
      }
      R[i]![j] = dot;
      for (let k = 0; k < n; k++) {
        Q[k]![j] = Q[k]![j]! - dot * Q[k]![i]!;
      }
    }

    // Normalize
    let norm = 0;
    for (let k = 0; k < n; k++) {
      norm += Q[k]![j]! * Q[k]![j]!;
    }
    norm = Math.sqrt(norm);
    R[j]![j] = norm;

    if (norm > 1e-14) {
      for (let k = 0; k < n; k++) {
        Q[k]![j] = Q[k]![j]! / norm;
      }
    }
  }

  return { Q, R };
}

// ============================================================================
// Printing and Formatting Functions
// ============================================================================

/**
 * Set printing options for array output.
 */
export function set_printoptions(options: Partial<formattingOps.PrintOptions>): void {
  formattingOps.set_printoptions(options);
}

/**
 * Get current print options.
 */
export function get_printoptions(): formattingOps.PrintOptions {
  return formattingOps.get_printoptions();
}

/**
 * Context manager for temporarily setting print options.
 */
export function printoptions(
  options: Partial<formattingOps.PrintOptions>
): ReturnType<typeof formattingOps.printoptions> {
  return formattingOps.printoptions(options);
}

/**
 * Convert an array to a string representation.
 *
 * @param a - Input array
 * @param max_line_width - Maximum line width
 * @param precision - Number of digits of precision
 * @param suppress_small - Suppress small floating point values
 * @param separator - Separator between elements
 * @param prefix - Prefix string
 * @param suffix - Suffix string
 * @param threshold - Threshold for summarization
 * @param edgeitems - Number of edge items when summarizing
 * @returns String representation of the array
 */
export function array2string(
  a: NDArray,
  options?: {
    max_line_width?: number | null;
    precision?: number | null;
    suppress_small?: boolean | null;
    separator?: string;
    prefix?: string;
    suffix?: string;
    threshold?: number | null;
    edgeitems?: number | null;
  }
): string {
  const opts = options || {};
  return formattingOps.array2string(
    a.storage,
    opts.max_line_width ?? null,
    opts.precision ?? null,
    opts.suppress_small ?? null,
    opts.separator ?? ' ',
    opts.prefix ?? '',
    opts.suffix ?? '',
    opts.threshold ?? null,
    opts.edgeitems ?? null
  );
}

/**
 * Return the string representation of an array (includes 'array()' wrapper).
 */
export function array_repr(
  a: NDArray,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null
): string {
  return formattingOps.array_repr(a.storage, max_line_width, precision, suppress_small);
}

/**
 * Return a string representation of the data in an array.
 */
export function array_str(
  a: NDArray,
  max_line_width: number | null = null,
  precision: number | null = null,
  suppress_small: boolean | null = null
): string {
  return formattingOps.array_str(a.storage, max_line_width, precision, suppress_small);
}

/**
 * Return a string representation of a number in the given base.
 *
 * @param number - Number to convert
 * @param base - Base for representation (2-36, default: 2)
 * @param padding - Minimum number of digits
 * @returns String representation in the given base
 */
export function base_repr(number: number, base: number = 2, padding: number = 0): string {
  return formattingOps.base_repr(number, base, padding);
}

/**
 * Return the binary representation of the input number as a string.
 *
 * @param num - Integer to convert
 * @param width - Minimum width of result (uses two's complement for negatives)
 * @returns Binary string representation
 */
export function binary_repr(num: number, width: number | null = null): string {
  return formattingOps.binary_repr(num, width);
}

/**
 * Format a floating-point number in positional notation.
 */
export function format_float_positional(
  x: number,
  precision?: number | null,
  unique?: boolean,
  fractional?: boolean,
  trim?: 'k' | '.' | '0' | '-',
  sign?: '-' | '+' | ' ',
  pad_left?: number | null,
  pad_right?: number | null,
  min_digits?: number | null
): string {
  return formattingOps.format_float_positional(
    x,
    precision,
    unique,
    fractional,
    trim,
    sign,
    pad_left,
    pad_right,
    min_digits
  );
}

/**
 * Format a floating-point number in scientific notation.
 */
export function format_float_scientific(
  x: number,
  precision?: number | null,
  unique?: boolean,
  trim?: 'k' | '.' | '0' | '-',
  sign?: '-' | '+' | ' ',
  pad_left?: number | null,
  exp_digits?: number,
  min_digits?: number | null
): string {
  return formattingOps.format_float_scientific(
    x,
    precision,
    unique,
    trim,
    sign,
    pad_left,
    exp_digits,
    min_digits
  );
}
