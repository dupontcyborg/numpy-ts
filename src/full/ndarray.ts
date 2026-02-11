/**
 * NDArray - NumPy-compatible multidimensional array
 *
 * Core array class providing NumPy-like API
 */

import { parseSlice, normalizeSlice } from '../common/slicing';
import {
  type DType,
  type TypedArray,
  getTypedArrayConstructor,
  getDTypeSize,
  isBigIntDType,
  isComplexDType,
} from '../common/dtype';
import { Complex } from '../common/complex';
import { ArrayStorage } from '../common/storage';
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
import * as gradientOps from '../common/ops/gradient';

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
