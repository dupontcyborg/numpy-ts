/**
 * Declarative NDArray method definitions for code generation.
 *
 * Each entry describes a method on the NDArray class.
 * The generator uses these to produce src/full/ndarray.ts.
 *
 * Run with: npm run generate
 */

// ============================================================
// Types
// ============================================================

export type MethodPattern =
  | 'unary'          // fn(): NDArray → up(core.fn(this))
  | 'binary'         // fn(other): NDArray → up(core.fn(this, other))
  | 'reduction'      // fn(...): NDArray | scalar → r instanceof NDArrayCore ? up(r) : r
  | 'passthrough'    // fn(...): NDArray → up(core.fn(this, ...))
  | 'array_return'   // fn(): NDArray[] → core.fn(this).map(up)
  | 'tuple_return'   // fn(...): [NDArray, NDArray] → [up(r[0]), up(r[1])]
  | 'manual';        // Verbatim method body

export interface MethodDef {
  /** Method name on NDArray */
  name: string;
  /** Core function name if different (e.g., 'variance' for 'var') */
  coreName?: string;
  /** Which code template to use */
  pattern: MethodPattern;
  /** Parameter signature (excluding 'this') */
  params?: string;
  /** Override return type */
  returnType?: string;
  /** Extra args passed to core function after 'this' (for non-obvious mappings) */
  coreArgs?: string;
  /** For 'manual' pattern: verbatim method body */
  manualCode?: string;
  /** JSDoc comment (without the wrapping slashes) */
  doc?: string;
  /** Flags for special method types */
  flags?: {
    isGetter?: boolean;
    isOverride?: boolean;
    isStatic?: boolean;
    isGenerator?: boolean;
  };
}

// ============================================================
// Manual method bodies (embedded as strings)
// ============================================================

const CONSTRUCTOR = `\
constructor(storage: ArrayStorage, base?: NDArray) {
    super(storage, base);
    this._base = base;
  }`;

const FROM_STORAGE = `\
static override fromStorage(storage: ArrayStorage, base?: NDArray): NDArray {
    return new NDArray(storage, base);
  }`;

const BASE_GETTER = `\
override get base(): NDArray | null {
    return this._base ?? null;
  }`;

const T_GETTER = `\
get T(): NDArray {
    return this.transpose();
  }`;

const ITEMSIZE_GETTER = `\
override get itemsize(): number {
    return getDTypeSize(this._storage.dtype);
  }`;

const NBYTES_GETTER = `\
override get nbytes(): number {
    return this.size * this.itemsize;
  }`;

const FILL_METHOD = `\
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
  }`;

const ITERATOR_METHOD = `\
override *[Symbol.iterator](): Iterator<NDArray | number | bigint | Complex> {
    if (this.ndim === 0) {
      yield this._storage.iget(0);
    } else if (this.ndim === 1) {
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this._storage.iget(i);
      }
    } else {
      for (let i = 0; i < this.shape[0]!; i++) {
        yield this.slice(String(i));
      }
    }
  }`;

const GET_METHOD = `\
override get(indices: number[]): number | bigint | Complex {
    if (indices.length !== this.ndim) {
      throw new Error(
        \`Index has \${indices.length} dimensions, but array has \${this.ndim} dimensions\`
      );
    }

    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          \`Index \${idx} is out of bounds for axis \${dim} with size \${this.shape[dim]}\`
        );
      }
      return normalized;
    });

    return this._storage.get(...normalizedIndices);
  }`;

const SET_METHOD = `\
override set(
    indices: number[],
    value: number | bigint | Complex | { re: number; im: number }
  ): void {
    if (indices.length !== this.ndim) {
      throw new Error(
        \`Index has \${indices.length} dimensions, but array has \${this.ndim} dimensions\`
      );
    }

    const normalizedIndices = indices.map((idx, dim) => {
      let normalized = idx;
      if (normalized < 0) {
        normalized = this.shape[dim]! + normalized;
      }
      if (normalized < 0 || normalized >= this.shape[dim]!) {
        throw new Error(
          \`Index \${idx} is out of bounds for axis \${dim} with size \${this.shape[dim]}\`
        );
      }
      return normalized;
    });

    const currentDtype = this.dtype as DType;

    if (isComplexDType(currentDtype)) {
      this._storage.set(normalizedIndices, value);
    } else if (isBigIntDType(currentDtype)) {
      const numValue = value instanceof Complex ? value.re : Number(value);
      const convertedValue = typeof value === 'bigint' ? value : BigInt(Math.round(numValue));
      this._storage.set(normalizedIndices, convertedValue);
    } else if (currentDtype === 'bool') {
      const numValue = value instanceof Complex ? value.re : Number(value);
      const convertedValue = numValue ? 1 : 0;
      this._storage.set(normalizedIndices, convertedValue);
    } else {
      const convertedValue = value instanceof Complex ? value.re : Number(value);
      this._storage.set(normalizedIndices, convertedValue);
    }
  }`;

const COPY_METHOD = `\
override copy(): NDArray {
    return new NDArray(this._storage.copy());
  }`;

const ASTYPE_METHOD = `\
override astype(dtype: DType, copy: boolean = true): NDArray {
    const currentDtype = this.dtype as DType;

    if (currentDtype === dtype && !copy) {
      return this;
    }

    if (currentDtype === dtype && copy) {
      return this.copy();
    }

    const shape = Array.from(this.shape);
    const size = this.size;

    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) {
      throw new Error(\`Cannot convert to dtype \${dtype}\`);
    }
    const newData = new Constructor(size);
    const oldData = this.data;

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
    else if (!isBigIntDType(currentDtype) && isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = BigInt(
          Math.round(Number(typedOldData[i]))
        );
      }
    }
    else if (dtype === 'bool') {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Uint8Array)[i] = typedOldData[i] !== 0 ? 1 : 0;
      }
    }
    else if (currentDtype === 'bool' && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Uint8Array;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    }
    else if (!isBigIntDType(currentDtype) && !isBigIntDType(dtype)) {
      const typedOldData = oldData as Exclude<TypedArray, BigInt64Array | BigUint64Array>;
      for (let i = 0; i < size; i++) {
        (newData as Exclude<TypedArray, BigInt64Array | BigUint64Array>)[i] = typedOldData[i]!;
      }
    }
    else {
      const typedOldData = oldData as BigInt64Array | BigUint64Array;
      for (let i = 0; i < size; i++) {
        (newData as BigInt64Array | BigUint64Array)[i] = typedOldData[i]!;
      }
    }

    const storage = ArrayStorage.fromData(newData, shape, dtype);
    return new NDArray(storage);
  }`;

const SLICE_METHOD = `\
override slice(...sliceStrs: string[]): NDArray {
    if (sliceStrs.length === 0) {
      return this;
    }

    if (sliceStrs.length > this.ndim) {
      throw new Error(
        \`Too many indices for array: array is \${this.ndim}-dimensional, but \${sliceStrs.length} were indexed\`
      );
    }

    const sliceSpecs = sliceStrs.map((str, i) => {
      const spec = parseSlice(str);
      const normalized = normalizeSlice(spec, this.shape[i]!);
      return normalized;
    });

    while (sliceSpecs.length < this.ndim) {
      sliceSpecs.push({
        start: 0,
        stop: this.shape[sliceSpecs.length]!,
        step: 1,
        isIndex: false,
      });
    }

    const newShape: number[] = [];
    const newStrides: number[] = [];
    let newOffset = this._storage.offset;

    for (let i = 0; i < sliceSpecs.length; i++) {
      const spec = sliceSpecs[i]!;
      const stride = this._storage.strides[i]!;

      newOffset += spec.start * stride;

      if (!spec.isIndex) {
        let dimSize: number;
        if (spec.step > 0) {
          dimSize = Math.max(0, Math.ceil((spec.stop - spec.start) / spec.step));
        } else {
          dimSize = Math.max(0, Math.ceil((spec.start - spec.stop) / Math.abs(spec.step)));
        }
        newShape.push(dimSize);
        newStrides.push(stride * spec.step);
      }
    }

    const slicedStorage = ArrayStorage.fromData(
      this._storage.data,
      newShape,
      this._storage.dtype,
      newStrides,
      newOffset
    );

    const base = this._base ?? this;
    return new NDArray(slicedStorage, base);
  }`;

const RESHAPE_METHOD = `\
reshape(...shape: number[]): NDArray {
    const newShape = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    const resultStorage = core.reshape(this, newShape).storage;
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray.fromStorage(resultStorage, base);
  }`;

const RAVEL_METHOD = `\
ravel(): NDArray {
    const resultStorage = core.ravel(this).storage;
    const isView = resultStorage.data === this.data;
    const base = isView ? (this._base ?? this) : undefined;
    return NDArray.fromStorage(resultStorage, base);
  }`;

const ROW_METHOD = `\
row(i: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('row() requires at least 2 dimensions');
    }
    return this.slice(String(i), ':');
  }`;

const COL_METHOD = `\
col(j: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('col() requires at least 2 dimensions');
    }
    return this.slice(':', String(j));
  }`;

const ROWS_METHOD = `\
rows(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('rows() requires at least 2 dimensions');
    }
    return this.slice(\`\${start}:\${stop}\`, ':');
  }`;

const COLS_METHOD = `\
cols(start: number, stop: number): NDArray {
    if (this.ndim < 2) {
      throw new Error('cols() requires at least 2 dimensions');
    }
    return this.slice(':', \`\${start}:\${stop}\`);
  }`;

const TOSTRING_METHOD = `\
override toString(): string {
    return core.array_str(this);
  }`;

const TOARRAY_METHOD = `\
// eslint-disable-next-line @typescript-eslint/no-explicit-any
  override toArray(): any {
    if (this.ndim === 0) {
      return this._storage.iget(0);
    }

    const shape = this.shape;
    const ndim = shape.length;

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
  }`;

const TOLIST_METHOD = `\
// eslint-disable-next-line @typescript-eslint/no-explicit-any
  override tolist(): any {
    return this.toArray();
  }`;

const TOBYTES_METHOD = `\
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
  }`;

const ITEM_METHOD = `\
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
        throw new Error(\`index \${flatIdx} is out of bounds for size \${this.size}\`);
      }
      return this._storage.iget(flatIdx);
    }
    return this.get(args);
  }`;

const BYTESWAP_METHOD = `\
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
  }`;

const VIEW_METHOD = `\
view(dtype?: DType): NDArray {
    if (!dtype || dtype === this.dtype) {
      return NDArray.fromStorage(this._storage, this._base ?? this);
    }
    const oldSize = getDTypeSize(this.dtype as DType);
    const newSize = getDTypeSize(dtype);
    if (oldSize !== newSize) {
      throw new Error(
        'When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.'
      );
    }
    const Constructor = getTypedArrayConstructor(dtype);
    if (!Constructor) throw new Error(\`Unsupported dtype: \${dtype}\`);
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
    return NDArray.fromStorage(storage, this._base ?? this);
  }`;

const IINDEX_METHOD = `\
iindex(indices: NDArray | number[] | number[][], axis: number = 0): NDArray {
    let indexArray: number[];
    if (indices instanceof NDArray) {
      indexArray = [];
      for (let i = 0; i < indices.size; i++) {
        const val = indices.storage.iget(i);
        const numVal =
          typeof val === 'bigint' ? Number(val) : val instanceof Complex ? val.re : val;
        indexArray.push(numVal);
      }
    } else if (Array.isArray(indices) && indices.length > 0 && Array.isArray(indices[0])) {
      indexArray = (indices as number[][]).flat();
    } else {
      indexArray = indices as number[];
    }

    return this.take(indexArray, axis);
  }`;

const BINDEX_METHOD = `\
bindex(mask: NDArray, axis?: number): NDArray {
    return up(core.compress(mask, this, axis));
  }`;

const PUT_METHOD = `\
put(indices: number[], values: NDArray | number | bigint): void {
    const valuesStorage = values instanceof NDArray ? values._storage : values;
    core.put(this, indices, valuesStorage as never);
  }`;

const COMPRESS_METHOD = `\
compress(condition: NDArray | boolean[], axis?: number): NDArray {
    const condStorage =
      condition instanceof NDArray
        ? condition
        : NDArray.fromStorage(ArrayStorage.fromData(
            new Uint8Array(condition.map((b) => (b ? 1 : 0))),
            [condition.length],
            'bool'
          ));
    return up(core.compress(condStorage, this, axis));
  }`;

const CHOOSE_METHOD = `\
choose(choices: NDArray[]): NDArray {
    return up(core.choose(this, choices));
  }`;

const CLIP_METHOD = `\
clip(a_min: number | NDArray | null, a_max: number | NDArray | null): NDArray {
    return up(core.clip(this, a_min, a_max));
  }`;

const ROUND_METHOD = `\
round(decimals: number = 0): NDArray {
    return this.around(decimals);
  }`;

const CONJUGATE_METHOD = `\
conjugate(): NDArray {
    return this.conj();
  }`;

const AROUND_METHOD = `\
around(decimals: number = 0): NDArray {
    return up(core.around(this, decimals));
  }`;

const ALLCLOSE_METHOD = `\
allclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): boolean {
    return core.allclose(this, other, rtol, atol);
  }`;

const ISCLOSE_METHOD = `\
isclose(other: NDArray | number, rtol: number = 1e-5, atol: number = 1e-8): NDArray {
    return up(core.isclose(this, other, rtol, atol));
  }`;

const AVERAGE_METHOD = `\
average(weights?: NDArray, axis?: number): NDArray | number | Complex {
    const r = core.average(this, axis, weights);
    return r instanceof NDArrayCore ? up(r) : r;
  }`;

const DOT_METHOD = `\
dot(other: NDArray): NDArray | number | bigint | Complex {
    const r = core.dot(this, other);
    return r instanceof NDArrayCore ? up(r) : r;
  }`;

const TRACE_METHOD = `\
trace(): number | bigint | Complex {
    return core.trace(this);
  }`;

const INNER_METHOD = `\
inner(other: NDArray): NDArray | number | bigint | Complex {
    const r = core.inner(this, other);
    return r instanceof NDArrayCore ? up(r) : r;
  }`;

const TENSORDOT_METHOD = `\
tensordot(
    other: NDArray,
    axes: number | [number[], number[]] = 2
  ): NDArray | number | bigint | Complex {
    const r = core.tensordot(this, other, axes);
    return r instanceof NDArrayCore ? up(r) : r;
  }`;

const DIVMOD_METHOD = `\
divmod(divisor: NDArray | number): [NDArray, NDArray] {
    const r = core.divmod(this, divisor);
    return [up(r[0]), up(r[1])] as [NDArray, NDArray];
  }`;

const SEARCHSORTED_METHOD = `\
searchsorted(v: NDArray, side: 'left' | 'right' = 'left'): NDArray {
    return up(core.searchsorted(this, v, side));
  }`;

const TOFILE_METHOD = `\
tofile(_file: string, _sep: string = '', _format: string = ''): void {
    throw new Error(
      'tofile() requires file system access. Use the node module: import { save } from "numpy-ts/node"'
    );
  }`;

// ============================================================
// Meshgrid standalone function (appended after class)
// ============================================================

export const MESHGRID_FUNCTION = `\
export function meshgrid(...args: (NDArray | { indexing?: 'xy' | 'ij' })[]): NDArray[] {
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

  const sizes = arrays.map((a) => a.size);

  if (indexing === 'xy' && arrays.length >= 2) {
    arrays = [arrays[1]!, arrays[0]!, ...arrays.slice(2)];
    [sizes[0], sizes[1]] = [sizes[1]!, sizes[0]!];
  }

  const outputShape = sizes;
  const ndim = outputShape.length;

  const results: NDArray[] = [];

  for (let i = 0; i < arrays.length; i++) {
    const inputArr = arrays[i]!;
    const inputSize = inputArr.size;

    const broadcastShape: number[] = new Array(ndim).fill(1);
    broadcastShape[i] = inputSize;

    const reshaped = inputArr.reshape(...broadcastShape);
    const resultStorage = core.broadcast_to(reshaped, outputShape);
    const result = NDArray.fromStorage(resultStorage.storage.copy());
    results.push(result);
  }

  if (indexing === 'xy' && results.length >= 2) {
    [results[0], results[1]] = [results[1]!, results[0]!];
  }

  return results;
}`;

// ============================================================
// Method definitions
// ============================================================

export const METHOD_DEFS: MethodDef[] = [
  // ============================================================
  // Manual methods (core infrastructure)
  // ============================================================
  {
    name: '_base',
    pattern: 'manual',
    doc: 'Override _base with NDArray type',
    manualCode: 'protected override _base?: NDArray;',
  },
  {
    name: 'constructor',
    pattern: 'manual',
    manualCode: CONSTRUCTOR,
  },
  {
    name: 'fromStorage',
    pattern: 'manual',
    doc: 'Create NDArray from storage (for ops modules)\n@internal',
    manualCode: FROM_STORAGE,
    flags: { isStatic: true, isOverride: true },
  },
  {
    name: 'base',
    pattern: 'manual',
    doc: 'Base array if this is a view, null if this array owns its data\nSimilar to NumPy\'s base attribute',
    manualCode: BASE_GETTER,
    flags: { isGetter: true, isOverride: true },
  },
  {
    name: 'T',
    pattern: 'manual',
    doc: 'Transpose of the array (shorthand for transpose())\nReturns a view with axes reversed',
    manualCode: T_GETTER,
    flags: { isGetter: true },
  },
  {
    name: 'itemsize',
    pattern: 'manual',
    doc: 'Size of one array element in bytes',
    manualCode: ITEMSIZE_GETTER,
    flags: { isGetter: true, isOverride: true },
  },
  {
    name: 'nbytes',
    pattern: 'manual',
    doc: 'Total bytes consumed by the elements of the array',
    manualCode: NBYTES_GETTER,
    flags: { isGetter: true, isOverride: true },
  },
  {
    name: 'fill',
    pattern: 'manual',
    doc: 'Fill the array with a scalar value (in-place)\n@param value - Value to fill with',
    manualCode: FILL_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'Symbol.iterator',
    pattern: 'manual',
    doc: 'Iterator protocol - iterate over the first axis\nFor 1D arrays, yields elements; for ND arrays, yields (N-1)D subarrays',
    manualCode: ITERATOR_METHOD,
    flags: { isOverride: true, isGenerator: true },
  },
  {
    name: 'get',
    pattern: 'manual',
    doc: 'Get a single element from the array\n@param indices - Array of indices, one per dimension\n@returns The element value',
    manualCode: GET_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'set',
    pattern: 'manual',
    doc: 'Set a single element in the array\n@param indices - Array of indices, one per dimension\n@param value - Value to set',
    manualCode: SET_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'copy',
    pattern: 'manual',
    doc: 'Return a deep copy of the array',
    manualCode: COPY_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'astype',
    pattern: 'manual',
    doc: 'Cast array to a different dtype\n@param dtype - Target dtype\n@param copy - If false and dtype matches, return self\n@returns Array with specified dtype',
    manualCode: ASTYPE_METHOD,
    flags: { isOverride: true },
  },

  // ============================================================
  // Manual methods (slicing / indexing)
  // ============================================================
  {
    name: 'slice',
    pattern: 'manual',
    doc: 'Slice the array using NumPy-style string syntax\n@param sliceStrs - Slice specifications, one per dimension\n@returns Sliced view of the array',
    manualCode: SLICE_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'row',
    pattern: 'manual',
    doc: 'Get a single row (convenience method)\n@param i - Row index\n@returns Row as 1D or (n-1)D array',
    manualCode: ROW_METHOD,
  },
  {
    name: 'col',
    pattern: 'manual',
    doc: 'Get a single column (convenience method)\n@param j - Column index\n@returns Column as 1D or (n-1)D array',
    manualCode: COL_METHOD,
  },
  {
    name: 'rows',
    pattern: 'manual',
    doc: 'Get a range of rows (convenience method)\n@param start - Start row index\n@param stop - Stop row index (exclusive)\n@returns Rows as array',
    manualCode: ROWS_METHOD,
  },
  {
    name: 'cols',
    pattern: 'manual',
    doc: 'Get a range of columns (convenience method)\n@param start - Start column index\n@param stop - Stop column index (exclusive)\n@returns Columns as array',
    manualCode: COLS_METHOD,
  },

  // ============================================================
  // Manual methods (view-aware: reshape, ravel)
  // ============================================================
  {
    name: 'reshape',
    pattern: 'manual',
    doc: 'Reshape array to a new shape\nReturns a new array with the specified shape\n@param shape - New shape (must be compatible with current size)\n@returns Reshaped array',
    manualCode: RESHAPE_METHOD,
  },
  {
    name: 'ravel',
    pattern: 'manual',
    doc: 'Return a flattened array (view when possible, otherwise copy)\n@returns 1D array containing all elements',
    manualCode: RAVEL_METHOD,
  },

  // ============================================================
  // Manual methods (custom param handling)
  // ============================================================
  {
    name: 'put',
    pattern: 'manual',
    doc: 'Put values at specified indices (modifies array in-place)\n@param indices - Indices at which to place values\n@param values - Values to put',
    manualCode: PUT_METHOD,
  },
  {
    name: 'compress',
    pattern: 'manual',
    doc: 'Return selected slices of this array along given axis\n@param condition - Boolean array that selects which entries to return\n@param axis - Axis along which to take slices\n@returns Array with selected entries',
    manualCode: COMPRESS_METHOD,
  },
  {
    name: 'choose',
    pattern: 'manual',
    doc: 'Construct an array from an index array and a list of arrays to choose from\n@param choices - Array of NDArrays to choose from\n@returns New array with selected elements',
    manualCode: CHOOSE_METHOD,
  },
  {
    name: 'clip',
    pattern: 'manual',
    doc: 'Clip (limit) the values in an array\n@param a_min - Minimum value (null for no minimum)\n@param a_max - Maximum value (null for no maximum)\n@returns Array with values clipped to [a_min, a_max]',
    manualCode: CLIP_METHOD,
  },
  {
    name: 'iindex',
    pattern: 'manual',
    doc: 'Integer array indexing (fancy indexing)\n\nSelect elements using an array of indices.\n@param indices - Array of integer indices\n@param axis - Axis along which to index (default: 0)\n@returns New array with selected elements',
    manualCode: IINDEX_METHOD,
  },
  {
    name: 'bindex',
    pattern: 'manual',
    doc: 'Boolean array indexing (fancy indexing with mask)\n\nSelect elements where a boolean mask is true.\n@param mask - Boolean NDArray mask\n@param axis - Axis along which to apply the mask\n@returns New 1D array with selected elements',
    manualCode: BINDEX_METHOD,
  },

  // ============================================================
  // Manual methods (serialization)
  // ============================================================
  {
    name: 'toString',
    pattern: 'manual',
    doc: 'String representation of the array\n@returns String describing the array shape and dtype',
    manualCode: TOSTRING_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'toArray',
    pattern: 'manual',
    doc: 'Convert to nested JavaScript array\n@returns Nested JavaScript array representation',
    manualCode: TOARRAY_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'tolist',
    pattern: 'manual',
    doc: 'Return the array as a nested list (same as toArray)',
    manualCode: TOLIST_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'tobytes',
    pattern: 'manual',
    doc: 'Return the raw bytes of the array data',
    manualCode: TOBYTES_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'item',
    pattern: 'manual',
    doc: 'Copy an element of an array to a standard scalar and return it',
    manualCode: ITEM_METHOD,
    flags: { isOverride: true },
  },
  {
    name: 'byteswap',
    pattern: 'manual',
    doc: 'Swap the bytes of the array elements',
    manualCode: BYTESWAP_METHOD,
  },
  {
    name: 'view',
    pattern: 'manual',
    doc: 'Return a view of the array with a different dtype',
    manualCode: VIEW_METHOD,
  },
  {
    name: 'tofile',
    pattern: 'manual',
    doc: 'Write array to a file (stub)',
    manualCode: TOFILE_METHOD,
  },

  // ============================================================
  // Manual methods (aliases and special returns)
  // ============================================================
  {
    name: 'round',
    pattern: 'manual',
    doc: 'Round an array to the given number of decimals (alias for around)\n@param decimals - Number of decimal places to round to (default: 0)\n@returns New array with rounded values',
    manualCode: ROUND_METHOD,
  },
  {
    name: 'conjugate',
    pattern: 'manual',
    doc: 'Return the complex conjugate, element-wise (alias for conj)\n@returns Complex conjugate of the array',
    manualCode: CONJUGATE_METHOD,
  },
  {
    name: 'around',
    pattern: 'manual',
    doc: 'Round an array to the given number of decimals\n@param decimals - Number of decimal places to round to (default: 0)\n@returns New array with rounded values',
    manualCode: AROUND_METHOD,
  },
  {
    name: 'allclose',
    pattern: 'manual',
    doc: 'Element-wise comparison with tolerance\nReturns True where |a - b| <= (atol + rtol * |b|)\n@param other - Value or array to compare with\n@param rtol - Relative tolerance (default: 1e-5)\n@param atol - Absolute tolerance (default: 1e-8)\n@returns boolean',
    manualCode: ALLCLOSE_METHOD,
  },
  {
    name: 'isclose',
    pattern: 'manual',
    doc: 'Element-wise comparison with tolerance\nReturns True where |a - b| <= (atol + rtol * |b|)\n@param other - Value or array to compare with\n@param rtol - Relative tolerance (default: 1e-5)\n@param atol - Absolute tolerance (default: 1e-8)\n@returns Boolean array',
    manualCode: ISCLOSE_METHOD,
  },
  {
    name: 'average',
    pattern: 'manual',
    doc: 'Compute the weighted average along the specified axis\n@param weights - Array of weights (optional)\n@param axis - Axis along which to compute average\n@returns Weighted average of array elements',
    manualCode: AVERAGE_METHOD,
  },
  {
    name: 'dot',
    pattern: 'manual',
    doc: 'Dot product (matching NumPy behavior)\n@param other - Array to dot with\n@returns Result of dot product',
    manualCode: DOT_METHOD,
  },
  {
    name: 'trace',
    pattern: 'manual',
    doc: 'Sum of diagonal elements (trace)\n@returns Sum of diagonal elements',
    manualCode: TRACE_METHOD,
  },
  {
    name: 'inner',
    pattern: 'manual',
    doc: 'Inner product (contracts over last axes of both arrays)\n@param other - Array to compute inner product with\n@returns Inner product result',
    manualCode: INNER_METHOD,
  },
  {
    name: 'tensordot',
    pattern: 'manual',
    doc: 'Tensor dot product along specified axes\n@param other - Array to contract with\n@param axes - Axes to contract\n@returns Tensor dot product result',
    manualCode: TENSORDOT_METHOD,
  },
  {
    name: 'divmod',
    pattern: 'manual',
    doc: 'Returns both quotient and remainder (floor divide and modulo)\n@param divisor - Array or scalar divisor\n@returns Tuple of [quotient, remainder] arrays',
    manualCode: DIVMOD_METHOD,
  },
  {
    name: 'searchsorted',
    pattern: 'manual',
    doc: 'Find indices where elements should be inserted to maintain order\n@param v - Values to insert\n@param side - "left" or "right" side to insert\n@returns Indices where values should be inserted',
    manualCode: SEARCHSORTED_METHOD,
  },

  // ============================================================
  // Unary operations (self → NDArray)
  // ============================================================
  { name: 'sqrt', pattern: 'unary', doc: 'Square root of each element\nPromotes integer types to float64' },
  { name: 'exp', pattern: 'unary', doc: 'Natural exponential (e^x) of each element\nPromotes integer types to float64' },
  { name: 'exp2', pattern: 'unary', doc: 'Base-2 exponential (2^x) of each element\nPromotes integer types to float64' },
  { name: 'expm1', pattern: 'unary', doc: 'Exponential minus one (e^x - 1) of each element\nMore accurate than exp(x) - 1 for small x' },
  { name: 'log', pattern: 'unary', doc: 'Natural logarithm (ln) of each element\nPromotes integer types to float64' },
  { name: 'log2', pattern: 'unary', doc: 'Base-2 logarithm of each element\nPromotes integer types to float64' },
  { name: 'log10', pattern: 'unary', doc: 'Base-10 logarithm of each element\nPromotes integer types to float64' },
  { name: 'log1p', pattern: 'unary', doc: 'Natural logarithm of (1 + x) of each element\nMore accurate than log(1 + x) for small x' },
  { name: 'absolute', pattern: 'unary', doc: 'Absolute value of each element' },
  { name: 'negative', pattern: 'unary', doc: 'Numerical negative (element-wise negation)' },
  { name: 'sign', pattern: 'unary', doc: 'Sign of each element (-1, 0, or 1)' },
  { name: 'positive', pattern: 'unary', doc: 'Numerical positive (element-wise +x)\n@returns Copy of the array' },
  { name: 'reciprocal', pattern: 'unary', doc: 'Element-wise reciprocal (1/x)' },
  { name: 'ceil', pattern: 'unary', doc: 'Return the ceiling of the input, element-wise' },
  { name: 'fix', pattern: 'unary', doc: 'Round to nearest integer towards zero' },
  { name: 'floor', pattern: 'unary', doc: 'Return the floor of the input, element-wise' },
  { name: 'rint', pattern: 'unary', doc: 'Round elements to the nearest integer' },
  { name: 'trunc', pattern: 'unary', doc: 'Return the truncated value of the input, element-wise' },
  { name: 'sin', pattern: 'unary', doc: 'Sine of each element (in radians)\nPromotes integer types to float64' },
  { name: 'cos', pattern: 'unary', doc: 'Cosine of each element (in radians)\nPromotes integer types to float64' },
  { name: 'tan', pattern: 'unary', doc: 'Tangent of each element (in radians)\nPromotes integer types to float64' },
  { name: 'arcsin', pattern: 'unary', doc: 'Inverse sine of each element\nPromotes integer types to float64' },
  { name: 'arccos', pattern: 'unary', doc: 'Inverse cosine of each element\nPromotes integer types to float64' },
  { name: 'arctan', pattern: 'unary', doc: 'Inverse tangent of each element\nPromotes integer types to float64' },
  { name: 'degrees', pattern: 'unary', doc: 'Convert angles from radians to degrees' },
  { name: 'radians', pattern: 'unary', doc: 'Convert angles from degrees to radians' },
  { name: 'sinh', pattern: 'unary', doc: 'Hyperbolic sine of each element\nPromotes integer types to float64' },
  { name: 'cosh', pattern: 'unary', doc: 'Hyperbolic cosine of each element\nPromotes integer types to float64' },
  { name: 'tanh', pattern: 'unary', doc: 'Hyperbolic tangent of each element\nPromotes integer types to float64' },
  { name: 'arcsinh', pattern: 'unary', doc: 'Inverse hyperbolic sine of each element\nPromotes integer types to float64' },
  { name: 'arccosh', pattern: 'unary', doc: 'Inverse hyperbolic cosine of each element\nPromotes integer types to float64' },
  { name: 'arctanh', pattern: 'unary', doc: 'Inverse hyperbolic tangent of each element\nPromotes integer types to float64' },
  { name: 'bitwise_not', pattern: 'unary', doc: 'Bitwise NOT (inversion) element-wise' },
  { name: 'invert', pattern: 'unary', doc: 'Invert (bitwise NOT) element-wise - alias for bitwise_not' },
  { name: 'logical_not', pattern: 'unary', doc: 'Logical NOT element-wise\n@returns Boolean array (1 = true, 0 = false)' },
  { name: 'isfinite', pattern: 'unary', doc: 'Test element-wise for finiteness (not infinity and not NaN)' },
  { name: 'isinf', pattern: 'unary', doc: 'Test element-wise for positive or negative infinity' },
  { name: 'isnan', pattern: 'unary', doc: 'Test element-wise for NaN (Not a Number)' },
  { name: 'isnat', pattern: 'unary', doc: 'Test element-wise for NaT (Not a Time)\n@returns Boolean array (always false without datetime support)' },
  { name: 'signbit', pattern: 'unary', doc: 'Returns element-wise True where signbit is set (less than zero)' },
  { name: 'spacing', pattern: 'unary', doc: 'Return the distance between x and the nearest adjacent number' },
  { name: 'cbrt', pattern: 'unary', doc: 'Element-wise cube root\nPromotes integer types to float64' },
  { name: 'fabs', pattern: 'unary', doc: 'Element-wise absolute value (always returns float)' },
  { name: 'square', pattern: 'unary', doc: 'Element-wise square (x**2)' },
  { name: 'conj', pattern: 'unary', doc: 'Return the complex conjugate, element-wise' },
  { name: 'flatten', pattern: 'unary', doc: 'Return a flattened copy of the array\n@returns 1D array containing all elements' },
  { name: 'argwhere', pattern: 'unary', doc: 'Find the indices of array elements that are non-zero, grouped by element\n@returns 2D array of shape (N, ndim)' },

  // ============================================================
  // Binary operations (self, other → NDArray)
  // ============================================================
  { name: 'add', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise addition\n@param other - Array or scalar to add' },
  { name: 'subtract', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise subtraction\n@param other - Array or scalar to subtract' },
  { name: 'multiply', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise multiplication\n@param other - Array or scalar to multiply' },
  { name: 'divide', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise division\n@param other - Array or scalar to divide by' },
  { name: 'mod', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise modulo operation\n@param other - Array or scalar divisor' },
  { name: 'floor_divide', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise floor division\n@param other - Array or scalar to divide by' },
  { name: 'power', pattern: 'binary', params: 'exponent: NDArray | number', coreArgs: 'exponent', doc: 'Raise elements to power\n@param exponent - Power to raise to (array or scalar)' },
  { name: 'logaddexp', pattern: 'binary', params: 'x2: NDArray | number', coreArgs: 'x2', doc: 'Logarithm of the sum of exponentials: log(exp(x1) + exp(x2))\n@param x2 - Second operand' },
  { name: 'logaddexp2', pattern: 'binary', params: 'x2: NDArray | number', coreArgs: 'x2', doc: 'Logarithm base 2 of the sum of exponentials: log2(2^x1 + 2^x2)\n@param x2 - Second operand' },
  { name: 'arctan2', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise arc tangent of this/other choosing the quadrant correctly\n@param other - x-coordinates' },
  { name: 'hypot', pattern: 'binary', params: 'other: NDArray | number', doc: 'Given the "legs" of a right triangle, return its hypotenuse\n@param other - Second leg' },
  { name: 'greater', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise greater than comparison\n@returns Boolean array' },
  { name: 'greater_equal', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise greater than or equal comparison\n@returns Boolean array' },
  { name: 'less', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise less than comparison\n@returns Boolean array' },
  { name: 'less_equal', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise less than or equal comparison\n@returns Boolean array' },
  { name: 'equal', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise equality comparison\n@returns Boolean array' },
  { name: 'not_equal', pattern: 'binary', params: 'other: NDArray | number', doc: 'Element-wise not equal comparison\n@returns Boolean array' },
  { name: 'bitwise_and', pattern: 'binary', params: 'other: NDArray | number', doc: 'Bitwise AND element-wise\n@param other - Array or scalar (must be integer type)' },
  { name: 'bitwise_or', pattern: 'binary', params: 'other: NDArray | number', doc: 'Bitwise OR element-wise\n@param other - Array or scalar (must be integer type)' },
  { name: 'bitwise_xor', pattern: 'binary', params: 'other: NDArray | number', doc: 'Bitwise XOR element-wise\n@param other - Array or scalar (must be integer type)' },
  { name: 'left_shift', pattern: 'binary', params: 'shift: NDArray | number', coreArgs: 'shift', doc: 'Left shift elements by positions\n@param shift - Shift amount' },
  { name: 'right_shift', pattern: 'binary', params: 'shift: NDArray | number', coreArgs: 'shift', doc: 'Right shift elements by positions\n@param shift - Shift amount' },
  { name: 'logical_and', pattern: 'binary', params: 'other: NDArray | number', doc: 'Logical AND element-wise\n@returns Boolean array (1 = true, 0 = false)' },
  { name: 'logical_or', pattern: 'binary', params: 'other: NDArray | number', doc: 'Logical OR element-wise\n@returns Boolean array (1 = true, 0 = false)' },
  { name: 'logical_xor', pattern: 'binary', params: 'other: NDArray | number', doc: 'Logical XOR element-wise\n@returns Boolean array (1 = true, 0 = false)' },
  { name: 'copysign', pattern: 'binary', params: 'x2: NDArray | number', coreArgs: 'x2', doc: 'Change the sign of x1 to that of x2, element-wise\n@param x2 - Values whose sign is used' },
  { name: 'nextafter', pattern: 'binary', params: 'x2: NDArray | number', coreArgs: 'x2', doc: 'Return the next floating-point value after x1 towards x2, element-wise\n@param x2 - Direction to look' },
  { name: 'remainder', pattern: 'binary', params: 'divisor: NDArray | number', coreArgs: 'divisor', doc: 'Element-wise remainder (same as mod)\n@param divisor - Array or scalar divisor' },
  { name: 'heaviside', pattern: 'binary', params: 'x2: NDArray | number', coreArgs: 'x2', doc: 'Heaviside step function\n@param x2 - Value to use when this array element is 0' },
  { name: 'matmul', pattern: 'binary', params: 'other: NDArray', coreArgs: 'other', doc: 'Matrix multiplication\n@param other - Array to multiply with' },
  { name: 'outer', pattern: 'binary', params: 'other: NDArray', coreArgs: 'other', doc: 'Outer product (flattens inputs then computes a[i]*b[j])\n@param other - Array to compute outer product with' },

  // ============================================================
  // Reduction operations (→ NDArray | scalar)
  // ============================================================

  // axis? + keepdims? → NDArray | number | Complex
  { name: 'sum', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | bigint | Complex', doc: 'Sum array elements over a given axis' },
  { name: 'mean', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Compute the arithmetic mean along the specified axis' },
  { name: 'prod', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | bigint | Complex', doc: 'Product of array elements over a given axis' },
  { name: 'max', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Return the maximum along a given axis' },
  { name: 'min', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Return the minimum along a given axis' },
  { name: 'ptp', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Peak to peak (maximum - minimum) value along a given axis' },
  { name: 'nansum', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Return the sum of array elements, treating NaNs as zero' },
  { name: 'nanprod', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Return the product of array elements, treating NaNs as ones' },
  { name: 'nanmean', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Compute the arithmetic mean, ignoring NaNs' },
  { name: 'nanmin', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Return minimum of an array, ignoring NaNs' },
  { name: 'nanmax', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number | Complex', doc: 'Return maximum of an array, ignoring NaNs' },

  // axis? → NDArray | number (no keepdims)
  { name: 'argmin', pattern: 'reduction', params: 'axis?: number', returnType: 'NDArray | number', doc: 'Indices of the minimum values along an axis' },
  { name: 'argmax', pattern: 'reduction', params: 'axis?: number', returnType: 'NDArray | number', doc: 'Indices of the maximum values along an axis' },
  { name: 'nanargmin', pattern: 'reduction', params: 'axis?: number', returnType: 'NDArray | number', doc: 'Return the indices of the minimum values, ignoring NaNs' },
  { name: 'nanargmax', pattern: 'reduction', params: 'axis?: number', returnType: 'NDArray | number', doc: 'Return the indices of the maximum values, ignoring NaNs' },

  // axis? + ddof + keepdims? → NDArray | number
  { name: 'var', pattern: 'reduction', coreName: 'variance', params: 'axis?: number, ddof: number = 0, keepdims: boolean = false', returnType: 'NDArray | number', doc: 'Compute variance along the specified axis\n@param axis - Axis along which to compute variance\n@param ddof - Delta degrees of freedom (default: 0)\n@param keepdims - If true, reduced axes are left as dimensions with size 1' },
  { name: 'std', pattern: 'reduction', params: 'axis?: number, ddof: number = 0, keepdims: boolean = false', returnType: 'NDArray | number', doc: 'Compute standard deviation along the specified axis\n@param axis - Axis along which to compute std\n@param ddof - Delta degrees of freedom (default: 0)\n@param keepdims - If true, reduced axes are left as dimensions with size 1' },
  { name: 'nanvar', pattern: 'reduction', params: 'axis?: number, ddof: number = 0, keepdims: boolean = false', returnType: 'NDArray | number', doc: 'Compute the variance, ignoring NaNs\n@param axis - Axis along which to compute variance\n@param ddof - Delta degrees of freedom (default: 0)\n@param keepdims - If true, reduced axes are left as dimensions with size 1' },
  { name: 'nanstd', pattern: 'reduction', params: 'axis?: number, ddof: number = 0, keepdims: boolean = false', returnType: 'NDArray | number', doc: 'Compute the standard deviation, ignoring NaNs\n@param axis - Axis along which to compute std\n@param ddof - Delta degrees of freedom (default: 0)\n@param keepdims - If true, reduced axes are left as dimensions with size 1' },

  // axis? + keepdims? → NDArray | boolean
  { name: 'all', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | boolean', doc: 'Test whether all array elements along a given axis evaluate to True' },
  { name: 'any', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | boolean', doc: 'Test whether any array elements along a given axis evaluate to True' },

  // axis? + keepdims? → NDArray | number
  { name: 'median', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number', doc: 'Compute the median along the specified axis' },
  { name: 'nanmedian', pattern: 'reduction', params: 'axis?: number, keepdims: boolean = false', returnType: 'NDArray | number', doc: 'Compute the median, ignoring NaNs' },

  // q first, then axis? + keepdims? → NDArray | number
  { name: 'percentile', pattern: 'reduction', params: 'q: number, axis?: number, keepdims: boolean = false', coreArgs: 'q, axis, keepdims', returnType: 'NDArray | number', doc: 'Compute the q-th percentile of the data along the specified axis\n@param q - Percentile to compute (0-100)' },
  { name: 'quantile', pattern: 'reduction', params: 'q: number, axis?: number, keepdims: boolean = false', coreArgs: 'q, axis, keepdims', returnType: 'NDArray | number', doc: 'Compute the q-th quantile of the data along the specified axis\n@param q - Quantile to compute (0-1)' },
  { name: 'nanquantile', pattern: 'reduction', params: 'q: number, axis?: number, keepdims: boolean = false', coreArgs: 'q, axis, keepdims', returnType: 'NDArray | number', doc: 'Compute the q-th quantile, ignoring NaNs\n@param q - Quantile to compute (0-1)' },
  { name: 'nanpercentile', pattern: 'reduction', params: 'q: number, axis?: number, keepdims: boolean = false', coreArgs: 'q, axis, keepdims', returnType: 'NDArray | number', doc: 'Compute the q-th percentile, ignoring NaNs\n@param q - Percentile to compute (0-100)' },

  // ============================================================
  // Passthrough operations (always return NDArray)
  // ============================================================
  { name: 'cumsum', pattern: 'passthrough', params: 'axis?: number', doc: 'Return the cumulative sum of elements along a given axis' },
  { name: 'cumprod', pattern: 'passthrough', params: 'axis?: number', doc: 'Return the cumulative product of elements along a given axis' },
  { name: 'nancumsum', pattern: 'passthrough', params: 'axis?: number', doc: 'Return the cumulative sum of elements, treating NaNs as zero' },
  { name: 'nancumprod', pattern: 'passthrough', params: 'axis?: number', doc: 'Return the cumulative product of elements, treating NaNs as one' },
  { name: 'sort', pattern: 'passthrough', params: 'axis: number = -1', doc: 'Return a sorted copy of the array\n@param axis - Axis along which to sort. Default is -1 (last axis)' },
  { name: 'argsort', pattern: 'passthrough', params: 'axis: number = -1', doc: 'Returns the indices that would sort this array\n@param axis - Axis along which to sort. Default is -1 (last axis)' },
  { name: 'partition', pattern: 'passthrough', params: 'kth: number, axis: number = -1', doc: 'Partially sort the array\n@param kth - Element index to partition by\n@param axis - Axis along which to sort. Default is -1 (last axis)' },
  { name: 'argpartition', pattern: 'passthrough', params: 'kth: number, axis: number = -1', doc: 'Returns indices that would partition the array\n@param kth - Element index to partition by\n@param axis - Axis along which to sort. Default is -1 (last axis)' },
  { name: 'diagonal', pattern: 'passthrough', params: 'offset: number = 0, axis1: number = 0, axis2: number = 1', doc: 'Return specified diagonals\n@param offset - Offset of the diagonal from the main diagonal\n@param axis1 - First axis of the 2-D sub-arrays\n@param axis2 - Second axis of the 2-D sub-arrays' },
  { name: 'resize', pattern: 'passthrough', params: 'newShape: number[]', coreArgs: 'newShape', doc: 'Return a new array with the specified shape\nIf larger, filled with repeated copies of the original data\n@param newShape - Shape of the resized array' },
  { name: 'diff', pattern: 'passthrough', params: 'n: number = 1, axis: number = -1', doc: 'Calculate the n-th discrete difference along the given axis\n@param n - Number of times values are differenced (default: 1)\n@param axis - Axis along which to compute difference (default: -1)' },
  { name: 'take', pattern: 'passthrough', params: 'indices: number[], axis?: number', doc: 'Take elements from array along an axis\n@param indices - Indices of elements to take\n@param axis - Axis along which to take' },
  { name: 'repeat', pattern: 'passthrough', params: 'repeats: number | number[], axis?: number', doc: 'Repeat elements of an array\n@param repeats - Number of repetitions for each element\n@param axis - Axis along which to repeat' },
  { name: 'transpose', pattern: 'passthrough', params: 'axes?: number[]', doc: 'Transpose array (permute dimensions)\n@param axes - Permutation of axes. If undefined, reverse the dimensions\n@returns Transposed array (always a view)' },
  { name: 'squeeze', pattern: 'passthrough', params: 'axis?: number', doc: 'Remove axes of length 1\n@param axis - Axis to squeeze\n@returns Array with specified dimensions removed (always a view)' },
  { name: 'expand_dims', pattern: 'passthrough', params: 'axis: number', doc: 'Expand the shape by inserting a new axis of length 1\n@param axis - Position where new axis is placed\n@returns Array with additional dimension (always a view)' },
  { name: 'swapaxes', pattern: 'passthrough', params: 'axis1: number, axis2: number', doc: 'Swap two axes of an array\n@param axis1 - First axis\n@param axis2 - Second axis\n@returns Array with swapped axes (always a view)' },
  { name: 'moveaxis', pattern: 'passthrough', params: 'source: number | number[], destination: number | number[]', doc: 'Move axes to new positions\n@param source - Original positions of axes to move\n@param destination - New positions for axes\n@returns Array with moved axes (always a view)' },

  // ============================================================
  // Array return (→ NDArray[])
  // ============================================================
  { name: 'nonzero', pattern: 'array_return', doc: 'Return the indices of non-zero elements\n@returns Tuple of arrays, one for each dimension' },
];
