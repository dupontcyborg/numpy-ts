/**
 * Gathering a strided/broadcast view into a dense, C-contiguous buffer.
 *
 * This is the hot path behind `ArrayStorage.copy()`, and therefore behind every
 * `.copy()` of a transposed, sliced, or broadcast view. It used to run
 * `dest[i] = this.iget(i)` per element, which per element re-checked the dtype,
 * recomputed the trailing-dimension products in an O(ndim^2) loop with a
 * `Math.floor` division per dimension, and then stored through a TypedArray
 * union — a store site that goes megamorphic once more than four dtypes pass
 * through it, so every dtype pays.
 *
 * Instead, pick a strategy from the stride pattern and let the engine's native
 * bulk primitives do the work:
 *
 *   contiguous   one `.set()` for the whole array
 *   broadcast    `.fill()` per repeated element (trailing zero strides)
 *   row-major    `.set()` per contiguous run
 *   otherwise    precompute source offsets once, then a monomorphic gather
 *
 * Only the last strategy touches elements individually, and it dispatches on the
 * concrete TypedArray constructor so each loop stays monomorphic.
 */

import type { TypedArray } from '../dtype';

type Gather = (src: TypedArray, dst: TypedArray, idx: Int32Array) => void;

function gF64(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Float64Array;
  const b = d as Float64Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gF32(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Float32Array;
  const b = d as Float32Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gF16(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Float16Array;
  const b = d as Float16Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gI32(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Int32Array;
  const b = d as Int32Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gU32(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Uint32Array;
  const b = d as Uint32Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gI16(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Int16Array;
  const b = d as Int16Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gU16(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Uint16Array;
  const b = d as Uint16Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gI8(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Int8Array;
  const b = d as Int8Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gU8(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Uint8Array;
  const b = d as Uint8Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gI64(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as BigInt64Array;
  const b = d as BigInt64Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}
function gU64(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as BigUint64Array;
  const b = d as BigUint64Array;
  for (let i = 0; i < ix.length; i++) b[i] = a[ix[i]!]!;
}

const GATHER_LOOPS = new Map<unknown, Gather>([
  [Float64Array, gF64],
  [Float32Array, gF32],
  [Int32Array, gI32],
  [Uint32Array, gU32],
  [Int16Array, gI16],
  [Uint16Array, gU16],
  [Int8Array, gI8],
  [Uint8Array, gU8],
  [BigInt64Array, gI64],
  [BigUint64Array, gU64],
]);

// float16 only exists on engines that ship Float16Array.
if (typeof Float16Array !== 'undefined') {
  GATHER_LOOPS.set(Float16Array, gF16);
}

/** Untyped views of the bulk primitives — one megamorphic call per run, not per element. */
interface BulkDst {
  set(src: ArrayLike<number>, offset: number): void;
  fill(value: never, start: number, end: number): void;
}
interface BulkSrc {
  subarray(begin: number, end: number): ArrayLike<number>;
}

/**
 * Copy the logical elements of a strided view into `dst` in C order.
 *
 * `shape`, `strides` and `offset` are in element units; for complex dtypes they
 * count complex elements, and `isComplex` rescales them to the underlying
 * real pairs by appending a trailing `[2]` dimension of stride 1. That turns
 * complex into an ordinary contiguous-innermost case, so it needs no special
 * handling below.
 */
export function stridedCopyInto(
  src: TypedArray,
  dst: TypedArray,
  shapeIn: readonly number[],
  stridesIn: readonly number[],
  offsetIn: number,
  isComplex: boolean,
): void {
  let shape: readonly number[] = shapeIn;
  let strides: readonly number[] = stridesIn;
  let offset = offsetIn;

  if (isComplex) {
    shape = [...shapeIn, 2];
    strides = [...stridesIn.map((s) => s * 2), 1];
    offset = offsetIn * 2;
  }

  const nd = shape.length;
  let size = 1;
  for (let i = 0; i < nd; i++) size *= shape[i]!;
  if (size === 0) return;

  const bd = dst as unknown as BulkDst;
  const bs = src as unknown as BulkSrc;

  if (nd === 0) {
    bd.set(bs.subarray(offset, offset + 1), 0);
    return;
  }

  // Trailing zero-stride dimensions repeat one source element `repeat` times.
  let repeat = 1;
  let j = nd - 1;
  while (j >= 0 && strides[j] === 0) {
    repeat *= shape[j]!;
    j--;
  }

  if (repeat > 1) {
    // Broadcast: one native fill per distinct source element.
    fillWalk(src, bd, shape, strides, offset, j, repeat);
    return;
  }

  // Collapse the trailing dimensions that are already contiguous into one run.
  let runLen = 1;
  let k = nd - 1;
  while (k >= 0 && strides[k] === runLen) {
    runLen *= shape[k]!;
    k--;
  }

  if (k < 0) {
    // Fully contiguous — a single bulk copy, regardless of offset.
    bd.set(bs.subarray(offset, offset + size), 0);
    return;
  }

  // Below ~8 elements a native call per run costs more than it saves, so fall
  // through to the gather instead of issuing `size` one-element `.set()`s.
  if (runLen >= 8) {
    runWalk(bs, bd, shape, strides, offset, k, runLen);
    return;
  }

  const loop = GATHER_LOOPS.get((src as { constructor: unknown }).constructor);
  if (!loop) {
    throw new Error('stridedCopyInto: unsupported TypedArray');
  }
  loop(src, dst, buildOffsets(shape, strides, offset, size, nd));
}

/** One native `.set()` per contiguous run of `runLen` elements. */
function runWalk(
  bs: BulkSrc,
  bd: BulkDst,
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  k: number,
  runLen: number,
): void {
  const counter = new Int32Array(k + 1);
  let srcIdx = offset;
  let d = 0;
  let outer = 1;
  for (let i = 0; i <= k; i++) outer *= shape[i]!;

  for (let o = 0; o < outer; o++) {
    bd.set(bs.subarray(srcIdx, srcIdx + runLen), d);
    d += runLen;
    for (let i = k; i >= 0; i--) {
      counter[i]!++;
      srcIdx += strides[i]!;
      if (counter[i]! < shape[i]!) break;
      srcIdx -= strides[i]! * shape[i]!;
      counter[i] = 0;
    }
  }
}

/** One native `.fill()` per source element, for trailing broadcast dimensions. */
function fillWalk(
  src: TypedArray,
  bd: BulkDst,
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  j: number,
  repeat: number,
): void {
  let outer = 1;
  for (let i = 0; i <= j; i++) outer *= shape[i]!;

  const counter = new Int32Array(j + 1);
  let srcIdx = offset;
  let d = 0;

  for (let o = 0; o < outer; o++) {
    bd.fill(src[srcIdx] as never, d, d + repeat);
    d += repeat;
    for (let i = j; i >= 0; i--) {
      counter[i]!++;
      srcIdx += strides[i]!;
      if (counter[i]! < shape[i]!) break;
      srcIdx -= strides[i]! * shape[i]!;
      counter[i] = 0;
    }
  }
}

/** Source offsets in C order — one linear pass, no division. */
function buildOffsets(
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  size: number,
  nd: number,
): Int32Array {
  const idx = new Int32Array(size);
  const counter = new Int32Array(nd);
  let srcIdx = offset;
  for (let i = 0; i < size; i++) {
    idx[i] = srcIdx;
    for (let a = nd - 1; a >= 0; a--) {
      counter[a]!++;
      srcIdx += strides[a]!;
      if (counter[a]! < shape[a]!) break;
      srcIdx -= strides[a]! * shape[a]!;
      counter[a] = 0;
    }
  }
  return idx;
}

/**
 * Store `count` values at `start`, `start + step`, ... in `dst`.
 *
 * `values` is either a scalar or an array cycled by index. The constructor
 * switch exists so each store below is its own call site and stays monomorphic;
 * a single loop storing through the TypedArray union goes megamorphic as soon
 * as a fifth dtype reaches it, and then every dtype pays.
 */
export function stepStore(
  dst: TypedArray,
  start: number,
  step: number,
  count: number,
  read: (i: number) => number | bigint,
): void {
  const ctor = (dst as { constructor: unknown }).constructor;
  if (ctor === Float64Array) {
    const a = dst as Float64Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Float32Array) {
    const a = dst as Float32Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Int32Array) {
    const a = dst as Int32Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Uint32Array) {
    const a = dst as Uint32Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Int16Array) {
    const a = dst as Int16Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Uint16Array) {
    const a = dst as Uint16Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Int8Array) {
    const a = dst as Int8Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === Uint8Array) {
    const a = dst as Uint8Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else if (ctor === BigInt64Array) {
    const a = dst as BigInt64Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as bigint;
  } else if (ctor === BigUint64Array) {
    const a = dst as BigUint64Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as bigint;
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const a = dst as Float16Array;
    for (let i = 0; i < count; i++) a[start + i * step] = read(i) as number;
  } else {
    throw new Error('stepStore: unsupported TypedArray');
  }
}

type Scatter = (src: TypedArray, dst: TypedArray, idx: Int32Array) => void;

function sF64(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Float64Array;
  const b = d as Float64Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sF32(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Float32Array;
  const b = d as Float32Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sF16(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Float16Array;
  const b = d as Float16Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sI32(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Int32Array;
  const b = d as Int32Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sU32(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Uint32Array;
  const b = d as Uint32Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sI16(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Int16Array;
  const b = d as Int16Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sU16(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Uint16Array;
  const b = d as Uint16Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sI8(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Int8Array;
  const b = d as Int8Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sU8(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as Uint8Array;
  const b = d as Uint8Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sI64(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as BigInt64Array;
  const b = d as BigInt64Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}
function sU64(s: TypedArray, d: TypedArray, ix: Int32Array): void {
  const a = s as BigUint64Array;
  const b = d as BigUint64Array;
  for (let i = 0; i < ix.length; i++) b[ix[i]!] = a[i]!;
}

const SCATTER_LOOPS = new Map<unknown, Scatter>([
  [Float64Array, sF64],
  [Float32Array, sF32],
  [Int32Array, sI32],
  [Uint32Array, sU32],
  [Int16Array, sI16],
  [Uint16Array, sU16],
  [Int8Array, sI8],
  [Uint8Array, sU8],
  [BigInt64Array, sI64],
  [BigUint64Array, sU64],
]);

if (typeof Float16Array !== 'undefined') {
  SCATTER_LOOPS.set(Float16Array, sF16);
}

/**
 * The mirror of `stridedCopyInto`: write a dense, C-order `src` into a strided
 * region of `dst`. This is what concatenation does — each input occupies a
 * strided slice of the output — and it picks strategies the same way.
 */
export function stridedScatterFrom(
  src: TypedArray,
  dst: TypedArray,
  shapeIn: readonly number[],
  stridesIn: readonly number[],
  offsetIn: number,
  isComplex: boolean,
): void {
  let shape: readonly number[] = shapeIn;
  let strides: readonly number[] = stridesIn;
  let offset = offsetIn;

  if (isComplex) {
    shape = [...shapeIn, 2];
    strides = [...stridesIn.map((s) => s * 2), 1];
    offset = offsetIn * 2;
  }

  const nd = shape.length;
  let size = 1;
  for (let i = 0; i < nd; i++) size *= shape[i]!;
  if (size === 0) return;

  const bd = dst as unknown as { set(v: ArrayLike<number>, o: number): void };
  const bs = src as unknown as { subarray(b: number, e: number): ArrayLike<number> };

  if (nd === 0) {
    bd.set(bs.subarray(0, 1), offset);
    return;
  }

  let runLen = 1;
  let k = nd - 1;
  while (k >= 0 && strides[k] === runLen) {
    runLen *= shape[k]!;
    k--;
  }

  if (k < 0) {
    bd.set(bs.subarray(0, size), offset);
    return;
  }

  if (runLen >= 8) {
    const counter = new Int32Array(k + 1);
    let dstIdx = offset;
    let sPos = 0;
    let outer = 1;
    for (let i = 0; i <= k; i++) outer *= shape[i]!;

    for (let o = 0; o < outer; o++) {
      bd.set(bs.subarray(sPos, sPos + runLen), dstIdx);
      sPos += runLen;
      for (let i = k; i >= 0; i--) {
        counter[i]!++;
        dstIdx += strides[i]!;
        if (counter[i]! < shape[i]!) break;
        dstIdx -= strides[i]! * shape[i]!;
        counter[i] = 0;
      }
    }
    return;
  }

  // Before falling back to per-element offsets, check whether the remaining
  // dimensions are themselves regular — dimension d spans exactly dimension
  // d+1's extent when strides[d] === shape[d+1] * strides[d+1]. When they all
  // collapse, the destination is one arithmetic progression and needs no
  // offset array at all. dstack lands here: width-1 inputs concatenated on the
  // last axis write at a constant stride throughout.
  if (runLen === 1) {
    let count = shape[k]!;
    let d = k - 1;
    while (d >= 0 && strides[d] === shape[d + 1]! * strides[d + 1]!) {
      count *= shape[d]!;
      d--;
    }
    if (d < 0 && stepScatter(src, dst, offset, strides[k]!, count)) return;
  }

  const loop = SCATTER_LOOPS.get((dst as { constructor: unknown }).constructor);
  if (!loop) {
    throw new Error('stridedScatterFrom: unsupported TypedArray');
  }
  loop(src, dst, buildOffsets(shape, strides, offset, size, nd));
}

/**
 * Scatter a dense source into `dst` at `start`, `start + step`, ... — the case
 * where the destination pattern collapses to a single arithmetic progression.
 *
 * This is what `dstack` produces: concatenating width-1 inputs along the last
 * axis writes every element at a constant stride. Handling it here avoids
 * materialising one Int32Array offset per element just to walk a progression
 * that is fully described by two numbers. Constructor switch so each store is
 * its own monomorphic call site.
 */
export function stepScatter(
  src: TypedArray,
  dst: TypedArray,
  start: number,
  step: number,
  count: number,
): boolean {
  const ctor = (dst as { constructor: unknown }).constructor;
  if ((src as { constructor: unknown }).constructor !== ctor) return false;

  if (ctor === Float64Array) {
    const a = src as Float64Array;
    const b = dst as Float64Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Float32Array) {
    const a = src as Float32Array;
    const b = dst as Float32Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Int32Array) {
    const a = src as Int32Array;
    const b = dst as Int32Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Uint32Array) {
    const a = src as Uint32Array;
    const b = dst as Uint32Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Int16Array) {
    const a = src as Int16Array;
    const b = dst as Int16Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Uint16Array) {
    const a = src as Uint16Array;
    const b = dst as Uint16Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Int8Array) {
    const a = src as Int8Array;
    const b = dst as Int8Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === Uint8Array) {
    const a = src as Uint8Array;
    const b = dst as Uint8Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === BigInt64Array) {
    const a = src as BigInt64Array;
    const b = dst as BigInt64Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (ctor === BigUint64Array) {
    const a = src as BigUint64Array;
    const b = dst as BigUint64Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const a = src as Float16Array;
    const b = dst as Float16Array;
    for (let i = 0; i < count; i++) b[start + i * step] = a[i]!;
  } else {
    return false;
  }
  return true;
}
