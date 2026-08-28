/**
 * Widening and narrowing between a concrete TypedArray and a Float64Array.
 *
 * Working in f64 is the right idea for anything that has to do real arithmetic
 * across dtypes. Doing it with `new Float64Array(storage.data.subarray(...))` is
 * not: `storage.data` is a `Float64Array | Int8Array | ...` union, so that one
 * builtin call site sees every TypedArray type a process touches, goes
 * megamorphic, and drops off V8's fast TypedArray-to-TypedArray conversion onto a
 * generic per-element path. Measured across the full dtype sweep that cost more
 * than the loops it was meant to make monomorphic — `ediff1d` at 116.7x and
 * `tensordot` float32 at 110.1x were both this.
 *
 * The switch below dispatches once per call, not per element, and each branch
 * has narrowed the type so its loop compiles against a single map. That is the
 * same shape as CMP_LOOPS and stepStore, both of which held up under the full
 * suite.
 */

import type { TypedArray } from '../dtype';

/** Widen `n` elements of `src` from `srcOff` into `out`, which must hold `n`. */
export function widenToF64(src: TypedArray, srcOff: number, n: number, out: Float64Array): void {
  const ctor = (src as { constructor: unknown }).constructor;

  if (ctor === Float64Array) {
    const a = src as Float64Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Float32Array) {
    const a = src as Float32Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Int32Array) {
    const a = src as Int32Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Uint32Array) {
    const a = src as Uint32Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Int16Array) {
    const a = src as Int16Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Uint16Array) {
    const a = src as Uint16Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Int8Array) {
    const a = src as Int8Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === Uint8Array) {
    const a = src as Uint8Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else if (ctor === BigInt64Array) {
    const a = src as BigInt64Array;
    for (let i = 0; i < n; i++) out[i] = Number(a[srcOff + i]!);
  } else if (ctor === BigUint64Array) {
    const a = src as BigUint64Array;
    for (let i = 0; i < n; i++) out[i] = Number(a[srcOff + i]!);
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const a = src as Float16Array;
    for (let i = 0; i < n; i++) out[i] = a[srcOff + i]!;
  } else {
    throw new Error('widenToF64: unsupported TypedArray');
  }
}

/** Narrow `n` elements of `src` into `dst` at `dstOff`, wrapping as NumPy does. */
export function narrowFromF64(dst: TypedArray, dstOff: number, n: number, src: Float64Array): void {
  const ctor = (dst as { constructor: unknown }).constructor;

  if (ctor === Float64Array) {
    const b = dst as Float64Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Float32Array) {
    const b = dst as Float32Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Int32Array) {
    const b = dst as Int32Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Uint32Array) {
    const b = dst as Uint32Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Int16Array) {
    const b = dst as Int16Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Uint16Array) {
    const b = dst as Uint16Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Int8Array) {
    const b = dst as Int8Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === Uint8Array) {
    const b = dst as Uint8Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else if (ctor === BigInt64Array) {
    const b = dst as BigInt64Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = BigInt(Math.trunc(src[i]!));
  } else if (ctor === BigUint64Array) {
    const b = dst as BigUint64Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = BigInt(Math.trunc(src[i]!));
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const b = dst as Float16Array;
    for (let i = 0; i < n; i++) b[dstOff + i] = src[i]!;
  } else {
    throw new Error('narrowFromF64: unsupported TypedArray');
  }
}

/**
 * A sorted copy of `n` elements of `src` from `off`, at the source's own dtype.
 *
 * `TypedArray.prototype.sort` is native and numeric (NaN last), which is what
 * the JS comparators this replaces spelled out by hand. Sorting at the source
 * type also keeps 64-bit integers exact: widening first would let values above
 * 2^53 collapse onto the same double and reorder.
 */
export function sortedCopy(src: TypedArray, off: number, n: number): TypedArray {
  const ctor = (src as { constructor: unknown }).constructor;

  if (ctor === Float64Array) {
    const c = (src as Float64Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Float32Array) {
    const c = (src as Float32Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Int32Array) {
    const c = (src as Int32Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Uint32Array) {
    const c = (src as Uint32Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Int16Array) {
    const c = (src as Int16Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Uint16Array) {
    const c = (src as Uint16Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Int8Array) {
    const c = (src as Int8Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === Uint8Array) {
    const c = (src as Uint8Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === BigInt64Array) {
    const c = (src as BigInt64Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (ctor === BigUint64Array) {
    const c = (src as BigUint64Array).slice(off, off + n);
    c.sort();
    return c;
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const c = (src as Float16Array).slice(off, off + n);
    c.sort();
    return c;
  }
  throw new Error('sortedCopy: unsupported TypedArray');
}

/**
 * `out[i] = src[idx[i]]`, widening to f64, dispatched per source dtype.
 *
 * Lets a strided or broadcast operand be read straight into a plain
 * Float64Array. The alternative — materialising it with `.copy()` first —
 * allocates a WASM region per operand, and allocation pressure is what made
 * these ops slow once the heap was busy.
 */
export function widenGatherToF64(src: TypedArray, idx: Int32Array, out: Float64Array): void {
  const ctor = (src as { constructor: unknown }).constructor;
  const n = idx.length;

  if (ctor === Float64Array) {
    const a = src as Float64Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Float32Array) {
    const a = src as Float32Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Int32Array) {
    const a = src as Int32Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Uint32Array) {
    const a = src as Uint32Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Int16Array) {
    const a = src as Int16Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Uint16Array) {
    const a = src as Uint16Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Int8Array) {
    const a = src as Int8Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === Uint8Array) {
    const a = src as Uint8Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else if (ctor === BigInt64Array) {
    const a = src as BigInt64Array;
    for (let i = 0; i < n; i++) out[i] = Number(a[idx[i]!]!);
  } else if (ctor === BigUint64Array) {
    const a = src as BigUint64Array;
    for (let i = 0; i < n; i++) out[i] = Number(a[idx[i]!]!);
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const a = src as Float16Array;
    for (let i = 0; i < n; i++) out[i] = a[idx[i]!]!;
  } else {
    throw new Error('widenGatherToF64: unsupported TypedArray');
  }
}

/**
 * Write `n` real values from `src` into an interleaved complex buffer:
 * `dst[2i] = src[i]`, `dst[2i + 1] = 0`.
 *
 * Dispatched per source dtype so no widening buffer is needed in between. The
 * previous route — copy, widen to Float64Array, then write — allocated twice
 * per call, which measured 15% slower than the boxed version it replaced.
 */
export function writeInterleaved(
  src: TypedArray,
  dst: Float64Array | Float32Array,
  n: number,
): void {
  const ctor = (src as { constructor: unknown }).constructor;

  if (ctor === Float64Array) {
    const a = src as Float64Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Float32Array) {
    const a = src as Float32Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Int32Array) {
    const a = src as Int32Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Uint32Array) {
    const a = src as Uint32Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Int16Array) {
    const a = src as Int16Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Uint16Array) {
    const a = src as Uint16Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Int8Array) {
    const a = src as Int8Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === Uint8Array) {
    const a = src as Uint8Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else if (ctor === BigInt64Array || ctor === BigUint64Array) {
    const a = src as BigInt64Array | BigUint64Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = Number(a[i]!);
      dst[i * 2 + 1] = 0;
    }
  } else if (typeof Float16Array !== 'undefined' && ctor === Float16Array) {
    const a = src as Float16Array;
    for (let i = 0; i < n; i++) {
      dst[i * 2] = a[i]!;
      dst[i * 2 + 1] = 0;
    }
  } else {
    throw new Error('writeInterleaved: unsupported TypedArray');
  }
}
