/**
 * Adapter contract for benchmarking a third-party JS numerical library against
 * numpy-ts on a shared operation + input set.
 *
 * Design (see ../COVERAGE.md): coverage-aware. Each adapter declares only the
 * operations it genuinely supports; the runner benchmarks the intersection with
 * numpy-ts's spec list and reports per-library coverage (N / total).
 */

/**
 * The dtype a benchmark run targets. Every participant runs at this dtype, or is
 * skipped if it doesn't natively support it (no silent coercion). int64/uint64/
 * complex are effectively numpy-ts-only showcases.
 */
export type Dtype =
  | 'float64' | 'float32' | 'float16'
  | 'int8' | 'int16' | 'int32' | 'int64'
  | 'uint8' | 'uint16' | 'uint32' | 'uint64'
  | 'bool' | 'complex64' | 'complex128';

export const ALL_DTYPES: Dtype[] = [
  'float64', 'float32', 'float16',
  'int8', 'int16', 'int32', 'int64',
  'uint8', 'uint16', 'uint32', 'uint64',
  'bool', 'complex64', 'complex128',
];

/** A single input array, extracted from numpy-ts into a library-neutral form. */
export interface ArrayData {
  shape: number[];
  /** Row-major (C-order) flat values. */
  data: number[];
  dtype: string;
}

/** Library-neutral inputs for one benchmark spec, built once (outside timing). */
export interface SpecData {
  /** Array-valued operands keyed by name: 'a', 'b', 'c', ... */
  arrays: Record<string, ArrayData>;
  /** Scalar / structural params: axis, n, indices, dims, new_shape, ... */
  params: Record<string, unknown>;
}

/** One operation's implementation for a given library. */
export interface OpDef {
  /** Build native inputs from SpecData. Runs ONCE per spec, outside timing. */
  prepare: (d: SpecData) => unknown;
  /** Execute the op on prepared inputs. Runs inside the timed region. */
  run: (prepared: unknown) => unknown;
  /**
   * Force a lazy/async result to fully materialize. Awaited inside the timed
   * region. Omit for eager libraries (treated as identity).
   */
  materialize?: (result: unknown) => unknown | Promise<unknown>;
  /** Free a per-iteration result (e.g. jax-js dispose, tfjs is handled via tidy). */
  disposeResult?: (result: unknown) => void;
  /** Free the prepared native inputs once the spec is done. */
  disposePrepared?: (prepared: unknown) => void;
  /** Override the adapter-level dtype list for this op (rare). */
  dtypes?: string[];
}

export interface JsLibAdapter {
  /** Display name. */
  name: string;
  /** npm package(s) backing this adapter. */
  pkg: string;
  /** Installed version (filled at runtime). */
  version?: string;
  /**
   * numpy-equivalent dtypes this library natively supports. Doubles as the
   * participation gate: the library runs a dtype only if it's listed here.
   */
  dtypes: Dtype[];
  /** One-time async init (e.g. tfjs setBackend('wasm')). */
  init?: () => Promise<void>;
  /** operation-name -> implementation. Presence == supported. */
  ops: Record<string, OpDef>;
}

export interface LibTiming {
  mean_ms: number;
  median_ms: number;
  min_ms: number;
  ops_per_sec: number;
  samples: number;
}

export interface SpecResult {
  name: string;
  category: string;
  operation: string;
  /** library name -> timing (only libraries that ran this spec). */
  timings: Record<string, LibTiming>;
  /** library name -> ratio (lib mean / numpy-ts mean); >1 == slower than numpy-ts. */
  ratios: Record<string, number>;
}
