/**
 * Benchmark coverage guards.
 *
 * The benchmark suite generates dtype variants automatically (see
 * benchmarks/src/specs.ts). That generator is easy to defeat by accident: a
 * base spec that pins a dtype quietly produces a single variant, and nothing
 * fails. That is how every `bitwise` op — including `bitwise_count`, whose int8
 * kernel is ~20x faster than its uint32 one — ran at exactly one dtype while
 * the suite reported full coverage.
 *
 * These tests make coverage loss loud instead of silent:
 *   1. an op that yields one dtype must say why, in a declared list
 *   2. skip lists may not contain operations no spec uses (rot)
 *   3. integer-only categories must sweep every int/uint width
 *   4. every public function must be benchmarked, or listed as an exception
 *
 * They are static — no NumPy, no timing, no WASM. Cheap enough for every run.
 */

import { describe, expect, it } from 'vitest';
import {
  CATEGORY_DTYPE_SUPPORT,
  FAMILY_VARIANTS,
  getBenchmarkSpecs,
  PINNED_INDEX_DTYPE_OPERATIONS,
  SKIP_COMPLEX_OPERATIONS,
  SKIP_DTYPE_OPERATIONS,
  SKIP_FLOAT16_OPERATIONS,
  SKIP_INT_OPERATIONS,
  SKIP_NARROW_INT_OPERATIONS,
  SKIP_UINT_OPERATIONS,
} from '../../benchmarks/src/specs';
import * as np from '../../src';

const ALL_DTYPES = [
  'complex128',
  'complex64',
  'float16',
  'float32',
  'float64',
  'int8',
  'int16',
  'int32',
  'int64',
  'uint8',
  'uint16',
  'uint32',
  'uint64',
] as const;

/** `full` is the widest mode, so it is the one that must be fully covered. */
const specs = getBenchmarkSpecs('full');

/** Trailing dtype token in a spec name; absent means the float64 base. */
function dtypeOf(name: string): string {
  for (const dt of ALL_DTYPES) if (name.endsWith(` ${dt}`)) return dt;
  return 'float64';
}

/** operation -> { dtypes swept, category } */
const byOperation = (() => {
  const m = new Map<string, { dtypes: Set<string>; category: string }>();
  for (const spec of specs) {
    let entry = m.get(spec.operation);
    if (!entry) {
      entry = { dtypes: new Set(), category: spec.category };
      m.set(spec.operation, entry);
    }
    entry.dtypes.add(dtypeOf(spec.name));
  }
  return m;
})();

const SKIP_LISTS: [string, ReadonlySet<string>][] = [
  ['SKIP_DTYPE_OPERATIONS', SKIP_DTYPE_OPERATIONS],
  ['SKIP_INT_OPERATIONS', SKIP_INT_OPERATIONS],
  ['SKIP_UINT_OPERATIONS', SKIP_UINT_OPERATIONS],
  ['SKIP_COMPLEX_OPERATIONS', SKIP_COMPLEX_OPERATIONS],
  ['SKIP_FLOAT16_OPERATIONS', SKIP_FLOAT16_OPERATIONS],
  ['SKIP_NARROW_INT_OPERATIONS', SKIP_NARROW_INT_OPERATIONS],
  ['PINNED_INDEX_DTYPE_OPERATIONS', PINNED_INDEX_DTYPE_OPERATIONS],
];

describe('benchmark dtype coverage', () => {
  it('every single-dtype operation is declared, so no sweep is lost silently', () => {
    const undeclared: string[] = [];
    for (const [operation, { dtypes, category }] of byOperation) {
      if (dtypes.size > 1) continue;
      // Categories absent from CATEGORY_DTYPE_SUPPORT (random, utilities) are
      // intentionally never swept.
      if (!CATEGORY_DTYPE_SUPPORT[category]) continue;
      if (SKIP_DTYPE_OPERATIONS.has(operation)) continue;
      if (PINNED_INDEX_DTYPE_OPERATIONS.has(operation)) continue;
      undeclared.push(`${operation} (category=${category}, only ${[...dtypes]})`);
    }
    expect(
      undeclared,
      'These operations produce exactly one dtype variant but are not declared ' +
        'anywhere. Either the base spec pins a dtype that blocks the sweep (fix ' +
        'the spec), or the single dtype is intentional — in which case add it to ' +
        'SKIP_DTYPE_OPERATIONS or PINNED_INDEX_DTYPE_OPERATIONS with a reason.\n',
    ).toEqual([]);
  });

  it('skip lists contain no operations that no spec uses', () => {
    const known = new Set(specs.map((s) => s.operation));
    const dead: string[] = [];
    for (const [listName, list] of SKIP_LISTS) {
      for (const operation of list) {
        if (!known.has(operation)) dead.push(`${listName}: '${operation}'`);
      }
    }
    expect(
      dead,
      'These skip-list entries reference operations that no benchmark spec ' +
        'defines. They are dead weight and hide the fact that the real op (if ' +
        'renamed) is now unguarded. Remove them or fix the spelling.\n',
    ).toEqual([]);
  });

  it('integer-only categories sweep every int and uint width', () => {
    const expected = [
      ...FAMILY_VARIANTS.int.map((v) => v.dtype),
      ...FAMILY_VARIANTS.uint.map((v) => v.dtype),
    ].sort();

    const intOnly = Object.entries(CATEGORY_DTYPE_SUPPORT)
      .filter(([, fams]) => !fams.includes('float') && !fams.includes('complex'))
      .map(([cat]) => cat);
    expect(intOnly.length, 'expected at least one integer-only category').toBeGreaterThan(0);

    const short: string[] = [];
    for (const [operation, { dtypes, category }] of byOperation) {
      if (!intOnly.includes(category)) continue;
      if (SKIP_DTYPE_OPERATIONS.has(operation)) continue;
      if (PINNED_INDEX_DTYPE_OPERATIONS.has(operation)) continue;
      const missing = expected.filter(
        (d) =>
          !dtypes.has(d) &&
          !(SKIP_NARROW_INT_OPERATIONS.has(operation) && /8$|16$/.test(d)) &&
          !(SKIP_UINT_OPERATIONS.has(operation) && d.startsWith('uint')),
      );
      if (missing.length) short.push(`${operation} missing ${missing.join(',')}`);
    }
    expect(
      short,
      'Integer-only categories (e.g. bitwise) have no float64 base, so their ' +
        'int32/uint32 base dtype must remain sweepable. If these are short, the ' +
        'generator is treating a category default as a semantic dtype pin.\n',
    ).toEqual([]);
  });

  it('bitwise_count is swept across widths (its int8 kernel is ~20x its uint32 one)', () => {
    const entry = byOperation.get('bitwise_count');
    expect(entry, 'bitwise_count has no benchmark at all').toBeDefined();
    for (const dt of ['int8', 'uint8', 'int16', 'uint16', 'int32', 'int64']) {
      expect(entry?.dtypes.has(dt), `bitwise_count is not benchmarked at ${dt}`).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// Function coverage
// ---------------------------------------------------------------------------

/**
 * Public functions that will never have a benchmark, with the reason. These are
 * not timing-relevant: they do no array work, or they are thin aliases measured
 * under their canonical name.
 */
const NOT_BENCHMARKABLE = new Set<string>([
  // Classes, constructors and error types
  'Complex',
  'NDArray',
  'NDArrayCore',
  'InvalidNpyError',
  'UnsupportedDTypeError',
  // Runtime config and error state
  'configureWasm',
  'wasmFreeBytes',
  'seterr',
  'geterr',
  'set_printoptions',
  'get_printoptions',
  'printoptions',
  // Dtype introspection — no array compute
  'can_cast',
  'isdtype',
  'issubdtype',
  'result_type',
  'promote_types',
  'min_scalar_type',
  'mintypecode',
  'common_type',
  'typename',
  // Array introspection — O(1) metadata
  'ndim',
  'shape',
  'size',
  'isscalar',
  'iterable',
  'isfortran',
  'iscomplexobj',
  'isrealobj',
  'isnat',
  'may_share_memory',
  'shares_memory',
  // String and formatting
  'array2string',
  'array_repr',
  'array_str',
  'base_repr',
  'binary_repr',
  'format_float_positional',
  'format_float_scientific',
  // Filesystem and buffer IO — dominated by IO, not compute. The pure
  // serialize/parse paths that are compute-bound *are* benchmarked (io category).
  'frombuffer',
  'fromfile',
  'fromregex',
  'fromregexFile',
  'fromregexFileSync',
  'fromstring',
  'genfromtxt',
  'genfromtxtFile',
  'genfromtxtFileSync',
  'load',
  'loadNpy',
  'loadNpySync',
  'loadNpz',
  'loadNpzFile',
  'loadNpzFileSync',
  'loadNpzSync',
  'loadSync',
  'loadtxt',
  'loadtxtSync',
  'parseNpyData',
  'parseNpyHeader',
  'parseNpz',
  'parseTxt',
  'save',
  'saveNpy',
  'saveNpySync',
  'saveNpzFile',
  'saveNpzFileSync',
  'saveSync',
  'savetxt',
  'savetxtSync',
  'savez',
  'savez_compressed',
  'serializeNpz',
  'serializeTxt',
  'tobytes',
  'tofile',
  'fill',
  'byteswap',
  'copyto',
  // Indexing accessors, not standalone ops
  'bindex',
  'iindex',
  'vindex',
  'view',
  // Aliases measured under their canonical name
  'abs',
  'acos',
  'acosh',
  'asin',
  'asinh',
  'atan',
  'atan2',
  'atanh',
  'conj',
  'conjugate',
  'true_divide',
  'bitwise_invert',
  'bitwise_left_shift',
  'bitwise_right_shift',
  'pow',
  'permute_dims',
  'cumulative_sum',
  'cumulative_prod',
  'row_stack',
  'in1d',
  'degrees',
  'radians',
  'variance',
  'delete',
  'delete_',
  'around',
  'amax',
  'amin',
  'array',
  'asanyarray',
  'asarray',
  'ascontiguousarray',
  'asfortranarray',
]);

/**
 * Ratchet: public functions that *should* have a benchmark but do not yet.
 * This list may only shrink. Adding a benchmark for one of these requires
 * removing it here, and a new uncovered function fails the test rather than
 * silently joining the backlog.
 *
 * Notable clusters: the comparison family (equal/less/greater/...), the
 * rounding family (ceil/floor/round/rint/trunc/fix), and the nan* reductions.
 */
const MISSING_BENCHMARK = new Set<string>([
  'allclose',
  'angle',
  'append',
  'apply_along_axis',
  'apply_over_axes',
  'array_equal',
  'array_equiv',
  'array_split',
  'atleast_1d',
  'atleast_2d',
  'atleast_3d',
  'broadcast_arrays',
  'broadcast_shapes',
  'ceil',
  'choose',
  'column_stack',
  'diag_indices_from',
  'diagflat',
  'dsplit',
  'dstack',
  'ediff1d',
  'einsum_path',
  'empty_like',
  'equal',
  'expand_dims',
  'expm1',
  'fill_diagonal',
  'fix',
  'fliplr',
  'flipud',
  'floor',
  'fromfunction',
  'fromiter',
  'full_like',
  'greater',
  'greater_equal',
  'histogramdd',
  'hsplit',
  'imag',
  'insert',
  'intersect1d',
  'isclose',
  'iscomplex',
  'isin',
  'isinf',
  'ix_',
  'less',
  'less_equal',
  'log1p',
  'logaddexp2',
  'mask_indices',
  'meshgrid',
  'moveaxis',
  'nanargmax',
  'nanargmin',
  'nancumprod',
  'nancumsum',
  'nanmedian',
  'nanprod',
  'nanstd',
  'nanvar',
  'nextafter',
  'not_equal',
  'ones_like',
  'place',
  'put',
  'put_along_axis',
  'putmask',
  'real',
  'real_if_close',
  'resize',
  'rint',
  'rollaxis',
  'round',
  'select',
  'setdiff1d',
  'setxor1d',
  'sort_complex',
  'spacing',
  'split',
  'squeeze',
  'tensordot',
  'tril_indices_from',
  'triu_indices_from',
  'trunc',
  'union1d',
  'unique',
  'unique_all',
  'unique_inverse',
  'vander',
  'vsplit',
]);

describe('benchmark function coverage', () => {
  const publicFunctions = Object.entries(np)
    .filter(([, v]) => typeof v === 'function')
    .map(([name]) => name);

  /** Bench operations are slugs: `linalg.eigh` is `linalg_eigh`. */
  const benchmarked = (() => {
    const s = new Set(specs.map((x) => x.operation));
    for (const op of [...s]) s.add(op.replace(/^linalg_/, ''));
    return s;
  })();

  it('finds a non-trivial public surface (guards against a broken import)', () => {
    expect(publicFunctions.length).toBeGreaterThan(300);
  });

  it('every public function is benchmarked or explicitly excepted', () => {
    const unaccounted = publicFunctions
      .filter((f) => !benchmarked.has(f))
      .filter((f) => !NOT_BENCHMARKABLE.has(f) && !MISSING_BENCHMARK.has(f))
      .sort();
    expect(
      unaccounted,
      'These public functions have no benchmark and are not declared. Add a ' +
        'spec in benchmarks/src/specs.ts, or add them to NOT_BENCHMARKABLE ' +
        '(with a reason) or MISSING_BENCHMARK (the shrink-only backlog).\n',
    ).toEqual([]);
  });

  it('the MISSING_BENCHMARK backlog has no stale entries', () => {
    const nowCovered = [...MISSING_BENCHMARK].filter((f) => benchmarked.has(f)).sort();
    expect(
      nowCovered,
      'These now have benchmarks — remove them from MISSING_BENCHMARK so the ' +
        'backlog keeps shrinking and stays trustworthy.\n',
    ).toEqual([]);
  });

  it('exception lists do not overlap and reference real exports', () => {
    const both = [...MISSING_BENCHMARK].filter((f) => NOT_BENCHMARKABLE.has(f));
    expect(both, 'listed as both permanently excluded and a todo').toEqual([]);

    const known = new Set(Object.keys(np));
    const unknown = [...NOT_BENCHMARKABLE, ...MISSING_BENCHMARK]
      .filter((f) => !known.has(f))
      .sort();
    expect(
      unknown,
      'These exception entries are not exported by numpy-ts. They are stale ' +
        '(renamed or removed) and silently excuse nothing.\n',
    ).toEqual([]);
  });
});
