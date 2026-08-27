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

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CATEGORY_DTYPE_SUPPORT,
  DATA_ARRAY_KEYS,
  FAMILY_VARIANTS,
  getBenchmarkSpecs,
  INT_ONLY_OPERATIONS,
  PINNED_INDEX_DTYPE_OPERATIONS,
  SKIP_COMPLEX_OPERATIONS,
  SKIP_DTYPE_OPERATIONS,
  SKIP_FLOAT16_OPERATIONS,
  SKIP_INT_OPERATIONS,
  SKIP_INT64_BROADCAST_OPERATIONS,
  SKIP_INT64_OPERATIONS,
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
  const m = new Map<
    string,
    {
      dtypes: Set<string>;
      category: string;
      /** A `value:` fill blocks complex variants (np.full with complex is broken). */
      hasValueFill: boolean;
      /** Mixed operand shapes block complex variants (complex broadcasting is buggy). */
      shapesDiffer: boolean;
      /** A negative `value:` fill blocks uint variants (NumPy raises OverflowError). */
      hasNegativeValue: boolean;
    }
  >();
  for (const spec of specs) {
    let entry = m.get(spec.operation);
    if (!entry) {
      // The first spec seen for an operation is its base; variants follow it.
      const data = Object.entries(spec.setup).filter(([k]) => DATA_ARRAY_KEYS.has(k));
      entry = {
        dtypes: new Set(),
        category: spec.category,
        hasValueFill: data.some(([, e]) => e.value !== undefined),
        shapesDiffer: new Set(data.map(([, e]) => JSON.stringify(e.shape))).size > 1,
        hasNegativeValue: data.some(([, e]) => typeof e.value === 'number' && e.value < 0),
      };
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
  ['SKIP_INT64_OPERATIONS', SKIP_INT64_OPERATIONS],
  ['SKIP_INT64_BROADCAST_OPERATIONS', SKIP_INT64_BROADCAST_OPERATIONS],
  ['INT_ONLY_OPERATIONS', INT_ONLY_OPERATIONS],
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

  it('every dtype absent from an operation is explained by a skip list', () => {
    // Rather than re-deriving what the generator should emit (which would just
    // duplicate its bugs), assert the weaker but sharper property: for each
    // operation, any dtype its category supports but the suite does not run
    // must be accounted for by a named skip list. A silently dropped dtype has
    // no explanation and fails here.
    const FAMILY_OF: Record<string, 'float' | 'int' | 'uint' | 'complex'> = {
      float64: 'float',
      float32: 'float',
      float16: 'float',
      int8: 'int',
      int16: 'int',
      int32: 'int',
      int64: 'int',
      uint8: 'uint',
      uint16: 'uint',
      uint32: 'uint',
      uint64: 'uint',
      complex128: 'complex',
      complex64: 'complex',
    };
    const NARROW = new Set(['int8', 'int16', 'uint8', 'uint16']);
    const WIDE64 = new Set(['int64', 'uint64']);

    const unexplained: string[] = [];
    for (const [operation, meta] of byOperation) {
      const { dtypes, category } = meta;
      const families = CATEGORY_DTYPE_SUPPORT[category];
      if (!families) continue; // category never swept (random, utilities)
      // Declared single-dtype operations are covered by their own test above.
      if (SKIP_DTYPE_OPERATIONS.has(operation)) continue;
      if (PINNED_INDEX_DTYPE_OPERATIONS.has(operation)) continue;
      // Handwritten complex specs already target complex128 directly.
      if (operation.startsWith('complex_')) continue;
      // Integer-only ops (gcd/lcm) take no float or complex variants by design.
      const intOnly = INT_ONLY_OPERATIONS.has(operation);
      const complexBlockedBySetup = meta.hasValueFill || meta.shapesDiffer;

      for (const family of families) {
        for (const { dtype } of FAMILY_VARIANTS[family]) {
          if (dtypes.has(dtype)) continue;
          if (FAMILY_OF[dtype] !== family) continue;
          if (intOnly && family !== 'int' && family !== 'uint') continue;
          const excused =
            (family === 'complex' &&
              (SKIP_COMPLEX_OPERATIONS.has(operation) || complexBlockedBySetup)) ||
            (family === 'uint' && meta.hasNegativeValue) ||
            ((family === 'int' || family === 'uint') && SKIP_INT_OPERATIONS.has(operation)) ||
            (family === 'uint' && SKIP_UINT_OPERATIONS.has(operation)) ||
            (dtype === 'float16' && SKIP_FLOAT16_OPERATIONS.has(operation)) ||
            (NARROW.has(dtype) && SKIP_NARROW_INT_OPERATIONS.has(operation)) ||
            (WIDE64.has(dtype) && SKIP_INT64_OPERATIONS.has(operation)) ||
            (WIDE64.has(dtype) &&
              SKIP_INT64_BROADCAST_OPERATIONS.has(operation) &&
              meta.shapesDiffer);
          if (!excused) unexplained.push(`${operation} (${category}) is missing ${dtype}`);
        }
      }
    }
    expect(
      unexplained,
      'These operations skip a dtype their category supports, with nothing to ' +
        'explain it. Either the spec pins a dtype that blocks the sweep, or the ' +
        'dtype genuinely cannot work — in which case add the operation to the ' +
        'matching SKIP_* list so the omission is on the record.\n',
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
 * Public functions that cannot be cross-validated against NumPy, so they get no
 * benchmark. Distinct from NOT_BENCHMARKABLE: these *do* real work and would be
 * worth measuring — the harness just cannot check them for correctness, and an
 * unvalidated benchmark is worse than none.
 */
// Empty: `einsum_path` was the only entry, and its path now matches NumPy's
// shape exactly (marker string first, contraction pairs addressed against the
// shrinking operand list), so it is cross-validatable like everything else.
const NOT_CROSS_VALIDATABLE = new Set<string>([]);

/**
 * Public functions whose *timing* against NumPy is not a like-for-like
 * measurement of this library, so a ratio would be misleading rather than
 * informative. They still do real array work and are validated for correctness
 * elsewhere — they just get no benchmark.
 */
const NOT_PERF_COMPARABLE = new Set<string>([
  // NumPy calls the callback exactly once, with broadcast index arrays, and
  // does the per-element work in C. Our signature is
  // `(...indices: number[]) => number`, so the callback runs once per element
  // by design. The benchmark therefore measures N JS callback invocations
  // against one vectorised NumPy call — a language-semantics difference, not an
  // engine one. Correctness is covered in tests/validation.
  'fromfunction',
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

  it('every public function is benchmarked, or excepted with a reason', () => {
    const unaccounted = publicFunctions
      .filter((f) => !benchmarked.has(f))
      .filter(
        (f) =>
          !NOT_BENCHMARKABLE.has(f) && !NOT_CROSS_VALIDATABLE.has(f) && !NOT_PERF_COMPARABLE.has(f),
      )
      .sort();
    expect(
      unaccounted,
      'These public functions have no benchmark. Add a spec in ' +
        'benchmarks/src/specs.ts (plus the operation in bench-utils.ts, ' +
        'numpy_benchmark.py, validation.ts and validation.py), or add them to ' +
        'NOT_BENCHMARKABLE / NOT_CROSS_VALIDATABLE / NOT_PERF_COMPARABLE with ' +
        'a reason.\n',
    ).toEqual([]);
  });

  it('no exception entry is stale (all still lack a benchmark)', () => {
    const nowCovered = [...NOT_BENCHMARKABLE, ...NOT_CROSS_VALIDATABLE, ...NOT_PERF_COMPARABLE]
      .filter((f) => benchmarked.has(f))
      .sort();
    expect(
      nowCovered,
      'These are listed as exceptions but now have benchmarks. Remove them so ' +
        'the exception lists stay meaningful.\n',
    ).toEqual([]);
  });

  it('exception lists do not overlap and reference real exports', () => {
    const both = [...NOT_CROSS_VALIDATABLE, ...NOT_PERF_COMPARABLE].filter((f) =>
      NOT_BENCHMARKABLE.has(f),
    );
    expect(both, 'listed in both exception sets').toEqual([]);

    const known = new Set(Object.keys(np));
    const unknown = [...NOT_BENCHMARKABLE, ...NOT_CROSS_VALIDATABLE, ...NOT_PERF_COMPARABLE]
      .filter((f) => !known.has(f))
      .sort();
    expect(
      unknown,
      'These exception entries are not exported by numpy-ts. They are stale ' +
        '(renamed or removed) and silently excuse nothing.\n',
    ).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Coverage baseline
// ---------------------------------------------------------------------------

/**
 * Committed snapshot of which dtypes each operation is benchmarked at.
 *
 * The assertions above compare the generated specs against the generator's own
 * config, so they cannot catch a change to that config — dropping a dtype
 * family from CATEGORY_DTYPE_SUPPORT lowers both the output and the
 * expectation, and nothing fails. This baseline is the independent record:
 * any change in coverage, from any cause, shows up as a diff that has to be
 * reviewed and committed on purpose.
 *
 * Regenerate deliberately (never to "make the test pass"):
 *   UPDATE_BENCH_COVERAGE=1 npx vitest run --project=unit tests/unit/benchmark-coverage.test.ts
 */
// Lives outside tests/unit/ because the unit project's include glob is
// `tests/unit/**` — a data file there is picked up as a (test-less) test file.
const BASELINE_PATH = join(__dirname, '../../benchmarks/dtype-coverage-baseline.json');

describe('benchmark coverage baseline', () => {
  const current: Record<string, string[]> = {};
  for (const [operation, { dtypes }] of [...byOperation].sort(([a], [b]) => (a < b ? -1 : 1))) {
    current[operation] = [...dtypes].sort();
  }

  it('matches the committed dtype-coverage baseline', () => {
    if (process.env.UPDATE_BENCH_COVERAGE) {
      writeFileSync(BASELINE_PATH, `${JSON.stringify(current, null, 2)}\n`);
      return;
    }
    const baseline: Record<string, string[]> = JSON.parse(readFileSync(BASELINE_PATH, 'utf8'));

    const lost: string[] = [];
    const gained: string[] = [];
    for (const [operation, dtypes] of Object.entries(baseline)) {
      const now = new Set(current[operation] ?? []);
      const missing = dtypes.filter((d) => !now.has(d));
      if (!current[operation]) lost.push(`${operation}: benchmark removed entirely`);
      else if (missing.length) lost.push(`${operation}: lost ${missing.join(',')}`);
    }
    for (const [operation, dtypes] of Object.entries(current)) {
      const was = new Set(baseline[operation] ?? []);
      const added = dtypes.filter((d) => !was.has(d));
      if (!baseline[operation])
        gained.push(`${operation}: new benchmark (${dtypes.length} dtypes)`);
      else if (added.length) gained.push(`${operation}: gained ${added.join(',')}`);
    }

    expect(
      lost,
      'Benchmark coverage went DOWN. If that is intended (a dtype genuinely ' +
        'cannot work), add the operation to the matching SKIP_* list and ' +
        'regenerate the baseline with UPDATE_BENCH_COVERAGE=1.\n',
    ).toEqual([]);
    expect(
      gained,
      'Benchmark coverage went UP — nice. Regenerate the baseline with ' +
        'UPDATE_BENCH_COVERAGE=1 so it keeps protecting the new coverage.\n',
    ).toEqual([]);
  });
});
