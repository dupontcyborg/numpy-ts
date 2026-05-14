/**
 * Dtype cast edge-case sweep, validated byte-for-byte against NumPy.
 *
 * For each (sourceDtype × extremeValue × targetDtype) combination, NumPy is
 * the oracle and ndarray-ts must produce the same result. Boundary values:
 *   ±Infinity, NaN, ±0, ±2^31, ±2^63, uint64_max, int32_max+1, ±1.7
 *
 * Uses its own Python runner instead of the shared numpy-oracle because JSON
 * silently rounds anything past 2^53. Here values are pulled out one at a
 * time and emitted in text form: floats via repr() (preserving NaN/±Inf),
 * integers via Python int() so 18446744073709551615 survives round-trip.
 */

import { execSync } from 'node:child_process';
import { unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { array, ones } from '../../src';
import type { DType } from '../../src/common/dtype';
import { checkNumPyAvailable } from './numpy-oracle';

const PYTHON_CMD = process.env.NUMPY_PYTHON || 'python3';

const TARGET_DTYPES: DType[] = [
  'float64',
  'float32',
  'int64',
  'int32',
  'int16',
  'int8',
  'uint64',
  'uint32',
  'uint16',
  'uint8',
  'bool',
];

const BIGINT_DTYPES = new Set<DType>(['int64', 'uint64']);
const FLOAT_DTYPES = new Set<DType>(['float64', 'float32']);
const INT_DTYPES = new Set<DType>([
  'int8',
  'int16',
  'int32',
  'int64',
  'uint8',
  'uint16',
  'uint32',
  'uint64',
]);

/**
 * NumPy 2.x routes float→int through the platform's native fp-to-int intrinsic.
 * x86 `cvttsd2si` wraps / returns INT_MIN for invalid; arm64 `fcvtzs` saturates
 * / returns 0. So the entire space of "float that doesn't fit losslessly into
 * the target's integer range" is platform-dependent: the same NumPy version
 * produces different oracle values across CI runners.
 *
 * Three kinds of float→int are platform-dependent:
 *   1. NaN / ±Inf → any integer.
 *   2. Negative float (truncated) → unsigned integer.
 *   3. Float whose truncated value falls outside the target's signed range.
 *
 * For these cases we skip the comparison rather than pretend NumPy is an
 * oracle. The library's own behavior (arm64-style saturation, NaN→0) is
 * documented in docs/next/guides/dtypes.mdx; cross-platform parity is not
 * achievable so long as NumPy itself differs.
 */
const PATHOLOGICAL_FLOAT_LABELS = new Set<string>([
  'f64 NaN',
  'f64 +Inf',
  'f64 -Inf',
  'f32 NaN',
  'f32 +Inf',
  'f32 -Inf',
]);

const INT_RANGES: Record<string, { min: bigint; max: bigint }> = {
  int8: { min: -128n, max: 127n },
  int16: { min: -32768n, max: 32767n },
  int32: { min: -2147483648n, max: 2147483647n },
  int64: { min: -9223372036854775808n, max: 9223372036854775807n },
  uint8: { min: 0n, max: 255n },
  uint16: { min: 0n, max: 65535n },
  uint32: { min: 0n, max: 4294967295n },
  uint64: { min: 0n, max: 18446744073709551615n },
};

/**
 * True if casting `c.value` to `tgt` is platform-dependent in NumPy 2.x:
 * NaN/Inf, or a finite float whose truncated value falls outside `tgt`'s range.
 */
function isPlatformDependentFloatToInt(c: Case, tgt: DType): boolean {
  if (!FLOAT_DTYPES.has(c.src) || !INT_DTYPES.has(tgt)) return false;
  if (PATHOLOGICAL_FLOAT_LABELS.has(c.label)) return true;
  const v = c.value as number;
  if (!Number.isFinite(v)) return true;
  const truncated = BigInt(Math.trunc(v));
  const range = INT_RANGES[tgt]!;
  return truncated < range.min || truncated > range.max;
}

interface Case {
  label: string;
  src: DType;
  /** Source value passed to numpy-ts. BigInt for int64/uint64. */
  value: number | bigint;
  /** Python expression returning the same scalar, with matching np dtype. */
  pyValue: string;
}

const CASES: Case[] = [
  { label: 'f64 NaN', src: 'float64', value: NaN, pyValue: 'np.float64("nan")' },
  { label: 'f64 +Inf', src: 'float64', value: Infinity, pyValue: 'np.float64("inf")' },
  { label: 'f64 -Inf', src: 'float64', value: -Infinity, pyValue: 'np.float64("-inf")' },
  { label: 'f64 -0', src: 'float64', value: -0, pyValue: 'np.float64(-0.0)' },
  { label: 'f64 1.7', src: 'float64', value: 1.7, pyValue: 'np.float64(1.7)' },
  { label: 'f64 -1.7', src: 'float64', value: -1.7, pyValue: 'np.float64(-1.7)' },
  {
    label: 'f64 int32_max+1',
    src: 'float64',
    value: 2147483648,
    pyValue: 'np.float64(2147483648.0)',
  },
  {
    label: 'f64 -int32_max-1',
    src: 'float64',
    value: -2147483649,
    pyValue: 'np.float64(-2147483649.0)',
  },
  { label: 'f32 NaN', src: 'float32', value: NaN, pyValue: 'np.float32("nan")' },
  { label: 'f32 +Inf', src: 'float32', value: Infinity, pyValue: 'np.float32("inf")' },
  { label: 'f32 -Inf', src: 'float32', value: -Infinity, pyValue: 'np.float32("-inf")' },

  // Integer source boundaries — bigint literals so JS doesn't truncate them.
  {
    label: 'i64 max',
    src: 'int64',
    value: 9223372036854775807n,
    pyValue: 'np.int64(9223372036854775807)',
  },
  {
    label: 'i64 min',
    src: 'int64',
    value: -9223372036854775808n,
    pyValue: 'np.int64(-9223372036854775808)',
  },
  {
    label: 'u64 max',
    src: 'uint64',
    value: 18446744073709551615n,
    pyValue: 'np.uint64(18446744073709551615)',
  },
  { label: 'i32 max', src: 'int32', value: 2147483647, pyValue: 'np.int32(2147483647)' },
  { label: 'i32 min', src: 'int32', value: -2147483648, pyValue: 'np.int32(-2147483648)' },
];

/**
 * Known cast divergences from NumPy 2.x — keep empty unless a new divergence
 * surfaces. Each entry is `${case.label}__${tgt}`; matched entries become
 * `it.todo` so they show in CI output without failing the suite.
 *
 * Two earlier clusters (float→narrow-int saturation; bigint→narrow-int via
 * Number()) were resolved in src/common/ndarray-core.ts: floatToInt now
 * saturates to int32 range before bit-truncation (NumPy 2.x's actual path),
 * and the bigint→non-bigint cast masks to 32 bits in BigInt-space before
 * Number() so the TypedArray store can finish the truncation.
 */
const KNOWN_DIVERGENCES = new Set<string>();

interface OracleEntry {
  /** Raw text from Python — int literal, float repr, or one of the markers. */
  text: string;
}

/**
 * Run a batch of cast expressions in one Python subprocess. Each snippet sets
 * `result` to a 1-element NumPy array; we extract `result[0]` and emit a
 * key=text line per entry so JSON precision never enters the picture.
 */
function pythonCastBatch(
  snippets: Record<string, { src: DType; tgt: DType; py: string }>,
): Map<string, OracleEntry> {
  const entries = Object.entries(snippets);
  if (entries.length === 0) return new Map();

  const lines: string[] = [
    'import numpy as np, sys, math, warnings',
    'warnings.simplefilter("ignore")  # silence float→int overflow warnings',
    'def emit(key, v):',
    '    if isinstance(v, (np.floating, float)):',
    '        if math.isnan(v): t = "__NaN__"',
    '        elif math.isinf(v): t = "__+Inf__" if v > 0 else "__-Inf__"',
    '        elif v == 0.0 and math.copysign(1.0, v) < 0: t = "__-0__"',
    '        else: t = repr(float(v))',
    '    elif isinstance(v, (np.bool_, bool)):',
    '        t = "1" if bool(v) else "0"',
    '    elif isinstance(v, (np.integer, int)):',
    '        t = str(int(v))',
    '    else:',
    '        t = repr(v)',
    '    print(f"{key}\\t{t}")',
  ];
  for (const [key, snip] of entries) {
    lines.push('try:');
    lines.push(`    result = np.array([${snip.py}], dtype='${snip.src}').astype('${snip.tgt}')`);
    lines.push(`    emit(${JSON.stringify(key)}, result[0])`);
    lines.push('except Exception as e:');
    lines.push(`    print(f"${key.replace(/"/g, '\\"')}\\t__ERR__\\t{e}")`);
  }

  const tmp = join(tmpdir(), `cast-sweep-${Date.now()}-${Math.random().toString(36).slice(2)}.py`);
  writeFileSync(tmp, lines.join('\n'), 'utf-8');
  try {
    const out = execSync(`${PYTHON_CMD} ${tmp}`, { encoding: 'utf-8' });
    const map = new Map<string, OracleEntry>();
    for (const line of out.split('\n')) {
      if (!line) continue;
      const [key, ...rest] = line.split('\t');
      map.set(key!, { text: rest.join('\t') });
    }
    return map;
  } finally {
    try {
      unlinkSync(tmp);
    } catch {}
  }
}

function parseOracle(text: string, dtype: DType): number | bigint {
  if (text === '__NaN__') return NaN;
  if (text === '__+Inf__') return Infinity;
  if (text === '__-Inf__') return -Infinity;
  if (text === '__-0__') return -0;
  if (text.startsWith('__ERR__')) throw new Error(`oracle error: ${text}`);
  if (BIGINT_DTYPES.has(dtype)) return BigInt(text);
  return Number(text);
}

function readScalar(arr: ReturnType<typeof array>, dtype: DType): number | bigint {
  const v = arr.get([0]) as number | bigint;
  if (BIGINT_DTYPES.has(dtype)) {
    return typeof v === 'bigint' ? v : BigInt(v as number);
  }
  return typeof v === 'number' ? v : Number(v);
}

function buildSourceArray(c: Case): ReturnType<typeof array> {
  if (BIGINT_DTYPES.has(c.src)) {
    // array([bigint], 'uint64') may pre-coerce in some paths; build via
    // ones() + set() so the TypedArray holds the raw 2^63-class value.
    const src = ones([1], c.src);
    src.set([0], c.value as bigint);
    return src;
  }
  return array([c.value as number], c.src);
}

describe('NumPy Validation: dtype cast edge-case sweep', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
  });

  const snippets: Record<string, { src: DType; tgt: DType; py: string }> = {};
  for (const c of CASES) {
    for (const tgt of TARGET_DTYPES) {
      snippets[`${c.label}__${tgt}`] = { src: c.src, tgt, py: c.pyValue };
    }
  }
  const oracle = pythonCastBatch(snippets);

  for (const c of CASES) {
    describe(`${c.src} (${c.label})`, () => {
      for (const tgt of TARGET_DTYPES) {
        const key = `${c.label}__${tgt}`;
        const fn = () => {
          const entry = oracle.get(key);
          if (!entry) throw new Error(`oracle missing entry for ${key}`);
          const want = parseOracle(entry.text, tgt);

          const src = buildSourceArray(c);
          const cast = src.astype(tgt);
          expect(cast.dtype).toBe(tgt);
          const got = readScalar(cast, tgt);

          if (BIGINT_DTYPES.has(tgt)) {
            expect(got, `${c.label} → ${tgt}`).toBe(want);
          } else {
            expect(
              Object.is(got, want),
              `${c.label} → ${tgt}: got ${String(got)}, want ${String(want)}`,
            ).toBe(true);
          }
        };

        if (KNOWN_DIVERGENCES.has(key)) {
          // Surface the divergence in test output but don't fail the suite.
          // Re-evaluate periodically: when the cast path is fixed, remove the
          // entry from KNOWN_DIVERGENCES and this becomes a normal `it`.
          it.todo(`→ ${tgt} (known divergence — saturation/narrowing path)`);
        } else if (isPlatformDependentFloatToInt(c, tgt)) {
          // NumPy 2.x diverges across x86 / arm64 here (see header comment by
          // isPlatformDependentFloatToInt). Skip rather than pretend we have
          // a stable oracle.
          it.skip(`→ ${tgt} (platform-dependent in NumPy 2.x)`);
        } else {
          it(`→ ${tgt}`, fn);
        }

        // FLOAT_DTYPES is consulted by buildSourceArray's else branch (string
        // dtypes) but tsc flags it as "declared and never read" if we lose
        // the reference. Keep the set to document intent.
        void FLOAT_DTYPES;
      }
    });
  }
});
