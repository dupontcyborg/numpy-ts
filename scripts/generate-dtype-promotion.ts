/**
 * Code Generator for compile-time dtype result-type tables.
 *
 * Single source of truth: the runtime result-dtype functions in
 * src/common/dtype.ts (each validated against NumPy). This script enumerates
 * dtypes and emits a TYPES-ONLY module (`src/common/dtype-promotion.ts`) so the
 * rules are mirrored at the type level with zero runtime cost.
 *
 * Emits one binary table (`Promote`) plus a set of unary tables (MathResult,
 * ReductionAccum, TrueDivide, …) and composed aliases (Power, Divide, MatMul).
 *
 * Run with: npx tsx scripts/generate-dtype-promotion.ts
 *
 * The committed output is guarded by tests/unit/dtype-promotion.test.ts, which
 * regenerates in-memory and fails if it drifts from the checked-in file.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  absResultDtype,
  angleResultDtype,
  boolArithmeticDtype,
  type DType,
  fftRealResultDtype,
  fftResultDtype,
  getComplexComponentDType,
  hasFloat16,
  isComplexDType,
  mathResultDtype,
  promoteDTypes,
  reductionAccumDtype,
  stdVarResultDtype,
  trueDivideResultDtype,
} from '../src/common/dtype';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const OUTPUT_FILE = path.join(__dirname, '../src/common/dtype-promotion.ts');

/** All dtypes, in declaration order (matches the DType union in dtype.ts). */
export const DTYPES: readonly DType[] = [
  'float64',
  'float32',
  'float16',
  'complex128',
  'complex64',
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

/** complex → real component float; every other dtype preserved. */
function complexComponentDtype(dtype: DType): DType {
  return isComplexDType(dtype) ? getComplexComponentDType(dtype) : dtype;
}

/**
 * Registry of unary result-dtype rules. Each becomes a flat type table
 * `interface <table> { [dtype]: result }` plus a `type <alias><D> = <table>[D]`.
 * `fn` is the runtime SSOT — the table is built by calling it over every dtype,
 * so the generated types can never drift from runtime behavior.
 */
export const UNARY_RULES: {
  table: string;
  alias: string;
  fn: (d: DType) => DType;
  doc: string;
}[] = [
  {
    table: 'MathResultTable',
    alias: 'MathResult',
    fn: mathResultDtype,
    doc: 'unary float math (sin, sqrt, exp, …)',
  },
  {
    table: 'ReductionAccumTable',
    alias: 'ReductionAccum',
    fn: reductionAccumDtype,
    doc: 'reduction accumulation (sum, prod, cumsum, …)',
  },
  {
    table: 'TrueDivideTable',
    alias: 'TrueDivide',
    fn: trueDivideResultDtype,
    doc: 'true division / mean / median',
  },
  {
    table: 'StdVarTable',
    alias: 'StdVar',
    fn: stdVarResultDtype,
    doc: 'std / var (spread statistics)',
  },
  { table: 'AbsTable', alias: 'Abs', fn: absResultDtype, doc: 'absolute value' },
  { table: 'AngleTable', alias: 'Angle', fn: angleResultDtype, doc: 'angle (always real float)' },
  {
    table: 'BoolArithTable',
    alias: 'BoolArith',
    fn: boolArithmeticDtype,
    doc: 'bool → int8 arithmetic promotion',
  },
  { table: 'FftResultTable', alias: 'FftResult', fn: fftResultDtype, doc: 'complex-output FFT' },
  {
    table: 'FftRealTable',
    alias: 'FftReal',
    fn: fftRealResultDtype,
    doc: 'real-output FFT (hfft)',
  },
  {
    table: 'ComplexComponentTable',
    alias: 'ComplexComponent',
    fn: complexComponentDtype,
    doc: 'complex → real component (real, imag)',
  },
];

/** Build the full binary promotion matrix by calling the runtime SSOT for every pair. */
export function buildPromotionTable(): Record<DType, Record<DType, DType>> {
  const table = {} as Record<DType, Record<DType, DType>>;
  for (const a of DTYPES) {
    table[a] = {} as Record<DType, DType>;
    for (const b of DTYPES) {
      table[a][b] = promoteDTypes(a, b);
    }
  }
  return table;
}

/** Build a flat unary table by calling `fn` over every dtype. */
export function buildUnaryTable(fn: (d: DType) => DType): Record<DType, DType> {
  const table = {} as Record<DType, DType>;
  for (const d of DTYPES) table[d] = fn(d);
  return table;
}

function renderPromotion(): string {
  const table = buildPromotionTable();
  const rows = DTYPES.map((a) => {
    const cols = DTYPES.map((b) => `    ${b}: '${table[a][b]}';`).join('\n');
    return `  ${a}: {\n${cols}\n  };`;
  }).join('\n');
  return `/**
 * Full dtype promotion matrix. \`PromotionTable[A][B]\` is the dtype NumPy
 * promotes \`A\` and \`B\` to.
 */
export interface PromotionTable {
${rows}
}

/**
 * Result dtype of a binary op between arrays of dtype \`A\` and \`B\`.
 * @example type T = Promote<'int32', 'float32'>; // 'float64'
 */
export type Promote<A extends DType, B extends DType> = PromotionTable[A][B];`;
}

function renderUnary(rule: (typeof UNARY_RULES)[number]): string {
  const table = buildUnaryTable(rule.fn);
  const rows = DTYPES.map((d) => `  ${d}: '${table[d]}';`).join('\n');
  return `/** Result-dtype table for ${rule.doc}. */
export interface ${rule.table} {
${rows}
}
export type ${rule.alias}<D extends DType> = ${rule.table}[D];`;
}

/** Render the complete types-only source module. */
export function generatePromotionSource(): string {
  // The math/real-fft tables encode the canonical float16 rule; guard against
  // baking the float32 fallback on an engine that lacks Float16Array.
  if (!hasFloat16) {
    throw new Error(
      'generate-dtype-promotion: Float16Array is unavailable in this runtime; ' +
        'regenerate on an engine with native float16 so math-result tables stay canonical.',
    );
  }

  const composed = `/**
 * Result dtype for power / mod / floor_divide / remainder: promote, then apply
 * the bool → int8 arithmetic rule.
 */
export type Power<A extends DType, B extends DType> = BoolArith<Promote<A, B>>;

/** Result dtype for true division of two arrays: promote, then true-divide. */
export type Divide<A extends DType, B extends DType> = TrueDivide<Promote<A, B>>;

/** Result dtype for matmul / dot / inner / outer: NumPy promotion. */
export type MatMul<A extends DType, B extends DType> = Promote<A, B>;

/**
 * Result dtype for float-returning binary math (arctan2, hypot, copysign,
 * logaddexp, …): promote, then apply the unary float-math rule.
 */
export type MathBinary<A extends DType, B extends DType> = MathResult<Promote<A, B>>;

/**
 * Result dtype for float_power: promote, then bump to a minimum precision of
 * float64 (integers/float16/float32 → float64; complex → complex128). Modelled
 * as promotion against float64 (float64 preserves floats, upgrades ints/complex).
 */
export type FloatPower<A extends DType, B extends DType> = Promote<Promote<A, B>, 'float64'>;`;

  return `/**
 * AUTO-GENERATED — DO NOT EDIT BY HAND.
 *
 * Compile-time mirror of the runtime result-dtype functions (src/common/dtype.ts),
 * which are the single source of truth. Regenerate with:
 *
 *   npx tsx scripts/generate-dtype-promotion.ts
 *
 * This module is types-only: it contributes zero bytes to the runtime bundle.
 */

import type { DType } from './dtype';

${renderPromotion()}

${UNARY_RULES.map(renderUnary).join('\n\n')}

${composed}
`;
}

/** Write the generated module to disk. */
export function writePromotionSource(): void {
  fs.writeFileSync(OUTPUT_FILE, generatePromotionSource(), 'utf8');
}

// Run when invoked directly (not when imported by tests).
if (import.meta.url === `file://${process.argv[1]}`) {
  writePromotionSource();
  // eslint-disable-next-line no-console
  console.log(`Generated ${path.relative(process.cwd(), OUTPUT_FILE)}`);
}
