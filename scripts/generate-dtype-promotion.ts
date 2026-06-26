/**
 * Code Generator for compile-time dtype promotion types.
 *
 * Single source of truth: the runtime `promoteDTypes` function in
 * src/common/dtype.ts (validated against NumPy). This script enumerates all
 * dtype pairs and emits a TYPES-ONLY module so the promotion rules are mirrored
 * at the type level without any runtime cost (nothing is emitted to JS).
 *
 * Run with: npx tsx scripts/generate-dtype-promotion.ts
 *
 * The committed output is guarded by tests/unit/dtype-promotion.test.ts, which
 * regenerates in-memory and fails if it drifts from the checked-in file.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { type DType, promoteDTypes } from '../src/common/dtype';

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

/** Build the full promotion matrix by calling the runtime SSOT for every pair. */
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

/** Render the types-only source module from the promotion matrix. */
export function generatePromotionSource(): string {
  const table = buildPromotionTable();

  const rows = DTYPES.map((a) => {
    const cols = DTYPES.map((b) => `    ${b}: '${table[a][b]}';`).join('\n');
    return `  ${a}: {\n${cols}\n  };`;
  }).join('\n');

  return `/**
 * AUTO-GENERATED — DO NOT EDIT BY HAND.
 *
 * Compile-time mirror of the runtime \`promoteDTypes\` function (src/common/dtype.ts),
 * which is the single source of truth. Regenerate with:
 *
 *   npx tsx scripts/generate-dtype-promotion.ts
 *
 * This module is types-only: it contributes zero bytes to the runtime bundle.
 */

import type { DType } from './dtype';

/**
 * Full dtype promotion matrix as a type. \`PromotionTable[A][B]\` is the dtype
 * that NumPy promotes \`A\` and \`B\` to.
 */
export interface PromotionTable {
${rows}
}

/**
 * Result dtype of a binary operation between arrays of dtype \`A\` and \`B\`,
 * following NumPy's type-promotion rules.
 *
 * @example
 * type T = Promote<'int32', 'float32'>; // 'float64'
 */
export type Promote<A extends DType, B extends DType> = PromotionTable[A][B];
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
