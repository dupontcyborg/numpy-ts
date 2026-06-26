/**
 * Validates the generated compile-time promotion table (src/common/dtype-promotion.ts)
 * against its single source of truth, the runtime `promoteDTypes` function.
 *
 * Type-level assertions live in tests/types/dtype-promotion.test-d.ts, checked by
 * `pnpm run typecheck:types`. This file covers the runtime guarantees:
 *   1. The generated file is in sync with the generator (drift guard).
 *   2. The promotion matrix the generator emits matches promoteDTypes for all pairs.
 */

import * as fs from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  buildPromotionTable,
  DTYPES,
  generatePromotionSource,
  OUTPUT_FILE,
} from '../../scripts/generate-dtype-promotion';
import { type DType, promoteDTypes } from '../../src/common/dtype';

describe('dtype promotion codegen', () => {
  it('committed src/common/dtype-promotion.ts is up to date', () => {
    const onDisk = fs.readFileSync(OUTPUT_FILE, 'utf8');
    const regenerated = generatePromotionSource();
    expect(onDisk).toBe(regenerated);
  });

  it('generated matrix matches promoteDTypes for all 14x14 pairs', () => {
    const table = buildPromotionTable();
    for (const a of DTYPES) {
      for (const b of DTYPES) {
        expect(table[a]![b]).toBe(promoteDTypes(a, b));
      }
    }
  });

  it('covers every dtype exactly once', () => {
    expect(new Set(DTYPES).size).toBe(DTYPES.length);
    // DTYPES must include every member of the DType union — spot-check the count.
    expect(DTYPES.length).toBe(14);
  });

  it('promotion is commutative (table[a][b] === table[b][a])', () => {
    const table = buildPromotionTable();
    for (const a of DTYPES) {
      for (const b of DTYPES) {
        expect(table[a]![b]).toBe(table[b]![a]);
      }
    }
  });

  // A few anchored cases pinned directly to NumPy's documented behavior, so a
  // regression in promoteDTypes is caught here and not only via the type tests.
  it.each<[DType, DType, DType]>([
    ['int32', 'float32', 'float64'], // mantissa safety
    ['int64', 'uint64', 'float64'], // no wider int exists
    ['int8', 'uint8', 'int16'], // signed+unsigned same size bumps
    ['complex64', 'float64', 'complex128'],
    ['bool', 'int8', 'int8'], // bool yields to the other type
    ['float16', 'int8', 'float16'],
  ])('promoteDTypes(%s, %s) === %s', (a, b, expected) => {
    expect(promoteDTypes(a, b)).toBe(expected);
  });
});
