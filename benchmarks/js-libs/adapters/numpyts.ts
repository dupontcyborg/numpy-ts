/**
 * numpy-ts reference adapter. Rebuilds native numpy-ts arrays from the shared
 * SpecData and runs the canonical executeOperation() dispatch — full fidelity
 * with the project's own benchmark path. This is the baseline every competitor
 * ratio is measured against.
 */

import * as np from '../../../src/core';
import { executeOperation } from '../../src/bench-utils';
import type { OpDef, SpecData } from '../lib/types';

function releaseResult(r: unknown): void {
  if (r == null) return;
  const d = (r as any).dispose;
  if (typeof d === 'function') (r as any).dispose();
  else if (r instanceof Map) for (const v of r.values()) releaseResult(v);
  else if (Array.isArray(r)) for (const v of r) releaseResult(v);
  else if (typeof r === 'object' && !(ArrayBuffer.isView(r) || r instanceof ArrayBuffer)) {
    for (const v of Object.values(r as Record<string, unknown>)) releaseResult(v);
  }
}

/** Build a numpy-ts OpDef for one operation + its SpecData. */
export function numpytsOpDef(operation: string, d: SpecData): OpDef {
  return {
    prepare() {
      const record: Record<string, any> = { ...d.params };
      for (const [key, a] of Object.entries(d.arrays)) {
        const flat = np.array(a.data, a.dtype as any);
        record[key] = a.shape.length > 1 ? np.reshape(flat, a.shape) : flat;
      }
      return record;
    },
    run(record) {
      return executeOperation(operation, record as Record<string, any>);
    },
    // numpy-ts is eager — no materialize needed.
    disposeResult: releaseResult,
    disposePrepared(record) {
      for (const v of Object.values(record as Record<string, any>)) (v as any)?.dispose?.();
    },
  };
}
