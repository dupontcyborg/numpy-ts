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

function prepareRecord(d: SpecData): Record<string, any> {
  const record: Record<string, any> = { ...d.params };
  for (const [key, a] of Object.entries(d.arrays)) {
    const flat = np.array(a.data, a.dtype as any);
    record[key] = a.shape.length > 1 ? np.reshape(flat, a.shape) : flat;
  }
  return record;
}

const disposePrepared = (record: unknown) => {
  for (const v of Object.values(record as Record<string, any>)) (v as any)?.dispose?.();
};

/** Build a numpy-ts OpDef for one operation + its SpecData. */
export function numpytsOpDef(operation: string, d: SpecData): OpDef {
  // transpose is an O(1) view; materialize a contiguous copy so it's comparable
  // to the competitors (which copy), and dispose the intermediate view.
  if (operation === 'transpose') {
    return {
      prepare: () => prepareRecord(d),
      run(record) {
        const t = np.transpose((record as any).a);
        const c = np.copy(t);
        t.dispose();
        return c;
      },
      disposeResult: releaseResult,
      disposePrepared,
    };
  }
  return {
    prepare: () => prepareRecord(d),
    run: (record) => executeOperation(operation, record as Record<string, any>),
    disposeResult: releaseResult, // numpy-ts is eager — no materialize needed
    disposePrepared,
  };
}
