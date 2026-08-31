/**
 * Async-capable, auto-calibrated timer. Mirrors the methodology in
 * benchmarks/src/runtime-runner.ts (warmup -> calibrate ops/sample to hit a
 * target wall time -> take N samples of mean ms) but awaits an optional
 * materialize step so lazy/async libraries (tfjs, jax-js) are forced to fully
 * compute inside the timed region — the same logical work the eager libraries do.
 */

import type { LibTiming, OpDef } from './types';

const DEBUG = !!process.env['JSLIBS_DEBUG'];
// Defaults match the project's standard methodology; overridable for large sweeps.
const MIN_SAMPLE_TIME_MS = Number(process.env['JSLIBS_SAMPLE_MS'] ?? 100);
const TARGET_SAMPLES = Number(process.env['JSLIBS_SAMPLES'] ?? 5);
const MAX_OPS_PER_SAMPLE = 100_000;

const now = () => globalThis.performance.now();

function mean(a: number[]): number {
  return a.reduce((x, y) => x + y, 0) / a.length;
}
function median(a: number[]): number {
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1]! + s[m]!) / 2 : s[m]!;
}

/** Run + materialize + dispose one result. The timed unit of work. */
async function once(op: OpDef, prepared: unknown): Promise<void> {
  const r = op.run(prepared);
  if (op.materialize) await op.materialize(r);
  op.disposeResult?.(r);
}

async function calibrate(op: OpDef, prepared: unknown): Promise<number> {
  let opsPerSample = 1;
  for (let runs = 0; runs < 10; runs++) {
    const start = now();
    for (let i = 0; i < opsPerSample; i++) await once(op, prepared);
    const elapsed = now() - start;
    if (elapsed >= MIN_SAMPLE_TIME_MS) break;
    if (elapsed < MIN_SAMPLE_TIME_MS / 10) opsPerSample *= 10;
    else if (elapsed < MIN_SAMPLE_TIME_MS / 2) opsPerSample *= 2;
    else {
      opsPerSample = Math.max(
        Math.ceil((opsPerSample * MIN_SAMPLE_TIME_MS) / elapsed),
        opsPerSample + 1,
      );
      break;
    }
  }
  return Math.min(opsPerSample, MAX_OPS_PER_SAMPLE);
}

/** Time one operation. Returns null if the op throws (unsupported param combo). */
export async function timeOp(
  op: OpDef,
  prepared: unknown,
  warmup: number,
  label?: string,
): Promise<LibTiming | null> {
  try {
    for (let i = 0; i < warmup; i++) await once(op, prepared);
    const opsPerSample = await calibrate(op, prepared);
    const sampleMs: number[] = [];
    for (let s = 0; s < TARGET_SAMPLES; s++) {
      const start = now();
      for (let i = 0; i < opsPerSample; i++) await once(op, prepared);
      sampleMs.push((now() - start) / opsPerSample);
    }
    const meanMs = mean(sampleMs);
    return {
      mean_ms: meanMs,
      median_ms: median(sampleMs),
      min_ms: Math.min(...sampleMs),
      ops_per_sec: 1000 / meanMs,
      samples: TARGET_SAMPLES,
    };
  } catch (e) {
    if (DEBUG && label) console.error(`  [N/A] ${label}: ${(e as Error).message.slice(0, 70)}`);
    return null;
  }
}
