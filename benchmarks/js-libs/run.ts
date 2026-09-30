/**
 * Coverage-aware runner: numpy-ts vs third-party JS numerical libraries.
 *
 *   pnpm run bench:js-libs                       # float64, core ops
 *   pnpm run bench:js-libs -- --dtype float32,int32   # one run per dtype
 *   pnpm run bench:js-libs -- --size small --op add,multiply,matmul
 *
 * Each run targets one dtype: every participant runs at that dtype, and a library
 * runs only if it natively supports it (no coercion) — int64/uint64/complex are
 * numpy-ts-only showcases. For each spec it times numpy-ts (reference) plus every
 * supporting library via one auto-calibrated, async-aware timer. Headline is
 * absolute geomean ops/sec; ratios are lib-mean / numpy-ts-mean. Writes per-dtype
 * JSON to benchmarks/results/js-libs/.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getBenchmarkSpecs } from '../src/specs';
import type { SizeScale } from '../src/types';
import { ADAPTERS } from './adapters/index';
import { numpytsOpDef } from './adapters/numpyts';
import { extractSpecData } from './lib/specdata';
import { buildSummary, printReport } from './lib/report';
import { timeOp } from './lib/timing';
import type { Dtype, SpecResult } from './lib/types';

// Every operation implemented by at least one adapter. numpy-ts baselines any of
// them via executeOperation; specs for ops no adapter implements are skipped.
const IMPLEMENTED_OPS = new Set(ADAPTERS.flatMap((a) => Object.keys(a.ops)));

const NUMPYTS_DTYPES = 14;
const DEBUG = !!process.env['JSLIBS_DEBUG'];

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (flag: string, def?: string) => {
    const i = a.indexOf(flag);
    return i >= 0 && a[i + 1] ? a[i + 1]! : def;
  };
  // --dtype accepts a comma list (run once per dtype). --regime kept as an alias.
  const dtypeArg = get('--dtype') ?? get('--regime', 'float64')!;
  return {
    dtypes: dtypeArg.split(',').map((s) => s.trim()) as Dtype[],
    size: (get('--size', 'default') as SizeScale),
    ops: get('--op')?.split(',').map((s) => s.trim()),
    warmup: Number(get('--warmup', '8')),
  };
}

/** Dispose prepared inputs; never let a cleanup error abort the run. */
function safeDispose(op: { disposePrepared?: (p: unknown) => void }, prepared: unknown): void {
  try {
    op.disposePrepared?.(prepared);
  } catch { /* a double-dispose etc. must not crash the sweep */ }
}

/** A "base" spec carries no baked-in dtype — we override it to each target. */
function isBaseSpec(spec: { setup: Record<string, any> }): boolean {
  return !Object.values(spec.setup).some(
    (s) => s && typeof s === 'object' && 'shape' in s && (s as any).dtype,
  );
}

async function main() {
  const { dtypes, size, ops, warmup } = parseArgs();
  const opFilter = ops ? new Set(ops) : IMPLEMENTED_OPS;
  const here = dirname(fileURLToPath(import.meta.url));

  // Version + one-time init for every adapter (regardless of dtype).
  for (const a of ADAPTERS) {
    for (const base of [resolve(here, 'node_modules'), resolve(here, '../../node_modules')]) {
      try {
        a.version = JSON.parse(readFileSync(resolve(base, a.pkg, 'package.json'), 'utf-8')).version;
        break;
      } catch { /* try next / best-effort */ }
    }
    if (a.init) await a.init();
  }

  // Base specs only — dtype-variant specs are redundant once we override dtype.
  const baseSpecs = getBenchmarkSpecs('full', size).filter(
    (s) => opFilter.has(s.operation) && isBaseSpec(s),
  );

  for (const dtype of dtypes) {
    await runDtype(dtype, baseSpecs, warmup, here);
  }
}

async function runDtype(
  dtype: Dtype,
  baseSpecs: ReturnType<typeof getBenchmarkSpecs>,
  warmup: number,
  here: string,
) {
  // Participation gate: a library runs this dtype only if it natively supports it.
  const adapters = ADAPTERS.filter((a) => a.dtypes.includes(dtype));
  console.error(
    `\ndtype=${dtype} adapters=[${adapters.map((a) => a.name).join(', ') || '(numpy-ts only — capability showcase)'}]`,
  );

  const results: SpecResult[] = [];
  for (const spec of baseSpecs) {
    const data = extractSpecData(spec, dtype);
    if (!data) continue;

    const refDef = numpytsOpDef(spec.operation, data);
    let refPrepared: unknown;
    try {
      refPrepared = refDef.prepare(data);
    } catch {
      continue;
    }
    const ref = await timeOp(refDef, refPrepared, warmup);
    safeDispose(refDef, refPrepared);
    if (!ref) continue;

    const timings: SpecResult['timings'] = { 'numpy-ts': ref };
    const ratios: SpecResult['ratios'] = {};

    for (const a of adapters) {
      const op = a.ops[spec.operation];
      if (!op) continue;
      const label = `${a.name} ${spec.name}`;
      let prepared: unknown;
      try {
        prepared = op.prepare(data);
      } catch (e) {
        if (DEBUG) console.error(`  [N/A] ${label}: ${(e as Error).message.slice(0, 70)}`);
        continue;
      }
      const t = await timeOp(op, prepared, warmup, label);
      safeDispose(op, prepared);
      if (t) {
        timings[a.name] = t;
        ratios[a.name] = t.mean_ms / ref.mean_ms;
      }
    }

    results.push({ name: spec.name, category: spec.category, operation: spec.operation, timings, ratios });
    process.stderr.write('.');
  }
  process.stderr.write('\n');

  const summary = buildSummary(dtype, results, adapters, NUMPYTS_DTYPES);
  printReport(summary, results);

  const outDir = resolve(here, '../results/js-libs');
  mkdirSync(outDir, { recursive: true });
  const out = {
    dtype,
    node: process.version,
    libs: adapters.map((a) => ({ name: a.name, pkg: a.pkg, version: a.version, dtypes: a.dtypes })),
    summary,
    results,
  };
  const file = resolve(outDir, `latest-${dtype}.json`);
  writeFileSync(file, JSON.stringify(out, null, 2));
  console.error(`wrote ${file}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
