/**
 * Coverage-aware runner: numpy-ts vs third-party JS numerical libraries.
 *
 *   npm run bench:js-libs                  # float64 regime, core ops
 *   npm run bench:js-libs -- --regime float32
 *   npm run bench:js-libs -- --size small --op add,multiply,matmul
 *
 * For each benchmarked spec it times numpy-ts (reference) plus every library
 * whose adapter declares support, using one auto-calibrated, async-aware timer.
 * Ratios are lib-mean / numpy-ts-mean. Results print to console and write to
 * benchmarks/results/js-libs/.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getBenchmarkSpecs } from '../src/specs';
import type { SizeScale } from '../src/types';
import { ADAPTERS } from './adapters/index';
import { numpytsOpDef } from './adapters/numpyts';
import { extractSpecData } from './lib/specdata';
import { buildSummary, printReport, type ReportSummary } from './lib/report';
import { timeOp } from './lib/timing';
import type { JsLibAdapter, Regime, SpecResult } from './lib/types';

// Operations implemented across the adapters (the fair-comparison core).
// Extend by adding the op to the relevant adapters' `ops` maps.
const CORE_OPS = new Set([
  'add', 'subtract', 'multiply', 'divide', 'power', 'negative', 'absolute', 'square',
  'sqrt', 'exp', 'log', 'log2', 'log10', 'sin', 'cos', 'tan',
  'sum', 'mean', 'max', 'min', 'prod', 'std', 'var',
  'matmul', 'transpose', 'dot',
]);

const NUMPYTS_DTYPES = 14;

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (flag: string, def?: string) => {
    const i = a.indexOf(flag);
    return i >= 0 && a[i + 1] ? a[i + 1]! : def;
  };
  return {
    regime: (get('--regime', 'float64') as Regime),
    size: (get('--size', 'default') as SizeScale),
    ops: get('--op')?.split(',').map((s) => s.trim()),
    warmup: Number(get('--warmup', '8')),
  };
}

async function main() {
  const { regime, size, ops, warmup } = parseArgs();
  const opFilter = ops ? new Set(ops) : CORE_OPS;

  const here = dirname(fileURLToPath(import.meta.url));
  const adapters = ADAPTERS.filter((a) => a.regimes.includes(regime));
  for (const a of adapters) {
    try {
      const pj = resolve(here, 'node_modules', a.pkg, 'package.json');
      a.version = JSON.parse(readFileSync(pj, 'utf-8')).version;
    } catch { /* version is best-effort */ }
    if (a.init) await a.init();
  }
  console.error(`regime=${regime} size=${size} adapters=[${adapters.map((a) => a.name).join(', ')}]`);

  const specs = getBenchmarkSpecs('full', size).filter((s) => opFilter.has(s.operation));
  const results: SpecResult[] = [];

  for (const spec of specs) {
    const data = extractSpecData(spec, regime);
    if (!data) continue;

    // numpy-ts reference timing.
    const refDef = numpytsOpDef(spec.operation, data);
    let refPrepared: unknown;
    try {
      refPrepared = refDef.prepare(data);
    } catch {
      continue;
    }
    const ref = await timeOp(refDef, refPrepared, warmup);
    refDef.disposePrepared?.(refPrepared);
    if (!ref) continue;

    const timings: SpecResult['timings'] = { 'numpy-ts': ref };
    const ratios: SpecResult['ratios'] = {};

    for (const a of adapters) {
      const op = a.ops[spec.operation];
      if (!op) continue;
      let prepared: unknown;
      try {
        prepared = op.prepare(data);
      } catch {
        continue;
      }
      const t = await timeOp(op, prepared, warmup);
      op.disposePrepared?.(prepared);
      if (t) {
        timings[a.name] = t;
        ratios[a.name] = t.mean_ms / ref.mean_ms;
      }
    }

    results.push({
      name: spec.name,
      category: spec.category,
      operation: spec.operation,
      timings,
      ratios,
    });
    process.stderr.write('.');
  }
  process.stderr.write('\n');

  const summary = buildSummary(regime, results, adapters, NUMPYTS_DTYPES);
  printReport(summary, results);
  writeResults(regime, summary, results, adapters);
}

function writeResults(
  regime: Regime,
  summary: ReportSummary,
  results: SpecResult[],
  adapters: JsLibAdapter[],
) {
  const here = dirname(fileURLToPath(import.meta.url));
  const outDir = resolve(here, '../results/js-libs');
  mkdirSync(outDir, { recursive: true });
  const out = {
    regime,
    node: process.version,
    libs: adapters.map((a) => ({ name: a.name, pkg: a.pkg, version: a.version, dtypes: a.dtypes })),
    summary,
    results,
  };
  const file = resolve(outDir, `latest-${regime}.json`);
  writeFileSync(file, JSON.stringify(out, null, 2));
  console.error(`wrote ${file}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
