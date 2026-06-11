/** Console + JSON reporting for the JS-library comparison. */

import type { JsLibAdapter, Regime, SpecResult } from './types';

export function geomean(values: number[]): number {
  const v = values.filter((x) => Number.isFinite(x) && x > 0);
  if (v.length === 0) return Number.NaN;
  return Math.exp(v.reduce((s, x) => s + Math.log(x), 0) / v.length);
}

/** geomean of ops/sec for a participant across the specs it ran. */
function geoOps(results: SpecResult[], name: string): number {
  return geomean(results.map((r) => r.timings[name]?.ops_per_sec).filter((x): x is number => x != null));
}

const pad = (s: string, n: number) => s.padEnd(n);
const padL = (s: string, n: number) => s.padStart(n);

/** Human ops/sec: 1.2M, 345K, 87. */
function fmtOps(x: number): string {
  if (!Number.isFinite(x)) return '·';
  if (x >= 1e6) return `${(x / 1e6).toFixed(2)}M`;
  if (x >= 1e3) return `${(x / 1e3).toFixed(1)}K`;
  return x.toFixed(0);
}

export interface LibSummary {
  name: string;
  pkg: string;
  version: string;
  nativeDtypes: number;
  coverage: number;
  /** geomean ops/sec over the specs this library ran (independent, absolute). */
  geomeanOpsPerSec: number;
  /** geomean of per-spec (lib / numpy-ts) ratio over shared specs; >1 == slower. */
  geomeanRatio: number;
}

export interface ReportSummary {
  regime: Regime;
  totalSpecs: number;
  reference: { name: string; nativeDtypes: number; geomeanOpsPerSec: number };
  libs: LibSummary[];
}

export function buildSummary(
  regime: Regime,
  results: SpecResult[],
  adapters: JsLibAdapter[],
  numpytsDtypes: number,
): ReportSummary {
  const libs: LibSummary[] = adapters
    .filter((a) => a.regimes.includes(regime))
    .map((a) => {
      const ratios: number[] = [];
      for (const r of results) if (r.ratios[a.name] != null) ratios.push(r.ratios[a.name]!);
      return {
        name: a.name,
        pkg: a.pkg,
        version: a.version ?? '?',
        nativeDtypes: a.dtypes.length,
        coverage: ratios.length,
        geomeanOpsPerSec: geoOps(results, a.name),
        geomeanRatio: geomean(ratios),
      };
    })
    .filter((l) => l.coverage > 0);
  return {
    regime,
    totalSpecs: results.length,
    reference: { name: 'numpy-ts', nativeDtypes: numpytsDtypes, geomeanOpsPerSec: geoOps(results, 'numpy-ts') },
    libs,
  };
}

export function printReport(summary: ReportSummary, results: SpecResult[]): void {
  const { regime, totalSpecs, reference, libs } = summary;
  console.log(`\n${'='.repeat(76)}`);
  console.log(`  numpy-ts vs JS numerical libraries — regime: ${regime}`);
  console.log(`  ${totalSpecs} benchmarked specs · headline = geomean ops/sec (higher is faster)`);
  console.log('='.repeat(76));

  // Headline: each party independent (absolute ops/sec), ratio vs numpy-ts as a side note.
  console.log(
    '\n' + pad('library', 14) + padL('version', 10) + padL('coverage', 11) +
    padL('dtypes', 8) + padL('ops/sec', 11) + '   vs numpy-ts',
  );
  console.log('-'.repeat(76));
  console.log(
    pad('numpy-ts', 14) + padL('—', 10) + padL(`${totalSpecs}/${totalSpecs}`, 11) +
    padL(String(reference.nativeDtypes), 8) + padL(fmtOps(reference.geomeanOpsPerSec), 11) + '   (baseline)',
  );
  for (const l of [...libs].sort((a, b) => b.geomeanOpsPerSec - a.geomeanOpsPerSec)) {
    const g = l.geomeanRatio;
    const vs = g < 1 ? `${(1 / g).toFixed(2)}x faster` : `${g.toFixed(2)}x slower`;
    console.log(
      pad(l.name, 14) + padL(l.version, 10) + padL(`${l.coverage}/${totalSpecs}`, 11) +
      padL(String(l.nativeDtypes), 8) + padL(fmtOps(l.geomeanOpsPerSec), 11) + `   ${vs}`,
    );
  }

  // Per-category grid of absolute geomean ops/sec.
  const cats = [...new Set(results.map((r) => r.category))].sort();
  const names = ['numpy-ts', ...libs.map((l) => l.name)];
  console.log('\nPer-category geomean ops/sec (higher = faster; · = not supported):');
  console.log('-'.repeat(76));
  console.log(pad('category', 13) + names.map((n) => padL(n.slice(0, 9), 11)).join(''));
  for (const cat of cats) {
    const rows = results.filter((r) => r.category === cat);
    const cells = names.map((n) =>
      padL(fmtOps(geomean(rows.map((r) => r.timings[n]?.ops_per_sec).filter((x): x is number => x != null))), 11),
    );
    console.log(pad(cat, 13) + cells.join(''));
  }
  console.log('');
}
