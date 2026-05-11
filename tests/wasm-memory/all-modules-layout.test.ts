/**
 * Static-layout invariant: every checked-in WASM module shares one linear
 * memory (env.memory). Modules compiled with --global-base place their data
 * segments at distinct offsets so initialization does not clobber another
 * module's constants. This test parses each `bins/*.wasm.ts` binary, extracts
 * its active data-segment ranges, and asserts pairwise disjointness across
 * all modules. It also pins the upper bound below MIN_HEAP_BASE so we notice
 * if the cumulative static footprint creeps toward the runtime heap floor.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const BINS_DIR = join(__dirname, '..', '..', 'src', 'common', 'wasm', 'bins');
const MIN_HEAP_BASE = 8 * 1024 * 1024; // must match runtime.ts

function extractB64FromModule(filePath: string): string {
  const src = readFileSync(filePath, 'utf-8');
  const match = src.match(/const B64\s*=\s*\n?\s*'([^']+)'/);
  if (!match) throw new Error(`Could not locate B64 constant in ${filePath}`);
  return match[1]!;
}

function decodeB64(b64: string): Uint8Array {
  const raw =
    typeof atob !== 'undefined' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

interface DataRange {
  module: string;
  start: number;
  end: number; // exclusive
}

class Reader {
  pos = 0;
  constructor(public buf: Uint8Array) {}
  u8(): number {
    return this.buf[this.pos++]!;
  }
  uleb(): number {
    let result = 0;
    let shift = 0;
    while (true) {
      const b = this.u8();
      result |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) return result >>> 0;
      shift += 7;
    }
  }
  sleb(): number {
    let result = 0;
    let shift = 0;
    let b = 0;
    do {
      b = this.u8();
      result |= (b & 0x7f) << shift;
      shift += 7;
    } while (b & 0x80);
    if (shift < 32 && (b & 0x40) !== 0) result |= -(1 << shift);
    return result;
  }
}

function parseDataSegments(bytes: Uint8Array, name: string): DataRange[] {
  const r = new Reader(bytes);
  // magic + version (8 bytes); validate magic so a corrupt embed fails loudly.
  if (r.u8() !== 0x00 || r.u8() !== 0x61 || r.u8() !== 0x73 || r.u8() !== 0x6d) {
    throw new Error(`${name}: not a wasm module`);
  }
  r.pos = 8;

  const ranges: DataRange[] = [];

  while (r.pos < r.buf.length) {
    const id = r.u8();
    const size = r.uleb();
    const sectionEnd = r.pos + size;

    if (id === 11) {
      const count = r.uleb();
      for (let i = 0; i < count; i++) {
        const flags = r.uleb();
        if ((flags & 0x01) === 0) {
          if ((flags & 0x02) !== 0) r.uleb(); // explicit memory index
          const op = r.u8();
          let offset = 0;
          if (op === 0x41) {
            offset = r.sleb();
          } else if (op === 0x23) {
            // global.get init — not expected for our modules; bail on this
            // segment but keep parsing.
            r.uleb();
          }
          // consume init-expression terminator (and any extra ops)
          while (r.u8() !== 0x0b) {}
          const segSize = r.uleb();
          if (segSize > 0 && op === 0x41) {
            ranges.push({ module: name, start: offset, end: offset + segSize });
          }
          r.pos += segSize;
        } else {
          const segSize = r.uleb();
          r.pos += segSize;
        }
      }
    }

    r.pos = sectionEnd;
  }

  return ranges;
}

describe('WASM bins static-region layout', () => {
  const files = readdirSync(BINS_DIR).filter((f) => f.endsWith('.wasm.ts'));

  it('discovers all checked-in modules', () => {
    expect(files.length).toBeGreaterThan(100);
  });

  it('every module parses as a valid wasm binary', () => {
    for (const f of files) {
      const bytes = decodeB64(extractB64FromModule(join(BINS_DIR, f)));
      // parseDataSegments validates the magic header and walks sections.
      expect(() => parseDataSegments(bytes, f)).not.toThrow();
    }
  });

  it('no two modules have overlapping data segments', () => {
    const all: DataRange[] = [];
    for (const f of files) {
      const bytes = decodeB64(extractB64FromModule(join(BINS_DIR, f)));
      all.push(...parseDataSegments(bytes, f));
    }

    all.sort((a, b) => a.start - b.start || a.end - b.end);
    const overlaps: string[] = [];
    for (let i = 1; i < all.length; i++) {
      const prev = all[i - 1]!;
      const cur = all[i]!;
      if (cur.module === prev.module) continue; // intra-module segments may abut
      if (cur.start < prev.end) {
        overlaps.push(
          `${prev.module} [${prev.start}..${prev.end}) overlaps ` +
            `${cur.module} [${cur.start}..${cur.end})`,
        );
      }
    }
    expect(overlaps).toEqual([]);
  });

  it('all data segments fit below MIN_HEAP_BASE (8 MiB)', () => {
    // If any module's static data crosses MIN_HEAP_BASE, the runtime heap
    // would overlap that module's constants. Bump MIN_HEAP_BASE in
    // runtime.ts (and re-derive MIN_HEAP_BYTES margin) before crossing this.
    let worst: DataRange | null = null;
    for (const f of files) {
      const bytes = decodeB64(extractB64FromModule(join(BINS_DIR, f)));
      for (const r of parseDataSegments(bytes, f)) {
        if (!worst || r.end > worst.end) worst = r;
      }
    }
    expect(worst, 'expected at least one data segment across all bins').not.toBeNull();
    expect(worst!.end, `largest segment end from ${worst!.module}`).toBeLessThan(MIN_HEAP_BASE);
  });
});
