/**
 * DEFLATE Benchmark: Zig WASM (ReleaseFast / ReleaseSmall) vs Compression Streams API
 *
 * Measures:
 *   - Binary size (.wasm)
 *   - Compression speed (MB/s, end-to-end including WASM memory copies)
 *   - Decompression speed (MB/s, end-to-end)
 *   - Compression ratio (compressed / original)
 *
 * Test data: pseudo-random float64 arrays (NPZ-like) at various sizes.
 */

import { readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { deflateRawSync, inflateRawSync } from 'node:zlib';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DIST_DIR = resolve(__dirname, '..', 'dist');
const RESULTS_DIR = resolve(__dirname, '..', 'results');

// ─── Test data generation ────────────────────────────────────────────────────

/** Generate NPZ-like float64 data (somewhat compressible — adjacent floats share exponent bits) */
function generateFloat64Data(numFloats: number): Uint8Array {
  const buf = new ArrayBuffer(numFloats * 8);
  const f64 = new Float64Array(buf);
  // Simple PRNG (mulberry32)
  let seed = 0xdeadbeef;
  for (let i = 0; i < numFloats; i++) {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    f64[i] = (u - 0.5) * 200; // range [-100, 100]
  }
  return new Uint8Array(buf);
}

/** Generate highly compressible data (zeros with occasional values) */
function generateSparseData(numBytes: number): Uint8Array {
  const data = new Uint8Array(numBytes);
  let seed = 42;
  for (let i = 0; i < numBytes; i += 64) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    data[i] = seed & 0xff;
  }
  return data;
}

// ─── WASM loader ─────────────────────────────────────────────────────────────

interface DeflateWasm {
  memory: WebAssembly.Memory;
  baseOffset: number;
  inflate_raw: (src: number, src_len: number, dst: number, dst_cap: number) => number;
  deflate_raw: (src: number, src_len: number, dst: number, dst_cap: number) => number;
  crc32: (data: number, len: number) => number;
}

async function loadDeflateWasm(filename: string): Promise<DeflateWasm | null> {
  const wasmPath = resolve(DIST_DIR, filename);
  let wasmBytes: Buffer;
  try {
    wasmBytes = readFileSync(wasmPath);
  } catch {
    console.warn(`  [skip] ${filename} not found`);
    return null;
  }

  const module = await WebAssembly.compile(wasmBytes);
  const memory = new WebAssembly.Memory({ initial: 1024, maximum: 65536 }); // 64MB initial

  const imports = WebAssembly.Module.imports(module);
  const memImport = imports.find((i) => i.kind === 'memory');

  let importObj: WebAssembly.Imports = {};
  if (memImport) {
    importObj = { [memImport.module]: { [memImport.name]: memory } };
  }

  const instance = await WebAssembly.instantiate(module, importObj);
  const exp = instance.exports as Record<string, unknown>;

  // Use exported memory if available, otherwise the one we provided
  const expMemory = WebAssembly.Module.exports(module).find((e) => e.kind === 'memory');
  const mem = expMemory ? (exp[expMemory.name] as WebAssembly.Memory) : memory;

  // Record initial memory size as base offset (stack + data segments live there)
  const baseOffset = mem.buffer.byteLength;

  return {
    memory: mem,
    baseOffset,
    inflate_raw: exp.inflate_raw as DeflateWasm['inflate_raw'],
    deflate_raw: exp.deflate_raw as DeflateWasm['deflate_raw'],
    crc32: exp.crc32 as DeflateWasm['crc32'],
  };
}

// ─── Benchmark helpers ───────────────────────────────────────────────────────

function ensureWasmMemory(wasm: DeflateWasm, needed: number) {
  const current = wasm.memory.buffer.byteLength;
  if (needed > current) {
    const pages = Math.ceil((needed - current) / 65536);
    wasm.memory.grow(pages);
  }
}

function wasmCompress(wasm: DeflateWasm, data: Uint8Array): Uint8Array {
  const srcLen = data.length;
  const dstCap = srcLen + 1024; // worst case: incompressible data + overhead
  const totalNeeded = wasm.baseOffset + srcLen + dstCap;
  ensureWasmMemory(wasm, totalNeeded);

  const srcPtr = wasm.baseOffset;
  const dstPtr = srcPtr + srcLen;

  new Uint8Array(wasm.memory.buffer).set(data, srcPtr);
  const compLen = wasm.deflate_raw(srcPtr, srcLen, dstPtr, dstCap);
  if (compLen === 0) throw new Error('WASM deflate_raw returned 0');

  return new Uint8Array(wasm.memory.buffer.slice(dstPtr, dstPtr + compLen));
}

function wasmDecompress(wasm: DeflateWasm, compressed: Uint8Array, originalLen: number): Uint8Array {
  const srcLen = compressed.length;
  const totalNeeded = wasm.baseOffset + srcLen + originalLen;
  ensureWasmMemory(wasm, totalNeeded);

  const srcPtr = wasm.baseOffset;
  const dstPtr = srcPtr + srcLen;

  new Uint8Array(wasm.memory.buffer).set(compressed, srcPtr);
  const outLen = wasm.inflate_raw(srcPtr, srcLen, dstPtr, originalLen);
  if (outLen === 0) throw new Error('WASM inflate_raw returned 0');

  return new Uint8Array(wasm.memory.buffer.slice(dstPtr, dstPtr + outLen));
}

// Compression Streams API
async function csCompress(data: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream('deflate-raw');
  const writer = cs.writable.getWriter();
  const copy = new Uint8Array(data.length);
  copy.set(data);
  void writer.write(copy);
  void writer.close();

  const reader = cs.readable.getReader();
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

async function csDecompress(compressed: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  const copy = new Uint8Array(compressed.length);
  copy.set(compressed);
  void writer.write(copy);
  void writer.close();

  const reader = ds.readable.getReader();
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

// node:zlib (synchronous)
function zlibCompress(data: Uint8Array): Uint8Array {
  return deflateRawSync(data);
}

function zlibDecompress(compressed: Uint8Array, _originalLen?: number): Uint8Array {
  return inflateRawSync(compressed);
}

// ─── Timing ──────────────────────────────────────────────────────────────────

interface TimingResult {
  median_ms: number;
  min_ms: number;
  max_ms: number;
  throughput_MBs: number; // MB/s based on median
}

function benchSync(fn: () => void, dataSize: number, iterations: number, warmup: number): TimingResult {
  for (let i = 0; i < warmup; i++) fn();

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)]!;
  const sizeMB = dataSize / (1024 * 1024);

  return {
    median_ms: median,
    min_ms: times[0]!,
    max_ms: times[times.length - 1]!,
    throughput_MBs: sizeMB / (median / 1000),
  };
}

async function benchAsync(fn: () => Promise<void>, dataSize: number, iterations: number, warmup: number): Promise<TimingResult> {
  for (let i = 0; i < warmup; i++) await fn();

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)]!;
  const sizeMB = dataSize / (1024 * 1024);

  return {
    median_ms: median,
    min_ms: times[0]!,
    max_ms: times[times.length - 1]!,
    throughput_MBs: sizeMB / (median / 1000),
  };
}

// ─── Main ────────────────────────────────────────────────────────────────────

interface BenchResult {
  engine: string;
  dataType: string;
  dataSize: number;
  dataSizeHuman: string;
  compressTime: TimingResult;
  decompressTime: TimingResult;
  compressedSize: number;
  ratio: number; // compressed / original
  binarySize?: number;
  binarySizeGzip?: number;
}

const DATA_CONFIGS = [
  { name: 'float64 1K',   gen: () => generateFloat64Data(1_000),    type: 'float64' },
  { name: 'float64 10K',  gen: () => generateFloat64Data(10_000),   type: 'float64' },
  { name: 'float64 100K', gen: () => generateFloat64Data(100_000),  type: 'float64' },
  { name: 'float64 1M',   gen: () => generateFloat64Data(1_000_000), type: 'float64' },
  { name: 'sparse 100KB', gen: () => generateSparseData(100_000),   type: 'sparse' },
  { name: 'sparse 1MB',   gen: () => generateSparseData(1_000_000), type: 'sparse' },
];

function iterationsForSize(size: number): { iterations: number; warmup: number } {
  if (size < 50_000) return { iterations: 50, warmup: 10 };
  if (size < 500_000) return { iterations: 20, warmup: 5 };
  if (size < 5_000_000) return { iterations: 10, warmup: 3 };
  return { iterations: 5, warmup: 2 };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║          DEFLATE Benchmark: Zig WASM vs Compression Streams ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  // Load WASM modules
  const wasmFast = await loadDeflateWasm('deflate_fast.wasm');
  const wasmSmall = await loadDeflateWasm('deflate_small.wasm');

  // Binary sizes
  const fastSize = statSync(resolve(DIST_DIR, 'deflate_fast.wasm')).size;
  const smallSize = statSync(resolve(DIST_DIR, 'deflate_small.wasm')).size;
  const fastGzip = deflateRawSync(readFileSync(resolve(DIST_DIR, 'deflate_fast.wasm'))).length;
  const smallGzip = deflateRawSync(readFileSync(resolve(DIST_DIR, 'deflate_small.wasm'))).length;

  console.log('Binary sizes:');
  console.log(`  ReleaseFast:  ${formatBytes(fastSize)} (${formatBytes(fastGzip)} gzipped)`);
  console.log(`  ReleaseSmall: ${formatBytes(smallSize)} (${formatBytes(smallGzip)} gzipped)`);
  console.log(`  Compression Streams: built-in (0 B)\n`);

  const results: BenchResult[] = [];

  for (const config of DATA_CONFIGS) {
    const data = config.gen();
    const { iterations, warmup } = iterationsForSize(data.length);

    console.log(`━━━ ${config.name} (${formatBytes(data.length)}, ${iterations} iters) ━━━`);

    // --- Zig WASM ReleaseFast ---
    if (wasmFast) {
      const compressed = wasmCompress(wasmFast, data);
      // Verify roundtrip
      const decompressed = wasmDecompress(wasmFast, compressed, data.length);
      if (decompressed.length !== data.length) throw new Error('ReleaseFast roundtrip length mismatch');
      for (let i = 0; i < 100 && i < data.length; i++) {
        if (decompressed[i] !== data[i]) throw new Error(`ReleaseFast roundtrip mismatch at byte ${i}`);
      }

      const compTime = benchSync(() => wasmCompress(wasmFast, data), data.length, iterations, warmup);
      const decompTime = benchSync(() => wasmDecompress(wasmFast, compressed, data.length), data.length, iterations, warmup);

      const res: BenchResult = {
        engine: 'Zig ReleaseFast',
        dataType: config.type,
        dataSize: data.length,
        dataSizeHuman: config.name,
        compressTime: compTime,
        decompressTime: decompTime,
        compressedSize: compressed.length,
        ratio: compressed.length / data.length,
        binarySize: fastSize,
        binarySizeGzip: fastGzip,
      };
      results.push(res);
      console.log(`  Zig Fast   compress: ${compTime.median_ms.toFixed(2)}ms (${compTime.throughput_MBs.toFixed(1)} MB/s)  decompress: ${decompTime.median_ms.toFixed(2)}ms (${decompTime.throughput_MBs.toFixed(1)} MB/s)  ratio: ${(res.ratio * 100).toFixed(1)}%`);
    }

    // --- Zig WASM ReleaseSmall ---
    if (wasmSmall) {
      const compressed = wasmCompress(wasmSmall, data);
      const decompressed = wasmDecompress(wasmSmall, compressed, data.length);
      if (decompressed.length !== data.length) throw new Error('ReleaseSmall roundtrip length mismatch');

      const compTime = benchSync(() => wasmCompress(wasmSmall, data), data.length, iterations, warmup);
      const decompTime = benchSync(() => wasmDecompress(wasmSmall, compressed, data.length), data.length, iterations, warmup);

      const res: BenchResult = {
        engine: 'Zig ReleaseSmall',
        dataType: config.type,
        dataSize: data.length,
        dataSizeHuman: config.name,
        compressTime: compTime,
        decompressTime: decompTime,
        compressedSize: compressed.length,
        ratio: compressed.length / data.length,
        binarySize: smallSize,
        binarySizeGzip: smallGzip,
      };
      results.push(res);
      console.log(`  Zig Small  compress: ${compTime.median_ms.toFixed(2)}ms (${compTime.throughput_MBs.toFixed(1)} MB/s)  decompress: ${decompTime.median_ms.toFixed(2)}ms (${decompTime.throughput_MBs.toFixed(1)} MB/s)  ratio: ${(res.ratio * 100).toFixed(1)}%`);
    }

    // --- Compression Streams API (async) ---
    {
      const compressed = await csCompress(data);
      const decompressed = await csDecompress(compressed);
      if (decompressed.length !== data.length) throw new Error('CS API roundtrip length mismatch');

      const compTime = await benchAsync(async () => { await csCompress(data); }, data.length, iterations, warmup);
      const decompTime = await benchAsync(async () => { await csDecompress(compressed); }, data.length, iterations, warmup);

      const res: BenchResult = {
        engine: 'Compression Streams',
        dataType: config.type,
        dataSize: data.length,
        dataSizeHuman: config.name,
        compressTime: compTime,
        decompressTime: decompTime,
        compressedSize: compressed.length,
        ratio: compressed.length / data.length,
      };
      results.push(res);
      console.log(`  CS API     compress: ${compTime.median_ms.toFixed(2)}ms (${compTime.throughput_MBs.toFixed(1)} MB/s)  decompress: ${decompTime.median_ms.toFixed(2)}ms (${decompTime.throughput_MBs.toFixed(1)} MB/s)  ratio: ${(res.ratio * 100).toFixed(1)}%`);
    }

    // --- node:zlib (sync, as reference) ---
    {
      const compressed = zlibCompress(data);
      const decompressed = zlibDecompress(compressed);
      if (decompressed.length !== data.length) throw new Error('zlib roundtrip length mismatch');

      const compTime = benchSync(() => zlibCompress(data), data.length, iterations, warmup);
      const decompTime = benchSync(() => zlibDecompress(compressed), data.length, iterations, warmup);

      const res: BenchResult = {
        engine: 'node:zlib (sync)',
        dataType: config.type,
        dataSize: data.length,
        dataSizeHuman: config.name,
        compressTime: compTime,
        decompressTime: decompTime,
        compressedSize: compressed.length,
        ratio: compressed.length / data.length,
      };
      results.push(res);
      console.log(`  node:zlib  compress: ${compTime.median_ms.toFixed(2)}ms (${compTime.throughput_MBs.toFixed(1)} MB/s)  decompress: ${decompTime.median_ms.toFixed(2)}ms (${decompTime.throughput_MBs.toFixed(1)} MB/s)  ratio: ${(res.ratio * 100).toFixed(1)}%`);
    }

    console.log();
  }

  // Save JSON results
  mkdirSync(RESULTS_DIR, { recursive: true });
  const outPath = resolve(RESULTS_DIR, 'deflate-bench.json');
  writeFileSync(outPath, JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2));
  console.log(`Results saved to ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
