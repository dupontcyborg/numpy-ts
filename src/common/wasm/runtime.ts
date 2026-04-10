/**
 * WASM Runtime — zero-copy memory management for all WASM kernels.
 *
 * All array data lives in a single fixed-size WebAssembly.Memory instance.
 * A persistent free-list allocator (alloc.zig) manages long-lived ArrayStorage
 * data. A scratch bump allocator handles temporary copy-ins for JS-fallback
 * arrays during kernel calls.
 *
 * Memory layout:
 *   [WASM statics][--- persistent heap (free-list) ---][--- scratch (bump) ---]
 *   0          heapBase                            scratchBase          maxBytes
 *
 * Because initial === maximum pages, memory.buffer is never detached by grow().
 * All TypedArray views into it remain valid for the program lifetime.
 */

import type { TypedArray } from '../dtype';
import { wasmMemoryConfig, type ConfigureWasmOptions } from './config';

// FinalizationRegistry is available in all target environments (Node 14+,
// modern browsers) but not in ES2020 lib typings.
declare class FinalizationRegistry<T> {
  constructor(callback: (heldValue: T) => void);
  register(target: object, heldValue: T, unregisterToken?: object): void;
  unregister(unregisterToken: object): void;
}
import { heap_init, heap_malloc, heap_free, heap_free_bytes } from './bins/alloc.wasm';

// ---------------------------------------------------------------------------
// Shared memory — fixed-size, never grows
// ---------------------------------------------------------------------------

const PAGE_SIZE = 65536;
let memory: WebAssembly.Memory | null = null;
let heapBase = 0;
let heapInitialized = false;

// Scratch region state
let scratchBase = 0;
let scratchOffset = 0;

// Temp heap allocations made when scratch is too small.
// Freed on next resetScratchAllocator() call.
let tempHeapPtrs: number[] = [];

/**
 * Get the shared WebAssembly.Memory instance.
 * Allocated once with initial === maximum pages so the buffer never detaches.
 */
export function getSharedMemory(): WebAssembly.Memory {
  if (!memory) {
    const pages = Math.ceil(wasmMemoryConfig.maxMemoryBytes / PAGE_SIZE);
    memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
  }
  return memory;
}

/**
 * Set the heap base offset. Called by each WASM module on first init
 * (reads __heap_base from exports). The highest value wins.
 */
export function setHeapBase(base: number): void {
  if (base > heapBase) {
    heapBase = base;
    // If the heap was already initialized with a lower base, re-init with the
    // higher base so that WASM modules' static data regions are not overlapped.
    if (heapInitialized) {
      const maxBytes = wasmMemoryConfig.maxMemoryBytes;
      scratchBase = maxBytes - wasmMemoryConfig.scratchBytes;
      const heapSize = scratchBase - heapBase;
      if (heapSize > 0) {
        heap_init(heapBase, heapSize);
      }
      scratchOffset = scratchBase;
    }
  }
}

/**
 * Initialize the persistent allocator and scratch region.
 * Called lazily on first wasmMalloc or scratch use.
 */
function ensureHeapInitialized(): void {
  if (heapInitialized) return;
  heapInitialized = true;

  // Ensure memory is created
  getSharedMemory();

  // Each WASM module is compiled with a unique --global-base (64 KiB stride),
  // placing its static data at a non-overlapping offset in shared memory.
  // With ~90 modules × 64 KiB = ~6 MiB, the heap must start above all of them.
  // 8 MiB provides headroom for future modules.
  const MIN_HEAP_BASE = 8 * 1024 * 1024;
  if (heapBase < MIN_HEAP_BASE) heapBase = MIN_HEAP_BASE;

  // Layout: [heapBase ... scratchBase ... maxBytes]
  const maxBytes = wasmMemoryConfig.maxMemoryBytes;
  scratchBase = maxBytes - wasmMemoryConfig.scratchBytes;
  const heapSize = scratchBase - heapBase;

  if (heapSize > 0) {
    heap_init(heapBase, heapSize);
  }

  scratchOffset = scratchBase;
}

// ---------------------------------------------------------------------------
// Public configuration API
// ---------------------------------------------------------------------------

/**
 * Configure WASM memory settings. Must be called before any array operations
 * (i.e. before the WASM memory is initialized on first use).
 *
 * @example
 * ```ts
 * import { configureWasm } from 'numpy-ts';
 * configureWasm({ maxMemory: 512 * 1024 * 1024 }); // 512 MiB
 * ```
 */
export function configureWasm(options: ConfigureWasmOptions): void {
  if (heapInitialized) {
    throw new Error(
      'configureWasm() must be called before any array operations. ' +
        'WASM memory has already been initialized.'
    );
  }
  if (options.maxMemory !== undefined) {
    if (options.maxMemory <= 0) {
      throw new Error('maxMemory must be a positive number of bytes.');
    }
    wasmMemoryConfig.maxMemoryBytes = options.maxMemory;
    // Auto-scale scratch to 1/16 of maxMemory, capped at 32 MiB
    if (options.scratchSize === undefined) {
      wasmMemoryConfig.scratchBytes = Math.min(
        Math.floor(options.maxMemory / 16),
        32 * 1024 * 1024
      );
    }
  }
  if (options.scratchSize !== undefined) {
    if (options.scratchSize <= 0) {
      throw new Error('scratchSize must be a positive number of bytes.');
    }
    wasmMemoryConfig.scratchBytes = options.scratchSize;
  }
}

// ---------------------------------------------------------------------------
// WasmRegion — refcounted handle to a persistent WASM allocation
// ---------------------------------------------------------------------------

export class WasmRegion {
  readonly ptr: number;
  readonly byteLength: number;
  private _refCount: number = 1;

  constructor(ptr: number, byteLength: number) {
    this.ptr = ptr;
    this.byteLength = byteLength;
  }

  retain(): void {
    this._refCount++;
  }

  release(): void {
    if (this._refCount <= 0) return; // already freed — prevent double-free
    if (--this._refCount === 0) {
      if (heapInitialized && this.ptr >= 64 && scratchBase > 0 && this.ptr < scratchBase) {
        heap_free(this.ptr);
      }
    }
  }

  get refCount(): number {
    return this._refCount;
  }
}

// ---------------------------------------------------------------------------
// FinalizationRegistry — GC-based cleanup for WASM regions
// ---------------------------------------------------------------------------

const regionRegistry = new FinalizationRegistry<WasmRegion>((region) => {
  region.release();
});

/**
 * Register an object (typically ArrayStorage) so that when it is garbage
 * collected, the associated WasmRegion's refcount is decremented.
 * The instance itself is used as the unregister token, allowing
 * eager cleanup via unregisterCleanup() to prevent double-free.
 */
export function registerForCleanup(instance: object, region: WasmRegion): void {
  regionRegistry.register(instance, region, instance);
}

/**
 * Unregister an object from the FinalizationRegistry.
 * Must be called when eagerly releasing WASM memory to prevent
 * the GC callback from double-freeing the region.
 */
export function unregisterCleanup(instance: object): void {
  regionRegistry.unregister(instance);
}

// ---------------------------------------------------------------------------
// Persistent allocator — for ArrayStorage backing data
// ---------------------------------------------------------------------------

/**
 * Allocate `bytes` from the persistent WASM heap.
 * Returns a WasmRegion, or null if out of memory (caller should fall back to JS).
 */
export function wasmMalloc(bytes: number): WasmRegion | null {
  if (bytes <= 0) return null;
  ensureHeapInitialized();

  const ptr = heap_malloc(bytes);
  if (ptr === 0) {
    if (typeof process !== 'undefined' && process.env?.['LOG_HEAP']) {
      const free = heap_free_bytes();
      console.error(`[wasm] malloc failed: requested ${bytes} bytes, ${free} bytes free`);
    }
    return null;
  }
  return new WasmRegion(ptr, bytes);
}

/**
 * Total free bytes in the persistent heap.
 */
export function wasmFreeBytes(): number {
  ensureHeapInitialized();
  return heap_free_bytes();
}

// ---------------------------------------------------------------------------
// Scratch allocator — temporary copy-in for JS-fallback arrays
// ---------------------------------------------------------------------------

/**
 * Reset the scratch bump allocator. Call before each kernel invocation
 * that may need to copy JS-fallback inputs into WASM memory.
 * Also frees any temp heap allocations from the previous kernel call.
 */
export function resetScratchAllocator(): void {
  ensureHeapInitialized();
  scratchOffset = scratchBase;
  if (tempHeapPtrs.length > 0) {
    for (const ptr of tempHeapPtrs) {
      heap_free(ptr);
    }
    tempHeapPtrs = [];
  }
}

/**
 * Bump-allocate `bytes` from the scratch region. Returns byte offset.
 * Always 8-byte aligned. Falls back to heap if scratch space is exhausted.
 */
export function scratchAlloc(bytes: number): number {
  ensureHeapInitialized();
  const aligned = (scratchOffset + 7) & ~7;
  const newOffset = aligned + bytes;
  if (newOffset > wasmMemoryConfig.maxMemoryBytes) {
    // Scratch region too small — allocate on the persistent heap instead.
    // The pointer is tracked and freed on the next resetScratchAllocator() call.
    const ptr = heap_malloc(bytes);
    if (ptr === 0) {
      throw new Error(
        `WASM OOM: scratch full (${wasmMemoryConfig.scratchBytes} bytes) ` +
          `and heap malloc failed for ${bytes} bytes`
      );
    }
    tempHeapPtrs.push(ptr);
    return ptr;
  }
  scratchOffset = newOffset;
  return aligned;
}

/**
 * Copy a JS TypedArray into the scratch region. Returns byte offset.
 */
export function scratchCopyIn(src: TypedArray): number {
  const ptr = scratchAlloc(src.byteLength);
  const mem = getSharedMemory();
  new Uint8Array(mem.buffer, ptr, src.byteLength).set(
    new Uint8Array(src.buffer, src.byteOffset, src.byteLength)
  );
  return ptr;
}

// ---------------------------------------------------------------------------
// Input resolution — unified pointer resolution for kernel wrappers
// ---------------------------------------------------------------------------

/**
 * Resolve an ArrayStorage input to a WASM pointer for kernel use.
 * If the storage is WASM-backed, returns its direct pointer (zero-copy).
 * If JS-backed, copies data into the scratch region.
 *
 * @param data - The TypedArray from storage.data
 * @param isWasmBacked - Whether the storage is backed by WASM memory
 * @param wasmPtr - The WASM pointer (only valid if isWasmBacked)
 * @param offset - Element offset into the data
 * @param elementCount - Number of elements to resolve
 * @param bpe - Bytes per element
 */
export function resolveInputPtr(
  data: TypedArray,
  isWasmBacked: boolean,
  wasmPtr: number,
  offset: number,
  elementCount: number,
  bpe: number
): number {
  if (isWasmBacked) {
    return wasmPtr + offset * bpe;
  }
  // JS-fallback: copy to scratch
  const src = data.subarray(offset, offset + elementCount) as TypedArray;
  return scratchCopyIn(src);
}

/**
 * Resolve a TypedArray to a WASM pointer. If the array is already a view
 * into WASM memory, returns its byte offset directly (zero-copy).
 * Otherwise copies to scratch. Useful when you have a TypedArray but
 * don't know if it's WASM-backed (e.g. from getContiguousData).
 */
export function resolveTypedArrayPtr(data: TypedArray): number {
  const mem = getSharedMemory();
  if (data.buffer === mem.buffer) {
    return data.byteOffset;
  }
  return scratchCopyIn(data);
}

// ---------------------------------------------------------------------------
// Backward compatibility — old API mapped to scratch allocator
// These are used by kernel wrappers that haven't been migrated yet.
// ---------------------------------------------------------------------------
// Optimized f16 conversion helpers (zero JS allocation)
// ---------------------------------------------------------------------------

/**
 * Convert f16 input to f32 in the scratch region using .set() — no JS allocation.
 * Creates Float32Array view on WASM scratch memory and uses .set(f16Data) which
 * converts f16→f32 in-place. 1.2x–3x faster than f16ToF32Input + scratchCopyIn.
 *
 * @param a - The ArrayStorage with f16 data
 * @param size - Number of elements
 * @returns WASM byte offset of the f32 scratch data
 */
export function f16InputToScratchF32(
  a: { data: TypedArray; isWasmBacked: boolean; wasmPtr: number; offset: number },
  size: number
): number {
  const mem = getSharedMemory();
  const ptr = scratchAlloc(size * 4);
  const f32View = new Float32Array(mem.buffer, ptr, size);
  if (a.isWasmBacked) {
    f32View.set(new Float16Array(mem.buffer, a.wasmPtr + a.offset * 2, size));
  } else {
    f32View.set(a.data.subarray(a.offset, a.offset + size) as unknown as ArrayLike<number>);
  }
  return ptr;
}

/**
 * Convert f32 kernel output to f16 in a new WASM region using .set() — no JS round-trip.
 * Allocates a persistent f16 WasmRegion, creates Float16Array + Float32Array views
 * on WASM memory, and uses .set() to convert in-place.
 * Replaces: copyOut + f32ToF16Output + fromData (saves 2 copies).
 *
 * @param outRegion - The WasmRegion containing f32 output from the kernel
 * @param size - Number of elements
 * @returns New WasmRegion with f16 data, or null on OOM. Caller must release outRegion.
 */
export function f32OutputToF16Region(outRegion: WasmRegion, size: number): WasmRegion | null {
  const f16Region = wasmMalloc(size * 2);
  if (!f16Region) return null;
  const mem = getSharedMemory();
  const f32View = new Float32Array(mem.buffer, outRegion.ptr, size);
  const f16View = new Float16Array(mem.buffer, f16Region.ptr, size);
  f16View.set(f32View);
  return f16Region;
}

/**
 * Convert f32 WASM output to f16 in-place within the same region.
 * Safe because f16[i] at byte 2i never reaches unread f32[j>i] at byte 4j.
 * Avoids extra wasmMalloc + release overhead of f32OutputToF16Region.
 * The region retains its original (f32-sized) allocation; the extra bytes are unused.
 */
export function f32ToF16InPlace(outRegion: WasmRegion, size: number): void {
  const mem = getSharedMemory();
  const f32View = new Float32Array(mem.buffer, outRegion.ptr, size);
  const f16View = new Float16Array(mem.buffer, outRegion.ptr, size);
  for (let i = 0; i < size; i++) {
    f16View[i] = f32View[i]!;
  }
}
