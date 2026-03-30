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
import { hasFloat16 } from '../dtype';
import { wasmConfig, wasmMemoryConfig } from './config';

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
 */
export function resetScratchAllocator(): void {
  ensureHeapInitialized();
  scratchOffset = scratchBase;
}

/**
 * Bump-allocate `bytes` from the scratch region. Returns byte offset.
 * Always 8-byte aligned. Throws if scratch space is exhausted.
 */
export function scratchAlloc(bytes: number): number {
  ensureHeapInitialized();
  const aligned = (scratchOffset + 7) & ~7;
  scratchOffset = aligned + bytes;
  if (scratchOffset > wasmMemoryConfig.maxMemoryBytes) {
    throw new Error(
      `WASM scratch OOM: need ${scratchOffset - scratchBase} bytes, ` +
        `have ${wasmMemoryConfig.scratchBytes}`
    );
  }
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

// ---------------------------------------------------------------------------
// Backward compatibility — old API mapped to scratch allocator
// These are used by kernel wrappers that haven't been migrated yet.
// ---------------------------------------------------------------------------

/**
 * @deprecated Use scratchAlloc region or wasmMalloc for persistent.
 * Ensure WASM memory can hold `bytes` above heapBase.
 * With fixed-size memory, this is a no-op (memory is pre-allocated).
 */
export function ensureMemory(_bytes: number): void {
  ensureHeapInitialized();
  // No-op: memory is fixed-size and pre-allocated
}

/**
 * @deprecated Use resetScratchAllocator.
 * Reset the bump allocator. For backward compat, resets the scratch region.
 */
export function resetAllocator(): void {
  ensureHeapInitialized();
  resetScratchAllocator();
  wasmConfig.wasmCallCount++;
}

/**
 * @deprecated Use scratchAlloc.
 * Bump-allocate from scratch region (backward compat).
 */
export function alloc(bytes: number): number {
  return scratchAlloc(bytes);
}

/**
 * @deprecated Use scratchCopyIn.
 * Copy a JS TypedArray into scratch region (backward compat).
 */
export function copyIn(src: TypedArray): number {
  return scratchCopyIn(src);
}

/**
 * @deprecated Output should be a WASM-backed view, not a copy.
 * Copy data from WASM memory into a new JS TypedArray (backward compat).
 */
export function copyOut<T extends TypedArray>(
  ptr: number,
  length: number,
  Ctor: new (buffer: ArrayBuffer, byteOffset: number, length: number) => T
): T {
  const mem = getSharedMemory();
  const result = new Ctor(
    new ArrayBuffer(length * (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT),
    0,
    length
  );
  new Uint8Array(result.buffer, 0, result.byteLength).set(
    new Uint8Array(mem.buffer, ptr, result.byteLength)
  );
  return result;
}

// ---------------------------------------------------------------------------
// Float16 conversion helpers (unchanged)
// ---------------------------------------------------------------------------

/**
 * Convert Float16Array data to Float32Array for WASM kernel input.
 * WASM kernels operate on f32 for float16 data (no native f16 SIMD).
 */
export function f16ToF32Input(data: TypedArray, dtype: string): TypedArray {
  if (dtype === 'float16' && hasFloat16) {
    return new Float32Array(data as unknown as ArrayLike<number>) as unknown as TypedArray;
  }
  return data;
}

/**
 * Convert Float32Array WASM output back to Float16Array.
 */
export function f32ToF16Output(data: TypedArray, dtype: string): TypedArray {
  if (dtype === 'float16' && hasFloat16) {
    const f16 = new Float16Array(data.length);
    f16.set(data as Exclude<TypedArray, BigInt64Array | BigUint64Array>);
    return f16 as unknown as TypedArray;
  }
  return data;
}
