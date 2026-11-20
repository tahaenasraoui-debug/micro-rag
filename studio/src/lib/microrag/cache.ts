/** Arena LRU semantic cache, ported from micro-rag `cache.rs`. */

import { cosineSimilarity, VectorError } from "./math";

export type CacheHit = {
  matchedQuery: string;
  completion: string;
  similarity: number;
  totalHits: number;
};

type CacheEntry = {
  queryText: string;
  queryEmbedding: number[];
  completion: string;
  hits: number;
};

type ArenaNode = {
  entry: CacheEntry;
  prev: number | null;
  next: number | null;
};

export type CacheProbe = {
  hit: CacheHit | null;
  bestSimilarity: number;
};

export class SemanticCache {
  readonly capacity: number;
  readonly dim: number;
  similarityThreshold: number;
  private nodes: (ArenaNode | null)[] = [];
  private freeSlots: number[] = [];
  private head: number | null = null;
  private tail: number | null = null;
  private count = 0;
  totalLookups = 0;
  totalHits = 0;

  constructor(capacity: number, dim: number, similarityThreshold: number) {
    if (dim === 0) throw new VectorError("Vector cannot be empty (zero dimensions)");
    if (capacity <= 0) throw new VectorError("Cache capacity must be greater than 0");
    if (similarityThreshold < 0 || similarityThreshold > 1) {
      throw new VectorError("Similarity threshold must be between 0.0 and 1.0");
    }
    this.capacity = capacity;
    this.dim = dim;
    this.similarityThreshold = similarityThreshold;
  }

  get length(): number {
    return this.count;
  }

  hitRate(): number {
    if (this.totalLookups === 0) return 0;
    return this.totalHits / this.totalLookups;
  }

  get(queryEmbedding: number[]): CacheProbe {
    if (queryEmbedding.length !== this.dim) {
      throw new VectorError(`Dimension mismatch: expected ${this.dim}, got ${queryEmbedding.length}`);
    }
    this.totalLookups += 1;
    const found = this.findClosest(queryEmbedding);
    if (found && found.sim >= this.similarityThreshold) {
      this.promoteToHead(found.slot);
      this.totalHits += 1;
      const node = this.nodes[found.slot];
      if (!node) throw new VectorError("Corrupted LRU arena slot");
      node.entry.hits += 1;
      return {
        hit: {
          matchedQuery: node.entry.queryText,
          completion: node.entry.completion,
          similarity: found.sim,
          totalHits: node.entry.hits,
        },
        bestSimilarity: found.sim,
      };
    }
    return { hit: null, bestSimilarity: found ? found.sim : -1 };
  }

  put(queryText: string, queryEmbedding: number[], completion: string): void {
    if (queryEmbedding.length !== this.dim) {
      throw new VectorError(`Dimension mismatch: expected ${this.dim}, got ${queryEmbedding.length}`);
    }
    cosineSimilarity(queryEmbedding, queryEmbedding);

    const existing = this.findClosest(queryEmbedding);
    if (existing && existing.sim >= this.similarityThreshold) {
      const node = this.nodes[existing.slot];
      if (!node) throw new VectorError("Corrupted LRU arena slot");
      node.entry.queryText = queryText;
      node.entry.queryEmbedding = queryEmbedding.slice();
      node.entry.completion = completion;
      this.promoteToHead(existing.slot);
      return;
    }

    if (this.count === this.capacity) this.evictLru();

    const slot = this.freeSlots.pop();
    const node: ArenaNode = {
      entry: {
        queryText,
        queryEmbedding: queryEmbedding.slice(),
        completion,
        hits: 0,
      },
      prev: null,
      next: this.head,
    };
    let index: number;
    if (slot !== undefined) {
      this.nodes[slot] = node;
      index = slot;
    } else {
      index = this.nodes.length;
      this.nodes.push(node);
    }
    if (this.head !== null) {
      const head = this.nodes[this.head];
      if (head) head.prev = index;
    }
    this.head = index;
    if (this.tail === null) this.tail = index;
    this.count += 1;
  }

  /** MRU → LRU. */
  entries(): CacheEntry[] {
    const out: CacheEntry[] = [];
    let cursor = this.head;
    while (cursor !== null) {
      const node = this.nodes[cursor];
      if (!node) break;
      out.push(node.entry);
      cursor = node.next;
    }
    return out;
  }

  clear(): void {
    this.nodes = [];
    this.freeSlots = [];
    this.head = null;
    this.tail = null;
    this.count = 0;
    this.totalLookups = 0;
    this.totalHits = 0;
  }

  private findClosest(queryEmbedding: number[]): { slot: number; sim: number } | null {
    let bestSlot: number | null = null;
    let bestSim = -1;
    let cursor = this.head;
    while (cursor !== null) {
      const node = this.nodes[cursor];
      if (!node) throw new VectorError("Corrupted LRU arena slot");
      const sim = cosineSimilarity(queryEmbedding, node.entry.queryEmbedding);
      if (sim > bestSim) {
        bestSim = sim;
        bestSlot = cursor;
      }
      cursor = node.next;
    }
    if (bestSlot === null) return null;
    return { slot: bestSlot, sim: bestSim };
  }

  private detach(slot: number): void {
    const node = this.nodes[slot];
    if (!node) return;
    const { prev, next } = node;
    if (prev !== null) {
      const p = this.nodes[prev];
      if (p) p.next = next;
    } else {
      this.head = next;
    }
    if (next !== null) {
      const n = this.nodes[next];
      if (n) n.prev = prev;
    } else {
      this.tail = prev;
    }
    node.prev = null;
    node.next = null;
  }

  private promoteToHead(slot: number): void {
    if (this.head === slot) return;
    this.detach(slot);
    const node = this.nodes[slot];
    if (!node) return;
    node.prev = null;
    node.next = this.head;
    if (this.head !== null) {
      const head = this.nodes[this.head];
      if (head) head.prev = slot;
    }
    this.head = slot;
    if (this.tail === null) this.tail = slot;
  }

  private evictLru(): void {
    if (this.tail === null) return;
    const slot = this.tail;
    this.detach(slot);
    this.nodes[slot] = null;
    this.freeSlots.push(slot);
    this.count -= 1;
  }
}
