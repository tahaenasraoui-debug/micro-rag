/**
 * Flat scan and KD-tree, ported from micro-rag `index.rs`.
 * Cosine search on the tree uses the identity ||u-v||^2 = 2-2 cos(u,v)
 * after unit-normalizing, then prunes with the current k-th radius.
 */

import { cosineSimilarity, l2Normalize, magnitudeSq, squaredEuclidean, VectorError } from "./math";

export type IndexType = "flat" | "kdtree";

export type DocumentRecord = {
  id: string;
  vector: number[];
  text: string;
  metadata: Record<string, string>;
};

export type SearchHit = {
  id: string;
  score: number;
  text: string;
  metadata: Record<string, string>;
};

type Scored = { score: number; docIdx: number };

class MinHeap {
  private data: Scored[] = [];

  get size(): number {
    return this.data.length;
  }

  peek(): Scored | undefined {
    return this.data[0];
  }

  push(item: Scored): void {
    this.data.push(item);
    this.bubble(this.data.length - 1);
  }

  pop(): Scored | undefined {
    if (this.data.length === 0) return undefined;
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0 && last) {
      this.data[0] = last;
      this.sink(0);
    }
    return top;
  }

  private less(a: Scored, b: Scored): boolean {
    if (a.score !== b.score) return a.score < b.score;
    return a.docIdx < b.docIdx;
  }

  private bubble(i: number): void {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(this.data[i], this.data[p])) break;
      [this.data[i], this.data[p]] = [this.data[p], this.data[i]];
      i = p;
    }
  }

  private sink(i: number): void {
    const n = this.data.length;
    while (true) {
      let smallest = i;
      const l = i * 2 + 1;
      const r = l + 1;
      if (l < n && this.less(this.data[l], this.data[smallest])) smallest = l;
      if (r < n && this.less(this.data[r], this.data[smallest])) smallest = r;
      if (smallest === i) break;
      [this.data[i], this.data[smallest]] = [this.data[smallest], this.data[i]];
      i = smallest;
    }
  }
}

type KdNode = {
  docIdx: number;
  splitAxis: number;
  splitVal: number;
  normVector: number[];
  left: KdNode | null;
  right: KdNode | null;
};

function selectNth(entries: { idx: number; vec: number[] }[], n: number, axis: number): void {
  let left = 0;
  let right = entries.length - 1;
  while (left < right) {
    const pivot = partition(entries, left, right, axis);
    if (pivot === n) return;
    if (pivot < n) left = pivot + 1;
    else right = pivot - 1;
  }
}

function partition(entries: { idx: number; vec: number[] }[], left: number, right: number, axis: number): number {
  const mid = left + ((right - left) >> 1);
  swap(entries, mid, right);
  const pivotVal = entries[right].vec[axis];
  let store = left;
  for (let i = left; i < right; i++) {
    if (entries[i].vec[axis] < pivotVal) {
      swap(entries, i, store);
      store += 1;
    }
  }
  swap(entries, store, right);
  return store;
}

function swap<T>(arr: T[], i: number, j: number): void {
  if (i === j) return;
  const tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}

export class VectorIndex {
  readonly dim: number;
  indexType: IndexType;
  documents: DocumentRecord[] = [];
  private idToIdx = new Map<string, number>();
  private kdRoot: KdNode | null = null;
  private dirty = false;

  constructor(dim: number, indexType: IndexType) {
    if (dim === 0) throw new VectorError("Vector cannot be empty (zero dimensions)");
    this.dim = dim;
    this.indexType = indexType;
  }

  get length(): number {
    return this.documents.length;
  }

  upsert(record: DocumentRecord): void {
    if (record.vector.length !== this.dim) {
      throw new VectorError(`Dimension mismatch: expected ${this.dim}, got ${record.vector.length}`);
    }
    if (magnitudeSq(record.vector) <= Number.EPSILON) {
      throw new VectorError("Zero-magnitude vector encountered during normalization");
    }
    const existing = this.idToIdx.get(record.id);
    if (existing !== undefined) {
      this.documents[existing] = record;
    } else {
      this.idToIdx.set(record.id, this.documents.length);
      this.documents.push(record);
    }
    if (this.indexType === "kdtree") this.dirty = true;
  }

  rebuildIndex(): void {
    if (this.indexType !== "kdtree") return;
    const entries = this.documents.map((doc, idx) => {
      const vec = doc.vector.slice();
      l2Normalize(vec);
      return { idx, vec };
    });
    this.kdRoot = buildKd(entries, 0, this.dim);
    this.dirty = false;
  }

  search(query: number[], k: number): { hits: SearchHit[]; scanned: number } {
    if (query.length !== this.dim) {
      throw new VectorError(`Dimension mismatch: expected ${this.dim}, got ${query.length}`);
    }
    if (k === 0 || this.documents.length === 0) return { hits: [], scanned: 0 };
    if (magnitudeSq(query) <= Number.EPSILON) {
      throw new VectorError("Zero-magnitude vector encountered during cosine similarity");
    }
    if (this.indexType === "flat") return this.searchFlat(query, k);
    if (this.dirty || !this.kdRoot) this.rebuildIndex();
    return this.searchKd(query, k);
  }

  private searchFlat(query: number[], k: number): { hits: SearchHit[]; scanned: number } {
    const heap = new MinHeap();
    for (let idx = 0; idx < this.documents.length; idx++) {
      const score = cosineSimilarity(query, this.documents[idx].vector);
      heap.push({ score, docIdx: idx });
      if (heap.size > k) heap.pop();
    }
    return { hits: this.drain(heap), scanned: this.documents.length };
  }

  private searchKd(query: number[], k: number): { hits: SearchHit[]; scanned: number } {
    const target = query.slice();
    l2Normalize(target);
    const heap = new MinHeap();
    const scanned = { n: 0 };
    if (this.kdRoot) knn(this.kdRoot, target, k, heap, scanned);
    const hits: SearchHit[] = [];
    while (heap.size) {
      const item = heap.pop();
      if (!item) break;
      const doc = this.documents[item.docIdx];
      hits.push({
        id: doc.id,
        score: cosineSimilarity(query, doc.vector),
        text: doc.text,
        metadata: doc.metadata,
      });
    }
    hits.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    return { hits, scanned: scanned.n };
  }

  private drain(heap: MinHeap): SearchHit[] {
    const hits: SearchHit[] = [];
    while (heap.size) {
      const item = heap.pop();
      if (!item) break;
      const doc = this.documents[item.docIdx];
      hits.push({ id: doc.id, score: item.score, text: doc.text, metadata: doc.metadata });
    }
    hits.reverse();
    return hits;
  }
}

function buildKd(entries: { idx: number; vec: number[] }[], depth: number, dim: number): KdNode | null {
  if (entries.length === 0) return null;
  const axis = depth % dim;
  const mid = Math.floor(entries.length / 2);
  selectNth(entries, mid, axis);
  const chosen = entries[mid];
  return {
    docIdx: chosen.idx,
    splitAxis: axis,
    splitVal: chosen.vec[axis],
    normVector: chosen.vec,
    left: buildKd(entries.slice(0, mid), depth + 1, dim),
    right: buildKd(entries.slice(mid + 1), depth + 1, dim),
  };
}

function knn(node: KdNode, target: number[], k: number, heap: MinHeap, scanned: { n: number }): void {
  scanned.n += 1;
  const distSq = squaredEuclidean(target, node.normVector);
  const neg = -distSq;
  if (heap.size < k) {
    heap.push({ score: neg, docIdx: node.docIdx });
  } else {
    const worst = heap.peek();
    if (worst && neg > worst.score) {
      heap.pop();
      heap.push({ score: neg, docIdx: node.docIdx });
    }
  }

  const diff = target[node.splitAxis] - node.splitVal;
  const near = diff <= 0 ? node.left : node.right;
  const far = diff <= 0 ? node.right : node.left;
  if (near) knn(near, target, k, heap, scanned);

  const plane = diff * diff;
  let searchFar = heap.size < k;
  if (!searchFar) {
    const worst = heap.peek();
    searchFar = worst ? plane <= -worst.score : true;
  }
  if (searchFar && far) knn(far, target, k, heap, scanned);
}
