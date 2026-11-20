import { VectorIndex, type DocumentRecord } from "./index";

const MASK = (1n << 64n) - 1n;

class XorShift64 {
  state: bigint;

  constructor(seed: bigint) {
    this.state = seed === 0n ? 0xdeadbeefcafebaben : seed;
  }

  nextF32(): number {
    let x = this.state;
    x = (x ^ (x >> 12n)) & MASK;
    x = (x ^ ((x << 25n) & MASK)) & MASK;
    x = (x ^ (x >> 27n)) & MASK;
    this.state = x;
    const val = (x * 0x2545f4914f6cdd1dn) & MASK;
    const top = Number((val >> 40n) & 0xffffffn);
    return (top / 16777216) * 2 - 1;
  }

  nextVec(dim: number): number[] {
    const out = new Array<number>(dim);
    for (let i = 0; i < dim; i++) out[i] = this.nextF32();
    return out;
  }
}

export type BenchResult = {
  vectors: number;
  dim: number;
  topK: number;
  queries: number;
  buildMs: number;
  flatMs: number;
  kdMs: number;
  flatUs: number;
  kdUs: number;
  recall: number;
  scannedFlat: number;
  scannedKd: number;
};

function yieldThread(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

export async function runBench(
  vectors: number,
  dim: number,
  topK: number,
  queries: number,
  onPhase?: (phase: string) => void,
): Promise<BenchResult> {
  onPhase?.("Sampling vectors");
  await yieldThread();
  const rng = new XorShift64(42n);
  const flat = new VectorIndex(dim, "flat");
  const kd = new VectorIndex(dim, "kdtree");
  for (let i = 0; i < vectors; i++) {
    const doc: DocumentRecord = {
      id: `vec_${i}`,
      vector: rng.nextVec(dim),
      text: `Synthetic passage ${i}`,
      metadata: {},
    };
    flat.upsert({ ...doc, vector: doc.vector.slice() });
    kd.upsert(doc);
  }

  onPhase?.("Building KD-tree");
  await yieldThread();
  const buildStart = performance.now();
  kd.rebuildIndex();
  const buildMs = performance.now() - buildStart;

  const qs: number[][] = [];
  for (let i = 0; i < queries; i++) qs.push(rng.nextVec(dim));

  onPhase?.("Scanning flat index");
  await yieldThread();
  const flatHits: string[][] = [];
  let scannedFlat = 0;
  const flatStart = performance.now();
  for (const q of qs) {
    const found = flat.search(q, topK);
    scannedFlat += found.scanned;
    flatHits.push(found.hits.map((h) => h.id));
  }
  const flatMs = performance.now() - flatStart;

  onPhase?.("Searching KD-tree");
  await yieldThread();
  const kdHits: string[][] = [];
  let scannedKd = 0;
  const kdStart = performance.now();
  for (const q of qs) {
    const found = kd.search(q, topK);
    scannedKd += found.scanned;
    kdHits.push(found.hits.map((h) => h.id));
  }
  const kdMs = performance.now() - kdStart;

  let matches = 0;
  for (let i = 0; i < queries; i++) {
    const a = flatHits[i];
    const b = kdHits[i];
    const n = Math.min(a.length, b.length);
    for (let j = 0; j < n; j++) if (a[j] === b[j]) matches += 1;
  }
  const recall = (matches / (queries * topK)) * 100;

  return {
    vectors,
    dim,
    topK,
    queries,
    buildMs,
    flatMs,
    kdMs,
    flatUs: (flatMs * 1000) / queries,
    kdUs: (kdMs * 1000) / queries,
    recall,
    scannedFlat,
    scannedKd,
  };
}
