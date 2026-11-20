import { SemanticCache } from "./cache";
import { SEED_CORPUS } from "./corpus";
import { EMBED_DIM, embedText } from "./embed";
import { VectorIndex, type DocumentRecord, type IndexType } from "./index";
import { corruptFrame, frameFromRecord, verifyFrame, type WalFrame } from "./wal";

export type MapPoint = {
  id: string;
  domain: string;
  x: number;
  y: number;
  score: number | null;
};

export type Trace = {
  id: number;
  query: string;
  outcome: "hit" | "miss";
  answer: string;
  matchedQuery: string | null;
  similarity: number;
  threshold: number;
  scanned: number;
  total: number;
  indexType: IndexType;
  us: { embed: number; cache: number; search: number };
  hits: { id: string; score: number; text: string; domain: string }[];
  follow: string | null;
};

export type CacheChip = {
  query: string;
  hits: number;
  preview: string;
  mru: boolean;
};

export type DocCard = {
  id: string;
  domain: string;
  text: string;
};

export type Snap = {
  docs: DocCard[];
  cache: CacheChip[];
  wal: WalFrame[];
  threshold: number;
  topK: number;
  indexType: IndexType;
  lookups: number;
  cacheHits: number;
  trace: Trace | null;
  replayError: string | null;
  points: MapPoint[];
  queryPoint: { x: number; y: number } | null;
  noteCount: number;
};

const CACHE_CAP = 6;
const TOPIC_COUNT = 7;

function completionFrom(hits: { text: string }[]): string {
  if (hits.length === 0) return "No note was close enough to answer that.";
  return hits[0].text;
}

/** Place a vector on the topic axis it activates. Percentages, origin top-left. */
function placeVector(vec: number[], id: string): { x: number; y: number } {
  let axis = 0;
  let best = 0;
  const limit = Math.min(TOPIC_COUNT, vec.length);
  for (let i = 0; i < limit; i++) {
    const mag = Math.abs(vec[i]);
    if (mag > best) {
      best = mag;
      axis = i;
    }
  }
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 33 + id.charCodeAt(i)) >>> 0;
  const jitter = ((hash % 100) / 100 - 0.5) * 0.38;
  const angle = -Math.PI / 2 + (axis + jitter) * ((Math.PI * 2) / TOPIC_COUNT);
  const radius = best > 0.2 ? 34 : 8;
  return {
    x: 50 + Math.cos(angle) * radius,
    y: 50 + Math.sin(angle) * radius,
  };
}

export class Engine {
  index: VectorIndex;
  cache: SemanticCache;
  wal: WalFrame[] = [];
  topK = 2;
  trace: Trace | null = null;
  replayError: string | null = null;
  private traceSeq = 0;
  private noteSeq = 0;
  private lastQueryVec: number[] | null = null;
  private lastQueryId = "query";

  private constructor(index: VectorIndex, cache: SemanticCache) {
    this.index = index;
    this.cache = cache;
  }

  static boot(): Engine {
    const index = new VectorIndex(EMBED_DIM, "kdtree");
    const cache = new SemanticCache(CACHE_CAP, EMBED_DIM, 0.92);
    const engine = new Engine(index, cache);
    for (const doc of SEED_CORPUS) {
      engine.ingest(doc.id, doc.domain, doc.text);
    }
    engine.index.rebuildIndex();
    return engine;
  }

  reset(): void {
    const fresh = Engine.boot();
    this.index = fresh.index;
    this.cache = fresh.cache;
    this.wal = fresh.wal;
    this.topK = 2;
    this.trace = null;
    this.replayError = null;
    this.traceSeq = 0;
    this.noteSeq = 0;
    this.lastQueryVec = null;
  }

  setThreshold(value: number): void {
    this.cache.similarityThreshold = Math.min(0.99, Math.max(0.5, value));
  }

  setTopK(value: number): void {
    this.topK = value;
  }

  setIndexType(type: IndexType): void {
    if (this.index.indexType === type) return;
    const next = new VectorIndex(EMBED_DIM, type);
    for (const doc of this.index.documents) next.upsert(doc);
    if (type === "kdtree") next.rebuildIndex();
    this.index = next;
  }

  addNote(text: string): void {
    const clean = text.trim();
    if (!clean) return;
    this.noteSeq += 1;
    this.ingest(`note-${this.noteSeq}`, "note", clean);
    this.index.rebuildIndex();
    this.replayError = null;
  }

  query(text: string, follow: string | null): Trace {
    const started = performance.now();
    const embedding = embedText(text);
    const embedUs = (performance.now() - started) * 1000;

    const cacheStart = performance.now();
    const probe = this.cache.get(embedding);
    const cacheUs = (performance.now() - cacheStart) * 1000;

    let answer: string;
    let hits: Trace["hits"] = [];
    let scanned = 0;
    let searchUs = 0;
    let outcome: Trace["outcome"];
    let similarity: number;
    let matched: string | null = null;

    if (probe.hit) {
      outcome = "hit";
      answer = probe.hit.completion;
      similarity = probe.hit.similarity;
      matched = probe.hit.matchedQuery;
    } else {
      const searchStart = performance.now();
      const found = this.index.search(embedding, this.topK);
      searchUs = (performance.now() - searchStart) * 1000;
      scanned = found.scanned;
      hits = found.hits.map((hit) => ({
        id: hit.id,
        score: hit.score,
        text: hit.text,
        domain: hit.metadata.domain ?? "note",
      }));
      answer = completionFrom(hits);
      this.cache.put(text, embedding, answer);
      outcome = "miss";
      similarity = probe.bestSimilarity;
    }

    this.lastQueryVec = embedding;
    this.lastQueryId = text;
    this.traceSeq += 1;
    this.trace = {
      id: this.traceSeq,
      query: text,
      outcome,
      answer,
      matchedQuery: matched,
      similarity,
      threshold: this.cache.similarityThreshold,
      scanned,
      total: this.index.length,
      indexType: this.index.indexType,
      us: { embed: embedUs, cache: cacheUs, search: searchUs },
      hits,
      follow,
    };
    return this.trace;
  }

  compact(): void {
    this.wal = this.index.documents.map((doc, i) => frameFromRecord(i + 1, doc));
    this.replayError = null;
  }

  corruptLast(): void {
    if (this.wal.length === 0) return;
    const next = this.wal.slice();
    next[next.length - 1] = corruptFrame(next[next.length - 1]);
    this.wal = next;
  }

  replay(): string | null {
    const rebuilt = new VectorIndex(EMBED_DIM, this.index.indexType);
    for (const frame of this.wal) {
      const check = verifyFrame(frame);
      if (!check.ok) {
        this.replayError = `Checksum mismatch on frame ${frame.n}: stored ${frame.crc.toString(16).padStart(8, "0")}, computed ${check.actual.toString(16).padStart(8, "0")}. Replay stopped.`;
        this.index = rebuilt;
        if (this.index.indexType === "kdtree") this.index.rebuildIndex();
        return this.replayError;
      }
      const parsed = JSON.parse(frame.payload) as {
        id: string;
        text: string;
        metadata: Record<string, string>;
        vector: number[];
      };
      rebuilt.upsert({
        id: parsed.id,
        text: parsed.text,
        metadata: parsed.metadata,
        vector: parsed.vector,
      });
    }
    if (rebuilt.indexType === "kdtree") rebuilt.rebuildIndex();
    this.index = rebuilt;
    this.replayError = null;
    return null;
  }

  snapshot(): Snap {
    const scoreById = new Map<string, number>();
    if (this.trace?.outcome === "miss") {
      for (const hit of this.trace.hits) scoreById.set(hit.id, hit.score);
    }
    const points = this.index.documents.map((doc) => {
      const placed = placeVector(doc.vector, doc.id);
      return {
        id: doc.id,
        domain: doc.metadata.domain ?? "note",
        x: placed.x,
        y: placed.y,
        score: scoreById.get(doc.id) ?? null,
      };
    });
    const queryPoint = this.lastQueryVec ? placeVector(this.lastQueryVec, this.lastQueryId) : null;
    const chips = this.cache.entries();
    return {
      docs: this.index.documents.map((doc) => ({
        id: doc.id,
        domain: doc.metadata.domain ?? "note",
        text: doc.text,
      })),
      cache: chips.map((entry, i) => ({
        query: entry.queryText,
        hits: entry.hits,
        preview: entry.completion,
        mru: i === 0,
      })),
      wal: this.wal,
      threshold: this.cache.similarityThreshold,
      topK: this.topK,
      indexType: this.index.indexType,
      lookups: this.cache.totalLookups,
      cacheHits: this.cache.totalHits,
      trace: this.trace,
      replayError: this.replayError,
      points,
      queryPoint,
      noteCount: this.noteSeq,
    };
  }

  private ingest(id: string, domain: string, text: string): void {
    const record: DocumentRecord = {
      id,
      text,
      vector: embedText(text),
      metadata: { domain },
    };
    this.wal = [...this.wal, frameFromRecord(this.wal.length + 1, record)];
    this.index.upsert(record);
  }
}

export function formatUs(us: number): string {
  if (!Number.isFinite(us) || us < 1) return "< 1 µs";
  if (us < 1000) return `${us.toFixed(us < 10 ? 1 : 0)} µs`;
  return `${(us / 1000).toFixed(2)} ms`;
}
