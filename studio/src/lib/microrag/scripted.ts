/** The README `demo` subcommand, with the same 4-D vectors. */

import { SemanticCache } from "./cache";
import { cosineSimilarity } from "./math";
import { VectorIndex, type DocumentRecord } from "./index";
import { crc32Ieee } from "./wal";

export type DemoLine = { kind: "info" | "miss" | "hit"; text: string };

const CORPUS: DocumentRecord[] = [
  {
    id: "sys-01",
    vector: [0.95, 0.2, 0.05, 0.01],
    text: "Linux namespaces and seccomp-bpf restrict syscall surfaces in sandboxes.",
    metadata: { domain: "security" },
  },
  {
    id: "math-02",
    vector: [0.1, 0.95, 0.15, 0.05],
    text: "The Mean Value Theorem (TAF) states f'(c) = (f(b)-f(a))/(b-a) on [a,b].",
    metadata: { domain: "analysis" },
  },
  {
    id: "ml-03",
    vector: [0.05, 0.12, 0.96, 0.18],
    text: "Semantic caches store embeddings in an LRU arena to bypass redundant LLM calls.",
    metadata: { domain: "ml-systems" },
  },
];

export function runScriptedDemo(threshold = 0.92): DemoLine[] {
  const index = new VectorIndex(4, "kdtree");
  const cache = new SemanticCache(128, 4, threshold);
  const lines: DemoLine[] = [{ kind: "info", text: "=== micro-rag end-to-end pipeline ===" }];
  let bytes = 0;
  for (const doc of CORPUS) {
    index.upsert(doc);
    bytes += new TextEncoder().encode(JSON.stringify({ id: doc.id, vector: doc.vector })).length;
  }
  index.rebuildIndex();
  const checksum = crc32Ieee(new TextEncoder().encode(CORPUS.map((d) => d.id).join("|")));
  lines.push({
    kind: "info",
    text: `[1] Ingested ${index.length} documents into a KD-tree. WAL frames ${bytes} bytes, crc32 ${checksum.toString(16)}.`,
  });

  const q1 = "How do seccomp filters isolate processes on Linux?";
  const v1 = [0.93, 0.24, 0.04, 0.02];
  const t0 = performance.now();
  const miss = cache.get(v1);
  const searched = index.search(v1, 2);
  const top = searched.hits[0];
  const completion = `Synthesized answer for '${q1}': [${top.id}] ${top.text}`;
  cache.put(q1, v1, completion);
  const missUs = (performance.now() - t0) * 1000;
  lines.push({
    kind: "miss",
    text: `[2] Cold query — cache MISS in ${missUs.toFixed(1)} µs (best cached cosine ${miss.bestSimilarity < 0 ? "n/a" : miss.bestSimilarity.toFixed(4)}).\n    Top hit: ${top.id} (score ${top.score.toFixed(4)}, scanned ${searched.scanned}/${index.length})\n    Output: ${completion}`,
  });

  const q2 = "How does Linux seccomp-bpf sandbox syscalls?";
  const v2 = [0.96, 0.18, 0.06, 0.01];
  const t1 = performance.now();
  const hit = cache.get(v2);
  const hitUs = (performance.now() - t1) * 1000;
  const sim = cosineSimilarity(v1, v2);
  if (hit.hit) {
    lines.push({
      kind: "hit",
      text: `[3] Paraphrase — semantic cache HIT in ${hitUs.toFixed(1)} µs.\n    Matched '${hit.hit.matchedQuery}' (cosine ${hit.hit.similarity.toFixed(4)} >= ${threshold.toFixed(2)}, probe ${sim.toFixed(4)})\n    Output: ${hit.hit.completion}`,
    });
  } else {
    lines.push({
      kind: "miss",
      text: `[3] Expected a cache hit but cosine was ${sim.toFixed(4)} against threshold ${threshold.toFixed(2)}.`,
    });
  }
  return lines;
}
