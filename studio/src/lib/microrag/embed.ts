/**
 * Local hashing embedder so typed questions land in the same space as the corpus.
 * Topic axes keep paraphrases close; the hashing trick keeps distinct notes apart.
 * This is the demo's stand-in for an embedding model — retrieval itself is micro-rag.
 */

import { l2Normalize } from "./math";

export const EMBED_DIM = 48;

const AXES: { phrases: string[] }[] = [
  { phrases: ["seccomp", "syscall", "syscalls", "sandbox", "sandboxes", "namespace", "namespaces", "bpf", "isolate", "isolates", "isolation", "seccomp-bpf"] },
  { phrases: ["sound", "audio", "alsa", "pipewire", "pulseaudio", "speaker", "speakers", "headphone", "headphones", "volume"] },
  { phrases: ["mean value", "theorem", "derivative", "rolle", "calculus", "slope"] },
  { phrases: ["semantic cache", "lru", "embedding", "embeddings", "cosine", "llm", "paraphrase", "paraphrases"] },
  { phrases: ["kd-tree", "kdtree", "k-d", "prune", "prunes", "pruning", "nearest", "flat scan", "flat index"] },
  { phrases: ["write-ahead", "wal", "crc32", "checksum", "replay", "crash", "compaction", "torn write"] },
  { phrases: ["borrow checker", "ownership", "lifetimes", "rust"] },
];

const STOP = new Set([
  "a", "an", "the", "to", "of", "and", "on", "my", "how", "do", "i", "is", "for", "in",
  "with", "it", "that", "this", "what", "does", "or", "be", "not", "are", "was", "as",
  "by", "from", "at", "into", "your", "you", "me", "we", "can", "if", "when", "why",
  "about", "just", "than", "then", "so", "but", "its", "their",
]);

function hash32(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/isn't/g, "is not")
    .replace(/don't/g, "do not")
    .replace(/[^a-z0-9+.-]+/g, " ")
    .split(/\s+/)
    .map((t) => t.replace(/^[.+-]+|[.+-]+$/g, ""))
    .filter((t) => t.length >= 3 && !STOP.has(t));
}

export function embedText(text: string): number[] {
  const v = new Array<number>(EMBED_DIM).fill(0);
  const lower = text.toLowerCase();
  const axisCount = AXES.length;

  for (let axis = 0; axis < AXES.length; axis++) {
    for (const phrase of AXES[axis].phrases) {
      let from = 0;
      let hits = 0;
      while (from <= lower.length) {
        const at = lower.indexOf(phrase, from);
        if (at < 0) break;
        const before = at === 0 || !/[a-z0-9]/.test(lower[at - 1]);
        const after = at + phrase.length >= lower.length || !/[a-z0-9]/.test(lower[at + phrase.length]);
        if (before && after) hits += 1;
        from = at + Math.max(1, phrase.length);
      }
      if (hits > 0) v[axis] += 7.5 * hits;
    }
  }

  const tokens = tokenize(text);
  const rest = EMBED_DIM - axisCount;
  for (const token of tokens) {
    const h = hash32(token);
    const idx = axisCount + (h % rest);
    const sign = (h & 1) === 0 ? 1 : -1;
    v[idx] += sign * 0.72;
  }
  for (let i = 0; i < tokens.length - 1; i++) {
    const h = hash32(`${tokens[i]}_${tokens[i + 1]}`);
    const idx = axisCount + (h % rest);
    const sign = (h & 2) === 0 ? 1 : -1;
    v[idx] += sign * 0.45;
  }

  let blank = true;
  for (let i = 0; i < v.length; i++) {
    if (v[i] !== 0) {
      blank = false;
      break;
    }
  }
  if (blank) {
    const h = hash32(lower || "empty");
    v[axisCount + (h % rest)] = 1;
  }
  l2Normalize(v);
  return v;
}
