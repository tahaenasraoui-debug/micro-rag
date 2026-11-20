export type SeedDoc = {
  id: string;
  domain: string;
  text: string;
};

/** Notes the playground retrieves over. Wording tracks the upstream README story. */
export const SEED_CORPUS: SeedDoc[] = [
  {
    id: "sys-01",
    domain: "security",
    text: "Linux namespaces and seccomp-bpf restrict syscall surfaces so a sandbox can isolate a process.",
  },
  {
    id: "sys-02",
    domain: "security",
    text: "Seccomp filters isolate processes by blocking dangerous syscalls before they reach the kernel.",
  },
  {
    id: "audio-01",
    domain: "audio",
    text: "When sound is not working, restart PipeWire and check that the ALSA device is the default speaker.",
  },
  {
    id: "audio-02",
    domain: "audio",
    text: "PulseAudio driver repairs: confirm the card is unmuted, then reload the audio module if the speaker stays silent.",
  },
  {
    id: "math-02",
    domain: "analysis",
    text: "The mean value theorem says there is a c in (a, b) where the derivative equals (f(b) - f(a)) / (b - a).",
  },
  {
    id: "math-03",
    domain: "analysis",
    text: "Rolle's theorem is the mean value theorem in the special case where f(a) equals f(b), so the slope is zero.",
  },
  {
    id: "ml-03",
    domain: "cache",
    text: "A semantic cache stores embeddings in an LRU arena and returns a previous completion when cosine similarity is high enough to skip the LLM.",
  },
  {
    id: "ml-04",
    domain: "cache",
    text: "Paraphrases should hit the semantic cache: same meaning, different words, no second model call.",
  },
  {
    id: "idx-01",
    domain: "index",
    text: "A KD-tree partitions vector space and prunes any branch whose splitting plane is farther than the current nearest neighbors.",
  },
  {
    id: "idx-02",
    domain: "index",
    text: "A flat index scans every vector. It is exact and simple, and its cost grows linearly with the corpus.",
  },
  {
    id: "wal-01",
    domain: "storage",
    text: "The write-ahead log appends every upsert as a JSON frame with a CRC32 checksum so a crash can replay cleanly.",
  },
  {
    id: "wal-02",
    domain: "storage",
    text: "Compaction rewrites the WAL down to the latest live documents and drops torn or duplicate frames.",
  },
  {
    id: "rs-01",
    domain: "rust",
    text: "Rust's borrow checker enforces ownership and lifetimes so memory safety does not need a garbage collector.",
  },
];

export const PROMPTS: { q: string; follow: string }[] = [
  {
    q: "How do seccomp filters isolate processes on Linux?",
    follow: "How does Linux seccomp-bpf sandbox syscalls?",
  },
  {
    q: "My sound isn't working, how do I repair it?",
    follow: "ALSA and PipeWire: no audio from the speakers",
  },
  {
    q: "What does the mean value theorem say?",
    follow: "Explain the derivative (f(b) - f(a)) / (b - a)",
  },
  {
    q: "Why keep a semantic cache in front of an LLM?",
    follow: "How does an LRU embedding cache skip duplicate generations?",
  },
];
