# micro-rag

[![CI](https://github.com/tahaenasraoui-debug/micro-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/tahaenasraoui-debug/micro-rag/actions/workflows/ci.yml)
[![Rust Edition](https://img.shields.io/badge/rust-2021-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2021/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

In-memory vector index, semantic query cache, and write-ahead log for local RAG pipelines.

Pure safe Rust (`#![forbid(unsafe_code)]`). No vector-database client and no vector-database server. You compute the embedding. On a cache miss the "answer" is a formatted top hit, not a language-model completion.

## What you get

- **Flat index.** Exact scan. Top-k kept in a heap of size k.
- **KD-tree.** Median split on axis `depth % dim`. Cosine search stores a unit-length copy in the tree (your `DocumentRecord::vector` is not rewritten) and prunes with squared Euclidean distance, which matches cosine order on the unit sphere: `‖u − v‖² = 2 − 2 cos(u, v)`. The score you get back is cosine. Dot-product search ignores the tree and scans. Upsert marks the tree dirty; the next search rebuilds it.
- **Semantic cache.** Arena list. Lookup walks every live entry with **cosine**, even if the index metric is Euclidean or dot product. Best score `>= threshold` wins. A tie keeps the newer entry. `put` overwrites an entry already within the threshold instead of inserting a second one.
- **WAL.** One JSON line per upsert, IEEE CRC32 over the operation bytes. Replay applies frames in order. Two upserts of the same id leave one document (the later text) and the returned count is the number of frames, not the number of ids.

## What you do not get

- An embedding model or an LLM.
- An HTTP server.
- Delete. Upsert by id replaces the record. There is no tombstone.
- Cache invalidation when a document changes. A cached completion can outlive the text it was built from.
- `File::sync_all`. Append flushes a `BufWriter`. Compact renames a temp file and does not fsync the file or the directory. A power loss can drop the tail.
- A torn-tail repair. A bad last line fails replay (`Serialization` if the JSON is cut, `ChecksumMismatch` if the CRC does not match). The file is not truncated. `MicroRagEngine::new` returns that error and drops the partial index. A direct `replay_into` caller still has the earlier frames applied.

`SemanticCache::new` panics if `capacity == 0` or if the threshold is outside `0.0..=1.0`.

## Requirements

- Stable Rust, edition 2021. Minimum supported Rust version has not been measured.
- Dependencies: `serde`, `serde_json`, `thiserror`, `clap`. `clap` is a normal dependency because the binary lives in the same package. It is not behind a feature flag.

## Build

```bash
git clone https://github.com/tahaenasraoui-debug/micro-rag.git
cd micro-rag
cargo build --release
```

The release profile is `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`.

## Demo

```bash
cargo run --release -- demo --wal-path ./demo_data.wal --threshold 0.92
```

Ingests three 4-dimensional records, rebuilds the KD-tree, runs a cold query, then a nearby query that hits the cache, then deletes the WAL file. `0.92` is the threshold you passed. It is not a quality claim.

Output from one release run on Linux, `rustc 1.99.0`. Timings will differ on your machine.

```text
=== micro-rag End-to-End Pipeline Demo ===
[1] Ingested 3 documents into KD-Tree and flushed CRC32 frames to "./demo_data.wal"
[2] Cold Query (Cache MISS in 3.41µs):
    Top Hit: sys-01 (score: 0.9989)
    Output : Synthesized answer for 'How do seccomp filters isolate processes on Linux?': [sys-01] Linux namespaces and seccomp-bpf restrict syscall surfaces in sandboxes.
[3] Paraphrased Query (Semantic Cache HIT in 277ns):
    Matched: 'How do seccomp filters isolate processes on Linux?' (cosine: 0.9975 >= 0.92)
    Output : Synthesized answer for 'How do seccomp filters isolate processes on Linux?': [sys-01] Linux namespaces and seccomp-bpf restrict syscall surfaces in sandboxes.
```

## Bench

```bash
cargo run --release -- bench --num-vectors 10000 --dim 32 --top-k 5
```

Same seeded vectors in a flat index and a KD-tree. Reports tree build time, 100 queries against each, and how often the top-k ids agree. Defaults if you omit the flags: 5000 vectors, 64 dimensions, top-5.

One release run, same toolchain as the demo. Not a multi-run benchmark.

```text
=== micro-rag Benchmark (10000 vectors, 32 dims, top-5) ===
KD-Tree Build Time : 3.883194ms
Flat Search (100q) : 39.189728ms (391.90 us/query)
KD-Tree (100q)     : 23.824765ms (238.25 us/query) | Recall@5: 100.0%
```

Regenerate either block by re-running the command above it.

## Use it as a library

```rust
use micro_rag::index::{DocumentRecord, IndexType};
use micro_rag::math::Metric;
use micro_rag::{MicroRagEngine, RagResponse};
use std::collections::BTreeMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = MicroRagEngine::new(
        4,
        Metric::Cosine,
        IndexType::KdTree,
        256,
        0.92,
        Some(Path::new("production.wal")),
    )?;

    engine.ingest(DocumentRecord {
        id: "arch-wiki-01".into(),
        vector: vec![0.91, 0.22, 0.10, 0.05],
        text: "Use pacman -Syu to synchronize and update all system packages.".into(),
        metadata: BTreeMap::from([("distro".into(), "arch".into())]),
    })?;

    // Optional. A dirty KD-tree rebuilds itself on the next non-dot-product search.
    engine.index.rebuild_index()?;

    match engine.query(
        "How do I update packages on Arch Linux?",
        &[0.90, 0.25, 0.08, 0.04],
        3,
    )? {
        RagResponse::CacheHit(hit) => {
            println!("cache hit {:.4}: {}", hit.similarity, hit.completion);
        }
        RagResponse::Retrieved { context, completion } => {
            println!("retrieved {}: {completion}", context.len());
        }
    }
    Ok(())
}
```

`ingest` checks the dimension, finiteness (rejecting `NaN`/`Inf`), and, for cosine, a non-zero L2 norm before it appends. A rejected record is not written to the log.

Scores, higher is more similar:

| Metric | Score |
| --- | --- |
| `Metric::Cosine` | cosine in [-1, 1] |
| `Metric::DotProduct` | dot product |
| `Metric::Euclidean` | negative L2 distance |

`dot_product` and `squared_euclidean_distance` use 4-wide sums so LLVM can autovectorize. `cosine_similarity` does not. There are no SIMD intrinsics.

## Layout

```text
micro-rag/
├── .github/workflows/ci.yml
├── .gitignore
├── Cargo.lock
├── Cargo.toml
├── LICENSE
├── README.md
├── rustfmt.toml
├── src/
│   ├── lib.rs               # MicroRagEngine
│   ├── main.rs              # demo and bench
│   ├── math.rs
│   ├── index.rs             # Flat and KD-tree
│   ├── storage.rs           # WAL
│   └── cache.rs
└── studio/                  # standalone TypeScript web playground (not backed by the Rust crate)
```

`studio/` is a standalone TypeScript/React web playground (TanStack Start + Vite) that mirrors the algorithms in TypeScript for interactive browser exploration; it is not backed by the Rust crate.

## Tests

```bash
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

## License

MIT. Copyright (c) 2026 Mohamed Taha Enasraoui. See [LICENSE](LICENSE).
