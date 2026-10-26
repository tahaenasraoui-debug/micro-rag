# `micro-rag` 🦀

```text
       _~^~^~_
   \) /  o o  \ (/
     '_   -   _'
     / '-----' \
   [ Zero-Bloat Rust Vector Engine ]
```

[![CI Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Rust Edition](https://img.shields.io/badge/rust-2021-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Zero Unsafe](https://img.shields.io/badge/unsafe-0%25-success.svg)]()

`micro-rag` is an ultra-lightweight, zero-bloat embedded **Vector Retrieval Engine** and **Semantic Query Cache** written in pure, safe Rust. It is engineered for edge deployments, local AI assistants, and low-latency Retrieval-Augmented Generation (RAG) pipelines where spinning up a heavyweight external database (like Qdrant, Milvus, or Postgres `pgvector`) is overkill.

---

## 📸 Visual Demo & Preview


<p align="center">
  <img src="./demo.jpg" alt="micro-rag Terminal Demo" width="800" />
</p>

*Above: Running `micro-rag demo` and `micro-rag bench` — demonstrating WAL ingestion, KD-Tree search, and sub-millisecond LRU Semantic Cache hits.*

---

## 🧠 What Does This Actually Do? (For Normal People)

When you ask an AI a question about your own documents—like *"How do I fix my Linux audio drivers?"*—two expensive things usually happen behind the scenes:
1. The computer searches through thousands of pages of notes to find the exact paragraphs about audio drivers.
2. It sends those paragraphs to a Large Language Model (LLM) to write a human-friendly answer.

If 50 people ask the same question phrased slightly differently (*"My sound isn't working on Linux, how to repair?"*), a standard system repeats all that slow, expensive work 50 times because the words aren't an exact letter-for-letter match.

**`micro-rag` solves this with four clever tricks:**

1. **Numbers Instead of Words (Vector Search):** Instead of matching exact words like `Ctrl+F`, it turns sentences into lists of numbers (coordinates on a giant map) based on their *meaning*. Sentences with similar meanings land right next to each other on the map.
2. **A Smart Shortcut Map (`KD-Tree`):** Instead of checking every single document one by one (`Flat Search`), `micro-rag` slices the map in half repeatedly—like playing *"20 Questions"* with geography—so it can jump straight to the right neighborhood of documents in microseconds.
3. **A "Déjà Vu" Memory Bank (`Semantic Cache`):** Before doing any heavy lifting, `micro-rag` checks if someone recently asked a question that *means* the same thing (even if phrased differently). If the meaning is at least **92% similar**, it instantly hands back the previous answer in less than a millisecond—saving time, battery, and API costs. To keep memory tidy, it automatically forgets the oldest, least-popular questions (`LRU Eviction`).
4. **A Crash-Proof Notebook (`WAL Persistence`):** Every time you add a document, `micro-rag` scribbles it at the very end of a digital ledger on your hard drive along with a security stamp (`CRC32 checksum`). If your computer suddenly loses power, `micro-rag` reads that ledger when it wakes back up and rebuilds your database without losing a single byte.

---

## 🏗️ System Architecture

```text
                        +---------------------------------------+
                        |       Incoming Query + Embedding      |
                        |   "How do seccomp filters work?"      |
                        |        [0.93, 0.24, 0.04, 0.02]       |
                        +-------------------+-------------------+
                                            |
                                            v
                        +---------------------------------------+
                        |       LRU Semantic Cache Layer        |
                        |  Arena Doubly-Linked List (O(1) Move) |
                        |     Cosine Similarity >= Threshold    |
                        +---------+-------------------+---------+
                                  |                   |
                         HIT (>= 0.92)           MISS (< 0.92)
                                  |                   |
                                  v                   v
      +---------------------------+     +-------------+-------------------------+
      | Return Cached Completion  |     |         Core Vector Index             |
      |   & Promote Node to MRU   |     |                                       |
      |   (Latency: ~1-5 us)      |     |  +---------------+  +--------------+  |
      +---------------------------+     |  |  Exact Flat   |  |   KD-Tree    |  |
                                        |  | 4-Way Unroll  |  | Median-Split |  |
                                        |  |  SIMD Kernels |  |  Branch/Bnd  |  |
                                        |  +---------------+  +--------------+  |
                                        +-------------+-------------------------+
                                                      |
                                                      | Top-K Documents (Min-Heap)
                                                      v
                                        +---------------------------------------+
                                        |   Downstream Synthesis & Cache Fill   |
                                        |  Inserts new completion at MRU Head   |
                                        +---------------------------------------+
                                                      ^
                                                      | Replay / Append / Compact
                                        +-------------+-------------------------+
                                        |     Append-Only WAL Persistence       |
                                        |   JSONL Frames + IEEE CRC32 Checksum  |
                                        +---------------------------------------+
```

---

## ⚙️ Deep-Dive Engineering & Mathematical Foundations

### 1. Hardware-Friendly Distance Kernels (`src/math.rs`)
All similarity and distance operations are implemented with **4-way loop unrolling** (`sum0`, `sum1`, `sum2`, `sum3`) to break single-instruction floating-point data dependency chains, allowing LLVM to auto-vectorize loops into `AVX2` / `AVX-512` or ARM `NEON` SIMD instructions without requiring nightly intrinsics.

| Metric | Mathematical Formula | Unified Ranking Score | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Cosine Similarity** | $$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\Vert{}\mathbf{A}\Vert{}_2 \Vert{}\mathbf{B}\Vert{}_2}$$ | $\cos(\theta) \in [-1.0, 1.0]$ | Normalized text embeddings where direction matters more than magnitude. |
| **Dot Product** | $$\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{d} A_i B_i$$ | $\mathbf{A} \cdot \mathbf{B} \in (-\infty, \infty)$ | Pre-normalized unit embeddings (fastest execution) or magnitude-weighted vectors. |
| **Euclidean ($L_2$)** | $$d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{d} (A_i - B_i)^2}$$ | $-d(\mathbf{A}, \mathbf{B}) \in (-\infty, 0.0]$ | Spatial coordinates, image feature maps, and dense low-dimensional manifolds. |

* **Edge-Case Safety:** Explicitly guards against empty vectors (`VectorError::EmptyVector`), dimension mismatches (`VectorError::DimensionMismatch`), and zero-norm vectors (`VectorError::ZeroMagnitudeVector` when $\Vert{}\mathbf{v}\Vert{}_2^2 \le \epsilon$).

### 2. Dual Indexing Strategies (`src/index.rs`)
* **Exact Flat Index (`IndexType::Flat`):** Performs a linear scan over contiguous memory using a bounded **Min-Heap of size $K$** (`std::collections::BinaryHeap` with inverted `Ord`). This bounds memory overhead to $O(K)$ and reduces sorting complexity from $O(N \log N)$ to $O(N \log K)$.
* **Median-Split KD-Tree (`IndexType::KdTree`):**
  * **Construction:** Partitions vectors recursively across cycling coordinate axes ($\text{axis} = \text{depth} \bmod d$) in $O(N \log N)$ time using Quickselect (`slice::select_nth_unstable_by`).
  * **Cosine-to-Euclidean Isomorphism:** Because Cosine similarity does not natively satisfy the coordinate-axis hyperplane pruning rules of a KD-Tree, `micro-rag` automatically projects all vectors onto the unit hypersphere ($\Vert{}\hat{\mathbf{u}}\Vert{}_2 = \Vert{}\hat{\mathbf{v}}\Vert{}_2 = 1$). On the unit hypersphere, squared Euclidean distance is strictly monotonically decreasing with respect to Cosine similarity:
    $$\Vert{}\hat{\mathbf{u}} - \hat{\mathbf{v}}\Vert{}_2^2 = \Vert{}\hat{\mathbf{u}}\Vert{}_2^2 + \Vert{}\hat{\mathbf{v}}\Vert{}_2^2 - 2(\hat{\mathbf{u}} \cdot \hat{\mathbf{v}}) = 2 - 2\cos(\theta)$$
    This enables exact branch-and-bound hyperplane pruning (`plane_dist_sq <= worst_dist_sq`) while returning exact Cosine scores.

### 3. Arena-Backed LRU Semantic Cache (`src/cache.rs`)
Standard doubly-linked lists in Rust often require `Rc<RefCell<Node>>` (heap allocation overhead) or `unsafe` raw pointers (`*mut Node`). `micro-rag` implements a **100% safe Arena Doubly-Linked List**:
* Nodes live inside a contiguous `Vec<Option<ArenaNode>>` indexed by `usize` handles.
* Evicted slots are recycled via a `free_slots: Vec<usize>` stack in $O(1)$ time with zero heap reallocations once warm.
* Promoting a cache hit to the Most Recently Used (`head`) position or evicting the Least Recently Used (`tail`) node requires only $O(1)$ integer index swaps.

### 4. Crash-Safe Write-Ahead Log (`src/storage.rs`)
Every ingested or updated `DocumentRecord` is serialized into a newline-delimited JSON frame paired with a dependency-free **IEEE 802.3 CRC32** checksum:
```json
{"crc32":3489210412,"op":{"Upsert":{"id":"sys-01","vector":[0.95,0.2,0.05,0.01],"text":"Linux namespaces...","metadata":{"domain":"security"}}}}
```
* **Replay Verification:** On startup, `WalStorage::replay_into` recomputes the CRC32 of every operation payload and halts with `StorageError::ChecksumMismatch` if a torn write or disk corruption is detected.
* **Atomic Compaction:** Calling `WalStorage::compact` deduplicates historical overwrites by writing live records to a `.wal.tmp` file, flushing buffers, and issuing an atomic POSIX filesystem `rename`.

---

## 📂 Repository Structure

```text
micro-rag/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI: fmt, clippy, unit tests, CLI smoke tests
├── assets/
│   └── demo.png            # Place your terminal screenshot or GIF here
├── Cargo.toml              # Package manifest (clap, serde, serde_json, thiserror)
├── README.md               # Architecture, plain-English guide, and API documentation
└── src/
    ├── lib.rs              # High-level MicroRagEngine facade combining Index, Cache & WAL
    ├── main.rs             # CLI binary (`demo` and `bench` subcommands + Xorshift64 PRNG)
    ├── math.rs             # Unrolled SIMD-friendly Cosine, DotProduct, Euclidean kernels
    ├── index.rs            # Flat & KD-Tree vector indexes + Min-Heap top-K collector
    ├── storage.rs          # Append-only WAL with IEEE CRC32 verification & compaction
    └── cache.rs            # Safe arena-allocated doubly-linked LRU Semantic Cache
```

---

## 🚀 Quickstart & CLI Usage

### Prerequisites
* **Rust Toolchain:** `rustc >= 1.75` (stable) and `cargo`.

### 1. Build the Project
```bash
git clone [https://github.com/your-username/micro-rag.git](https://github.com/your-username/micro-rag.git)
cd micro-rag
cargo build --release
```

### 2. Run the End-to-End Pipeline Demo
The `demo` command initializes a WAL file, ingests sample documents into a KD-Tree, runs a cold query (Cache Miss -> Vector Retrieval), and immediately follows up with a paraphrased query to demonstrate a sub-millisecond **Semantic Cache Hit**:

```bash
cargo run --release -- demo --wal-path ./demo_data.wal --threshold 0.92
```

**Expected Output:**
```text
=== micro-rag End-to-End Pipeline Demo ===
[1] Ingested 3 documents into KD-Tree and flushed CRC32 frames to "./demo_data.wal"
[2] Cold Query (Cache MISS in 18.42µs):
    Top Hit: sys-01 (score: 0.9987)
    Output : Synthesized answer for 'How do seccomp filters isolate processes on Linux?': [sys-01] Linux namespaces and seccomp-bpf restrict syscall surfaces in sandboxes.
[3] Paraphrased Query (Semantic Cache HIT in 1.15µs):
    Matched: 'How do seccomp filters isolate processes on Linux?' (cosine: 0.9971 >= 0.92)
    Output : Synthesized answer for 'How do seccomp filters isolate processes on Linux?': [sys-01] Linux namespaces and seccomp-bpf restrict syscall surfaces in sandboxes.
```

### 3. Run the Latency & Recall Benchmark
The `bench` command generates deterministic pseudo-random vectors (via an internal `XorShift64*` PRNG) and compares `IndexType::Flat` against `IndexType::KdTree`, reporting per-query latency in microseconds and verifying `Recall@K`:

```bash
cargo run --release -- bench --num-vectors 10000 --dim 32 --top-k 5
```

**Sample Output (Release Profile, LTO = fat):**
```text
=== micro-rag Benchmark (10000 vectors, 32 dims, top-5) ===
KD-Tree Build Time : 8.41ms
Flat Search (100q) : 14.20ms (142.00 us/query)
KD-Tree (100q)     : 4.85ms (48.50 us/query) | Recall@5: 100.0%
```

---

## 💻 Library API Example (Embedding `micro-rag` in Rust)

You can import `micro_rag` directly as a library inside your own Rust services:

```rust
use micro_rag::index::{DocumentRecord, IndexType};
use micro_rag::math::Metric;
use micro_rag::{MicroRagEngine, RagResponse};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize a 4-dimensional engine with KD-Tree indexing,
    // an LRU cache capacity of 256 queries, a 0.92 cosine threshold, and WAL persistence.
    let mut engine = MicroRagEngine::new(
        4,
        Metric::Cosine,
        IndexType::KdTree,
        256,
        0.92,
        Some(Path::new("production.wal")),
    )?;

    // Ingest a document (automatically checksummed and appended to production.wal)
    engine.ingest(DocumentRecord {
        id: "arch-wiki-01".into(),
        vector: vec![0.91, 0.22, 0.10, 0.05],
        text: "Use pacman -Syu to synchronize and update all system packages.".into(),
        metadata: HashMap::from([("distro".into(), "arch".into())]),
    })?;

    // Rebuild the KD-Tree after bulk ingestion
    engine.index.rebuild_index()?;

    // Query the engine
    let response = engine.query(
        "How do I update packages on Arch Linux?",
        &[0.90, 0.25, 0.08, 0.04],
        3,
    )?;

    match response {
        RagResponse::CacheHit(hit) => {
            println!("Served from Semantic Cache! (Sim: {:.4}): {}", hit.similarity, hit.completion);
        }
        RagResponse::Retrieved { context, completion } => {
            println!("Retrieved {} docs. Answer: {}", context.len(), completion);
        }
    }

    Ok(())
}
```

---

## 🧪 Testing & Continuous Integration

`micro-rag` includes unit tests covering orthogonal/collinear vectors, zero-magnitude error handling, Flat vs. KD-Tree recall equivalence, WAL replay & compaction, and LRU cache promotion/eviction order.

```bash
# Run all unit and integration tests
cargo test --all-targets --all-features

# Run strict linter checks
cargo clippy --all-targets --all-features -- -D warnings

# Verify formatting
cargo fmt --all -- --check
```

---

## 📄 License

Distributed under the **MIT License**. See `Cargo.toml` for metadata details.
