use clap::{Parser, Subcommand};
use micro_rag::index::{DocumentRecord, IndexType, VectorIndex};
use micro_rag::math::Metric;
use micro_rag::{MicroRagEngine, RagResponse};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "micro-rag",
    version = "0.1.0",
    about = "Embedded Rust Vector Engine, KD-Tree Indexer, WAL Storage & LRU Semantic Cache"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an end-to-end demonstration of WAL ingestion, KD-Tree retrieval, and Semantic Caching
    Demo {
        #[arg(short, long, default_value = "micro_rag_demo.wal")]
        wal_path: PathBuf,
        #[arg(short, long, default_value_t = 0.92)]
        threshold: f32,
    },
    /// Run deterministic latency & recall benchmarks comparing Exact Flat vs KD-Tree and Semantic Cache speedup
    Bench {
        #[arg(short, long, default_value_t = 5_000)]
        num_vectors: usize,
        #[arg(short, long, default_value_t = 64)]
        dim: usize,
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
}

/// Deterministic Xorshift64* PRNG for reproducible synthetic embeddings without external crate bloat.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let val = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((val >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn next_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

fn run_demo(wal_path: PathBuf, threshold: f32) {
    println!("=== micro-rag End-to-End Pipeline Demo ===");
    let _ = std::fs::remove_file(&wal_path);

    let mut engine = MicroRagEngine::new(
        4,
        Metric::Cosine,
        IndexType::KdTree,
        128,
        threshold,
        Some(&wal_path),
    )
    .expect("Failed to initialize MicroRagEngine");

    let corpus = vec![
        DocumentRecord {
            id: "sys-01".into(),
            vector: vec![0.95, 0.20, 0.05, 0.01],
            text: "Linux namespaces and seccomp-bpf restrict syscall surfaces in sandboxes.".into(),
            metadata: HashMap::from([("domain".into(), "security".into())]),
        },
        DocumentRecord {
            id: "math-02".into(),
            vector: vec![0.10, 0.95, 0.15, 0.05],
            text: "The Mean Value Theorem (TAF) states f'(c) = (f(b)-f(a))/(b-a) on [a,b].".into(),
            metadata: HashMap::from([("domain".into(), "analysis".into())]),
        },
        DocumentRecord {
            id: "ml-03".into(),
            vector: vec![0.05, 0.12, 0.96, 0.18],
            text: "Semantic caches store embeddings in an LRU arena to bypass redundant LLM calls."
                .into(),
            metadata: HashMap::from([("domain".into(), "ml-systems".into())]),
        },
    ];

    for doc in corpus {
        engine.ingest(doc).expect("Failed to ingest document");
    }
    engine.index.rebuild_index().unwrap();
    println!(
        "[1] Ingested {} documents into KD-Tree and flushed CRC32 frames to {:?}",
        engine.index.len(),
        wal_path
    );

    // Query 1: Cold query (Misses cache, hits KD-Tree)
    let q1_text = "How do seccomp filters isolate processes on Linux?";
    let q1_vec = vec![0.93, 0.24, 0.04, 0.02];
    let t0 = Instant::now();
    match engine.query(q1_text, &q1_vec, 2).unwrap() {
        RagResponse::Retrieved {
            context,
            completion,
        } => {
            println!(
                "[2] Cold Query (Cache MISS in {:?}):\n    Top Hit: {} (score: {:.4})\n    Output : {}",
                     t0.elapsed(),
                     context[0].id,
                     context[0].score,
                     completion
            );
        }
        _ => unreachable!(),
    }

    // Query 2: Paraphrased query with cosine similarity > 0.92 (Hits LRU Semantic Cache)
    let q2_text = "How does Linux seccomp-bpf sandbox syscalls?";
    let q2_vec = vec![0.96, 0.18, 0.06, 0.01];
    let t1 = Instant::now();
    match engine.query(q2_text, &q2_vec, 2).unwrap() {
        RagResponse::CacheHit(hit) => {
            println!(
                "[3] Paraphrased Query (Semantic Cache HIT in {:?}):\n    Matched: '{}' (cosine: {:.4} >= {:.2})\n    Output : {}",
                     t1.elapsed(),
                     hit.matched_query,
                     hit.similarity,
                     threshold,
                     hit.completion
            );
        }
        _ => panic!("Expected semantic cache hit!"),
    }

    let _ = std::fs::remove_file(&wal_path);
}

fn run_bench(num_vectors: usize, dim: usize, top_k: usize) {
    println!(
        "=== micro-rag Benchmark ({} vectors, {} dims, top-{}) ===",
        num_vectors, dim, top_k
    );
    let mut rng = XorShift64::new(42);

    let mut flat_index = VectorIndex::new(dim, Metric::Cosine, IndexType::Flat).unwrap();
    let mut kd_index = VectorIndex::new(dim, Metric::Cosine, IndexType::KdTree).unwrap();

    for i in 0..num_vectors {
        let doc = DocumentRecord {
            id: format!("vec_{}", i),
            vector: rng.next_vec(dim),
            text: format!("Synthetic passage {}", i),
            metadata: HashMap::new(),
        };
        flat_index.upsert(doc.clone()).unwrap();
        kd_index.upsert(doc).unwrap();
    }

    let build_start = Instant::now();
    kd_index.rebuild_index().unwrap();
    let build_time = build_start.elapsed();

    let num_queries = 100;
    let queries: Vec<Vec<f32>> = (0..num_queries).map(|_| rng.next_vec(dim)).collect();

    let flat_start = Instant::now();
    let mut flat_results = Vec::with_capacity(num_queries);
    for q in &queries {
        flat_results.push(flat_index.search(q, top_k).unwrap());
    }
    let flat_elapsed = flat_start.elapsed();

    let kd_start = Instant::now();
    let mut kd_results = Vec::with_capacity(num_queries);
    for q in &queries {
        kd_results.push(kd_index.search(q, top_k).unwrap());
    }
    let kd_elapsed = kd_start.elapsed();

    // Verify exact top-K recall agreement
    let mut matches = 0usize;
    for (f_list, k_list) in flat_results.iter().zip(kd_results.iter()) {
        for (f_item, k_item) in f_list.iter().zip(k_list.iter()) {
            if f_item.id == k_item.id {
                matches += 1;
            }
        }
    }
    let recall = (matches as f64) / ((num_queries * top_k) as f64) * 100.0;

    println!("KD-Tree Build Time : {:?}", build_time);
    println!(
        "Flat Search (100q) : {:?} ({:.2} us/query)",
        flat_elapsed,
        flat_elapsed.as_secs_f64() * 1e6 / num_queries as f64
    );
    println!(
        "KD-Tree (100q)     : {:?} ({:.2} us/query) | Recall@{}: {:.1}%",
        kd_elapsed,
        kd_elapsed.as_secs_f64() * 1e6 / num_queries as f64,
        top_k,
        recall
    );
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo {
            wal_path,
            threshold,
        } => run_demo(wal_path, threshold),
        Commands::Bench {
            num_vectors,
            dim,
            top_k,
        } => run_bench(num_vectors, dim, top_k),
    }
}
