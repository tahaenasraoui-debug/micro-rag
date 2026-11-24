#![forbid(unsafe_code)]

pub mod cache;
pub mod index;
pub mod math;
pub mod storage;

use cache::{CacheHit, SemanticCache};
use index::{DocumentRecord, IndexType, SearchResult, VectorIndex};
use math::{validate_finite, Metric, VectorError};
use std::path::Path;
use storage::{StorageError, WalStorage};

#[derive(Debug, Clone)]
pub enum RagResponse {
    /// Returned from the semantic cache. The index was not searched.
    CacheHit(CacheHit),
    /// Cache missed. `completion` is a formatted top hit, not model output.
    Retrieved {
        context: Vec<SearchResult>,
        completion: String,
    },
}

/// In-memory index, optional WAL, and semantic cache.
///
/// There is no embedding model and no language model in this type.
pub struct MicroRagEngine {
    pub index: VectorIndex,
    pub cache: SemanticCache,
    pub wal: Option<WalStorage>,
}

impl MicroRagEngine {
    pub fn new(
        dim: usize,
        metric: Metric,
        index_type: IndexType,
        cache_capacity: usize,
        cache_threshold: f32,
        wal_path: Option<&Path>,
    ) -> Result<Self, StorageError> {
        let mut index = VectorIndex::new(dim, metric, index_type)?;
        let cache = SemanticCache::new(cache_capacity, dim, cache_threshold)?;

        let wal = if let Some(path) = wal_path {
            let storage = WalStorage::open(path)?;
            storage.replay_into(&mut index)?;
            Some(storage)
        } else {
            None
        };

        Ok(Self { index, cache, wal })
    }

    /// Rejects a record the index would refuse, then appends it and indexes it.
    ///
    /// The WAL write happens only after the dimension, finiteness, and cosine-norm
    /// checks, so a rejected vector is not left in the log. `upsert` still repeats
    /// those checks.
    pub fn ingest(&mut self, record: DocumentRecord) -> Result<(), StorageError> {
        if record.vector.len() != self.index.dimension() {
            return Err(VectorError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: record.vector.len(),
            }
            .into());
        }
        validate_finite(&record.vector)?;
        if self.index.metric() == Metric::Cosine {
            let norm_sq: f32 = record.vector.iter().map(|x| x * x).sum();
            if norm_sq <= f32::EPSILON {
                return Err(VectorError::ZeroMagnitudeVector.into());
            }
        }

        if let Some(ref mut wal) = self.wal {
            wal.append_upsert(&record)?;
        }
        self.index.upsert(record)?;
        Ok(())
    }

    /// 1. Scan the cache with cosine similarity (always cosine, even if the index metric differs).
    /// 2. On a miss, search the index for `top_k` documents.
    /// 3. Format a completion from the top hit and store that pair in the cache.
    pub fn query(
        &mut self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<RagResponse, VectorError> {
        if let Some(hit) = self.cache.get(query_embedding)? {
            return Ok(RagResponse::CacheHit(hit));
        }

        let context = self.index.search(query_embedding, top_k)?;
        let top_snippet = context
            .first()
            .map(|doc| format!("[{}] {}", doc.id, doc.text))
            .unwrap_or_else(|| "No relevant context found.".to_string());

        let completion = format!("Synthesized answer for '{query_text}': {top_snippet}");

        self.cache.put(
            query_text.to_string(),
            query_embedding.to_vec(),
            completion.clone(),
        )?;

        Ok(RagResponse::Retrieved {
            context,
            completion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::VectorIndex;
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static ENGINE_TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_wal_path(label: &str) -> PathBuf {
        let seq = ENGINE_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "micro_rag_lib_{}_{}_{}.wal",
            label,
            std::process::id(),
            seq
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn rejected_ingest_does_not_append_wal() {
        let path = unique_wal_path("reject_zero");

        let mut engine = MicroRagEngine::new(
            2,
            Metric::Cosine,
            IndexType::Flat,
            4,
            0.9,
            Some(path.as_path()),
        )
        .unwrap();

        let rejected = engine.ingest(DocumentRecord {
            id: "bad".into(),
            vector: vec![0.0, 0.0],
            text: "zero".into(),
            metadata: BTreeMap::new(),
        });
        assert!(matches!(
            rejected,
            Err(StorageError::Vector(VectorError::ZeroMagnitudeVector))
        ));
        assert_eq!(engine.index.len(), 0);

        engine
            .ingest(DocumentRecord {
                id: "ok".into(),
                vector: vec![1.0, 0.0],
                text: "kept".into(),
                metadata: BTreeMap::new(),
            })
            .unwrap();
        assert_eq!(engine.index.len(), 1);
        drop(engine);

        let mut recovered = VectorIndex::new(2, Metric::Cosine, IndexType::Flat).unwrap();
        let wal = WalStorage::open(&path).unwrap();
        assert_eq!(wal.replay_into(&mut recovered).unwrap(), 1);
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered.documents()[0].id, "ok");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejected_non_finite_ingest_does_not_append_wal() {
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::Euclidean] {
            let path = unique_wal_path("reject_non_finite");
            let mut engine =
                MicroRagEngine::new(2, metric, IndexType::KdTree, 4, 0.9, Some(path.as_path()))
                    .unwrap();

            for bad_vec in [
                vec![f32::NAN, 1.0],
                vec![f32::INFINITY, 0.0],
                vec![0.0, f32::NEG_INFINITY],
            ] {
                let err = engine.ingest(DocumentRecord {
                    id: "bad".into(),
                    vector: bad_vec,
                    text: "invalid".into(),
                    metadata: BTreeMap::new(),
                });
                assert!(matches!(
                    err,
                    Err(StorageError::Vector(VectorError::NonFiniteValue))
                ));
                assert_eq!(engine.index.len(), 0);
            }

            engine
                .ingest(DocumentRecord {
                    id: "good".into(),
                    vector: vec![1.0, 0.0],
                    text: "valid".into(),
                    metadata: BTreeMap::new(),
                })
                .unwrap();
            drop(engine);

            let mut recovered = VectorIndex::new(2, metric, IndexType::Flat).unwrap();
            let wal = WalStorage::open(&path).unwrap();
            assert_eq!(wal.replay_into(&mut recovered).unwrap(), 1);
            assert_eq!(recovered.documents()[0].id, "good");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn engine_query_handles_empty_cold_cache_hit_miss_and_invalid_queries() {
        let mut engine =
            MicroRagEngine::new(3, Metric::Cosine, IndexType::KdTree, 4, 0.92, None).unwrap();

        // Query against empty index returns empty context and fallback snippet
        match engine.query("empty lookup", &[0.0, 0.0, 1.0], 2).unwrap() {
            RagResponse::Retrieved {
                context,
                completion,
            } => {
                assert!(context.is_empty());
                assert!(completion.contains("No relevant context found."));
            }
            RagResponse::CacheHit(_) => panic!("expected miss on empty cache"),
        }

        engine
            .ingest(DocumentRecord {
                id: "doc-sys".into(),
                vector: vec![0.95, 0.20, 0.05],
                text: "Seccomp restricts Linux syscalls.".into(),
                metadata: BTreeMap::from([("domain".into(), "security".into())]),
            })
            .unwrap();
        engine
            .ingest(DocumentRecord {
                id: "doc-db".into(),
                vector: vec![0.05, 0.95, 0.10],
                text: "WAL frames protect writes with CRC32.".into(),
                metadata: BTreeMap::from([("domain".into(), "storage".into())]),
            })
            .unwrap();

        // Cold query hits index and populates cache
        let cold_completion = match engine
            .query("How does seccomp work?", &[0.94, 0.22, 0.04], 2)
            .unwrap()
        {
            RagResponse::Retrieved {
                context,
                completion,
            } => {
                assert_eq!(context.len(), 2);
                assert_eq!(context[0].id, "doc-sys");
                assert!(completion.contains("[doc-sys] Seccomp restricts Linux syscalls."));
                completion
            }
            RagResponse::CacheHit(_) => panic!("expected cache miss for cold query"),
        };

        // Paraphrased nearby query hits the semantic cache
        match engine
            .query("Explain Linux seccomp filtering", &[0.96, 0.18, 0.05], 2)
            .unwrap()
        {
            RagResponse::CacheHit(hit) => {
                assert_eq!(hit.matched_query, "How does seccomp work?");
                assert_eq!(hit.completion, cold_completion);
                assert!(hit.similarity >= 0.92);
                assert_eq!(hit.total_hits, 1);
            }
            RagResponse::Retrieved { .. } => panic!("expected semantic cache hit"),
        }

        // Distant query misses cache and retrieves doc-db
        match engine
            .query("How does WAL integrity work?", &[0.04, 0.96, 0.09], 1)
            .unwrap()
        {
            RagResponse::Retrieved {
                context,
                completion,
            } => {
                assert_eq!(context.len(), 1);
                assert_eq!(context[0].id, "doc-db");
                assert!(completion.contains("[doc-db] WAL frames protect writes with CRC32."));
            }
            RagResponse::CacheHit(_) => panic!("expected cache miss for unrelated query"),
        }

        // Non-finite and wrong-dimension queries are rejected
        assert_eq!(
            engine
                .query("nan query", &[f32::NAN, 0.0, 1.0], 1)
                .unwrap_err(),
            VectorError::NonFiniteValue
        );
        assert_eq!(
            engine
                .query("inf query", &[0.0, f32::INFINITY, 1.0], 1)
                .unwrap_err(),
            VectorError::NonFiniteValue
        );
        assert_eq!(
            engine.query("short query", &[1.0, 0.0], 1).unwrap_err(),
            VectorError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn engine_new_fails_on_torn_wal_tail() {
        let path = unique_wal_path("engine_torn_tail");
        {
            let mut engine = MicroRagEngine::new(
                2,
                Metric::Cosine,
                IndexType::Flat,
                4,
                0.9,
                Some(path.as_path()),
            )
            .unwrap();
            engine
                .ingest(DocumentRecord {
                    id: "ok".into(),
                    vector: vec![1.0, 0.0],
                    text: "valid".into(),
                    metadata: BTreeMap::new(),
                })
                .unwrap();
        }

        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            file.write_all(b"{\"crc32\":999,\"op\":").unwrap();
            file.flush().unwrap();
        }

        let err = match MicroRagEngine::new(
            2,
            Metric::Cosine,
            IndexType::Flat,
            4,
            0.9,
            Some(path.as_path()),
        ) {
            Ok(_) => panic!("expected MicroRagEngine::new to fail on torn tail"),
            Err(err) => err,
        };
        assert!(matches!(err, StorageError::Serialization { line: 2, .. }));

        let _ = std::fs::remove_file(&path);
    }
}
