#![forbid(unsafe_code)]

pub mod cache;
pub mod index;
pub mod math;
pub mod storage;

use cache::{CacheHit, SemanticCache};
use index::{DocumentRecord, IndexType, SearchResult, VectorIndex};
use math::{Metric, VectorError};
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
    /// The WAL write happens only after the dimension and cosine-norm checks, so a
    /// rejected vector is not left in the log. `upsert` still repeats those checks.
    pub fn ingest(&mut self, record: DocumentRecord) -> Result<(), StorageError> {
        if record.vector.len() != self.index.dimension() {
            return Err(VectorError::DimensionMismatch {
                expected: self.index.dimension(),
                actual: record.vector.len(),
            }
            .into());
        }
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
    use std::collections::HashMap;

    #[test]
    fn rejected_ingest_does_not_append_wal() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "micro_rag_reject_{}_{}.wal",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&path);

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
            metadata: HashMap::new(),
        });
        assert!(rejected.is_err());
        assert_eq!(engine.index.len(), 0);

        engine
            .ingest(DocumentRecord {
                id: "ok".into(),
                vector: vec![1.0, 0.0],
                text: "kept".into(),
                metadata: HashMap::new(),
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
}
