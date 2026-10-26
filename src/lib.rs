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
    /// Returned directly from the LRU Semantic Cache in O(C * D) without hitting retrieval or LLM generation.
    CacheHit(CacheHit),
    /// Cache missed; retrieved top-K context documents and synthesized a new completion.
    Retrieved {
        context: Vec<SearchResult>,
        completion: String,
    },
}

/// Unified facade combining the VectorIndex, WAL persistence engine, and LRU Semantic Cache.
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

    /// Persists a document to the WAL (if configured) and indexes it in memory.
    pub fn ingest(&mut self, record: DocumentRecord) -> Result<(), StorageError> {
        if let Some(ref mut wal) = self.wal {
            wal.append_upsert(&record)?;
        }
        self.index.upsert(record)?;
        Ok(())
    }

    /// Executes the full RAG pipeline:
    /// 1. Checks the LRU Semantic Cache (`cosine_similarity >= threshold`).
    /// 2. On cache miss, queries the vector index (Flat or KD-Tree) for top-K documents.
    /// 3. Synthesizes an answer and populates the semantic cache.
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

        let completion = format!("Synthesized answer for '{}': {}", query_text, top_snippet);

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
