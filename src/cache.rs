use crate::math::{cosine_similarity, VectorError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub query_text: String,
    pub query_embedding: Vec<f32>,
    pub completion: String,
    pub hits: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CacheHit {
    pub matched_query: String,
    pub completion: String,
    pub similarity: f32,
    pub total_hits: u64,
}

#[derive(Debug, Clone)]
struct ArenaNode {
    entry: CacheEntry,
    prev: Option<usize>,
    next: Option<usize>,
}

/// An Arena-allocated Doubly-Linked-List LRU Semantic Cache.
///
/// Unlike key-value hash caches that require bit-identical query strings, `SemanticCache`
/// computes the cosine similarity between an incoming query embedding and cached query vectors.
/// If the best match meets or exceeds `similarity_threshold` (e.g. `0.92`), the node is promoted
/// to the MRU (Most Recently Used) head in O(1) pointer swaps and the cached completion is returned.
pub struct SemanticCache {
    capacity: usize,
    dim: usize,
    similarity_threshold: f32,
    nodes: Vec<Option<ArenaNode>>,
    free_slots: Vec<usize>,
    head: Option<usize>, // Most Recently Used (MRU)
    tail: Option<usize>, // Least Recently Used (LRU)
    len: usize,
    total_lookups: u64,
    total_hits: u64,
}

impl SemanticCache {
    pub fn new(
        capacity: usize,
        dim: usize,
        similarity_threshold: f32,
    ) -> Result<Self, VectorError> {
        if dim == 0 {
            return Err(VectorError::EmptyVector);
        }
        assert!(capacity > 0, "Cache capacity must be greater than 0");
        assert!(
            (0.0..=1.0).contains(&similarity_threshold),
            "Similarity threshold must be between 0.0 and 1.0"
        );

        Ok(Self {
            capacity,
            dim,
            similarity_threshold,
            nodes: Vec::with_capacity(capacity),
            free_slots: Vec::new(),
            head: None,
            tail: None,
            len: 0,
            total_lookups: 0,
            total_hits: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn hit_rate(&self) -> f32 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.total_hits as f32 / self.total_lookups as f32
        }
    }

    /// Probes the cache for a semantically equivalent query embedding.
    /// If the highest cosine similarity is `>= self.similarity_threshold`, promotes the entry
    /// to MRU and returns `Ok(Some(CacheHit))`.
    pub fn get(&mut self, query_embedding: &[f32]) -> Result<Option<CacheHit>, VectorError> {
        if query_embedding.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                actual: query_embedding.len(),
            });
        }

        self.total_lookups += 1;

        let mut best_slot: Option<usize> = None;
        let mut best_sim: f32 = -1.0;

        // Walk from MRU (head) to LRU (tail) so ties naturally favor fresher entries
        let mut cursor = self.head;
        while let Some(idx) = cursor {
            let node = self.nodes[idx].as_ref().expect("Corrupted LRU arena slot");
            let sim = cosine_similarity(query_embedding, &node.entry.query_embedding)?;
            if sim >= self.similarity_threshold && sim > best_sim {
                best_sim = sim;
                best_slot = Some(idx);
            }
            cursor = node.next;
        }

        if let Some(slot) = best_slot {
            self.promote_to_head(slot);
            self.total_hits += 1;
            let node = self.nodes[slot].as_mut().unwrap();
            node.entry.hits += 1;

            return Ok(Some(CacheHit {
                matched_query: node.entry.query_text.clone(),
                completion: node.entry.completion.clone(),
                similarity: best_sim,
                total_hits: node.entry.hits,
            }));
        }

        Ok(None)
    }

    /// Inserts a new query + completion into the cache, or updates and promotes an existing
    /// entry if an existing vector is already within `>= similarity_threshold`.
    /// Evicts the Least Recently Used (tail) entry if at capacity.
    pub fn put(
        &mut self,
        query_text: String,
        query_embedding: Vec<f32>,
        completion: String,
    ) -> Result<(), VectorError> {
        if query_embedding.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                actual: query_embedding.len(),
            });
        }
        // Validate non-zero magnitude
        let _ = cosine_similarity(&query_embedding, &query_embedding)?;

        // Check if a semantically identical entry already exists; if so, update in-place & promote
        let mut existing_slot: Option<usize> = None;
        let mut best_sim: f32 = -1.0;
        let mut cursor = self.head;

        while let Some(idx) = cursor {
            let node = self.nodes[idx].as_ref().unwrap();
            let sim = cosine_similarity(&query_embedding, &node.entry.query_embedding)?;
            if sim >= self.similarity_threshold && sim > best_sim {
                best_sim = sim;
                existing_slot = Some(idx);
            }
            cursor = node.next;
        }

        if let Some(slot) = existing_slot {
            let node = self.nodes[slot].as_mut().unwrap();
            node.entry.query_text = query_text;
            node.entry.query_embedding = query_embedding;
            node.entry.completion = completion;
            self.promote_to_head(slot);
            return Ok(());
        }

        // Evict LRU tail if at capacity
        if self.len == self.capacity {
            self.evict_lru();
        }

        let new_node = ArenaNode {
            entry: CacheEntry {
                query_text,
                query_embedding,
                completion,
                hits: 0,
            },
            prev: None,
            next: self.head,
        };

        let slot = if let Some(free_idx) = self.free_slots.pop() {
            self.nodes[free_idx] = Some(new_node);
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(new_node));
            idx
        };

        if let Some(old_head) = self.head {
            self.nodes[old_head].as_mut().unwrap().prev = Some(slot);
        }
        self.head = Some(slot);
        if self.tail.is_none() {
            self.tail = Some(slot);
        }
        self.len += 1;

        Ok(())
    }

    /// Returns cached queries ordered from Most Recently Used (MRU) to Least Recently Used (LRU).
    pub fn iter_mru_to_lru(&self) -> Vec<&CacheEntry> {
        let mut out = Vec::with_capacity(self.len);
        let mut cursor = self.head;
        while let Some(idx) = cursor {
            let node = self.nodes[idx].as_ref().unwrap();
            out.push(&node.entry);
            cursor = node.next;
        }
        out
    }

    fn detach(&mut self, slot: usize) {
        let (prev, next) = {
            let node = self.nodes[slot].as_ref().unwrap();
            (node.prev, node.next)
        };

        if let Some(p) = prev {
            self.nodes[p].as_mut().unwrap().next = next;
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            self.nodes[n].as_mut().unwrap().prev = prev;
        } else {
            self.tail = prev;
        }

        let node = self.nodes[slot].as_mut().unwrap();
        node.prev = None;
        node.next = None;
    }

    fn promote_to_head(&mut self, slot: usize) {
        if self.head == Some(slot) {
            return;
        }
        self.detach(slot);

        let old_head = self.head;
        {
            let node = self.nodes[slot].as_mut().unwrap();
            node.prev = None;
            node.next = old_head;
        }
        if let Some(h) = old_head {
            self.nodes[h].as_mut().unwrap().prev = Some(slot);
        }
        self.head = Some(slot);
        if self.tail.is_none() {
            self.tail = Some(slot);
        }
    }

    fn evict_lru(&mut self) {
        if let Some(lru_slot) = self.tail {
            self.detach(lru_slot);
            self.nodes[lru_slot] = None;
            self.free_slots.push(lru_slot);
            self.len -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_cache_hit_and_threshold() {
        let mut cache = SemanticCache::new(3, 3, 0.92).unwrap();

        cache
            .put(
                "How do I configure Arch Linux pacman mirrors?".into(),
                vec![0.98, 0.15, 0.05],
                "Use reflector to rank mirrors in /etc/pacman.d/mirrorlist.".into(),
            )
            .unwrap();

        // Paraphrased query with very close embedding (cosine similarity > 0.99)
        let near_query = vec![0.97, 0.18, 0.06];
        let hit = cache.get(&near_query).unwrap().expect("Should hit cache");
        assert!(hit.similarity >= 0.92);
        assert_eq!(hit.total_hits, 1);

        // Orthogonal query (cosine similarity ~ 0.05) -> Cache Miss
        let miss_query = vec![0.0, 0.1, 0.99];
        assert!(cache.get(&miss_query).unwrap().is_none());
    }

    #[test]
    fn test_lru_eviction_and_promotion_order() {
        // Use a high threshold (0.95) so orthogonal basis embeddings remain distinct keys
        let mut cache = SemanticCache::new(2, 3, 0.95).unwrap();

        // Insert Q1 and Q2
        cache
            .put("Q1".into(), vec![1.0, 0.0, 0.0], "A1".into())
            .unwrap();
        cache
            .put("Q2".into(), vec![0.0, 1.0, 0.0], "A2".into())
            .unwrap();

        // Order is now: Q2 (MRU) -> Q1 (LRU)
        assert_eq!(cache.iter_mru_to_lru()[0].query_text, "Q2");

        // Access Q1 via slightly perturbed vector (sim > 0.99) to promote Q1 to MRU
        let q1_probe = vec![0.99, 0.05, 0.0];
        assert!(cache.get(&q1_probe).unwrap().is_some());

        // Order is now: Q1 (MRU) -> Q2 (LRU)
        assert_eq!(cache.iter_mru_to_lru()[0].query_text, "Q1");

        // Insert Q3 -> Should evict Q2 (the current LRU)
        cache
            .put("Q3".into(), vec![0.0, 0.0, 1.0], "A3".into())
            .unwrap();

        assert_eq!(cache.len(), 2);
        let remaining: Vec<&str> = cache
            .iter_mru_to_lru()
            .iter()
            .map(|e| e.query_text.as_str())
            .collect();
        assert_eq!(remaining, vec!["Q3", "Q1"]);
        assert!(cache.get(&[0.0, 1.0, 0.0]).unwrap().is_none());
    }
}
