use crate::math::{
    cosine_similarity, dot_product, l2_normalize, squared_euclidean_distance, Metric, VectorError,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

/// Helper wrapper to turn `std::collections::BinaryHeap` (a max-heap) into a min-heap
/// of top-K highest scores, so we can evict the *lowest* score in O(log K) time.
#[derive(Debug, Clone)]
struct ScoredItem {
    score: f32,
    doc_idx: usize,
}

impl PartialEq for ScoredItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.doc_idx == other.doc_idx
    }
}
impl Eq for ScoredItem {}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed ordering for Min-Heap behavior on `score`
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.doc_idx.cmp(&self.doc_idx))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    Flat,
    KdTree,
}

#[derive(Debug, Clone)]
struct KdNode {
    doc_idx: usize,
    split_axis: usize,
    split_val: f32,
    /// Unit-normalized copy of the vector (used when Metric::Cosine is active on KD-Tree)
    norm_vector: Vec<f32>,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

pub struct VectorIndex {
    dim: usize,
    metric: Metric,
    index_type: IndexType,
    documents: Vec<DocumentRecord>,
    id_to_idx: HashMap<String, usize>,
    kd_root: Option<Box<KdNode>>,
    dirty_tree: bool,
}

impl VectorIndex {
    pub fn new(dim: usize, metric: Metric, index_type: IndexType) -> Result<Self, VectorError> {
        if dim == 0 {
            return Err(VectorError::EmptyVector);
        }
        Ok(Self {
            dim,
            metric,
            index_type,
            documents: Vec::new(),
            id_to_idx: HashMap::new(),
            kd_root: None,
            dirty_tree: false,
        })
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn documents(&self) -> &[DocumentRecord] {
        &self.documents
    }

    /// Inserts or updates a document in the vector store.
    pub fn upsert(&mut self, record: DocumentRecord) -> Result<(), VectorError> {
        if record.vector.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                actual: record.vector.len(),
            });
        }
        if self.metric == Metric::Cosine {
            let norm_sq: f32 = record.vector.iter().map(|x| x * x).sum();
            if norm_sq <= f32::EPSILON {
                return Err(VectorError::ZeroMagnitudeVector);
            }
        }

        if let Some(&existing_idx) = self.id_to_idx.get(&record.id) {
            self.documents[existing_idx] = record;
        } else {
            let idx = self.documents.len();
            self.id_to_idx.insert(record.id.clone(), idx);
            self.documents.push(record);
        }

        if self.index_type == IndexType::KdTree {
            self.dirty_tree = true;
        }
        Ok(())
    }

    /// Explicitly rebuilds the spatial partitioning tree if `IndexType::KdTree` is selected.
    pub fn rebuild_index(&mut self) -> Result<(), VectorError> {
        if self.index_type != IndexType::KdTree {
            return Ok(());
        }

        let mut entries: Vec<(usize, Vec<f32>)> = Vec::with_capacity(self.documents.len());
        for (idx, doc) in self.documents.iter().enumerate() {
            let mut vec_repr = doc.vector.clone();
            // For Cosine similarity, unit-normalizing vectors maps monotonic Cosine ranking
            // directly onto Euclidean distance in R^d: ||u - v||^2 = 2 - 2 * cos(u, v).
            if self.metric == Metric::Cosine {
                l2_normalize(&mut vec_repr)?;
            }
            entries.push((idx, vec_repr));
        }

        self.kd_root = Self::build_kdtree(&mut entries, 0, self.dim);
        self.dirty_tree = false;
        Ok(())
    }

    fn build_kdtree(
        entries: &mut [(usize, Vec<f32>)],
        depth: usize,
        dim: usize,
    ) -> Option<Box<KdNode>> {
        if entries.is_empty() {
            return None;
        }

        let axis = depth % dim;
        let mid = entries.len() / 2;

        // Partition around the median along `axis` in O(N) average time
        entries.select_nth_unstable_by(mid, |a, b| {
            a.1[axis].partial_cmp(&b.1[axis]).unwrap_or(Ordering::Equal)
        });

        let (doc_idx, norm_vector) = entries[mid].clone();
        let split_val = norm_vector[axis];

        let left = Self::build_kdtree(&mut entries[..mid], depth + 1, dim);
        let right = Self::build_kdtree(&mut entries[mid + 1..], depth + 1, dim);

        Some(Box::new(KdNode {
            doc_idx,
            split_axis: axis,
            split_val,
            norm_vector,
            left,
            right,
        }))
    }

    /// Searches the index for the top `k` most similar records.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if k == 0 || self.documents.is_empty() {
            return Ok(Vec::new());
        }

        // Validate zero magnitude early if using Cosine
        if self.metric == Metric::Cosine {
            let norm_sq: f32 = query.iter().map(|x| x * x).sum();
            if norm_sq <= f32::EPSILON {
                return Err(VectorError::ZeroMagnitudeVector);
            }
        }

        match self.index_type {
            IndexType::Flat => self.search_flat(query, k),
            IndexType::KdTree => {
                // DotProduct is not a metric space (violates triangle inequality), so fallback to Flat
                if self.metric == Metric::DotProduct {
                    return self.search_flat(query, k);
                }
                if self.dirty_tree {
                    self.rebuild_index()?;
                }
                self.search_kdtree(query, k)
            }
        }
    }

    fn search_flat(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorError> {
        let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::with_capacity(k + 1);

        for (idx, doc) in self.documents.iter().enumerate() {
            let score = match self.metric {
                Metric::Cosine => cosine_similarity(query, &doc.vector)?,
                Metric::DotProduct => dot_product(query, &doc.vector)?,
                Metric::Euclidean => -squared_euclidean_distance(query, &doc.vector)?.sqrt(),
            };

            heap.push(ScoredItem {
                score,
                doc_idx: idx,
            });
            if heap.len() > k {
                heap.pop();
            }
        }

        Ok(self.drain_heap_to_results(heap))
    }

    fn search_kdtree(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorError> {
        let mut search_target = query.to_vec();
        if self.metric == Metric::Cosine {
            l2_normalize(&mut search_target)?;
        }

        // Max-heap of squared Euclidean distances (stored as negative in `ScoredItem` so the
        // largest squared distance sits at `.peek()` and gets evicted first).
        let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::with_capacity(k + 1);

        if let Some(ref root) = self.kd_root {
            Self::kdtree_knn_recursive(root, &search_target, k, &mut heap)?;
        }

        // Convert internal KD-Tree scores back to user-facing metric scores
        let mut results: Vec<SearchResult> = Vec::with_capacity(heap.len());
        while let Some(item) = heap.pop() {
            let doc = &self.documents[item.doc_idx];
            let exact_score = self.metric.score(query, &doc.vector)?;
            results.push(SearchResult {
                id: doc.id.clone(),
                score: exact_score,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    fn kdtree_knn_recursive(
        node: &KdNode,
        target: &[f32],
        k: usize,
        heap: &mut BinaryHeap<ScoredItem>,
    ) -> Result<(), VectorError> {
        let dist_sq = squared_euclidean_distance(target, &node.norm_vector)?;
        let neg_dist_sq = -dist_sq;

        if heap.len() < k {
            heap.push(ScoredItem {
                score: neg_dist_sq,
                doc_idx: node.doc_idx,
            });
        } else if let Some(worst) = heap.peek() {
            if neg_dist_sq > worst.score {
                heap.pop();
                heap.push(ScoredItem {
                    score: neg_dist_sq,
                    doc_idx: node.doc_idx,
                });
            }
        }

        let diff = target[node.split_axis] - node.split_val;
        let (near, far) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref near_child) = near {
            Self::kdtree_knn_recursive(near_child, target, k, heap)?;
        }

        // Prune far branch only if the splitting hyperplane is further than our current k-th best radius
        let plane_dist_sq = diff * diff;
        let should_search_far = if heap.len() < k {
            true
        } else if let Some(worst) = heap.peek() {
            let worst_dist_sq = -worst.score;
            plane_dist_sq <= worst_dist_sq
        } else {
            true
        };

        if should_search_far {
            if let Some(ref far_child) = far {
                Self::kdtree_knn_recursive(far_child, target, k, heap)?;
            }
        }

        Ok(())
    }

    fn drain_heap_to_results(&self, mut heap: BinaryHeap<ScoredItem>) -> Vec<SearchResult> {
        let mut out = Vec::with_capacity(heap.len());
        while let Some(item) = heap.pop() {
            let doc = &self.documents[item.doc_idx];
            out.push(SearchResult {
                id: doc.id.clone(),
                score: item.score,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
            });
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<DocumentRecord> {
        vec![
            DocumentRecord {
                id: "doc_rust".into(),
                vector: vec![0.9, 0.1, 0.0, 0.0],
                text: "Rust systems programming and borrow checker".into(),
                metadata: HashMap::from([("lang".into(), "rust".into())]),
            },
            DocumentRecord {
                id: "doc_linux".into(),
                vector: vec![0.8, 0.2, 0.1, 0.0],
                text: "Linux kernel namespaces and seccomp filters".into(),
                metadata: HashMap::from([("topic".into(), "os".into())]),
            },
            DocumentRecord {
                id: "doc_cooking".into(),
                vector: vec![0.0, 0.1, 0.9, 0.2],
                text: "Spicy rice cakes tteokbokki recipe".into(),
                metadata: HashMap::from([("topic".into(), "food".into())]),
            },
            DocumentRecord {
                id: "doc_math".into(),
                vector: vec![0.1, 0.9, 0.1, 0.0],
                text: "Theoreme des accroissements finis and analysis".into(),
                metadata: HashMap::from([("topic".into(), "math".into())]),
            },
        ]
    }

    #[test]
    fn test_flat_vs_kdtree_equivalence_cosine() {
        let mut flat_idx = VectorIndex::new(4, Metric::Cosine, IndexType::Flat).unwrap();
        let mut kd_idx = VectorIndex::new(4, Metric::Cosine, IndexType::KdTree).unwrap();

        for doc in sample_docs() {
            flat_idx.upsert(doc.clone()).unwrap();
            kd_idx.upsert(doc).unwrap();
        }

        let query = vec![0.85, 0.15, 0.05, 0.0];
        let flat_res = flat_idx.search(&query, 3).unwrap();
        let kd_res = kd_idx.search(&query, 3).unwrap();

        assert_eq!(flat_res.len(), 3);
        assert_eq!(kd_res.len(), 3);
        for (f, k) in flat_res.iter().zip(kd_res.iter()) {
            assert_eq!(f.id, k.id);
            assert!((f.score - k.score).abs() < 1e-5);
        }
        assert_eq!(flat_res[0].id, "doc_rust");
        assert_eq!(flat_res[1].id, "doc_linux");
    }

    #[test]
    fn test_flat_vs_kdtree_equivalence_euclidean() {
        let mut flat_idx = VectorIndex::new(4, Metric::Euclidean, IndexType::Flat).unwrap();
        let mut kd_idx = VectorIndex::new(4, Metric::Euclidean, IndexType::KdTree).unwrap();

        for doc in sample_docs() {
            flat_idx.upsert(doc.clone()).unwrap();
            kd_idx.upsert(doc).unwrap();
        }

        let query = vec![0.0, 0.1, 0.85, 0.2];
        let flat_res = flat_idx.search(&query, 2).unwrap();
        let kd_res = kd_idx.search(&query, 2).unwrap();

        assert_eq!(flat_res[0].id, "doc_cooking");
        assert_eq!(kd_res[0].id, "doc_cooking");
        assert_eq!(flat_res[1].id, kd_res[1].id);
    }
}
