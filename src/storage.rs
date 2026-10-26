use crate::index::{DocumentRecord, VectorIndex};
use crate::math::VectorError;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("I/O error during persistence: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error on WAL line {line}: {source}")]
    Serialization {
        line: usize,
        source: serde_json::Error,
    },
    #[error("Checksum mismatch on WAL line {line}: expected {expected}, computed {actual}")]
    ChecksumMismatch {
        line: usize,
        expected: u32,
        actual: u32,
    },
    #[error("Vector engine error during replay: {0}")]
    Vector(#[from] VectorError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    Upsert(DocumentRecord),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WalFrame {
    crc32: u32,
    op: WalOperation,
}

/// Computes a fast, dependency-free CRC32 (IEEE 802.3) checksum over a byte slice
/// to detect torn writes or bit-rot in the append-only WAL.
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Append-only Write-Ahead Log (WAL) backed by newline-delimited JSON frames with CRC32 integrity.
pub struct WalStorage {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl WalStorage {
    /// Opens or creates an append-only WAL file at `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let path_buf = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path_buf)?;

        Ok(Self {
            path: path_buf,
            writer: BufWriter::new(file),
        })
    }

    /// Appends a document upsert operation to disk and flushes the OS buffer.
    pub fn append_upsert(&mut self, record: &DocumentRecord) -> Result<(), StorageError> {
        let op = WalOperation::Upsert(record.clone());
        let payload = serde_json::to_vec(&op)
            .map_err(|e| StorageError::Serialization { line: 0, source: e })?;
        let checksum = crc32_ieee(&payload);

        let frame = WalFrame {
            crc32: checksum,
            op,
        };

        serde_json::to_writer(&mut self.writer, &frame)
            .map_err(|e| StorageError::Serialization { line: 0, source: e })?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }

    /// Replays all frames from the WAL into the provided `VectorIndex`, verifying CRC32 per frame.
    pub fn replay_into(&self, index: &mut VectorIndex) -> Result<usize, StorageError> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut count = 0;

        for (idx, line_res) in reader.lines().enumerate() {
            let line_num = idx + 1;
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let frame: WalFrame =
                serde_json::from_str(trimmed).map_err(|e| StorageError::Serialization {
                    line: line_num,
                    source: e,
                })?;

            let payload =
                serde_json::to_vec(&frame.op).map_err(|e| StorageError::Serialization {
                    line: line_num,
                    source: e,
                })?;
            let computed_crc = crc32_ieee(&payload);

            if computed_crc != frame.crc32 {
                return Err(StorageError::ChecksumMismatch {
                    line: line_num,
                    expected: frame.crc32,
                    actual: computed_crc,
                });
            }

            match frame.op {
                WalOperation::Upsert(record) => {
                    index.upsert(record)?;
                    count += 1;
                }
            }
        }

        index.rebuild_index()?;
        Ok(count)
    }

    /// Compacts the WAL by rewriting only the latest live state of documents in `index`.
    pub fn compact(&mut self, index: &VectorIndex) -> Result<(), StorageError> {
        let tmp_path = self.path.with_extension("wal.tmp");
        {
            let tmp_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            let mut tmp_writer = BufWriter::new(tmp_file);

            for doc in index.documents() {
                let op = WalOperation::Upsert(doc.clone());
                let payload = serde_json::to_vec(&op)
                    .map_err(|e| StorageError::Serialization { line: 0, source: e })?;
                let frame = WalFrame {
                    crc32: crc32_ieee(&payload),
                    op,
                };
                serde_json::to_writer(&mut tmp_writer, &frame)
                    .map_err(|e| StorageError::Serialization { line: 0, source: e })?;
                tmp_writer.write_all(b"\n")?;
            }
            tmp_writer.flush()?;
        }

        std::fs::rename(&tmp_path, &self.path)?;
        let reopened = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(reopened);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexType;
    use crate::math::Metric;
    use std::collections::HashMap;

    #[test]
    fn test_wal_append_replay_and_compaction() {
        let mut temp_path = std::env::temp_dir();
        temp_path.push(format!("micro_rag_test_{}.wal", std::process::id()));
        let _ = std::fs::remove_file(&temp_path);

        {
            let mut wal = WalStorage::open(&temp_path).unwrap();
            let doc1 = DocumentRecord {
                id: "k1".into(),
                vector: vec![1.0, 0.0, 0.0],
                text: "Initial v1".into(),
                metadata: HashMap::new(),
            };
            let doc1_updated = DocumentRecord {
                id: "k1".into(),
                vector: vec![0.0, 1.0, 0.0],
                text: "Updated v2".into(),
                metadata: HashMap::new(),
            };
            wal.append_upsert(&doc1).unwrap();
            wal.append_upsert(&doc1_updated).unwrap();
        }

        let mut recovered_index = VectorIndex::new(3, Metric::Cosine, IndexType::KdTree).unwrap();
        let mut wal = WalStorage::open(&temp_path).unwrap();
        let replayed = wal.replay_into(&mut recovered_index).unwrap();

        assert_eq!(replayed, 2);
        assert_eq!(recovered_index.len(), 1);
        assert_eq!(recovered_index.documents()[0].text, "Updated v2");

        // Compact and verify only 1 frame remains
        wal.compact(&recovered_index).unwrap();
        let mut post_compact_index = VectorIndex::new(3, Metric::Cosine, IndexType::Flat).unwrap();
        let replayed_after = wal.replay_into(&mut post_compact_index).unwrap();
        assert_eq!(replayed_after, 1);

        let _ = std::fs::remove_file(&temp_path);
    }
}
