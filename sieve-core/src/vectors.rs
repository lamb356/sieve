use std::cmp::Ordering;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::{Result, SieveError};

const HOT_VECS_FILE: &str = "hot.vecs";
const HOT_META_FILE: &str = "hot.meta";
const EMBEDDED_FILE: &str = "embedded.bin";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorMeta {
    pub wal_entry_id: u64,
    pub source_path: String,
    pub line_range: (usize, usize),
    pub chunk_index: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorMatch {
    pub wal_entry_id: u64,
    pub source_path: String,
    pub line_range: (usize, usize),
    pub chunk_index: u32,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct HotVectorStore {
    dir: PathBuf,
    dimension: usize,
    metadata: Vec<VectorMeta>,
    embedded_set: RoaringTreemap,
}

impl HotVectorStore {
    pub fn open_or_create(dir: &Path, dimension: usize) -> Result<Self> {
        fs::create_dir_all(dir)?;
        ensure_file(&dir.join(HOT_VECS_FILE))?;
        ensure_file(&dir.join(HOT_META_FILE))?;
        ensure_file(&dir.join(EMBEDDED_FILE))?;

        let metadata = load_metadata(&dir.join(HOT_META_FILE))?;
        let embedded_set = load_embedded_set(&dir.join(EMBEDDED_FILE))?;
        let vector_count = vector_count(&dir.join(HOT_VECS_FILE), dimension)?;
        if vector_count != metadata.len() {
            return Err(SieveError::Io(std::io::Error::other(format!(
                "vector store metadata mismatch: {} vectors but {} metadata rows",
                vector_count,
                metadata.len()
            ))));
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            dimension,
            metadata,
            embedded_set,
        })
    }

    pub fn append(&mut self, vectors: &[Vec<f32>], metas: &[VectorMeta]) -> Result<()> {
        if vectors.len() != metas.len() {
            return Err(SieveError::Io(std::io::Error::other(
                "vector/meta length mismatch",
            )));
        }
        if vectors.iter().any(|vector| vector.len() != self.dimension) {
            return Err(SieveError::Io(std::io::Error::other(
                "vector dimension mismatch",
            )));
        }

        let vec_path = self.dir.join(HOT_VECS_FILE);
        let meta_path = self.dir.join(HOT_META_FILE);
        let embedded_path = self.dir.join(EMBEDDED_FILE);

        let mut vec_file = OpenOptions::new().append(true).open(&vec_path)?;
        for vector in vectors {
            for value in vector {
                vec_file.write_all(&value.to_ne_bytes())?;
            }
        }
        vec_file.flush()?;
        vec_file.sync_data()?;

        self.metadata.extend_from_slice(metas);
        persist_metadata(&meta_path, &self.metadata)?;

        for meta in metas {
            self.embedded_set.insert(meta.wal_entry_id);
        }
        persist_embedded_set(&embedded_path, &self.embedded_set)?;
        sync_dir(&self.dir)?;
        Ok(())
    }

    pub fn search_knn(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<VectorMatch>> {
        if query_vec.len() != self.dimension {
            return Err(SieveError::Io(std::io::Error::other(
                "query vector dimension mismatch",
            )));
        }
        if self.metadata.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let file = File::open(self.dir.join(HOT_VECS_FILE))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let chunk_bytes = self.dimension * std::mem::size_of::<f32>();
        let mut scored = Vec::with_capacity(self.metadata.len());

        for (index, meta) in self.metadata.iter().enumerate() {
            let start = index * chunk_bytes;
            let end = start + chunk_bytes;
            if end > mmap.len() {
                break;
            }
            let score = dot_product_bytes(&mmap[start..end], query_vec)?;
            scored.push(VectorMatch {
                wal_entry_id: meta.wal_entry_id,
                source_path: meta.source_path.clone(),
                line_range: meta.line_range,
                chunk_index: meta.chunk_index,
                score,
            });
        }

        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.source_path.cmp(&right.source_path))
                .then_with(|| left.line_range.0.cmp(&right.line_range.0))
        });
        if scored.len() > top_k {
            scored.truncate(top_k);
        }
        Ok(scored)
    }

    pub fn embedded_set(&self) -> &RoaringTreemap {
        &self.embedded_set
    }

    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

fn ensure_file(path: &Path) -> Result<()> {
    if !path.exists() {
        File::create(path)?;
    }
    Ok(())
}

fn load_metadata(path: &Path) -> Result<Vec<VectorMeta>> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    bincode::deserialize(&bytes)
        .map_err(|err| SieveError::Io(std::io::Error::other(err.to_string())))
}

fn persist_metadata(path: &Path, metadata: &[VectorMeta]) -> Result<()> {
    let bytes = bincode::serialize(metadata)
        .map_err(|err| SieveError::Io(std::io::Error::other(err.to_string())))?;
    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    file.sync_data()?;
    Ok(())
}

fn load_embedded_set(path: &Path) -> Result<RoaringTreemap> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.is_empty() {
        return Ok(RoaringTreemap::new());
    }
    let mut cursor = std::io::Cursor::new(bytes);
    RoaringTreemap::deserialize_from(&mut cursor)
        .map_err(|err| SieveError::Io(std::io::Error::other(err.to_string())))
}

fn persist_embedded_set(path: &Path, embedded_set: &RoaringTreemap) -> Result<()> {
    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
    embedded_set
        .serialize_into(&mut file)
        .map_err(|err| SieveError::Io(std::io::Error::other(err.to_string())))?;
    file.flush()?;
    file.sync_data()?;
    Ok(())
}

fn vector_count(path: &Path, dimension: usize) -> Result<usize> {
    if dimension == 0 {
        return Ok(0);
    }
    let bytes = fs::metadata(path)?.len() as usize;
    let vector_bytes = dimension * std::mem::size_of::<f32>();
    if !bytes.is_multiple_of(vector_bytes) {
        return Err(SieveError::Io(std::io::Error::other(format!(
            "vector file length {} is not divisible by vector width {}",
            bytes, vector_bytes
        ))));
    }
    Ok(bytes / vector_bytes)
}

fn dot_product_bytes(bytes: &[u8], query_vec: &[f32]) -> Result<f64> {
    let mut score = 0.0_f64;
    for (index, value) in query_vec.iter().enumerate() {
        let start = index * std::mem::size_of::<f32>();
        let raw = bytes
            .get(start..start + std::mem::size_of::<f32>())
            .ok_or_else(|| SieveError::Io(std::io::Error::other("truncated vector slice")))?;
        let stored = f32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
        score += (stored * value) as f64;
    }
    Ok(score)
}

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
