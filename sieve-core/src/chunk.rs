use memchr::{memchr, memrchr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub wal_entry_id: u64,
    pub chunk_id: u32,
    pub byte_start: u32,
    pub byte_end: u32,
    pub line_range: (usize, usize),
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingChunker {
    pub chunk_bytes: usize,
    pub overlap_bytes: usize,
    pub newline_snap: usize,
}

impl Default for SlidingChunker {
    fn default() -> Self {
        Self {
            chunk_bytes: 512,
            overlap_bytes: 256,
            newline_snap: 64,
        }
    }
}

impl SlidingChunker {
    pub fn new(chunk_bytes: usize, overlap_bytes: usize, newline_snap: usize) -> Self {
        Self {
            chunk_bytes: chunk_bytes.max(1),
            overlap_bytes: overlap_bytes.min(chunk_bytes.saturating_sub(1)),
            newline_snap,
        }
    }

    pub fn chunk_entry(&self, wal_entry_id: u64, content: &str) -> Vec<Chunk> {
        let bytes = content.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut chunk_id = 0_u32;
        let mut start = 0_usize;
        let content_len = bytes.len();

        while start < content_len {
            let raw_end = (start + self.chunk_bytes).min(content_len);
            let end = align_to_char_boundary(content, self.snap_end(bytes, start, raw_end));
            if end <= start {
                break;
            }

            let text = String::from_utf8_lossy(&bytes[start..end]).into_owned();
            chunks.push(Chunk {
                wal_entry_id,
                chunk_id,
                byte_start: start as u32,
                byte_end: end as u32,
                line_range: line_range_for_span(bytes, start, end),
                text,
            });

            if end == content_len {
                break;
            }

            let next_raw_start = end.saturating_sub(self.overlap_bytes);
            let next_start =
                align_to_char_boundary(content, self.snap_start(bytes, next_raw_start, end));
            if next_start <= start {
                start = end;
            } else {
                start = next_start;
            }
            chunk_id = chunk_id.saturating_add(1);
        }

        chunks
    }

    pub fn chunk(&self, wal_entry_id: u64, content: &str) -> Vec<Chunk> {
        self.chunk_entry(wal_entry_id, content)
    }

    fn snap_end(&self, bytes: &[u8], start: usize, raw_end: usize) -> usize {
        if raw_end >= bytes.len() {
            return bytes.len();
        }
        let snapped = nearest_newline_boundary(bytes, raw_end, self.newline_snap);
        snapped
            .filter(|boundary| *boundary > start)
            .unwrap_or(raw_end)
    }

    fn snap_start(&self, bytes: &[u8], raw_start: usize, previous_end: usize) -> usize {
        if raw_start == 0 {
            return 0;
        }
        let snapped = nearest_newline_boundary(bytes, raw_start, self.newline_snap);
        snapped
            .filter(|boundary| *boundary < previous_end)
            .unwrap_or(raw_start)
    }
}

fn nearest_newline_boundary(bytes: &[u8], boundary: usize, window: usize) -> Option<usize> {
    let search_start = boundary.saturating_sub(window);
    let search_end = (boundary + window).min(bytes.len());

    let backward = memrchr(b'\n', &bytes[search_start..boundary]).map(|idx| search_start + idx + 1);
    let forward = memchr(b'\n', &bytes[boundary..search_end]).map(|idx| boundary + idx + 1);

    match (backward, forward) {
        (Some(left), Some(right)) => {
            let left_distance = boundary.saturating_sub(left);
            let right_distance = right.saturating_sub(boundary);
            if left_distance <= right_distance {
                Some(left)
            } else {
                Some(right)
            }
        }
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn line_range_for_span(bytes: &[u8], start: usize, end: usize) -> (usize, usize) {
    let start_line = bytes[..start].iter().filter(|byte| **byte == b'\n').count() + 1;
    let effective_end = end.saturating_sub(1);
    let end_line = bytes[..=effective_end]
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + 1;
    (start_line, end_line.max(start_line))
}

fn align_to_char_boundary(content: &str, boundary: usize) -> usize {
    if boundary >= content.len() {
        return content.len();
    }
    let mut candidate = boundary;
    while candidate > 0 && !content.is_char_boundary(candidate) {
        candidate -= 1;
    }
    candidate
}
