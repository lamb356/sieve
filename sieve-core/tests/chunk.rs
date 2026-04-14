use sieve_core::chunk::{Chunk, SlidingChunker};

fn line_range(text: &str, start: usize, end: usize) -> (usize, usize) {
    let start_line = text[..start].bytes().filter(|b| *b == b'\n').count() + 1;
    let end_line = text[..end].bytes().filter(|b| *b == b'\n').count() + 1;
    (start_line, end_line)
}

#[test]
fn test_sliding_chunker_basic() {
    let content = "a".repeat(1200);
    let chunks = SlidingChunker::default().chunk_entry(7, &content);

    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].byte_start, 0);
    assert_eq!(chunks[0].byte_end, 512);
    assert_eq!(chunks[1].byte_start, 256);
    assert_eq!(chunks[1].byte_end, 768);
    assert_eq!(chunks[2].byte_start, 512);
    assert_eq!(chunks[2].byte_end, 1024);
    assert_eq!(chunks[3].byte_start, 768);
    assert_eq!(chunks[3].byte_end, 1200);
    for pair in chunks.windows(2) {
        let overlap = pair[0].byte_end - pair[1].byte_start;
        assert_eq!(overlap, 256);
    }
}

#[test]
fn test_sliding_chunker_newline_snap() {
    let content = format!(
        "{}\n{}\n{}",
        "a".repeat(500),
        "b".repeat(200),
        "c".repeat(200)
    );
    let chunks = SlidingChunker::default().chunk_entry(11, &content);

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].byte_end as usize, 501);
    assert_eq!(chunks[1].byte_start as usize, 245);
    assert_eq!(chunks[1].byte_end as usize, 702);
    assert_eq!(chunks[2].byte_start as usize, 501);
    assert_eq!(chunks[2].byte_end as usize, content.len());
}

#[test]
fn test_sliding_chunker_small_content() {
    let content = "first line\nsecond line\n";
    let chunks = SlidingChunker::default().chunk_entry(3, content);

    assert_eq!(
        chunks,
        vec![Chunk {
            wal_entry_id: 3,
            chunk_id: 0,
            byte_start: 0,
            byte_end: content.len() as u32,
            line_range: line_range(content, 0, content.len()),
            text: content.to_string(),
        }]
    );
}

#[test]
fn test_sliding_chunker_utf8_boundaries() {
    let content = "é".repeat(400);
    let chunks = SlidingChunker::default().chunk_entry(9, &content);

    assert!(chunks.len() >= 2);
    for chunk in chunks {
        assert!(content.is_char_boundary(chunk.byte_start as usize));
        assert!(content.is_char_boundary(chunk.byte_end as usize));
    }
}
