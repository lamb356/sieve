use std::fs;
use std::process::Command;

use tempfile::tempdir;

fn sieve_bin() -> &'static str {
    env!("CARGO_BIN_EXE_sieve-cli")
}

#[test]
fn index_command_builds_a_dot_sieve_index_in_target_directory() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("lib.rs"), "fn hello() {}\n").unwrap();
    fs::write(dir.path().join("README.md"), "authentication middleware\n").unwrap();

    let status = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();

    assert!(status.success());
    assert!(dir.path().join(".sieve/wal/wal.meta").exists());
    assert!(dir.path().join(".sieve/wal/wal.content").exists());
}

#[test]
fn search_command_finds_indexed_content_from_current_directory() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("handlers.rs"),
        "fn authentication_middleware() {}\n",
    )
    .unwrap();

    let index_status = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(index_status.success());

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "authentication"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("handlers.rs:1:fn authentication_middleware() {}"));
}

#[test]
fn search_without_existing_index_fails_and_does_not_create_dot_sieve() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("handlers.rs"),
        "fn authentication_middleware() {}\n",
    )
    .unwrap();

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "authentication"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(!dir.path().join(".sieve").exists());
}

#[test]
fn reindex_rebuilds_index_instead_of_accumulating_stale_results() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("handlers.rs");
    fs::write(&file_path, "fn hello() {}\n").unwrap();

    let first_index = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(first_index.success());

    fs::write(&file_path, "fn goodbye() {}\n").unwrap();

    let second_index = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(second_index.success());

    let hello = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "hello"])
        .output()
        .unwrap();
    assert!(hello.status.success());
    assert!(String::from_utf8_lossy(&hello.stdout).trim().is_empty());

    let goodbye = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "goodbye"])
        .output()
        .unwrap();
    assert!(goodbye.status.success());
    let stdout = String::from_utf8_lossy(&goodbye.stdout);
    assert_eq!(stdout.matches("handlers.rs:1:fn goodbye() {}").count(), 1);
}

#[test]
fn search_with_index_override_rejects_existing_non_index_directory() {
    let dir = tempdir().unwrap();
    let bogus = dir.path().join("bogus-index-root");
    fs::create_dir_all(&bogus).unwrap();

    let output = Command::new(sieve_bin())
        .args([
            "search",
            "authentication",
            "--index",
            bogus.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(!bogus.join("wal").exists());
}

#[test]
fn test_mtime_skip() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("handlers.rs");
    fs::write(&file_path, "fn hello() {}\n").unwrap();

    let first = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(first.success());

    let wal_meta = dir.path().join(".sieve/wal/wal.meta");
    let first_count = fs::read_to_string(&wal_meta).unwrap().lines().count();
    assert_eq!(first_count, 1);
    assert!(dir.path().join(".sieve/sources/manifest.json").exists());

    let second = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(second.success());

    let second_count = fs::read_to_string(&wal_meta).unwrap().lines().count();
    assert_eq!(
        second_count, 1,
        "unchanged files must not create new WAL entries"
    );
}

#[test]
fn test_json_output() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("handlers.rs"),
        "fn authentication_middleware() {}\n",
    )
    .unwrap();

    let index_status = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(index_status.success());

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "authentication", "--format", "json"])
        .output()
        .unwrap();
    assert!(output.status.success());

    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let arr = value.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["path"], "handlers.rs");
    assert_eq!(arr[0]["line"], 1);
    assert_eq!(arr[0]["layer"], "fused");
}

#[test]
fn test_status_command() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("handlers.rs"),
        "fn authentication_middleware() {}\n",
    )
    .unwrap();

    let index_status = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(index_status.success());

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["status"])
        .output()
        .unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Index root path:"));
    assert!(stdout.contains("WAL entries count:"));
    assert!(stdout.contains("Shard count:"));
    assert!(stdout.contains("Indexed entries count:"));
    assert!(stdout.contains("Unindexed entries count:"));
}

#[test]
fn test_context_lines() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("notes.txt"),
        "line one\nmatch target\nline three\n",
    )
    .unwrap();

    let index_status = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(index_status.success());

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "target", "--context", "1"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("line one"));
    assert!(stdout.contains("match target"));
    assert!(stdout.contains("line three"));
}

#[test]
fn test_deleted_file_is_pruned_on_reindex() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("obsolete.rs");
    fs::write(&file_path, "hello world\n").unwrap();

    let first = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(first.success());

    fs::remove_file(&file_path).unwrap();

    let second = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(second.success());

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "hello"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).trim().is_empty());
}

#[test]
fn test_corrupt_manifest_fails_loudly() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("handlers.rs"),
        "fn authentication_middleware() {}\n",
    )
    .unwrap();

    let index_status = Command::new(sieve_bin())
        .args(["index", dir.path().to_str().unwrap()])
        .status()
        .unwrap();
    assert!(index_status.success());

    fs::write(
        dir.path().join(".sieve/sources/manifest.json"),
        b"{ not json",
    )
    .unwrap();

    let output = Command::new(sieve_bin())
        .current_dir(dir.path())
        .args(["search", "authentication"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}
