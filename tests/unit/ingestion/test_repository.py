import hashlib
import subprocess
from pathlib import Path

import pytest

from src.ingestion.repository import GitRepository, RepositoryError


def _make_git_repo(path: Path, branch: str = "main") -> None:
    """Initialize a local git repo with one Go file committed."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", branch, str(path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(path), check=True, capture_output=True)
    (path / "main.go").write_text("package main\n")
    subprocess.run(["git", "add", "."], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(path), check=True, capture_output=True)


# ------------------------------------------------------------------
# list_go_files
# ------------------------------------------------------------------

def test_list_go_files_returns_go_files(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.go").write_text("package main")
    (repo_dir / "util.go").write_text("package main")
    (repo_dir / "README.md").write_text("# readme")

    repo = GitRepository("test", "file://x", "main", repo_dir)
    files = repo.list_go_files()

    assert sorted(str(f) for f in files) == ["main.go", "util.go"]


def test_list_go_files_includes_subdirectories(tmp_path):
    repo_dir = tmp_path / "repo"
    sub = repo_dir / "pkg" / "util"
    sub.mkdir(parents=True)
    (repo_dir / "main.go").write_text("package main")
    (sub / "util.go").write_text("package util")

    repo = GitRepository("test", "file://x", "main", repo_dir)
    files = [str(f) for f in repo.list_go_files()]

    assert "main.go" in files
    assert str(Path("pkg/util/util.go")) in files


def test_list_go_files_excludes_vendor(tmp_path):
    repo_dir = tmp_path / "repo"
    vendor = repo_dir / "vendor" / "lib"
    vendor.mkdir(parents=True)
    (repo_dir / "main.go").write_text("package main")
    (vendor / "lib.go").write_text("package lib")

    repo = GitRepository("test", "file://x", "main", repo_dir)
    files = [str(f) for f in repo.list_go_files()]

    assert "main.go" in files
    assert not any("vendor" in f for f in files)


def test_list_go_files_empty_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    repo = GitRepository("test", "file://x", "main", repo_dir)
    assert repo.list_go_files() == []


# ------------------------------------------------------------------
# file_content_hash
# ------------------------------------------------------------------

def test_file_content_hash_matches_sha256(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    content = b"package main\n\nfunc main() {}\n"
    (repo_dir / "main.go").write_bytes(content)

    repo = GitRepository("test", "file://x", "main", repo_dir)
    expected = hashlib.sha256(content).hexdigest()

    assert repo.file_content_hash(Path("main.go")) == expected


def test_file_content_hash_changes_on_modification(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    go_file = repo_dir / "main.go"
    go_file.write_text("package main")

    repo = GitRepository("test", "file://x", "main", repo_dir)
    hash_before = repo.file_content_hash(Path("main.go"))

    go_file.write_text("package main\n// changed")
    hash_after = repo.file_content_hash(Path("main.go"))

    assert hash_before != hash_after


def test_file_content_hash_stable_for_same_content(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.go").write_text("package main")

    repo = GitRepository("test", "file://x", "main", repo_dir)
    h1 = repo.file_content_hash(Path("main.go"))
    h2 = repo.file_content_hash(Path("main.go"))

    assert h1 == h2


# ------------------------------------------------------------------
# clone_or_pull
# ------------------------------------------------------------------

def test_clone_creates_directory(tmp_path):
    source = tmp_path / "source"
    _make_git_repo(source)

    clone_path = tmp_path / "clone"
    repo = GitRepository("test", str(source), "main", clone_path)
    repo.clone_or_pull()

    assert clone_path.exists()
    assert (clone_path / "main.go").exists()


def test_clone_bad_url_raises_repository_error(tmp_path):
    repo = GitRepository(
        "myrepo", "file:///nonexistent/path/to/repo", "main", tmp_path / "clone"
    )
    with pytest.raises(RepositoryError) as exc_info:
        repo.clone_or_pull()

    assert "myrepo" in str(exc_info.value)
    assert "clone" in str(exc_info.value)


def test_repository_error_wraps_called_process_error(tmp_path):
    repo = GitRepository(
        "myrepo", "file:///nonexistent/path/to/repo", "main", tmp_path / "clone"
    )
    with pytest.raises(RepositoryError) as exc_info:
        repo.clone_or_pull()

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, subprocess.CalledProcessError)


def test_pull_bad_remote_raises_repository_error(tmp_path):
    # Clone a valid repo, then break its remote URL
    source = tmp_path / "source"
    _make_git_repo(source)
    clone_path = tmp_path / "clone"
    repo = GitRepository("myrepo", str(source), "main", clone_path)
    repo.clone_or_pull()

    subprocess.run(
        ["git", "remote", "set-url", "origin", "file:///nonexistent"],
        cwd=str(clone_path), check=True, capture_output=True,
    )

    with pytest.raises(RepositoryError) as exc_info:
        repo.clone_or_pull()

    assert "myrepo" in str(exc_info.value)
    assert "pull" in str(exc_info.value)


def test_pull_fetches_new_commits(tmp_path):
    source = tmp_path / "source"
    _make_git_repo(source)

    clone_path = tmp_path / "clone"
    repo = GitRepository("test", str(source), "main", clone_path)
    repo.clone_or_pull()

    # Add a new file to source and commit
    (source / "new.go").write_text("package main")
    subprocess.run(["git", "add", "."], cwd=str(source), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add new.go"], cwd=str(source), check=True, capture_output=True)

    repo.clone_or_pull()

    assert (clone_path / "new.go").exists()
