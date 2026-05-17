import hashlib
import subprocess
from pathlib import Path


class GitRepository:
    def __init__(self, name: str, url: str, branch: str, clone_path: Path) -> None:
        self.name = name
        self.url = url
        self.branch = branch
        self.clone_path = Path(clone_path)

    def clone_or_pull(self) -> None:
        if self.clone_path.exists():
            subprocess.run(
                ["git", "-C", str(self.clone_path), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )
        else:
            self.clone_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git", "clone",
                    "--branch", self.branch,
                    "--single-branch",
                    self.url,
                    str(self.clone_path),
                ],
                check=True,
                capture_output=True,
            )

    def list_go_files(self) -> list[Path]:
        return [
            p.relative_to(self.clone_path)
            for p in self.clone_path.rglob("*.go")
            if p.is_file() and "vendor" not in p.parts
        ]

    def file_content_hash(self, rel_path: Path) -> str:
        data = (self.clone_path / rel_path).read_bytes()
        return hashlib.sha256(data).hexdigest()
