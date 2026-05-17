import json
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingFunction(Protocol):
    def __call__(self, input: list[str]) -> list[list[float]]:
        ...


class OllamaEmbeddingFunction:
    def __init__(
        self,
        model: str = "nomic-embed-text:v1.5",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        max_chars: int = 4000,
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._batch_size = batch_size
        self._max_chars = max_chars

    def __call__(self, input: list[str]) -> list[list[float]]:
        truncated = [t[: self._max_chars] for t in input]
        results: list[list[float]] = []
        for i in range(0, len(truncated), self._batch_size):
            batch = truncated[i : i + self._batch_size]
            results.extend(self._embed_batch(batch))
        return results

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": batch}).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            return data["embeddings"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {e.code} from {self._url} "
                f"(model={self._model!r}, inputs={len(batch)}): {body}"
            ) from e
