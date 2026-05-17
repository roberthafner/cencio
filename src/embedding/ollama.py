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
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embed"

    def __call__(self, input: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": input}).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]
