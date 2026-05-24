import json
import urllib.request
from typing import Protocol, runtime_checkable

from tokenizers import Tokenizer


@runtime_checkable
class EmbeddingFunction(Protocol):
    def __call__(self, input: list[str]) -> list[list[float]]:
        ...


# nomic-embed-text-v1.5 supports up to 8192 tokens
_DEFAULT_MAX_TOKENS = 8192


class OllamaEmbeddingFunction:
    def __init__(
        self,
        model: str = "nomic-embed-text:v1.5",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._tokenizer = Tokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

    def __call__(self, input: list[str]) -> list[list[float]]:
        truncated = [self._truncate_to_tokens(t) for t in input]
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

    def _truncate_to_tokens(self, text: str) -> str:
        """Truncate text to fit within the model's token limit.

        Uses token offsets to preserve original text formatting rather than
        decoding tokens back to a normalized form.
        """
        encoded = self._tokenizer.encode(text)
        if len(encoded.ids) <= self._max_tokens:
            return text
        # Use offsets to find where to cut the original text.
        # The offsets array includes [CLS] at start and [SEP] at end,
        # so we look at the token just before our limit.
        # Subtract 1 for [CLS], and leave room for [SEP] at the end.
        last_token_idx = self._max_tokens - 2  # -1 for [CLS], -1 for [SEP]
        if last_token_idx < 1:
            return ""
        end_offset = encoded.offsets[last_token_idx][1]
        return text[:end_offset]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        return len(self._tokenizer.encode(text).ids)
