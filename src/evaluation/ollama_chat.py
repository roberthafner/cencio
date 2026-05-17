import json
import re
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatFunction(Protocol):
    def __call__(self, prompt: str) -> str:
        ...


class OllamaChatFunction:
    def __init__(
        self,
        model: str = "devstral-small-2:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"

    def __call__(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
        }).encode()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            text = data["message"]["content"]
            # Strip reasoning blocks emitted by thinking models that ignore think=false
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            return text.strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {e.code} from {self._url} "
                f"(model={self._model!r}): {body}"
            ) from e
