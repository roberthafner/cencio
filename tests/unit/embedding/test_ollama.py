import json
from unittest.mock import MagicMock, patch

import pytest

from src.embedding.ollama import EmbeddingFunction, OllamaEmbeddingFunction


def _mock_urlopen(embeddings: list[list[float]]):
    """Build a mock context manager that returns the given embeddings."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"embeddings": embeddings}).encode()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp
    mock_cm.__exit__.return_value = False
    return mock_cm


# ------------------------------------------------------------------
# URL construction
# ------------------------------------------------------------------

def test_default_url():
    fn = OllamaEmbeddingFunction()
    assert fn._url == "http://localhost:11434/api/embed"


def test_custom_base_url():
    fn = OllamaEmbeddingFunction(base_url="http://myhost:11434")
    assert fn._url == "http://myhost:11434/api/embed"


def test_trailing_slash_stripped():
    fn = OllamaEmbeddingFunction(base_url="http://localhost:11434/")
    assert fn._url == "http://localhost:11434/api/embed"


# ------------------------------------------------------------------
# Request shape
# ------------------------------------------------------------------

def test_sends_post_request():
    fn = OllamaEmbeddingFunction()
    mock = MagicMock(return_value=_mock_urlopen([[0.1, 0.2]]))
    with patch("urllib.request.urlopen", mock):
        fn(["hello"])

    req = mock.call_args[0][0]
    assert req.get_method() == "POST"


def test_sends_correct_url():
    fn = OllamaEmbeddingFunction()
    mock = MagicMock(return_value=_mock_urlopen([[0.1, 0.2]]))
    with patch("urllib.request.urlopen", mock):
        fn(["hello"])

    req = mock.call_args[0][0]
    assert req.full_url == "http://localhost:11434/api/embed"


def test_sends_content_type_header():
    fn = OllamaEmbeddingFunction()
    mock = MagicMock(return_value=_mock_urlopen([[0.1, 0.2]]))
    with patch("urllib.request.urlopen", mock):
        fn(["hello"])

    req = mock.call_args[0][0]
    assert req.get_header("Content-type") == "application/json"


def test_sends_model_in_payload():
    fn = OllamaEmbeddingFunction(model="nomic-embed-text:v1.5")
    mock = MagicMock(return_value=_mock_urlopen([[0.1, 0.2]]))
    with patch("urllib.request.urlopen", mock):
        fn(["hello"])

    req = mock.call_args[0][0]
    payload = json.loads(req.data)
    assert payload["model"] == "nomic-embed-text:v1.5"


def test_sends_custom_model_in_payload():
    fn = OllamaEmbeddingFunction(model="mxbai-embed-large")
    mock = MagicMock(return_value=_mock_urlopen([[0.1, 0.2]]))
    with patch("urllib.request.urlopen", mock):
        fn(["hello"])

    req = mock.call_args[0][0]
    payload = json.loads(req.data)
    assert payload["model"] == "mxbai-embed-large"


def test_sends_input_texts_in_payload():
    fn = OllamaEmbeddingFunction()
    texts = ["func Foo() {}", "type Bar struct {}"]
    mock = MagicMock(return_value=_mock_urlopen([[0.1], [0.2]]))
    with patch("urllib.request.urlopen", mock):
        fn(texts)

    req = mock.call_args[0][0]
    payload = json.loads(req.data)
    assert payload["input"] == texts


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def test_returns_embeddings_from_response():
    fn = OllamaEmbeddingFunction()
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(expected)):
        result = fn(["text one", "text two"])

    assert result == expected


def test_single_input_returns_one_embedding():
    fn = OllamaEmbeddingFunction()
    expected = [[0.1, 0.2, 0.3, 0.4]]
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(expected)):
        result = fn(["hello world"])

    assert len(result) == 1
    assert result[0] == expected[0]


def test_batch_input_returns_one_embedding_per_text():
    fn = OllamaEmbeddingFunction()
    texts = [f"text {i}" for i in range(5)]
    embeddings = [[float(i)] * 4 for i in range(5)]
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(embeddings)):
        result = fn(texts)

    assert len(result) == 5


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------

def test_satisfies_embedding_function_protocol():
    fn = OllamaEmbeddingFunction()
    assert isinstance(fn, EmbeddingFunction)
