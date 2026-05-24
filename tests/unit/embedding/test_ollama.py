import json
from unittest.mock import MagicMock, patch

import pytest

from src.embedding.ollama import EmbeddingFunction, OllamaEmbeddingFunction, _DEFAULT_MAX_TOKENS


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
# Batching
# ------------------------------------------------------------------

def test_default_batch_size():
    fn = OllamaEmbeddingFunction()
    assert fn._batch_size == 32


def test_custom_batch_size():
    fn = OllamaEmbeddingFunction(batch_size=8)
    assert fn._batch_size == 8


def test_default_max_tokens():
    fn = OllamaEmbeddingFunction()
    assert fn._max_tokens == _DEFAULT_MAX_TOKENS


def test_inputs_truncated_to_max_tokens():
    fn = OllamaEmbeddingFunction(max_tokens=5)
    mock = MagicMock(return_value=_mock_urlopen([[0.1]]))
    with patch("urllib.request.urlopen", mock):
        fn(["hello world this is a longer text"])
    req = mock.call_args[0][0]
    payload = json.loads(req.data)
    # The tokenizer will truncate to 5 tokens and decode back
    # Exact output depends on tokenizer, but it should be shorter than original
    assert len(payload["input"][0]) < len("hello world this is a longer text")


def test_single_batch_when_inputs_fit():
    fn = OllamaEmbeddingFunction(batch_size=4)
    texts = ["a", "b", "c"]
    embeddings = [[float(i)] * 2 for i in range(3)]
    mock = MagicMock(return_value=_mock_urlopen(embeddings))
    with patch("urllib.request.urlopen", mock):
        fn(texts)
    assert mock.call_count == 1


def test_multiple_batches_when_inputs_exceed_batch_size():
    fn = OllamaEmbeddingFunction(batch_size=2)
    texts = ["a", "b", "c", "d", "e"]
    batch_embeddings = [
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6], [0.7, 0.8]],
        [[0.9, 1.0]],
    ]
    mock = MagicMock(side_effect=[_mock_urlopen(b) for b in batch_embeddings])
    with patch("urllib.request.urlopen", mock):
        result = fn(texts)
    assert mock.call_count == 3
    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]


def test_batch_payloads_contain_correct_inputs():
    fn = OllamaEmbeddingFunction(batch_size=2)
    texts = ["a", "b", "c"]
    batch_embeddings = [[[0.1], [0.2]], [[0.3]]]
    mock = MagicMock(side_effect=[_mock_urlopen(b) for b in batch_embeddings])
    with patch("urllib.request.urlopen", mock):
        fn(texts)
    first_req = mock.call_args_list[0][0][0]
    second_req = mock.call_args_list[1][0][0]
    assert json.loads(first_req.data)["input"] == ["a", "b"]
    assert json.loads(second_req.data)["input"] == ["c"]


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------

def test_satisfies_embedding_function_protocol():
    fn = OllamaEmbeddingFunction()
    assert isinstance(fn, EmbeddingFunction)


def test_count_tokens_returns_token_count():
    fn = OllamaEmbeddingFunction()
    # "hello world" should tokenize to a small number of tokens
    count = fn.count_tokens("hello world")
    assert isinstance(count, int)
    assert count > 0
    assert count < 10  # Should be just a few tokens


def test_count_tokens_longer_text_has_more_tokens():
    fn = OllamaEmbeddingFunction()
    short_count = fn.count_tokens("hello")
    long_count = fn.count_tokens("hello world this is a much longer piece of text")
    assert long_count > short_count
