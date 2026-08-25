from pathlib import Path

import numpy as np
import pytest

import search.embeddings as embeddings_module
from core.config import Settings
from search.embeddings import (
    FakeEmbeddingProvider,
    MiniLMEmbeddingProvider,
)

EMBEDDING_DIMENSION = Settings(_env_file=None).embedding_dimension


def test_fake_provider_is_deterministic_and_batch_aligned() -> None:
    provider = FakeEmbeddingProvider()

    first_result = provider.embed(["alpha", "beta", "gamma"])
    second_result = provider.embed(["alpha", "beta", "gamma"])

    assert first_result == second_result
    assert len(first_result) == 3
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in first_result)
    assert all(np.isclose(np.linalg.norm(vector), 1.0) for vector in first_result)


def test_empty_input_returns_empty_output() -> None:
    assert FakeEmbeddingProvider().embed([]) == []


def test_fake_batch_embeddings_match_individual_embeddings() -> None:
    provider = FakeEmbeddingProvider()
    texts = ["alpha", "beta", "gamma"]

    batch_result = provider.embed(texts)
    individual_result = [provider.embed([text])[0] for text in texts]

    assert np.allclose(batch_result, individual_result, rtol=0, atol=1e-12)


def test_minilm_provider_normalizes_batch_and_validates_dimension() -> None:
    model = _RecordingModel()
    provider = MiniLMEmbeddingProvider(model=model)

    result = provider.embed(["alpha", "beta"])

    assert model.calls == [["alpha", "beta"]]
    assert len(result) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in result)
    assert all(np.isclose(np.linalg.norm(vector), 1.0) for vector in result)


def test_minilm_provider_rejects_invalid_embedding_dimension() -> None:
    provider = MiniLMEmbeddingProvider(model=_RecordingModel(dimension=3))

    with pytest.raises(
        ValueError,
        match=f"expected {EMBEDDING_DIMENSION}",
    ):
        provider.embed(["alpha"])


def test_minilm_provider_loads_one_model_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def model_factory(*args: object, **kwargs: object) -> _RecordingModel:
        factory_calls.append((args, kwargs))
        return _RecordingModel()

    monkeypatch.setattr(embeddings_module, "SentenceTransformer", model_factory)
    settings = Settings(
        _env_file=None,
        model_cache_dir=Path("/tmp/nevis-model-cache"),
    )
    provider = MiniLMEmbeddingProvider(settings)

    for index in range(10):
        provider.embed([f"text-{index}"])

    assert len(factory_calls) == 1
    assert factory_calls[0][0] == (settings.embedding_model,)
    assert factory_calls[0][1]["cache_folder"] == str(settings.model_cache_dir)
    assert factory_calls[0][1]["device"] == "cpu"
    assert factory_calls[0][1]["local_files_only"] is True


class _RecordingModel:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.calls: list[list[str]] = []
        self.dimension = dimension

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True
        assert normalize_embeddings is False
        self.calls.append(sentences)
        return np.asarray(
            [
                [float(index + 1) for index in range(self.dimension)]
                for _ in sentences
            ]
        )
