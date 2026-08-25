from fastapi import Request

from search.embeddings import EmbeddingProvider, MiniLMEmbeddingProvider


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    provider = getattr(request.app.state, "embedding_provider", None)
    if provider is None:
        provider = MiniLMEmbeddingProvider()
        request.app.state.embedding_provider = provider
    return provider
