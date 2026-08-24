import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ready_file = Path(
        os.getenv("MODEL_READY_FILE", cache_dir / ".model-ready")
    )

    SentenceTransformer(
        settings.embedding_model,
        cache_folder=str(cache_dir),
        device="cpu",
    )

    ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_ready_file = ready_file.with_name(f"{ready_file.name}.tmp")
    temporary_ready_file.write_text(
        f"{settings.embedding_model}\n",
        encoding="utf-8",
    )
    temporary_ready_file.replace(ready_file)

    print(f"Embedding model is ready in {cache_dir}")


if __name__ == "__main__":
    main()
