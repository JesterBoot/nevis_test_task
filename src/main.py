import os

import uvicorn

from core.setup import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_CONTAINER_PORT", "8080")),
    )
