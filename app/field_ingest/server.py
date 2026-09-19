from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.field_ingest.api import create_router
from app.field_ingest.service import FieldIngestService


def create_receiver_app(service: FieldIngestService | None = None) -> FastAPI:
    field_service = service or FieldIngestService()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        field_service.close()

    app = FastAPI(
        title="RefereeLink Field Ingest Receiver",
        description="Receiver-only transport endpoint; inference is intentionally disabled.",
        lifespan=lifespan,
    )
    app.state.field_ingest = field_service
    app.include_router(create_router(field_service))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "field-ingest"}

    return app


app = create_receiver_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RefereeLink field ingest receiver only")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
