from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.config import get_settings
from app.backend.routers import predict
from app.backend.services.phobert import PhoBertService
from app.backend.services.sklearn_registry import SklearnRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.registry = SklearnRegistry(settings.artifacts_dir)
    app.state.registry.load()
    app.state.phobert = PhoBertService(settings.phobert_repo)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ViHSD Moderation Studio API", lifespan=lifespan)
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(predict.router)

    @app.get("/health")
    def health():
        return {
            "sklearn_loaded": bool(getattr(app.state, "registry", None)
                                   and app.state.registry.models),
            "phobert_available": bool(getattr(app.state, "phobert", None)
                                      and app.state.phobert.available),
        }

    return app


app = create_app()
