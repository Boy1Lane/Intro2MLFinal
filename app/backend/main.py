from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.config import get_settings
from app.backend.routers import predict, rewrite, batch, insights, monitor
from app.backend.services.monitor import MonitorService
from app.backend.services.phobert import PhoBertService
from app.backend.services.scheduler import MonitorScheduler
from app.backend.services.sklearn_registry import SklearnRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.registry = SklearnRegistry(settings.artifacts_dir)
    app.state.registry.load()
    app.state.phobert = PhoBertService(settings.phobert_repo)
    app.state.monitor = MonitorService(app.state.registry, app.state.phobert,
                                       settings)
    app.state.monitor.load()
    app.state.monitor_scheduler = MonitorScheduler(
        app.state.monitor, settings.monitor_interval_sec)
    app.state.monitor_scheduler.start()
    try:
        yield
    finally:
        app.state.monitor_scheduler.shutdown()


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
    app.include_router(rewrite.router)
    app.include_router(batch.router)
    app.include_router(insights.router)
    app.include_router(monitor.router)

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
