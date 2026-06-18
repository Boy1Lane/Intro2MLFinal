from fastapi import APIRouter

from app.backend.schemas import InsightsResponse
from app.backend.services.metrics import load_insights

router = APIRouter()


@router.get("/insights", response_model=InsightsResponse)
def insights() -> InsightsResponse:
    return InsightsResponse(**load_insights())
