from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class TokenScore(BaseModel):
    token: str
    score: float


class PredictResponse(BaseModel):
    label: int
    label_name: str
    proba: list[float]
    tokens: list[TokenScore]
    model: str


class ModelResult(BaseModel):
    name: str
    display_name: str
    label: int
    proba: list[float]
    latency_ms: float


class ShowdownResponse(BaseModel):
    models: list[ModelResult]


class Verdict(BaseModel):
    label: int
    label_name: str
    proba: list[float]


class RewriteResponse(BaseModel):
    rewritten: str
    before: Verdict
    after: Verdict


class BatchRow(BaseModel):
    text: str
    label: int
    label_name: str
    proba: list[float]


class BatchResponse(BaseModel):
    total: int
    counts: dict[str, int]
    toxic_ratio: float
    rows: list[BatchRow]


class ModelMetric(BaseModel):
    display_name: str
    accuracy: float
    precision_w: float
    recall_w: float
    f1_w: float
    f1_macro: float


class InsightsResponse(BaseModel):
    best: str
    models: list[ModelMetric]


class MonitorComment(BaseModel):
    text: str
    label: int
    label_name: str
    proba: list[float]
    toxic: bool
    model: str
    seen_at: str
    # top tokens that drove the (toxic) label; empty for clean comments.
    # Computed lazily from the sklearn LR coefficients, like /predict.
    tokens: list[TokenScore] = []


class WatchSummary(BaseModel):
    id: str
    url: str
    label: str | None = None
    created_at: str
    last_scan: str | None = None
    last_error: str | None = None
    alert_count: int
    total_comments: int
    toxic_count: int
    model: str = "PhoBERT"


class WatchDetail(WatchSummary):
    comments: list[MonitorComment]


class CreateWatchRequest(BaseModel):
    url: str = Field(min_length=1)
    label: str | None = None
    model: str = "PhoBERT"


class ModelOption(BaseModel):
    key: str
    name: str


class SourceInfo(BaseModel):
    name: str
    display: str
