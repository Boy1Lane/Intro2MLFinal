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
