import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "insights_data.json"


@lru_cache
def load_insights() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))
