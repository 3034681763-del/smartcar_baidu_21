import json
from pathlib import Path


def load_params(filename="param.json"):
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
