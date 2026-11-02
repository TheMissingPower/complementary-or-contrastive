"""The frozen datasets (data/datasets.json, made once by generate_datasets.py): get_dataset and the answer key."""

from __future__ import annotations

import json
from pathlib import Path

from slimconfig import build_output

from contract import AnswerKeyOutput, DatasetInput, DatasetOutput

DATA = Path(__file__).parent / "data"
DATASETS = DATA / "datasets.json"


def load_datasets() -> list[AnswerKeyOutput]:
    return [build_output(entry, AnswerKeyOutput) for entry in json.loads(DATASETS.read_text(encoding="utf-8"))]


def pin_dataset(given: DatasetInput, workspace: Path) -> AnswerKeyOutput:
    datasets = load_datasets()
    for key in datasets:
        if key.dataset_id == given.dataset_id:
            return key
    ids = ", ".join(key.dataset_id for key in datasets)
    raise ValueError(f"{given.dataset_id!r} is not one of this plugin's datasets: {ids}")


def get_dataset(given: DatasetInput, workspace: Path) -> DatasetOutput:
    key = pin_dataset(given, workspace)
    return DatasetOutput(dataset_id=key.dataset_id, target_sounds=key.target_sounds, words=key.words, blanks=key.blanks)
