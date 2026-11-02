"""The frozen datasets (data/datasets.json, made once by generate_datasets.py): get_dataset and the answer key."""

from __future__ import annotations

import json
from pathlib import Path

from slimconfig import build_output

from contract import AnswerKeyOutput, DatasetInput, DatasetOutput, ShownBlank, ShownWord
from verifier.rules import syllables

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
    """The dataset without its answer key, each word also split into syllables by the task's convention."""
    key = pin_dataset(given, workspace)
    sound = key.target_sounds[0]  # both target sounds are vowels, or both consonants
    return DatasetOutput(
        dataset_id=key.dataset_id,
        target_sounds=key.target_sounds,
        words=[ShownWord(word.form, syllables(word.form), word.gloss) for word in key.words],
        blanks=[
            ShownBlank(blank.blank_id, blank.form, syllables(blank.form, sound), blank.gloss) for blank in key.blanks
        ],
    )
