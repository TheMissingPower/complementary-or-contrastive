from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import AnswerKeyOutput, DatasetInput, Distribution, DistributionOutput, Environment
from plugin import get_dataset, pin_dataset
from plugin.datasets import DATASETS, load_datasets
from plugin.generate_datasets import COMPLEMENTARY, SEED, datasets, dumps, near_miss, uniqueness_problem, violations
from verifier.checks import SYMBOLS
from verifier.rules import BLANK, filled, realize, syllables

KEYS = load_datasets()


def ground_truth(key: AnswerKeyOutput) -> dict:
    return {
        "distribution": key.distribution.value,
        "minimal_pairs": [plain_output(pair) for pair in key.minimal_pairs],
        "predictions": [plain_output(answer) for answer in key.answers],
    }


def near_minimal_pairs(key: AnswerKeyOutput) -> list[tuple[str, str]]:
    """Shown words that differ in A vs B and in one other sound: distractors, not minimal pairs."""
    forms = [word.form for word in key.words]
    found = []
    for first, second in combinations(forms, 2):
        differ = [set(pair) for pair in zip(first, second) if pair[0] != pair[1]]
        if len(first) == len(second) and len(differ) == 2 and set(key.target_sounds) in differ:
            found.append((first, second))
    return found


def verdict(key: AnswerKeyOutput, output: dict, tmp_path: Path) -> dict:
    given = DatasetInput(dataset_id=key.dataset_id)
    reference = build_output(plain_output(pin_dataset(given, tmp_path)), AnswerKeyOutput)
    return verifier.verify_task("classify-distribution", given, build_output(output, DistributionOutput), reference, "")


def test_the_frozen_file_is_what_the_generator_makes() -> None:
    assert DATASETS.read_text(encoding="utf-8") == dumps(datasets(random.Random(SEED)))


def test_the_dataset_mix() -> None:
    assert [key.dataset_id for key in KEYS] == [f"lang-{n:02d}" for n in range(1, 21)]
    labels = Counter(key.distribution for key in KEYS)
    assert labels == {Distribution.COMPLEMENTARY: 12, Distribution.CONTRASTIVE: 8}
    environments = Counter(key.rule.environment for key in KEYS if key.rule)
    assert environments == Counter(spec.environment for spec in COMPLEMENTARY)
    assert set(environments) == set(Environment)
    for key in KEYS:
        assert len(key.words) == 30 and len(key.blanks) == 8


def test_every_symbol_is_one_sound_of_the_inventory() -> None:
    for key in KEYS:
        for form in [word.form for word in key.words] + [blank.form.replace(BLANK, "") for blank in key.blanks]:
            assert unicodedata.normalize("NFC", form) == form and set(form) <= SYMBOLS, form
        assert all(len(sound) == 1 for sound in key.target_sounds)


@pytest.mark.parametrize("key", KEYS, ids=[key.dataset_id for key in KEYS])
def test_the_ground_truth_passes_and_the_dataset_is_unique(key: AnswerKeyOutput, tmp_path: Path) -> None:
    assert uniqueness_problem(key) is None
    given = DatasetInput(dataset_id=key.dataset_id)
    assert verifier.verify_output("pin_dataset", given, key, "")["valid"] is True
    result = verdict(key, ground_truth(key), tmp_path)
    assert result["valid"] is True, result["message"]
    if key.distribution is Distribution.CONTRASTIVE:
        assert len(key.minimal_pairs) >= 2 and key.rule is None and not key.answers
        assert len(near_minimal_pairs(key)) >= 2
    else:
        assert not key.minimal_pairs and {answer.sound for answer in key.answers} == set(key.target_sounds)


def test_a_near_miss_rule_fails(tmp_path: Path) -> None:
    # A word-final-conditioned dataset answered by a rule that ignores word-finality, and so on: every blank's
    # near miss is ruled out by only 1 or 2 shown words, yet answering with it fails the dataset.
    key = next(key for key in KEYS if key.rule and key.rule.environment is Environment.BEFORE_WORD_FINAL)
    a, b = key.target_sounds
    forms = [word.form for word in key.words]
    for blank, answer in zip(key.blanks, key.answers):
        rule = near_miss(forms, a, b, blank.form, answer.sound)
        assert rule is not None and 1 <= violations(rule, forms) <= 2
        slot = blank.form.index(BLANK)
        predictions = [
            {
                "blank_id": other.blank_id,
                "sound": realize(rule, filled(other.form, rule.elsewhere))[other.form.index(BLANK)],
            }
            for other in key.blanks
        ]
        assert next(p for p in predictions if p["blank_id"] == blank.blank_id)["sound"] != answer.sound
        assert realize(rule, filled(blank.form, rule.elsewhere))[slot] != answer.sound
        output = {"distribution": "complementary", "minimal_pairs": [], "predictions": predictions}
        result = verdict(key, output, tmp_path)
        assert result["valid"] is False and result["details"]["blank_results"][blank.blank_id] is False


def test_get_dataset_gives_the_data_and_not_the_answer(tmp_path: Path) -> None:
    key = KEYS[0]
    data = plain_output(get_dataset(DatasetInput(dataset_id=key.dataset_id), tmp_path))
    assert set(data) == {"dataset_id", "target_sounds", "words", "blanks"}
    assert [(word["form"], word["gloss"]) for word in data["words"]] == [(word.form, word.gloss) for word in key.words]
    assert [blank["form"] for blank in data["blanks"]] == [blank.form for blank in key.blanks]
    assert "rule" not in json.dumps(data) and "answers" not in json.dumps(data)


@pytest.mark.parametrize(
    ("form", "sound", "split"),
    [
        ("demobri", None, "de.mob.ri"),
        ("tudtigihra", None, "tud.ti.gih.ra"),
        ("sirayawe", None, "si.ra.ya.we"),
        ("tagmid", None, "tag.mid"),
        ("mi_iya", "d", "mi._i.ya"),
        ("n_bhub", "i", "n_b.hub"),
        ("k_de", "u", "k_.de"),
    ],
)
def test_syllable_breaks_follow_the_convention(form, sound, split) -> None:
    assert syllables(form, sound) == split


def test_the_tools_refuse_an_unknown_dataset(tmp_path: Path) -> None:
    for tool in (get_dataset, pin_dataset):
        with pytest.raises(ValueError, match="not one of this plugin's datasets"):
            tool(DatasetInput(dataset_id="lang-99"), tmp_path)
