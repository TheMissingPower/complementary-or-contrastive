from __future__ import annotations

from itertools import combinations

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import (
    AnswerKeyOutput,
    DatasetInput,
    DatasetOutput,
    Distribution,
    DistributionOutput,
    NotesInput,
    NotesOutput,
)
from plugin.datasets import get_dataset, load_datasets, pin_dataset
from plugin.notes import save_notes

KEYS = load_datasets()
CONTRASTIVE = next(key for key in KEYS if key.distribution is Distribution.CONTRASTIVE)
COMPLEMENTARY = next(key for key in KEYS if key.distribution is Distribution.COMPLEMENTARY)


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verdict(key: AnswerKeyOutput, output: dict, tmp_path) -> dict:
    given = build_output({"dataset_id": key.dataset_id}, DatasetInput)
    reference = build_output(plain_output(pin_dataset(given, tmp_path)), AnswerKeyOutput)
    built = build_output(output, DistributionOutput)
    return verifier.verify_task("classify-distribution", given, built, reference, str(tmp_path))


def contrastive(*pairs: tuple[str, str], predictions=()) -> dict:
    return {
        "distribution": "contrastive",
        "minimal_pairs": [{"first": first, "second": second} for first, second in pairs],
        "predictions": list(predictions),
    }


def complementary(answers: dict[str, str], pairs=()) -> dict:
    return {
        "distribution": "complementary",
        "minimal_pairs": [{"first": first, "second": second} for first, second in pairs],
        "predictions": [{"blank_id": blank_id, "sound": sound} for blank_id, sound in answers.items()],
    }


def true_pairs(key: AnswerKeyOutput) -> list[tuple[str, str]]:
    return [(pair.first, pair.second) for pair in key.minimal_pairs]


def answers(key: AnswerKeyOutput) -> dict[str, str]:
    return {answer.blank_id: answer.sound for answer in key.answers}


def shown_pairs(differ) -> tuple[AnswerKeyOutput, tuple[str, str]]:
    """The first contrastive dataset with two shown words of equal length whose differing sounds pass ``differ``."""
    for key in KEYS:
        if key.distribution is not Distribution.CONTRASTIVE:
            continue
        for first, second in combinations([word.form for word in key.words], 2):
            if len(first) == len(second) and differ([(x, y) for x, y in zip(first, second) if x != y], key):
                return key, (first, second)
    raise AssertionError("no such pair in the frozen datasets")


def near_minimal(differences, key) -> bool:
    return len(differences) == 2 and any(set(pair) == set(key.target_sounds) for pair in differences)


def wrong_sounds(differences, key) -> bool:
    return len(differences) == 1 and set(differences[0]) != set(key.target_sounds)


def test_passes_a_contrastive_answer_with_valid_minimal_pairs(tmp_path) -> None:
    for pairs in (true_pairs(CONTRASTIVE), true_pairs(CONTRASTIVE)[:1], [p[::-1] for p in true_pairs(CONTRASTIVE)]):
        result = verdict(CONTRASTIVE, contrastive(*pairs), tmp_path)
        assert result["valid"] is True and result["score"] == 1.0, result["message"]


def test_passes_a_complementary_answer_with_every_blank_right(tmp_path) -> None:
    result = verdict(COMPLEMENTARY, complementary(answers(COMPLEMENTARY)), tmp_path)
    assert result["valid"] is True and result["score"] == 1.0
    assert result["details"]["blanks_correct"] == result["details"]["blanks_total"] == len(COMPLEMENTARY.blanks)


def test_fails_the_right_label_without_minimal_pairs(tmp_path) -> None:
    result = verdict(CONTRASTIVE, contrastive(), tmp_path)
    assert result["valid"] is False and "at least one minimal pair" in result["message"]


def test_fails_a_near_minimal_pair(tmp_path) -> None:
    key, pair = shown_pairs(near_minimal)
    result = verdict(key, contrastive(true_pairs(key)[0], pair), tmp_path)
    assert result["valid"] is False and "differ in 2 positions" in result["message"]


def test_fails_a_pair_that_differs_in_the_wrong_sounds(tmp_path) -> None:
    key, pair = shown_pairs(wrong_sounds)
    result = verdict(key, contrastive(pair), tmp_path)
    assert result["valid"] is False and "not " in result["message"] and "invalid minimal pair" in result["message"]


@pytest.mark.parametrize("which", [0, 1], ids=["first", "second"])
def test_fails_a_pair_with_a_word_not_in_the_dataset(which, tmp_path) -> None:
    first, second = true_pairs(CONTRASTIVE)[0]
    a = CONTRASTIVE.target_sounds[0]
    # A target sound appended: words the dataset does not have, still differing only in A and B.
    invented = (first + a, second + a)
    pair = (invented[0], second) if which == 0 else (first, invented[1])
    result = verdict(CONTRASTIVE, contrastive(pair), tmp_path)
    assert result["valid"] is False and "is not one of the shown words" in result["message"]


def test_fails_one_invalid_pair_among_valid_ones(tmp_path) -> None:
    first = true_pairs(CONTRASTIVE)[0][0]
    result = verdict(CONTRASTIVE, contrastive(*true_pairs(CONTRASTIVE), (first, first)), tmp_path)
    assert result["valid"] is False and "differ in 0 positions" in result["message"]


def test_fails_a_contrastive_answer_with_predictions(tmp_path) -> None:
    blank = CONTRASTIVE.blanks[0].blank_id
    output = contrastive(*true_pairs(CONTRASTIVE), predictions=[{"blank_id": blank, "sound": "p"}])
    result = verdict(CONTRASTIVE, output, tmp_path)
    assert result["valid"] is False and "no predictions" in result["message"]


def test_fails_one_wrong_blank_and_reports_partial_accuracy(tmp_path) -> None:
    given = answers(COMPLEMENTARY)
    blank = next(iter(given))
    a, b = COMPLEMENTARY.target_sounds
    given[blank] = b if given[blank] == a else a
    result = verdict(COMPLEMENTARY, complementary(given), tmp_path)
    assert result["valid"] is False and result["score"] == 0.0 and f"1 of 8 blanks wrong: {blank}" in result["message"]
    assert result["details"]["blanks_correct"] == 7 and result["details"]["blank_results"][blank] is False


def test_fails_a_missing_blank(tmp_path) -> None:
    given = answers(COMPLEMENTARY)
    blank = list(given)[-1]
    del given[blank]
    result = verdict(COMPLEMENTARY, complementary(given), tmp_path)
    assert result["valid"] is False and f"blanks with no prediction: {blank}" in result["message"]


def test_fails_a_duplicate_blank(tmp_path) -> None:
    output = complementary(answers(COMPLEMENTARY))
    output["predictions"].append(dict(output["predictions"][0]))
    result = verdict(COMPLEMENTARY, output, tmp_path)
    assert result["valid"] is False and "predicted more than once" in result["message"]


def test_fails_an_unknown_blank(tmp_path) -> None:
    result = verdict(
        COMPLEMENTARY, complementary({**answers(COMPLEMENTARY), "b99": COMPLEMENTARY.target_sounds[0]}), tmp_path
    )
    assert result["valid"] is False and "'b99' is not one of the blanks" in result["message"]


@pytest.mark.parametrize("sound", ["z", "", "_"])
def test_fails_a_prediction_that_is_neither_target_sound(sound, tmp_path) -> None:
    given = answers(COMPLEMENTARY)
    given[next(iter(given))] = sound
    result = verdict(COMPLEMENTARY, complementary(given), tmp_path)
    assert result["valid"] is False and "is neither" in result["message"]


def test_fails_a_wrong_label_either_way(tmp_path) -> None:
    called_contrastive = verdict(COMPLEMENTARY, contrastive(*true_pairs(CONTRASTIVE)), tmp_path)
    assert (
        called_contrastive["valid"] is False
        and "complementary distribution, not contrastive" in called_contrastive["message"]
    )
    called_complementary = verdict(CONTRASTIVE, complementary({}), tmp_path)
    assert called_complementary["valid"] is False
    assert "contrastive distribution, not complementary" in called_complementary["message"]


def test_fails_a_complementary_answer_that_lists_minimal_pairs(tmp_path) -> None:
    forms = [word.form for word in COMPLEMENTARY.words]
    result = verdict(COMPLEMENTARY, complementary(answers(COMPLEMENTARY), pairs=[(forms[0], forms[1])]), tmp_path)
    assert result["valid"] is False and "no minimal pairs" in result["message"]


def test_fails_an_answer_key_for_another_dataset(tmp_path) -> None:
    given = build_output({"dataset_id": COMPLEMENTARY.dataset_id}, DatasetInput)
    output = build_output(contrastive(*true_pairs(CONTRASTIVE)), DistributionOutput)
    result = verifier.verify_task("classify-distribution", given, output, CONTRASTIVE, str(tmp_path))
    assert result["valid"] is False and "answer key is for" in result["message"]


@pytest.mark.parametrize(
    "output",
    [
        {"distribution": "contrastive", "minimal_pairs": []},
        {"distribution": "contrastive", "predictions": []},
        {"minimal_pairs": [], "predictions": []},
        {"distribution": "allophonic", "minimal_pairs": [], "predictions": []},
        {"distribution": "Contrastive", "minimal_pairs": [], "predictions": []},
        {"distribution": None, "minimal_pairs": [], "predictions": []},
        {"distribution": "contrastive", "minimal_pairs": [["pa", "ba"]], "predictions": []},
        {"distribution": "contrastive", "minimal_pairs": [{"first": "pa"}], "predictions": []},
        {"distribution": "contrastive", "minimal_pairs": "pa ba", "predictions": []},
        {"distribution": "complementary", "minimal_pairs": [], "predictions": {"b1": "t"}},
        {"distribution": "complementary", "minimal_pairs": [], "predictions": [{"blank_id": "b1", "sound": 1}]},
        {"distribution": "complementary", "minimal_pairs": [], "predictions": [{"blank_id": "b1"}]},
        {"distribution": "complementary", "minimal_pairs": [], "predictions": [], "rule": "word-final"},
        None,
    ],
)
def test_malformed_output_never_reaches_the_verifier(output) -> None:
    # NanoAgent builds the submitted output into DistributionOutput first, so a malformed one fails the task there.
    with pytest.raises(ValueError):
        build_output(output, DistributionOutput)


def test_checks_get_dataset_calls(tmp_path) -> None:
    given = build_output({"dataset_id": COMPLEMENTARY.dataset_id}, DatasetInput)
    assert verifier.verify_input("get_dataset", given)["valid"] is True
    for bad in ["lang-1", "LANG-01", "lang-01 ", "dataset-01", ""]:
        assert verifier.verify_input("get_dataset", build_output({"dataset_id": bad}, DatasetInput))["valid"] is False
    returned = plain_output(get_dataset(given, tmp_path))
    assert verifier.verify_output("get_dataset", given, build_output(returned, DatasetOutput), "")["valid"] is True
    blank = returned["blanks"][0]
    for broken in [
        {**returned, "dataset_id": CONTRASTIVE.dataset_id},
        {**returned, "target_sounds": returned["target_sounds"][:1]},
        {**returned, "target_sounds": [returned["target_sounds"][0]] * 2},
        {**returned, "words": returned["words"] + returned["words"][:1]},
        {**returned, "words": [{"form": "pa1", "syllables": "pa1", "gloss": "x"}]},
        {**returned, "words": [{**returned["words"][0], "syllables": returned["words"][0]["form"]}]},
        {**returned, "blanks": [{**blank, "syllables": blank["form"]}]},
        {**returned, "blanks": [{**blank, "form": blank["form"].replace("_", "")}]},
        {**returned, "blanks": [{**blank, "form": blank["form"] + "_"}]},
        {**returned, "blanks": [blank, blank]},
    ]:
        built = build_output(broken, DatasetOutput)
        assert verifier.verify_output("get_dataset", given, built, "")["valid"] is False


def test_checks_save_notes_calls(tmp_path) -> None:
    assert verifier.verify_input("save_notes", build_output({"notes": "no minimal pair"}, NotesInput))["valid"] is True
    assert verifier.verify_input("save_notes", build_output({"notes": "  "}, NotesInput))["valid"] is False
    notes = []
    for text in ["no minimal pair", "r: ri.ke, before i (ends syllable)"]:
        given = build_output({"notes": text}, NotesInput)
        returned = build_output(plain_output(save_notes(given, tmp_path)), NotesOutput)
        notes.append(text)
        assert returned.notes == notes
        assert verifier.verify_output("save_notes", given, returned, str(tmp_path))["valid"] is True
    for broken in [
        {"path": "notes.md", "notes": notes},
        {"path": "notes.json", "notes": notes[:1]},
        {"path": "notes.json", "notes": ["something else", notes[-1]]},
    ]:
        built = build_output(broken, NotesOutput)
        assert verifier.verify_output("save_notes", given, built, str(tmp_path))["valid"] is False


def test_refuses_a_tampered_answer_key(tmp_path) -> None:
    for key in (COMPLEMENTARY, CONTRASTIVE):
        given = build_output({"dataset_id": key.dataset_id}, DatasetInput)
        pinned = plain_output(pin_dataset(given, tmp_path))
        assert verifier.verify_output("pin_dataset", given, build_output(pinned, AnswerKeyOutput), "")["valid"] is True
        a, b = pinned["target_sounds"]
        flipped = [{**answer, "sound": b if answer["sound"] == a else a} for answer in pinned["answers"]]
        tampered = [
            {**pinned, "dataset_id": "lang-99"},
            {**pinned, "distribution": "contrastive" if key is COMPLEMENTARY else "complementary"},
        ]
        if key is COMPLEMENTARY:
            tampered += [
                {**pinned, "answers": flipped[:1] + pinned["answers"][1:]},
                {**pinned, "answers": pinned["answers"][1:]},
                {**pinned, "rule": {**pinned["rule"], "sounds": ["p"], "environment": "before"}},
                {**pinned, "rule": None},
            ]
        else:
            tampered += [
                {**pinned, "answers": [{"blank_id": "b1", "sound": a}]},
                {**pinned, "minimal_pairs": pinned["minimal_pairs"][1:]},
            ]
        for entry in tampered:
            built = build_output(entry, AnswerKeyOutput)
            assert verifier.verify_output("pin_dataset", given, built, "")["valid"] is False, entry
