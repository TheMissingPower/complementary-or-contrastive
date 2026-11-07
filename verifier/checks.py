"""Checks shared by the verifiers. Pure functions of their arguments."""

import re
import unicodedata

from .rules import BLANK, VOWELS

DATASET_ID = re.compile(r"lang-\d\d")
CONSONANTS = "ptkqʔbdgmnŋsʃhlrwy"  # in a fixed order, for the generator
SYMBOLS = frozenset(CONSONANTS) | VOWELS
# Other spellings of inventory symbols, read as the symbol: Americanist š for IPA ʃ, the IPA script ɡ for g,
# and modifier-letter glottal stops for ʔ.
VARIANTS = str.maketrans({"š": "ʃ", "ɡ": "g", "ʼ": "ʔ", "ˀ": "ʔ"})


def canonical(text):
    """``text`` with variant spellings of inventory symbols replaced by the symbols (after NFC, so s + caron is š)."""
    return unicodedata.normalize("NFC", text).translate(VARIANTS)


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def dataset_id_problem(dataset_id):
    """What makes ``dataset_id`` no dataset id, or None."""
    if not DATASET_ID.fullmatch(dataset_id):
        return f"dataset_id {dataset_id!r} is not of the form lang-NN, e.g. lang-07"
    return None


def dataset_problem(data):
    """What is wrong with a dataset's public part (a DatasetOutput or AnswerKeyOutput), or None."""
    sounds = data.target_sounds
    if len(sounds) != 2 or sounds[0] == sounds[1] or not set(sounds) <= SYMBOLS:
        return f"target_sounds {sounds} are not two different sounds of the inventory"
    if (sounds[0] in VOWELS) != (sounds[1] in VOWELS):
        return f"target_sounds {sounds} are not both vowels or both consonants"
    forms = [word.form for word in data.words]
    if not forms or len(set(forms)) != len(forms):
        return "the words are empty or repeat a form"
    for form in forms:
        if not form or not set(form) <= SYMBOLS:
            return f"word {form!r} has a symbol outside the inventory"
    ids = [blank.blank_id for blank in data.blanks]
    if len(set(ids)) != len(ids):
        return "blank_ids repeat"
    for blank in data.blanks:
        rest = blank.form.replace(BLANK, "", 1)
        if blank.form.count(BLANK) != 1 or not set(rest) <= SYMBOLS:
            return f"blank {blank.blank_id} {blank.form!r} does not have exactly one _ among inventory symbols"
        if set(rest) & set(sounds):
            return f"blank {blank.blank_id} {blank.form!r} has a target sound besides its blank"
    return None
