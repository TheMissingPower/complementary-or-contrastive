"""Write data/datasets.json, the frozen datasets: 12 complementary and 8 contrastive, all invented languages.

Run it once, from the plugin root:

    uv run python -m plugin.generate_datasets

A fixed seed makes the same file every time, which a test checks. A dataset is drawn again until it passes
uniqueness_problem: the shown words decide every blank, and each blank is one a near-miss rule gets wrong.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import pairwise

from slimconfig import plain_output

from contract import AnswerKeyOutput, Blank, Distribution, Environment, MinimalPair, Prediction, Rule, Word
from plugin.datasets import DATASETS
from verifier.checks import CONSONANTS
from verifier.rules import (
    BLANK,
    BOUNDARY,
    SET_ENVIRONMENTS,
    VOWELS,
    applies,
    consistent_predictions,
    context,
    filled,
    fitting_rules,
    minimal_pairs,
    realize,
    targets,
)

SEED = 480
SHOWN = 30
TARGET_WORDS = 22  # shown words with a target sound; the rest have none
BLANKS = 8
DRAFTS = 6000
ATTEMPTS = 400
CANDIDATES = 40  # blanks that pass, to choose 8 from
INVENTORY_CONSONANTS = 10
EVERY_CONSONANT = "*"  # a rule's sounds: every consonant of the language
GLOSSES = [
    "water",
    "fire",
    "stone",
    "tree",
    "river",
    "moon",
    "sun",
    "star",
    "cloud",
    "rain",
    "wind",
    "snow",
    "sand",
    "hill",
    "road",
    "house",
    "door",
    "roof",
    "bread",
    "salt",
    "meat",
    "fish",
    "bird",
    "dog",
    "horse",
    "cow",
    "goat",
    "snake",
    "egg",
    "milk",
    "honey",
    "seed",
    "leaf",
    "root",
    "flower",
    "grass",
    "bone",
    "blood",
    "skin",
    "hair",
    "eye",
    "ear",
    "nose",
    "mouth",
    "tooth",
    "tongue",
    "hand",
    "foot",
    "knee",
    "heart",
    "belly",
    "mother",
    "father",
    "child",
    "friend",
    "chief",
    "guest",
    "name",
    "song",
    "dance",
    "game",
    "knife",
    "rope",
    "basket",
    "pot",
    "cup",
    "bowl",
    "boat",
    "net",
    "spear",
    "bow",
    "arrow",
    "drum",
    "mat",
    "bed",
    "fence",
    "field",
    "garden",
    "village",
    "market",
    "gift",
    "debt",
    "price",
    "story",
    "dream",
    "smoke",
    "ash",
    "shadow",
    "island",
    "lake",
    "cave",
    "forest",
    "valley",
    "bridge",
    "wall",
    "gate",
    "tower",
    "ring",
    "belt",
]

VOWEL_ORDER = "aiueo"


@dataclass(frozen=True)
class Spec:
    elsewhere: str
    special: str
    environment: Environment | None = None  # None: contrastive
    sounds: str = ""
    foil: tuple[Environment, str] | None = None  # the near-miss rule a few shown words rule out


COMPLEMENTARY = [
    Spec("s", "ʃ", Environment.BEFORE, "ie", (Environment.BEFORE, "ieu")),
    Spec("a", "e", Environment.BEFORE, "yʃ", (Environment.BEFORE, "yʃs")),
    Spec("k", "q", Environment.AFTER, "ao", (Environment.AFTER, "aou")),
    Spec("u", "o", Environment.AFTER, "qr", (Environment.AFTER, "qrk")),
    Spec("l", "d", Environment.WORD_INITIAL),
    Spec("t", "ʔ", Environment.WORD_FINAL, "", (Environment.CLOSED_SYLLABLE, "")),
    Spec("i", "e", Environment.CLOSED_SYLLABLE, "", (Environment.BEFORE_WORD_FINAL, EVERY_CONSONANT)),
    Spec("u", "o", Environment.CLOSED_SYLLABLE),
    Spec("i", "e", Environment.BEFORE_WORD_FINAL, "kq", (Environment.BEFORE, "kq")),
    Spec("s", "h", Environment.BEFORE_WORD_FINAL, "au", (Environment.BEFORE, "au")),
    Spec("a", "o", Environment.BEFORE_SYLLABLE_FINAL, "nŋ", (Environment.BEFORE_WORD_FINAL, "nŋ")),
    Spec("d", "r", Environment.BEFORE_SYLLABLE_FINAL, "ai", (Environment.BEFORE_WORD_FINAL, "ai")),
]
CONTRASTIVE = [Spec(a, b) for a, b in ["pb", "td", "sʃ", "nŋ", "ie", "uo", "lr", "kq"]]


def describe(environment: Environment, special: str, elsewhere: str, sounds: list[str]) -> str:
    named = " or ".join(sounds)
    where = {
        Environment.BEFORE: f"before {named}",
        Environment.AFTER: f"after {named}",
        Environment.WORD_INITIAL: "word-initially",
        Environment.WORD_FINAL: "word-finally",
        Environment.BEFORE_WORD_FINAL: f"before a word-final {named}",
        Environment.BEFORE_SYLLABLE_FINAL: f"before a syllable-final {named}",
        Environment.CLOSED_SYLLABLE: "in a closed syllable",
    }[environment]
    return f"/{elsewhere}/ is [{special}] {where}, and [{elsewhere}] elsewhere"


def make_rule(environment: Environment, special: str, elsewhere: str, sounds) -> Rule:
    sounds = sorted(sounds) if environment in SET_ENVIRONMENTS else []
    return Rule(special, elsewhere, environment, sounds, describe(environment, special, elsewhere, sounds))


def inventory(rng: random.Random, spec: Spec) -> tuple[list[str], list[str]]:
    """The language's consonants and vowels: every sound its spec needs, never the allophone, more at random."""
    banned = {spec.special} if spec.environment else set()
    needed = set(spec.elsewhere + spec.sounds + (spec.foil[1] if spec.foil else "") + spec.special) - banned
    needed.discard(EVERY_CONSONANT)
    extra = [c for c in CONSONANTS if c not in needed | banned]
    count = INVENTORY_CONSONANTS - len([c for c in needed if c in CONSONANTS])
    chosen = needed | set(rng.sample(extra, count))
    return [c for c in CONSONANTS if c in chosen], [v for v in VOWEL_ORDER if v not in banned]


def pick(rng: random.Random, symbols: list[str], favored: tuple[str, ...]) -> str:
    boosted = [s for s in symbols if s in favored]
    return rng.choice(boosted) if boosted and rng.random() < 0.25 else rng.choice(symbols)


def drafts(rng: random.Random, consonants: list[str], vowels: list[str], favored: tuple[str, ...]) -> list[str]:
    """Distinct words of 2 to 4 CV or CVC syllables, no two target sounds side by side."""
    found = {}
    for _ in range(DRAFTS):
        symbols = []
        for _ in range(rng.randint(2, 4)):
            symbols += [pick(rng, consonants, favored), pick(rng, vowels, favored)]
            if rng.random() < 0.45:
                symbols.append(pick(rng, consonants, favored))
        form = "".join(symbols)
        slots = targets(form, favored)
        if all(j - i > 1 for i, j in pairwise(slots)):
            found.setdefault(form, None)
    return list(found)


def violations(rule: Rule, forms: list[str]) -> int:
    """How many forms the rule does not fit: some target sound in them is not the one it gives."""
    pair = (rule.special, rule.elsewhere)
    return sum(
        any(applies(rule.environment, rule.sounds, form, i) != (form[i] == rule.special) for i in targets(form, pair))
        for form in forms
    )


def candidate_sounds(found, key, want_in):
    """Sets S worth trying for a near miss: the special sound's contexts, the blank's key forced in or out, and
    each of those with one more context toggled. ``found``: (word, context, is special) of every target."""
    seen = sorted({found_context for _, found_context, _ in found} - {None, BOUNDARY})
    base = {found_context for _, found_context, special in found if special} - {None, BOUNDARY}
    if key not in (None, BOUNDARY):
        base = base | {key} if want_in else base - {key}
    yield base
    for other in seen:
        if other != key:
            yield base ^ {other}


def near_miss(forms: list[str], a: str, b: str, blank_form: str, answer: str) -> Rule | None:
    """A rule that 1 or 2 shown words rule out and that gives the blank the wrong sound, or None if none is found.

    The search tries a few sets per environment, so a rule it finds is a real near miss; it may miss some.
    """
    wrong = b if answer == a else a
    slot = blank_form.index(BLANK)
    for environment in Environment:
        is_set = environment in SET_ENVIRONMENTS
        for special, elsewhere in ((a, b), (b, a)):
            found = [
                (n, context(environment, form, i), form[i] == special)
                for n, form in enumerate(forms)
                for i in targets(form, (a, b))
            ]
            key = context(environment, filled(blank_form, elsewhere), slot)
            for sounds in candidate_sounds(found, key, wrong == special) if is_set else [set()]:
                holds = key in sounds if is_set else key
                if (special if holds else elsewhere) != wrong:
                    continue
                ruled_out = {
                    n
                    for n, found_context, is_special in found
                    if (found_context in sounds if is_set else found_context) != is_special
                }
                if 1 <= len(ruled_out) <= 2:
                    return make_rule(environment, special, elsewhere, sounds)
    return None


def uniqueness_problem(key: AnswerKeyOutput) -> str | None:
    """What keeps the dataset from having exactly one defensible answer, or None."""
    a, b = key.target_sounds
    forms = [word.form for word in key.words]
    pairs = minimal_pairs(forms, a, b)
    if key.distribution is Distribution.CONTRASTIVE:
        if fitting_rules(forms, a, b):
            return "a rule fits a contrastive dataset"
        return None if len(pairs) >= 2 else "a contrastive dataset needs at least 2 minimal pairs"
    if pairs:
        return "a complementary dataset has a minimal pair"
    if not fitting_rules(forms, a, b):
        return "no rule fits a complementary dataset"
    answers = {answer.blank_id: answer.sound for answer in key.answers}
    for blank in key.blanks:
        answer = answers[blank.blank_id]
        if consistent_predictions(forms, a, b, blank.form) != {answer}:
            return f"the rules that fit disagree on blank {blank.blank_id}"
        if near_miss(forms, a, b, blank.form, answer) is None:
            return f"no near-miss rule gets blank {blank.blank_id} wrong"
    return None


def answer_key(rng, spec, forms, blank_forms, distribution, rule, answers) -> AnswerKeyOutput:
    glosses = rng.sample(GLOSSES, len(forms) + len(blank_forms))
    sounds = [spec.elsewhere, spec.special]
    rng.shuffle(sounds)
    return AnswerKeyOutput(
        dataset_id="",
        target_sounds=sounds,
        words=[Word(form, gloss) for form, gloss in zip(forms, glosses)],
        blanks=[
            Blank(f"b{n + 1}", form, gloss) for n, (form, gloss) in enumerate(zip(blank_forms, glosses[len(forms) :]))
        ],
        distribution=distribution,
        minimal_pairs=[MinimalPair(*pair) for pair in minimal_pairs(forms, spec.elsewhere, spec.special)],
        rule=rule,
        answers=[Prediction(f"b{n + 1}", sound) for n, sound in enumerate(answers)],
    )


def choose_blanks(rng, candidates, rule, foil, special_needed=3, separating_needed=2):
    """8 (blank form, answer) from evaluated candidates: some the foil gets wrong, and both answers 3 times at least."""
    chosen = []

    def take(keep, count):
        chosen.extend([c for c in candidates if keep(c) and c not in chosen][:count])

    if foil:
        take(lambda c: c[2], separating_needed)
    for sound in (rule.special, rule.elsewhere):
        take(lambda c, sound=sound: c[1] == sound, special_needed - sum(c[1] == sound for c in chosen))
    take(lambda c: True, BLANKS - len(chosen))
    enough = len(chosen) == BLANKS and (not foil or sum(c[2] for c in chosen) >= separating_needed)
    if not enough or min(sum(c[1] == s for c in chosen) for s in (rule.special, rule.elsewhere)) < special_needed:
        return None
    rng.shuffle(chosen)
    return [(form, answer) for form, answer, _ in chosen]


def complementary(rng: random.Random, spec: Spec) -> AnswerKeyOutput:
    consonants, vowels = inventory(rng, spec)
    rule = make_rule(spec.environment, spec.special, spec.elsewhere, spec.sounds)
    foil = None
    if spec.foil:
        environment, sounds = spec.foil
        foil = make_rule(environment, spec.special, spec.elsewhere, consonants if sounds == EVERY_CONSONANT else sounds)
    underlying = drafts(rng, consonants, vowels, (spec.elsewhere,))

    def separates(form):
        return foil is not None and realize(foil, form) != realize(rule, form)

    with_target = [form for form in underlying if 1 <= form.count(spec.elsewhere) <= 2]
    separating = [form for form in with_target if separates(form)]
    plain = [form for form in with_target if not separates(form)]
    without = [form for form in underlying if spec.elsewhere not in form]
    singles = [form for form in underlying if form.count(spec.elsewhere) == 1]
    for _ in range(ATTEMPTS):
        k = rng.choice((1, 2)) if foil else 0
        chosen = (
            rng.sample(separating, k) + rng.sample(plain, TARGET_WORDS - k) + rng.sample(without, SHOWN - TARGET_WORDS)
        )
        rng.shuffle(chosen)
        forms = [realize(rule, form) for form in chosen]
        surface = "".join(forms)
        if min(surface.count(spec.special), surface.count(spec.elsewhere)) < 4:
            continue
        shown = set(forms)
        candidates = []
        for form in rng.sample(singles, 300):
            slot = form.index(spec.elsewhere)
            blank_form = form[:slot] + BLANK + form[slot + 1 :]
            answer = realize(rule, form)[slot]
            if form in shown or filled(blank_form, spec.special) in shown or blank_form in [c[0] for c in candidates]:
                continue
            if consistent_predictions(forms, spec.elsewhere, spec.special, blank_form) != {answer}:
                continue
            if near_miss(forms, spec.elsewhere, spec.special, blank_form, answer) is None:
                continue
            candidates.append((blank_form, answer, separates(form)))
            if len(candidates) == CANDIDATES:
                break
        blanks = choose_blanks(rng, candidates, rule, foil)
        if blanks is None:
            continue
        key = answer_key(
            rng, spec, forms, [form for form, _ in blanks], Distribution.COMPLEMENTARY, rule, [a for _, a in blanks]
        )
        if uniqueness_problem(key) is None:
            return key
    raise RuntimeError(f"no dataset for {rule.description} in {ATTEMPTS} attempts")


def swap(form: str, slot: int, a: str, b: str) -> str:
    return form[:slot] + (b if form[slot] == a else a) + form[slot + 1 :]


def contrastive(rng: random.Random, spec: Spec) -> AnswerKeyOutput:
    a, b = spec.elsewhere, spec.special
    consonants, vowels = inventory(rng, spec)
    words = drafts(rng, consonants, vowels, (a, b))
    with_target = [form for form in words if targets(form, (a, b))]
    singles = [form for form in words if len(targets(form, (a, b))) == 1]
    for _ in range(ATTEMPTS):
        shown = []
        # Two true minimal pairs, then two near-minimal pairs: A for B and one other sound changed too.
        for near in (False, False, True, True):
            form = rng.choice(with_target)
            other = swap(form, rng.choice(targets(form, (a, b))), a, b)
            if near:
                slot = rng.choice([i for i, s in enumerate(other) if s not in (a, b)])
                same_class = vowels if other[slot] in VOWELS else consonants
                other = (
                    other[:slot]
                    + rng.choice([s for s in same_class if s not in (a, b, other[slot])])
                    + other[slot + 1 :]
                )
            shown += [form, other]
        shown += rng.sample(words, SHOWN - len(shown))
        if len(set(shown)) != SHOWN:
            continue
        rng.shuffle(shown)
        blank_forms = []
        for form in rng.sample(singles, 200):
            slot = targets(form, (a, b))[0]
            blank_form = form[:slot] + BLANK + form[slot + 1 :]
            if (
                filled(blank_form, a) not in shown
                and filled(blank_form, b) not in shown
                and blank_form not in blank_forms
            ):
                blank_forms.append(blank_form)
            if len(blank_forms) == BLANKS:
                break
        key = answer_key(rng, spec, shown, blank_forms, Distribution.CONTRASTIVE, None, [])
        if len(blank_forms) == BLANKS and uniqueness_problem(key) is None:
            return key
    raise RuntimeError(f"no contrastive dataset for {a} and {b} in {ATTEMPTS} attempts")


def datasets(rng: random.Random) -> list[dict]:
    keys = [complementary(rng, spec) for spec in COMPLEMENTARY] + [contrastive(rng, spec) for spec in CONTRASTIVE]
    rng.shuffle(keys)  # an id says nothing of the answer
    for n, key in enumerate(keys):
        key.dataset_id = f"lang-{n + 1:02d}"
    return [plain_output(key) for key in sorted(keys, key=lambda key: key.dataset_id)]


def dumps(entries: list[dict]) -> str:
    return json.dumps(entries, ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    DATASETS.write_text(dumps(datasets(random.Random(SEED))), encoding="utf-8", newline="\n")
    print(f"wrote {DATASETS}")


if __name__ == "__main__":
    main()
