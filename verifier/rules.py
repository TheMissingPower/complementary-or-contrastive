"""The rule space: syllables, environments and which rules the shown words allow. Pure functions of their arguments.

A rule says "the sound `special` is found in an environment, `elsewhere` everywhere else". A set environment
(BEFORE, AFTER, BEFORE_WORD_FINAL, BEFORE_SYLLABLE_FINAL) holds when a target's context, a neighboring symbol,
is in the rule's set S; the other environments are positions. Every context is a function of the rest of the
word, so a minimal pair rules out every rule.
"""

from itertools import combinations

from contract import Environment

VOWELS = frozenset("aiueo")
BLANK = "_"
BOUNDARY = "#"
SET_ENVIRONMENTS = (
    Environment.BEFORE,
    Environment.AFTER,
    Environment.BEFORE_WORD_FINAL,
    Environment.BEFORE_SYLLABLE_FINAL,
)


def syllable_final(form):
    """Whether each position ends its syllable: V.CV, VC.CV, and word edges are syllable edges."""
    vowel = [symbol in VOWELS for symbol in form]
    last = len(form) - 1
    final = []
    for i in range(len(form)):
        if i == last:
            final.append(True)
        elif vowel[i]:
            # A vowel ends its syllable unless a coda follows it: a consonant before a consonant or the word's end.
            final.append(vowel[i + 1] or (i + 2 <= last and vowel[i + 2]))
        else:
            final.append(not vowel[i + 1])
    return final


def context(environment, form, i):
    """Position i's context: a neighboring symbol, BOUNDARY or None for a set environment, else a bool."""
    last = len(form) - 1
    if environment is Environment.BEFORE:
        return form[i + 1] if i < last else BOUNDARY
    if environment is Environment.AFTER:
        return form[i - 1] if i > 0 else BOUNDARY
    if environment is Environment.BEFORE_WORD_FINAL:
        return form[i + 1] if i + 1 == last else None
    if environment is Environment.BEFORE_SYLLABLE_FINAL:
        return form[i + 1] if i < last and syllable_final(form)[i + 1] else None
    if environment is Environment.WORD_INITIAL:
        return i == 0
    if environment is Environment.WORD_FINAL:
        return i == last
    final = syllable_final(form)
    end = next(j for j in range(i, len(form)) if final[j])
    return form[end] not in VOWELS


def applies(environment, sounds, form, i):
    found = context(environment, form, i)
    return found in sounds if environment in SET_ENVIRONMENTS else found


def realize(rule, underlying):
    """The surface form of an underlying form: each `elsewhere` in the rule's environment becomes `special`."""
    return "".join(
        rule.special if symbol == rule.elsewhere and applies(rule.environment, rule.sounds, underlying, i) else symbol
        for i, symbol in enumerate(underlying)
    )


def targets(form, sounds):
    return [i for i, symbol in enumerate(form) if symbol in sounds]


def filled(blank_form, sound):
    return blank_form.replace(BLANK, sound)


def minimal_pairs(forms, a, b):
    """Every pair of forms that differ in exactly one position, where one has a and the other b."""
    pairs = []
    for first, second in combinations(forms, 2):
        if len(first) != len(second):
            continue
        differ = [(x, y) for x, y in zip(first, second) if x != y]
        if len(differ) == 1 and set(differ[0]) == {a, b}:
            pairs.append((first, second))
    return pairs


def occurrences(forms, a, b):
    """(word index, form, position) of every a or b in the forms."""
    return [(n, form, i) for n, form in enumerate(forms) for i in targets(form, (a, b))]


def fits(environment, special, elsewhere, forms):
    """Whether some rule of this environment and direction fits every form; for a set environment, the contexts
    S must hold (those of special) and must not (those of elsewhere), else None."""
    found = occurrences(forms, special, elsewhere)
    if environment not in SET_ENVIRONMENTS:
        return all(context(environment, form, i) == (form[i] == special) for _, form, i in found) or None
    inside = {context(environment, form, i) for _, form, i in found if form[i] == special}
    outside = {context(environment, form, i) for _, form, i in found if form[i] == elsewhere}
    if None in inside or BOUNDARY in inside or inside & outside:
        return None
    return inside, outside


def blank_prediction(environment, special, elsewhere, forms, blank_form):
    """The sounds the rules of one environment and direction that fit every form give the blank.

    Exact over every set S: None when no rule fits, else {special}, {elsewhere} or both.
    """
    fit = fits(environment, special, elsewhere, forms)
    if fit is None:
        return None
    slot = blank_form.index(BLANK)
    key = context(environment, filled(blank_form, elsewhere), slot)
    if environment not in SET_ENVIRONMENTS:
        return {special if key else elsewhere}
    inside, outside = fit
    if key is None or key == BOUNDARY or key in outside:
        return {elsewhere}
    if key in inside:
        return {special}
    return {special, elsewhere}


def consistent_predictions(forms, a, b, blank_form):
    """Every sound some rule fitting all the forms gives the blank, over every environment, direction and S."""
    predicted = set()
    for environment in Environment:
        for special, elsewhere in ((a, b), (b, a)):
            predicted |= blank_prediction(environment, special, elsewhere, forms, blank_form) or set()
    return predicted


def fitting_rules(forms, a, b):
    """(environment, special, elsewhere) of every family with some rule that fits all the forms."""
    return [
        (environment, special, elsewhere)
        for environment in Environment
        for special, elsewhere in ((a, b), (b, a))
        if fits(environment, special, elsewhere, forms) is not None
    ]
