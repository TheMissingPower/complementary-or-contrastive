"""pin_dataset's checks: the pinned answer key agrees with a fresh analysis of its own words."""

from contract import Distribution

from . import dataset
from .checks import dataset_problem, result
from .rules import BLANK, consistent_predictions, filled, fitting_rules, minimal_pairs, realize

verify_input = dataset.verify_input


def verify_output(input_value, output_value, workspace):
    key = output_value
    if key.dataset_id != input_value.dataset_id:
        return result(False, f"the answer key is for {key.dataset_id}, not {input_value.dataset_id}")
    problem = dataset_problem(key)
    if problem:
        return result(False, problem)
    a, b = key.target_sounds
    forms = [word.form for word in key.words]
    pairs = {frozenset(pair) for pair in minimal_pairs(forms, a, b)}
    if {frozenset((pair.first, pair.second)) for pair in key.minimal_pairs} != pairs:
        return result(False, "the answer key's minimal pairs are not the shown words' minimal pairs")
    if key.distribution is Distribution.CONTRASTIVE:
        if key.rule is not None or key.answers:
            return result(False, "a contrastive answer key has a rule or blank answers")
        if len(pairs) < 2:
            return result(False, f"a contrastive dataset needs at least 2 minimal pairs, it has {len(pairs)}")
        if fitting_rules(forms, a, b):
            return result(False, "a rule fits the shown words of a contrastive dataset")
        return result(True, f"contrastive, with {len(pairs)} minimal pairs and no rule that fits")
    rule = key.rule
    if rule is None or {rule.special, rule.elsewhere} != {a, b}:
        return result(False, "a complementary answer key needs a rule between the two target sounds")
    for form in forms:
        if realize(rule, form.replace(rule.special, rule.elsewhere)) != form:
            return result(False, f"the rule does not give the shown word {form!r}")
    answers = {answer.blank_id: answer.sound for answer in key.answers}
    if len(answers) != len(key.answers) or set(answers) != {blank.blank_id for blank in key.blanks}:
        return result(False, "the answers do not cover every blank exactly once")
    for blank in key.blanks:
        expected = realize(rule, filled(blank.form, rule.elsewhere))[blank.form.index(BLANK)]
        if answers[blank.blank_id] != expected:
            return result(
                False, f"blank {blank.blank_id}: the answer is {answers[blank.blank_id]}, the rule gives {expected}"
            )
        if consistent_predictions(forms, a, b, blank.form) != {expected}:
            return result(False, f"blank {blank.blank_id}: the shown words do not decide it")
    return result(True, f"complementary: {rule.description}", rule=rule.description)
