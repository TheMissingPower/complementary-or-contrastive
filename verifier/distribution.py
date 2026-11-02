"""The classify-distribution task's check: the right label, backed by valid minimal pairs or every blank filled right.

All or nothing: one invalid pair or one wrong blank fails the dataset. details always gives each blank's
correctness, so partial accuracy can still be reported.
"""

from contract import Distribution

from .checks import result


def pair_problem(pair, shown, a, b):
    """What makes ``pair`` no minimal pair for a and b among the shown words, or None."""
    first, second = pair.first, pair.second
    for form in (first, second):
        if form not in shown:
            return f"{form!r} is not one of the shown words"
    if len(first) != len(second):
        return f"{first} and {second} differ in length"
    differ = [(x, y) for x, y in zip(first, second) if x != y]
    if len(differ) != 1:
        return f"{first} and {second} differ in {len(differ)} positions, not 1"
    if set(differ[0]) != {a, b}:
        return f"{first} and {second} differ in {differ[0][0]} and {differ[0][1]}, not {a} and {b}"
    return None


def blank_results(output, reference):
    """Each blank_id -> whether the output's one prediction for it is right (False when missing or repeated)."""
    answers = {answer.blank_id: answer.sound for answer in reference.answers}
    given = {}
    for prediction in output.predictions:
        given.setdefault(prediction.blank_id, []).append(prediction.sound)
    return {blank_id: given.get(blank_id) == [sound] for blank_id, sound in answers.items()}


def verify_task(arguments, output, reference, workspace):
    """``output``: a contract DistributionOutput; ``reference``: the dataset's pinned AnswerKeyOutput."""
    if reference.dataset_id != arguments.dataset_id:
        return result(False, f"the answer key is for {reference.dataset_id}, not {arguments.dataset_id}")
    a, b = reference.target_sounds
    blanks = blank_results(output, reference)
    tally = {
        "blank_results": blanks,
        "blanks_correct": sum(blanks.values()),
        "blanks_total": len(blanks),
    }

    def fail(message):
        return result(False, message, **tally)

    if output.distribution is not reference.distribution:
        return fail(f"{a} and {b} are in {reference.distribution.value} distribution, not {output.distribution.value}")
    if reference.distribution is Distribution.CONTRASTIVE:
        if not output.minimal_pairs:
            return fail("contrastive needs at least one minimal pair as evidence")
        shown = {word.form for word in reference.words}
        for pair in output.minimal_pairs:
            problem = pair_problem(pair, shown, a, b)
            if problem:
                return fail(f"invalid minimal pair: {problem}")
        if output.predictions:
            return fail("a contrastive answer must have no predictions")
        return result(True, f"contrastive, with {len(output.minimal_pairs)} valid minimal pairs", **tally)
    if output.minimal_pairs:
        return fail("a complementary answer must list no minimal pairs")
    ids = [prediction.blank_id for prediction in output.predictions]
    for prediction in output.predictions:
        if prediction.blank_id not in blanks:
            return fail(f"{prediction.blank_id!r} is not one of the blanks")
        if prediction.sound not in (a, b):
            return fail(f"blank {prediction.blank_id}: {prediction.sound!r} is neither {a} nor {b}")
    repeated = sorted({blank_id for blank_id in ids if ids.count(blank_id) > 1})
    if repeated:
        return fail(f"blanks predicted more than once: {', '.join(repeated)}")
    missing = [blank_id for blank_id in blanks if blank_id not in ids]
    if missing:
        return fail(f"blanks with no prediction: {', '.join(missing)}")
    wrong = [blank_id for blank_id, right in blanks.items() if not right]
    if wrong:
        return fail(f"{len(wrong)} of {len(blanks)} blanks wrong: {', '.join(wrong)}")
    return result(True, f"complementary, with all {len(blanks)} blanks right", **tally)
