"""get_dataset's checks: a known id asked, and the dataset it returns is well formed and is the one asked for."""

from .checks import dataset_id_problem, dataset_problem, result


def verify_input(value):
    problem = dataset_id_problem(value.dataset_id)
    return result(False, problem) if problem else result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.dataset_id != input_value.dataset_id:
        return result(False, f"asked for {input_value.dataset_id}, got {output_value.dataset_id}")
    problem = dataset_problem(output_value)
    if problem:
        return result(False, problem)
    return result(
        True, f"{output_value.dataset_id}: {len(output_value.words)} words, {len(output_value.blanks)} blanks"
    )
