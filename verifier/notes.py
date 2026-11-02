"""save_notes's checks: non-empty notes, kept in notes.json, the call's own notes last."""

import json
from pathlib import Path

from .checks import result

NOTES = "notes.json"


def verify_input(value):
    if not value.notes.strip():
        return result(False, "notes must not be empty")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if output_value.path != NOTES:
        return result(False, f"the notes are saved to {output_value.path!r}, not {NOTES}")
    if not output_value.notes or output_value.notes[-1] != input_value.notes:
        return result(False, "the returned notes do not end with this call's notes")
    saved = json.loads((Path(workspace) / NOTES).read_text(encoding="utf-8"))
    if saved != output_value.notes:
        return result(False, f"{NOTES} does not hold the returned notes")
    return result(True, f"{len(output_value.notes)} notes saved")
