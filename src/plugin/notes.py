"""The save_notes tool: the agent's working notes, one entry per step, kept in the workspace between steps."""

from __future__ import annotations

import json
from pathlib import Path

from contract import NotesInput, NotesOutput

NOTES = "notes.json"


def save_notes(given: NotesInput, workspace: Path) -> NotesOutput:
    if not given.notes.strip():
        raise ValueError("notes must not be empty")
    path = workspace / NOTES
    notes = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    notes.append(given.notes)
    path.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    return NotesOutput(path=NOTES, notes=notes)
