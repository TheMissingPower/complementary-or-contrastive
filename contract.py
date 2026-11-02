"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    GET_DATASET = "get_dataset"
    PIN_DATASET = "pin_dataset"
    SAVE_NOTES = "save_notes"


class Task(Enum):
    CLASSIFY_DISTRIBUTION = "classify-distribution"


class Distribution(Enum):
    COMPLEMENTARY = "complementary"
    CONTRASTIVE = "contrastive"


class Environment(Enum):
    BEFORE = "before"
    AFTER = "after"
    WORD_INITIAL = "word-initial"
    WORD_FINAL = "word-final"
    BEFORE_WORD_FINAL = "before-word-final"
    BEFORE_SYLLABLE_FINAL = "before-syllable-final"
    CLOSED_SYLLABLE = "closed-syllable"


DATASET_ID = "The dataset's id, lang-01 to lang-20, e.g. lang-07"
TARGET_SOUNDS = 'The two target sounds A and B, one symbol each, e.g. ["t", "ʔ"]'
FORM = "A word, one symbol per sound, e.g. tapuʔ"
BLANK_ID = "The blank's id, as get_dataset gives it, e.g. b3"
SYLLABLES = "The form split into syllables with dots by the task's convention (V.CV, VC.CV), e.g. de.mob.ri"


@dataclass
class DatasetInput(Output):
    dataset_id: Annotated[str, DATASET_ID]


@dataclass
class Word(Output):
    form: Annotated[str, FORM]
    gloss: Annotated[str, "The word's English meaning"]


@dataclass
class Blank(Output):
    blank_id: Annotated[str, BLANK_ID]
    form: Annotated[str, "A held-out word with one _ where A or B belongs, e.g. ta_u"]
    gloss: Annotated[str, "The word's English meaning"]


@dataclass
class ShownWord(Output):
    form: Annotated[str, FORM]
    syllables: Annotated[str, SYLLABLES]
    gloss: Annotated[str, "The word's English meaning"]


@dataclass
class ShownBlank(Output):
    blank_id: Annotated[str, BLANK_ID]
    form: Annotated[str, "A held-out word with one _ where A or B belongs, e.g. ta_u"]
    syllables: Annotated[str, SYLLABLES + "; the _ counts as A or B"]
    gloss: Annotated[str, "The word's English meaning"]


@dataclass
class DatasetOutput(Output):
    dataset_id: Annotated[str, DATASET_ID]
    target_sounds: Annotated[list[str], TARGET_SOUNDS]
    words: Annotated[list[ShownWord], "The shown words of the language"]
    blanks: Annotated[list[ShownBlank], "The held-out words, each with one blank for A or B"]


@dataclass
class NotesInput(Output):
    notes: Annotated[str, "This step's findings, short lines, e.g. the environments of one target sound"]


@dataclass
class NotesOutput(Output):
    path: Annotated[str, "The notes file, relative to the workspace: notes.json"]
    notes: Annotated[list[str], "Every step's notes so far, in order"]


@dataclass
class MinimalPair(Output):
    first: Annotated[str, "A shown word"]
    second: Annotated[str, "Another shown word, the same as first except A for B in one position"]


@dataclass
class Prediction(Output):
    blank_id: Annotated[str, BLANK_ID]
    sound: Annotated[str, "The target sound that fills the blank: A or B"]


@dataclass
class DistributionOutput(Output):
    distribution: Annotated[Distribution, "complementary or contrastive: how A and B are distributed"]
    minimal_pairs: Annotated[
        list[MinimalPair], "Contrastive: minimal pairs from the shown words, at least one; complementary: []"
    ]
    predictions: Annotated[list[Prediction], "Complementary: the sound for every blank, once each; contrastive: []"]


@dataclass
class Rule(Output):
    special: Annotated[str, "The sound found in the environment"]
    elsewhere: Annotated[str, "The sound found everywhere else, the underlying sound"]
    environment: Annotated[Environment, "The kind of environment"]
    sounds: Annotated[list[str], "The environment's set of neighboring sounds S; [] for a positional environment"]
    description: Annotated[str, "The rule in words"]


@dataclass
class AnswerKeyOutput(Output):
    dataset_id: Annotated[str, DATASET_ID]
    target_sounds: Annotated[list[str], TARGET_SOUNDS]
    words: Annotated[list[Word], "The shown words of the language"]
    blanks: Annotated[list[Blank], "The held-out words, each with one blank for A or B"]
    distribution: Annotated[Distribution, "How A and B are distributed"]
    minimal_pairs: Annotated[list[MinimalPair], "Every minimal pair for A and B among the shown words"]
    rule: Annotated[Rule | None, "The hidden rule of a complementary dataset; null when contrastive"]
    answers: Annotated[list[Prediction], "The sound for every blank of a complementary dataset; [] when contrastive"]


TOOLS = {
    Tool.GET_DATASET: (DatasetInput, DatasetOutput),
    Tool.PIN_DATASET: (DatasetInput, AnswerKeyOutput),
    Tool.SAVE_NOTES: (NotesInput, NotesOutput),
}
TASKS = {
    Task.CLASSIFY_DISTRIBUTION: (DatasetInput, DistributionOutput),
}
