# Complementary or Contrastive

A `nanoagent.plugin/v0001` plugin, built with the structure of NanoAgent's reference plugin,
[arxiv-source](https://github.com/zeyuyang8/arxiv-source).

An agent gets a dataset from an invented language: words with English glosses, two
target sounds A and B, and 8 held-out words with a blank (`_`) where A or B belongs.
It must decide whether A and B are in **complementary** distribution (one occurs
in a predictable environment and the other everywhere else) or **contrastive**
distribution (they occur in the same environments and distinguish words). It must
also back up its answer with evidence that code can check:

- contrastive: minimal pairs from the shown words, two words that differ only in
  A vs B in one position;
- complementary: the sound that fills every blank.

A deterministic verifier, plain Python with no model, checks the answer. **Every
language is invented.** The words come from a seeded generator, and no
real-language data is used.

## Tools

| Tool | Does | Code |
|---|---|---|
| `get_dataset` | Gives a dataset by id (`lang-01` to `lang-20`): its target sounds, its 30 words with glosses, and its 8 blanks. Read-only; it analyzes nothing | `src/plugin/datasets.py` |
| `pin_dataset` | Pins a frozen dataset's answer key: the distribution, every minimal pair, the hidden rule and each blank's sound | `src/plugin/datasets.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `classify-distribution` | Decides complementary or contrastive for a dataset, with minimal pairs or blank fill-ins as evidence | `agent`, with `get_dataset`; reference `pin_dataset` |

Input: `{"dataset_id": str}`. Output:

```json
{
  "distribution": "complementary" | "contrastive",
  "minimal_pairs": [{"first": str, "second": str}],
  "predictions": [{"blank_id": str, "sound": str}]
}
```

`pin_dataset` is the task's reference tool. NanoAgent runs it before the agent
starts, so the agent never sees its output. The agent gets only `get_dataset`,
which returns the data and none of the answer key. Every dataset has blanks,
the contrastive ones too, so the shape of the data does not give the label
away. Ids are shuffled, so they do not give it away either.

The rules the agent is given:

- Every symbol in a word is exactly one sound. There are no multi-character sounds.
- Vowels are `a e i o u`. Every other symbol is a consonant.
- A single consonant between vowels begins the next syllable (V.CV). Two
  consonants between vowels are split (VC.CV). Word boundaries are also syllable
  boundaries.
- In a complementary dataset, the environment is one of these: before a sound from a set S; after
  a sound from a set S; word-initially; word-finally; before a word-final sound
  from S; before a syllable-final sound from S; in a closed syllable.

### Datasets

`src/plugin/data/datasets.json` holds 20 datasets, 12 complementary and 8
contrastive. Each has 30 shown words and 8 blanks. Every word is 2 to 4 CV or
CVC syllables.

| Id | Target sounds | Distribution | Hidden rule |
|---|---|---|---|
| lang-01 | e, a | complementary | /a/ is [e] before y or ʃ, and [a] elsewhere |
| lang-02 | e, i | contrastive | none: 2 minimal pairs, e.g. bekniraʃ / bekneraʃ |
| lang-03 | r, d | complementary | /d/ is [r] before a syllable-final a or i, and [d] elsewhere |
| lang-04 | ŋ, n | contrastive | none: 2 minimal pairs, e.g. nanone / nanoŋe |
| lang-05 | d, l | complementary | /l/ is [d] word-initially, and [l] elsewhere |
| lang-06 | ʃ, s | contrastive | none: 2 minimal pairs, e.g. warmoʃa / warmosa |
| lang-07 | ʃ, s | complementary | /s/ is [ʃ] before e or i, and [s] elsewhere |
| lang-08 | p, b | contrastive | none: 2 minimal pairs, e.g. ʔebubabi / ʔebubapi |
| lang-09 | t, d | contrastive | none: 2 minimal pairs, e.g. lugtuŋ / lugduŋ |
| lang-10 | e, i | complementary | /i/ is [e] in a closed syllable, and [i] elsewhere |
| lang-11 | o, u | complementary | /u/ is [o] after q or r, and [u] elsewhere |
| lang-12 | k, q | contrastive | none: 2 minimal pairs, e.g. gokupul / goqupul |
| lang-13 | u, o | complementary | /u/ is [o] in a closed syllable, and [u] elsewhere |
| lang-14 | r, l | contrastive | none: 2 minimal pairs, e.g. liyirʃil / liyirʃir |
| lang-15 | ʔ, t | complementary | /t/ is [ʔ] word-finally, and [t] elsewhere |
| lang-16 | i, e | complementary | /i/ is [e] before a word-final k or q, and [i] elsewhere |
| lang-17 | o, a | complementary | /a/ is [o] before a syllable-final n or ŋ, and [a] elsewhere |
| lang-18 | o, u | contrastive | none: 2 minimal pairs, e.g. nenuh / nenoh |
| lang-19 | k, q | complementary | /k/ is [q] after a or o, and [k] elsewhere |
| lang-20 | h, s | complementary | /s/ is [h] before a word-final a or u, and [s] elsewhere |

`src/plugin/generate_datasets.py` keeps a dataset only if it passes these
uniqueness checks. The tests check them again on the frozen file.

- **The rule space.** A rule says "B in environment E, A elsewhere", in either
  direction. E is one of the seven environment kinds above, and S can be any
  set of sounds. The checker covers every set S exactly, because a set
  environment fits the data only when the contexts of the two sounds do not
  overlap.
- **Complementary.** At least one rule fits all 30 shown words, and every rule
  that fits gives each blank the same sound. The shown words contain no minimal
  pair. Each blank is also one that a **near-miss** rule gets wrong. A near-miss
  rule fits all but 1 or 2 shown words: for example, "in a closed syllable"
  where the true rule is word-final, or "before k or q" where the true rule is
  "before a word-final k or q". So a blank tests the real generalization, not a
  nearby one.
- **Contrastive.** No rule fits the shown words, and there are at least 2 true
  minimal pairs. Each contrastive dataset also has 2 near-minimal pairs as
  distractors. These differ in A vs B *and* in one other sound.

### Verifier

`verify_task` (`verifier/distribution.py`) is all-or-nothing. It checks, in
order, and stops at the first failure with a reason:

1. NanoAgent has already built the output into `DistributionOutput`. A missing
   field, a wrong type, a label other than `complementary`/`contrastive`, or an
   extra key fails the task there, before the verifier runs.
2. The answer key is for the dataset asked about.
3. `distribution` equals the answer key's label.
4. Contrastive:
   - `minimal_pairs` has at least one pair.
   - Every pair is valid: both words are shown words, they have the same
     length, they differ in exactly one position, and the differing sounds are
     exactly A and B. One invalid pair fails the dataset.
   - `predictions` is empty.
5. Complementary:
   - `minimal_pairs` is empty.
   - Every prediction names a known blank and gives A or B.
   - No blank is predicted twice, and none is missing.
   - Every prediction matches the answer key.

`details` always carries `blank_results` (blank_id → correct), `blanks_correct`
and `blanks_total`, so partial accuracy can be reported even when the verdict
is a fail.

The tool checks:

- `get_dataset`: the id has the form `lang-NN`, and the returned dataset is the
  one asked for and is well formed (two target sounds, inventory symbols only,
  one `_` per blank).
- `pin_dataset`: the answer key is re-derived from its own words. The minimal
  pairs must be the shown words' minimal pairs. For a complementary dataset,
  the rule must produce every shown word, and every answer must be the one
  sound that every fitting rule gives. For a contrastive dataset, no rule may
  fit.

## Layout

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # environment variable NAMEs this plugin needs (none)
├── src/plugin/      # the MCP server: __main__.py, its IMPLEMENTATIONS, and the plugin's own modules
│   ├── datasets.py          # the frozen datasets, get_dataset and pin_dataset
│   ├── generate_datasets.py # writes data/datasets.json, once: languages, rules, uniqueness checks
│   └── data/datasets.json   # the 20 frozen datasets with their answer keys
├── tests/           # test_verifier.py (the verifier's cases) and test_plugin.py (data, uniqueness, tools)
├── ui/              # how the page draws the outputs: drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package; rules.py is the rule engine the generator shares
```

## Develop and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then,
from this folder:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

`uv.lock` pins every dependency, and the plugin runs with `uv run --frozen`. The
plugin and verifier need no network and no environment variables.

To regenerate the frozen datasets, run the command below. It uses a fixed seed
and writes the same file byte for byte, which a test checks:

```bash
uv run python -m plugin.generate_datasets
```

To use the plugin, publish it on RUSH Lab, open it, and ask its chat for a
dataset, for example "Classify the target sounds of lang-07". The result card
shows the verifier's verdict.

Publishing checks every plugin rule. Push the plugin to a public GitHub
repository and submit its URL on the RUSH Lab marketplace.
