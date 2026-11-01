# MReview

A CLI for reviewing model design and training code. It splits target code into units by code location, applies a fixed questionnaire, collects ledger evidence, and performs coverage verification. It also integrates deterministic tools such as `ruff` and `torchfix`, plus weight-file checks. Delivered as a pure CLI that any agent harness can invoke through the shell.

## Review Logic

1. **Location-based splitting**: AST-based splitting of target code into units (functions/methods, class bodies, module-level flows); every line belongs to exactly one unit; units longer than 150 lines are split at syntax seams.
2. **Fixed questionnaire**: each unit is answered against the same 10-question set, and every answer must cite evidence within the unit's line range.
3. **Ledger verification**: `verify` re-parses the source to check that (1) all 10 questions are covered per unit, (2) evidence line numbers fall within the unit's range, and (3) "ok" verdicts also carry line numbers. Gaps only mark that unit as "unverified" and list the missing `(unit, question)` pairs, without invalidating the whole ledger.
4. **External tool merge**: `ruff`/`torchfix` output is merged into findings automatically (deduplicated by file:line); missing tools are skipped and noted in the report.

## Review Modes

- **Static (default)**: no actual execution — runs external tools and read-only numeric checks on weight files.
- **Dynamic (`--dynamic`)**: actually loads the model (import + instantiate + load weights + one forward pass), checking output finiteness and magnitude; on failure it degrades to "unverified (reason)".

## Installation

```bash
# in the target Python environment (e.g. diffusion)
pip install -e .
```

No isolated venv, no uv; `doctor` only confirms that the review tools exist.

## Usage

```bash
mreview doctor                    # environment self-check: confirm torch/ruff/torchfix etc. exist
mreview split <target>            # split units, generate worksheet.json (units x 10-question ledger template)
mreview review <target> [--dynamic]  # fully automatic: split + external tools + weight checks
mreview verify <ledger.json> <target> # validate an agent-filled ledger, output merged report
```

Exit codes: `0` pass / `1` environment or argument error / `2` unverified units exist.

`<target>` may be a directory (auto-discovers `*.py` and `*.pt/.pth/.ckpt/.safetensors`) or a single file.

## Version Scoping (`tools/version_scope.py`)

`mreview` reviews whole files or directories and does not filter code by runtime version strings. When training code dispatches behavior through config-driven version values (e.g. `if 'v10.' in self.params.version`), use `tools/version_scope.py` to find which lines actually execute for a given config:

```bash
python tools/version_scope.py <config.yaml|configs_dir> <code.py|code_dir> [--units units.json]
```

- Reads every string scalar from the YAML config(s) - key names do not matter - scans Python sources for `'X' in expr`, `'X' not in expr` and `expr == 'X'` conditions, and reports every condition that evaluates TRUE for a config value.
- `--units` maps hit lines to mreview units (pass the `units.json` produced by `mreview split`).
- Pass directories to batch over configs x code files; `--json out.json` writes a structured report.

## Artifacts & Traceability

Each run creates `.mreview-results/<UTC timestamp>/`, containing:

| File | Contents |
|---|---|
| `units.json` | unit list (id, type, line range) |
| `worksheet.json` | questionnaire + ledger template |
| `findings.json` | structured findings (severity/category/file/line/cause/fix/evidence) |
| `REPORT.md` | human-readable report |
| `ledger-gate.json` | verify coverage results and gaps |
| `tools/*.txt` | raw tool outputs |
| `weights/*.json` | weight-check details |
| `meta.json` | command, arguments, environment |

## 10-Question Questionnaire (shared by all code units)

Input/output contracts / numerical stability / gradients & backpropagation / patterns & state / device & data placement / resources & lifecycle / error handling / boundaries & indexing / consistency & conventions / performance.

## Weight Checks

- **Static**: loadability, state_dict key alignment with the model definition, dtype/shape, NaN/Inf counts and locations, min/max/mean/std, all-zero/dead zones.
- **Dynamic (`--dynamic`)**: instantiate the model, load weights, run one forward pass, and check output finiteness and magnitude.

Every finding includes a **cause** and a **suggested fix** (e.g. NaN traced to a too-large learning rate / AMP overflow, with remediation steps).

## Ledger Format (agent-filled)

```json
{
  "units": [
    {
      "unit_id": "bad_model.py:BadNet.forward",
      "answers": [
        { "question_id": "q1", "verdict": "ok", "evidence": "L30-L34" },
        { "question_id": "q2", "verdict": "issue",
          "evidence": "L33", "title": "...", "cause": "...", "fix": "...",
          "category": "correctness", "severity": "P2" }
      ]
    }
  ]
}
```

`verdict` values: `ok / issue / skip`; `ok` and `skip` must also carry line-range evidence.

## Development

```bash
pip install -e .[dev]
pytest tests/          # unit split / ledger verification / weight-check tests
ruff check mreview tests
```

`fixtures/` contains known-defect samples and a fake ledger, used to exercise recall and verify's ability to catch fabricated evidence.

## Directory Structure

```
mreview/          # package source (cli / splitter / tools / weights / ledger / report / doctor / questionnaire)
tools/            # standalone helpers (version_scope.py: map config version values to code lines/units)
tests/            # pytest
fixtures/         # known-defect samples (bad_model.py, bad_train.py, bad_weights.pt, fake_ledger.json)
```

## Known Limitations

- torchfix / hotpath etc. are skipped when not installed and noted in the report; this does not block a run.
- Dynamic-check input shapes are guessed heuristically from the model's first layer (Conv2d -> 64x64, Linear -> in_features); failures are honestly marked "unverified".
- Without network access, the existence of torchfix/hotpath-ai/torch-preflight cannot be verified; `doctor` reports this honestly.
