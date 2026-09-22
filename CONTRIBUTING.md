# Contributing

## 1. One-time setup

```bash
git clone git@github.com:vks-g/cmapss-rul-hybrid.git
cd cmapss-rul-hybrid

# Use the email that is verified on YOUR GitHub account,
# otherwise your commits won't count in the contributors list.
git config user.name  "<your-github-username>"
git config user.email "<email-verified-on-your-github>"

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `import rul` work from notebooks and scripts
```

## 2. Branches

| Branch | Purpose | Who merges into it |
|---|---|---|
| `main` | Stable milestones only. Tagged `v1.0-phase1`, `v2.0-phase2`. | M1, from `dev` |
| `dev` | Integration branch. **All PRs target `dev`.** | M1, after review |
| `<type>/<handle>/<topic>` | One short-lived branch per task | You |

Branch types: `feat`, `fix`, `docs`, `exp` (experiment), `test`.
Examples: `feat/m3/rolling-features`, `docs/m5/setup-guide`, `exp/m2/lstm-window-length`.

## 3. Daily workflow

```bash
git switch dev && git pull
git switch -c feat/<handle>/<topic>

# ...work, commit small and often...
git add <files>
git commit -m "feat: add rolling slope features"

git push -u origin feat/<handle>/<topic>
# Open a PR on GitHub: base = dev, link the issue ("Closes #12")
```

Keep your branch current with `git pull origin dev` before opening the PR. Delete the branch after it is merged.

## 4. Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `<type>: <what changed>`

| Type | Use for |
|---|---|
| `feat` | new functionality |
| `fix` | bug fix |
| `exp` | experiment / notebook results |
| `docs` | README, report, comments |
| `test` | tests |
| `refactor` | restructuring without behaviour change |
| `chore` | config, dependencies, housekeeping |

Each commit should do one thing. "wip", "update", and "changes" are not commit messages.

## 5. Pull requests

- Base branch is `dev`. Never push directly to `dev` or `main`.
- Every PR needs **one approving review**. M1 reviews all PRs, and a second reviewer rotates among members.
- Merge with **"Create a merge commit"**. Do **not** squash: squashing collapses your commits into one and hides who did what.
- Every PR must link its issue. Issues are grouped under the `Phase 1` and `Phase 2` milestones.

## 6. Ownership rules

- Every notebook, `src/` module and report section has one owner, assigned through its GitHub Issue.
- **Never edit a notebook you don't own.** If you need a change in shared logic, change the `src/` module in a PR.
- Notebooks stay thin: reusable logic goes in `src/rul/`, and notebooks import it.
- Before committing a notebook, run **Restart & Run All** so its outputs match its code.
- Each report section is its own file in `reports/phaseX/sections/`, written and committed by its owner.

## 7. Code style

- PEP 8, type hints on public functions, a short docstring on every public function.
- No hard-coded absolute paths. Use paths relative to the repo root.
- Set random seeds for anything stochastic (splits, models, training).

## 8. Technical guardrails (non-negotiable for a fair comparison)

- **Split by engine (`unit`) only**, e.g. with `GroupKFold`. Never split by row.
- Fit scalers, regime clusters and flat-sensor selection on **training folds only**.
- Build sequence windows **inside engine boundaries**. A window never spans two engines.
- Training RUL = (final cycle of that engine) − (current cycle).
- Test engines are scored on their **last cycle** against `RUL_FDxxx.txt`.
- Uncapped RUL is the primary target. A capped target (e.g. 125 cycles) is reported only as an ablation, and is stated explicitly.
- Report RMSE, MAE and NASA score, plus error by true-RUL stage (early / mid / late), using `rul.evaluation.metrics` for every model.
