# Contributing

This is a research repository. The bar for a change is not "it runs" - it is
"the claim is traceable".

## Ground rules

1. **Prediction first.** Write the expected number down (in the task, the
   PR, or the research file) before running the experiment.
2. **Replication discipline.** One run is one draw. Anything that will be
   quoted in a document needs either a second run or an explicit caveat.
3. **No inferred values.** Every number in `docs/` must come from a run, a
   recorded-votes file, or a named tool. If it cannot be traced, it does not
   go in.
4. **Smoke over suites.** Tests are for invariants: `python -m unittest
   discover -s tests` (29 tests, stdlib-only) plus
   `python -m py_compile pando/*.py tools/*.py`. Prefer one meaningful
   end-to-end check over a pile of unit tests.

## Workflow

- Fork or branch (`feat/`/`codex/` prefixes are fine); keep PRs small and
  single-idea.
- Wire changes: update `docs/design/PROTOCOL-stem.md` in the same PR.
- Research work: add `docs/archive/research/YYYY-MM-DD-roundNNN.md` and regenerate
  `docs/archive/research/INDEX.md` via `python tools/build_research_index.py`.
- CI (`.github/workflows/ci.yml`) runs static analysis, the smoke suite,
  the demo, and distribution verification on every push - it must be green.

## Reporting results

State configuration, split, sample size, and the exact tool used. Recorded
votes belong under `assets/votes/` with provenance in its README; the
offline reproduction lives in `tools/reproduce_headlines.py`.
