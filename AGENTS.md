# Rules for agents working in this repository

You may be a large model or a very small one. These rules are written so a
small model can follow them mechanically: each rule is short, each has a
check, and every pattern names the file that shows it done right. If you
cannot tell whether a change breaks a rule, **stop and ask** - a question
in the PR is free, a wrong merge is not.

The project: many cheap devices, one mind. A fleet of heterogeneous
personal machines runs a swarm that votes, proposes, replicates its own
code and models, and improves itself. Everything below protects one goal:
**the fleet must keep running while the code keeps changing.**

## The five invariants

1. **Catalogue in code, wiring in config.** What *can* run is a table in
   the source (`pando/backend.py` registry, `pando/deploy.py` SERVICES,
   `pando/execute/main.py` _STAGERS, `tools/code_sync.py` CLIENT_UNITS).
   What *this host* runs lives in its untracked `host.toml` and
   `hosts.json`. Never merge the two directions.
2. **The repository names no machine.** No hostnames, IP addresses,
   usernames, absolute user paths, member names, or public keys of a real
   deployment in tracked files. Test fixtures use `node-a`, `/host/a/`,
   `10.1.0.x`. Historical incident records under `docs/archive/` keep their
   real names - they are evidence, not configuration.
3. **Fail closed, fail visible.** A gate that cannot verify refuses. A
   failure is recorded in a ledger row or a log line, never swallowed.
   `except Exception: pass` is a bug. So is a default that guesses.
4. **The ledger is the memory.** An action that leaves no row did not
   happen. One failing action must not kill the loop (`record the reason,
   continue`), but the reason must be in the row.
5. **Honest names.** A function says what it proves. `check_integrity`
   detects tampering; it does not verify authenticity, and it is not named
   as if it does. If you weaken a guarantee, rename the function.

## Hard rules - do not

- **No hardcoded machine facts.** Check: `git diff | grep -E
  "192\.168\.|100\.|/Users/|/home/[a-z]"` must be empty outside
  `docs/archive/`.
- **No silent fallback.** If a config value is missing, refuse with a
  message that names the missing key. Do not default to a value that only
  one machine ever had.
- **No if/elif dispatch chains on type, backend, or platform.** Two
  branches are fine; three means a table. Look at `BACKEND_BUILDERS` in
  `pando/worker.py` before writing the third branch.
- **No new dependency without a stated reason in the PR.** The library
  depends on PyNaCl only; configuration is stdlib TOML.
- **No process-global mutation.** No `urllib.request.install_opener`, no
  `os.environ` writes, no monkey-patching outside tests. HTTP goes through
  `pando.net.open_url` so proxy policy has exactly one home.
- **No test that cannot fail.** A test that only prints, or asserts
  `True`, or mocks the thing it claims to test, is deleted on sight.
- **Do not rewrite history records.** Incident dates, measured numbers,
  and member names in `docs/archive/`, docstring incident citations, and
  `tools/reproduce_headlines.py` are data. Changing them is falsification.
- **Do not import `tools/` from `pando/`** except through the existing
  importlib seam in `pando/execute/reconcile.py`.

## Required patterns - do

- **Dispatch by table.** Adding a backend: register a factory in
  `pando/backend.py` + one row in `BACKEND_BUILDERS`. Adding an executor
  action: one row in `_STAGERS`. Adding a service: one row in
  `deploy.SERVICES`. If your change does not fit the table, the table may
  be wrong - say so in the PR instead of working around it.
- **Config layering.** Adjacency (mailboxes, roots) → `hosts.json`.
  Identity and trust (pins, paths, deployment wiring) → `host.toml`, read
  via `pando/hostconfig.py`. One-shot launch overrides → environment
  variables, `PANDO_*`. Persistent settings must not be env vars.
- **Errors carry their type.** `raise PolicyError(f"...")`, not
  `raise Exception`. Ledger rows name the failure class
  (`record["failed"] = f"{type(exc).__name__}: {exc}"`).
- **Defensive checks at trust boundaries.** Input that crosses a process,
  network, or signature boundary gets validated at the boundary
  (`pando/spore.py` `_resolve_in` is the shape). Interior calls stay bare.
- **One source of truth per fact.** Version: `pyproject.toml`, read via
  `importlib.metadata`. Proxy rules: `pando/net.py`. If you find yourself
  copying a constant, stop.

## Tests

- Run before every commit: `python -m unittest discover -s tests`. All of
  it. "The unrelated test was already red" requires a link to the run that
  proved it.
- Write tests for invariants and contracts (the envelope shape, the
  ledger row, the renderer output), not for implementation details.
- One honest end-to-end test beats ten mock-heavy unit tests. The smoke
  suite boots real coordinators and stems on loopback; prefer joining that
  pattern.
- Temporary files use `tempfile.TemporaryDirectory` or
  `addCleanup(shutil.rmtree, ...)`. A test that leaks state breaks the
  next test.
- Deleting a low-value test is a good change. Say what it claimed to
  protect and why nothing is lost.

## Growth rules - when to stop and propose, not pile in

These thresholds exist so a small model never has to judge "is this file
too big" on taste:

- A `pando/` module past **~1500 lines**: propose a split before adding
  to it. (`coordinator.py` is over budget and grandfathered; do not use it
  as precedent.)
- A domain with **4+ modules** in the flat `pando/` directory earns a
  subpackage, the way `pando/execute/` grew. Below 4, stay flat.
- A single function past **~150 lines**: split it or table-drive it
  (`run_once` in `pando/execute/main.py` is the before/after).
- `tools/` scripts are research artifacts and may stay numerous, but a
  new tool must run standalone and say what it measures in its docstring.

## Documentation

- `docs/design/` is stateless: it describes what is true now, with no
  version labels. History goes in `docs/archive/` with a date.
- Numbers in docs come from a named run. Unverifiable claims are removed.
- README stays short; detail moves to `docs/`.

## When to stop and ask

Stop and ask the operator before:

- changing anything under `docs/archive/`;
- deleting a module, a ledger field, or a config key (migrations are
  proposed, not executed silently);
- adding a network endpoint, a grant action, or a signature verification
  path;
- weakening a refusal into a fallback, for any reason;
- a change whose blast radius you cannot enumerate (which files, which
  hosts, which running processes).

The answer to "is this over-engineered?" is: would a new machine join the
fleet with *fewer* lines of config because of this abstraction? If not,
do not build it.
