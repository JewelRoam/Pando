# DESIGN — the self-repair actuator (the one we do not have yet)

*2026-10-19, mac-codex. Written before any code, on purpose: this is the most
dangerous actuator in the set, and `CIVILIZATION-NOTES.md` says the boundary has
to be argued first.*

## The gap, stated as three incidents

All three were verified live on 2026-10-19, not recalled.

1. **Windows ran four commits behind and nothing said so.** The only way to
   learn its commit was to ask it; the moment no cloud operator is left on that
   box, "ask it" stops existing. `10f5043` added `code_head` to every member's
   capability so any peer can read it off `/nodes` — and the fix itself is
   useless on the machine that needed it, because that machine predates the
   field. Today all three Windows members still report **no `code_head` at all**.

2. **A stale reactor produced 36 hours of votes about a fixed point in the
   past.** `win-laya-int8` appended a reaction every two minutes; every target
   it named was stamped ~36 hours earlier. Root cause: that box predates
   `1542026`, the `--max-age` bound that stops a reactor from answering
   yesterday forever. The votes were cast, claimed, and worthless.

3. **The Mac has the same disease, just milder.** `mac-text` (`:8766`, started
   Oct 18 16:41) also reports no `code_head`. It is *healthy* — its answer-lag
   slope is −0.22, so it is keeping up. This is the counter-example that keeps
   us honest: **old code is not the same thing as broken code.** Windows is
   sick because it is old *specifically on the far side of one fix*, not
   because it is old.

So the colony can propose, vote, and judge. When the thing that is wrong is
**its own bytes**, it is helpless: `pando/execute.py`'s `ACTUATORS` is a closed
set of eight, and not one of them touches code or a process table.

## The line this design draws

The actuator may **bring a host onto a commit that already exists in the
sanctioned history**. It may not **produce** a commit.

That single sentence separates this from "the civilization rewrites itself".

| It may | It must never |
|---|---|
| `fetch` from the configured remote, then the peer bundle | resolve a merge, rebase, or conflict — a dirty or diverged tree is left exactly as it is |
| fast-forward, and only fast-forward | rewrite history, force, or discard local edits |
| restart the client processes its own supervisor owns | kill a process it does not supervise |
| restart the whole stack when the supervisor roster itself changed | reload a model when only client bytes moved |
| record the ledger row **before** the restart it triggers | take a diff, a patch, a module name, or a command from a proposal body |

The last row of the "never" column is the load-bearing one. `meta.propose`
already refuses `code` / `source` / `module` / `path` / `command` in a payload
for exactly this reason. Nothing here should reopen that door from the side.

## The self-reference trap

The process that would apply the fix is **inside the process table it has to
restart**. `tools/node_code_watch.py` is already written for this: it fires the
restart command and does not wait, because waiting means never writing the line
that explains the restart. On Windows the same trap appears one level up — the
watcher is *itself* a service the supervisor starts, so a supervisor that has
never been told about `code-watch` will never run it. That is why the Windows
bootstrap needs one human paste and why it is a **one-time** cost rather than a
recurring one: after that paste, the watcher is in the roster, and the roster is
what re-reads the roster.

The Mac has no watcher at all. That is fine today — a human shell lives here —
and it is exactly the assumption that stops being true the day it doesn't.

## Guards (all three are preconditions, not options)

1. **Experiment lock.** A restart during a measured run corrupts the run. If a
   transient measurement is in flight, the actuator records `deferred` and does
   nothing. (This is the reason the Mac's stale members were *not* restarted on
   2026-10-19 even though the shell was right there.)
2. **Ledger before action.** Row first, restart second — see the trap above.
3. **Cooldown.** One fast-forward, one restart. A supervisor that restarts itself
   in a loop looks identical to a healthy one from the outside.

## What would falsify this design

- After a Windows fast-forward, if `code_head` still does not appear in `/nodes`,
  the field is not carried by the process that registers, and the diagnosis
  "it runs old bytes" is wrong.
- If a deferred-with-lock run ever loses a vote that a non-deferred run would
  have cast, the guard costs more than it saves.
- If a host fast-forwards and its answer-lag slope does **not** fall back under
  `0.20`, the wedge was never about code age.

## Deliberately out of scope

Applying a diff authored by a proposal; hot-swapping a module under a live
process; letting `meta.propose` land code. Each of those makes the diff itself
the input, and this design's whole safety argument is that the input is a commit
hash that humans already reviewed upstream.
