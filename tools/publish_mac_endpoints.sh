#!/bin/bash
# Publish the Mac's *currently reachable* quick-tunnel URLs to the volatile
# `mac-endpoint` branch, so other machines never have to trust a cached URL.
#
# Why this exists as a mechanical script and not as an agent step:
# cloudflared quick tunnels get a fresh random hostname on every restart, and
# the agents that used to "remember" to republish are exactly the processes
# that die in the incidents this is meant to survive. On 2026-10-17 all five
# tunnels restarted at 04:21:37Z; `endpoint/mac.json` kept pointing at the
# 00:20Z hostnames, and win-codex reported the Mac as unreachable while every
# local service was perfectly healthy. The fix is to make URL publication a
# property of the host, not of anybody's memory.
#
# This script deliberately runs *outside* the dev checkout: it works in its own
# clone under $HOME/.pandosi, uses /usr/bin/python3, and never touches a
# TCC-protected folder. A launchd agent whose ProgramArguments point into
# ~/Downloads is denied by macOS ("Operation not permitted"); keeping the whole
# job inside ~/.pandosi is what makes it survive a reboot unattended.
#
# Usage:
#   tools/publish_mac_endpoints.sh            # publish if the URL set changed
#   tools/publish_mac_endpoints.sh --dry-run  # report only; touch nothing
#   tools/publish_mac_endpoints.sh --print    # print the URLs it reads
#
# Overridables: PANDOSI_HOME (~/.pandosi), PANDOSI_ENDPOINT_CLONE,
#               PANDOSI_ENDPOINT_BRANCH, PANDOSI_ENDPOINT_REMOTE,
#               PANDOSI_ENDPOINT_REMOTE_URL, PANDOSI_PY
#
# Exit codes: 0 = nothing to do / published / (dry-run) nothing changed
#            10 = dry-run found a change that would have been published
#             1 = could not collect a complete URL set (nothing published)
set -u

HOME_DIR="${PANDOSI_HOME:-$HOME/.pandosi}"
CLONE="${PANDOSI_ENDPOINT_CLONE:-$HOME_DIR/mac-endpoint-publish}"
BRANCH="${PANDOSI_ENDPOINT_BRANCH:-mac-endpoint}"
REMOTE="${PANDOSI_ENDPOINT_REMOTE:-origin}"
REMOTE_URL="${PANDOSI_ENDPOINT_REMOTE_URL:-git@github.com:JewelRoam/PandoSI.git}"
PY="${PANDOSI_PY:-/usr/bin/python3}"
LOG="$HOME_DIR/logs/publish-endpoints.log"

DRY_RUN=0
PRINT_ONLY=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  --print)   PRINT_ONLY=1 ;;
esac

mkdir -p "$HOME_DIR/logs"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

# Latest hostname cloudflared printed for a given tunnel log. cloudflared
# appends on restart, so the *last* match is the live one.
url_of() {
  grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$HOME_DIR/tunnel-$1.log" 2>/dev/null | tail -1
}

MB="$(url_of mailbox)"
RT="$(url_of root)"
L4="$(url_of 4b)"
T2="$(url_of 8bq6)"
L8="$(url_of 8b)"

if [[ "$PRINT_ONLY" == 1 ]]; then
  printf 'mailbox      %s\nisland_root  %s\nllm4b        %s\nllm_teacher2 %s\nllm8b        %s\n' \
    "$MB" "$RT" "$L4" "$T2" "$L8"
  exit 0
fi

# Refuse to publish an incomplete set: a partial push is worse than a stale one,
# because it turns "everyone is unreachable" into "some callers are unreachable".
for pair in "mailbox:$MB" "root:$RT" "llm4b:$L4" "teacher2:$T2"; do
  name="${pair%%:*}"; val="${pair#*:}"
  if [[ "$val" != https://* ]]; then
    log "abort: no URL for $name (mb=$MB rt=$RT 4b=$L4 t2=$T2 8b=$L8)"
    echo "publish_mac_endpoints: no URL for $name; nothing published" >&2
    exit 1
  fi
done

# Start time of the tunnel process says how fresh the URL set is. All five are
# started together by launchd, so the mailbox one speaks for the set.
STARTED=""
tpid="$(pgrep -f 'cloudflared.*8776' 2>/dev/null | head -1)"
if [[ -n "$tpid" ]]; then
  raw="$(LC_ALL=C ps -o lstart= -p "$tpid" 2>/dev/null | sed 's/  */ /g; s/^ //; s/ $//')"
  epoch="$(LC_ALL=C date -j -f "%a %b %d %T %Y" "$raw" "+%s" 2>/dev/null)"
  [[ -n "$epoch" ]] && STARTED="$(LC_ALL=C date -u -r "$epoch" "+%Y-%m-%dT%H:%M:%SZ")"
fi

# Dedicated shallow clone. It lives under ~/.pandosi (never ~/Downloads), and it
# is recreated if a reboot or cleanup wipes it.
if [[ ! -e "$CLONE/.git" ]]; then
  git clone -q --depth 1 --single-branch --branch "$BRANCH" "$REMOTE_URL" "$CLONE" 2>>"$LOG" \
    || { log "abort: clone failed ($REMOTE_URL#$BRANCH)"; exit 1; }
fi

cd "$CLONE" || exit 1
if [[ "$DRY_RUN" != 1 ]]; then
  git fetch -q --depth 1 "$REMOTE" "$BRANCH" 2>>"$LOG" || { log "abort: fetch failed"; exit 1; }
  git reset -q --hard FETCH_HEAD 2>>"$LOG" || { log "abort: reset failed"; exit 1; }
fi

PSI_MB="$MB" PSI_RT="$RT" PSI_L4="$L4" PSI_T2="$T2" PSI_L8="$L8" \
PSI_STARTED="$STARTED" PSI_NOW="$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
PSI_DRY="$DRY_RUN" "$PY" - <<'PY'
import json, os, pathlib, sys

p = pathlib.Path("endpoint/mac.json")
d = json.loads(p.read_text())
l8 = os.environ["PSI_L8"]
want = {
    "mailbox_url": os.environ["PSI_MB"],
    "island_root_url": os.environ["PSI_RT"],
    "llm4b_url": os.environ["PSI_L4"],
    "llm_teacher2_url": os.environ["PSI_T2"],
    "llm8b_url": l8,
    "llm8b_think_url": l8,
}
stale = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
if not stale:
    print("UNCHANGED")
    raise SystemExit(0)
for k, (old, new) in stale.items():
    print("stale %s: %s -> %s" % (k, old, new))
d.update(want)
d["updated"] = os.environ["PSI_NOW"]
if os.environ.get("PSI_STARTED"):
    d["tunnels_started_at"] = os.environ["PSI_STARTED"]
d["url_rotation_warning"] = (
    "quick-tunnel URLs are reissued whenever cloudflared restarts. "
    "If you cannot reach these URLs, compare updated/tunnels_started_at against now "
    "and re-read this branch; if it is stale, ask mac-codex to republish."
)
if os.environ.get("PSI_DRY") == "1":
    # dry-run must not modify anything, including the clone's working tree
    raise SystemExit(10)
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
raise SystemExit(10)
PY
rc=$?

if [[ "$rc" == 0 ]]; then
  log "unchanged (tunnels_started_at=${STARTED:-unknown})"
  exit 0
fi
if [[ "$rc" != 10 ]]; then
  log "abort: endpoint rewrite failed rc=$rc"
  exit 1
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "dry-run: endpoint/mac.json would be republished" >&2
  log "dry-run: change detected, nothing written"
  exit 10
fi

git add endpoint/mac.json
git -c user.name=mac-codex -c user.email=mac-codex@pandosi.local \
  commit -q -m "endpoint: republish quick-tunnel URLs (tunnels_started_at=$STARTED)" || {
    log "note: nothing staged; leaving as-is"; exit 0; }

if ! git push -q "$REMOTE" "HEAD:$BRANCH" 2>>"$LOG"; then
  # Someone else moved the branch; rewind onto theirs and try once more.
  git fetch -q --depth 1 "$REMOTE" "$BRANCH" 2>>"$LOG" || { log "push failed; fetch failed"; exit 1; }
  git rebase -q FETCH_HEAD 2>>"$LOG" || { log "push failed; rebase failed"; exit 1; }
  git push -q "$REMOTE" "HEAD:$BRANCH" 2>>"$LOG" || { log "push failed twice; will retry next tick"; exit 1; }
fi

head_sha="$(git rev-parse --short HEAD)"
log "published $head_sha (mb=$MB root=$RT 4b=$L4 t2=$T2)"
echo "published $head_sha" >&2
exit 0
