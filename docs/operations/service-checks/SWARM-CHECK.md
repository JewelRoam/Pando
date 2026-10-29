# Multi-process tokenized swarm validation

Validated on Windows, then designed to run unchanged on native Linux.

## Check

```bash
python tools/swarm_check.py --yes
```

The checker starts independent OS processes, not in-process test doubles:

- one `pando.coordinator` HTTP coordinator;
- five `pando.worker` workers with checkpoint diversity;
- pairing secret read from a temporary token file;
- root protected by the same pairing token;
- real loopback HTTP requests for registration and consensus.

It verifies:

1. `/register` without a token returns `403`;
2. three initial stems register and answer as a quorum;
3. two more stems join the running root and enlarge the quorum to five;
4. one stem is terminated;
5. heartbeat expiry removes it from root;
6. the quorum shrinks from 5 to 4 and still answers;
7. root logs contain only a token fingerprint;
8. all temporary OS processes are cleaned up.

## Windows evidence

Run: `python tools/swarm_check.py --yes`

```text
reject_missing_token: 403
registered: 5
decision_full: choice=approve, tier=auto, responded=5, quorum_size=5
shrink: removed swarm-stem-4, count=4
decision_shrunk: responded=4, quorum_size=4
root stdout: fingerprint only
status: OK
```

The deterministic mock decision returned `verify` after shrink, not an error:
losing one replica lowered observed agreement, so the root correctly routed
the result to verification instead of claiming high confidence.

## Ring 0 dependency fix

`worker --backend mock` previously imported `backend.py`, whose top-level
`import numpy` made the advertised dependency-free mock path fail on a clean
host. NumPy is now optional: `MockBackend` remains dependency-free, while the
ONNX backend explicitly reports that NumPy and ONNX Runtime are required.

CI now verifies a clean `PyNaCl`-only environment can import the mock backend
and run the five-process swarm lifecycle.
