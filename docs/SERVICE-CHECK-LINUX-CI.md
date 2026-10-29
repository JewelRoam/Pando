# Linux `syworker --user` service lifecycle validation

Validated on GitHub Actions `ubuntu-latest`, not simulated.

## Final passing run

- Commit: `0ffdc7b2cf8ddad425d41f992b10de52c95232b5`
- Workflow: **Service lifecycle**
- Run: <https://github.com/JewelRoam/PandoSI/actions/runs/38470391492>
- Result: **success**
- Platform: Linux
- Init path: user syworker, no administrator privilege
- Mode: loopback-only root
- Token: temporary file; not in argv

The isolated checker executed the full contract:

```text
install             0
start               0
verify              0
status              0
stop                0
stopped_verified    0
uninstall           0
uninstall_verified  0
pairing.required    true
```

`syworker status` showed the user unit active and the Python process
using `--pairing-token-file`, not a literal token argument.

The redirected coordinator log printed only:

```text
stem pairing token fingerprint: sha256:f997c837afe78f51
```

## Issues found and fixed by CI

1. **Too many syworker arguments**
   - The first install command merged `daemon-reload` and
     `enable --now` into one `systemctl` invocation.
   - Linux correctly rejected it with `Too many arguments`.
   - Fixed by running the two commands through an explicit `sh -c` chain.

2. **Token leaked into redirected logs**
   - One successful run exposed the ephemeral pairing token when coordinator
     printed it to syworker journald.
   - The token belonged to an already-deleted temporary service, but the
     behavior was still unsafe.
   - Fixed so interactive TTYs may show the token, while redirected logs
     print only a SHA-256 fingerprint unless
     `--reveal-pairing-token` is explicitly passed.

## Check command

Run on a native Linux host with a user syworker session:

```bash
python tools/service_check.py --platform linux --yes
```

The checker removes its temporary service, unit file, token, grant, and
audit artifacts after validating every lifecycle step.
