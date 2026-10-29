# Windows service lifecycle validation record.
Date: 2026-10-14 05:14 local (UTC+8)
Host platform: Windows, non-admin user session
Adapter: user Startup file + hidden PowerShell runner + PID file
Mode: loopback-only root
Token: temporary file, random value, not in argv
Grants: temporary `service_autostart`; no `lan_listen`
Task/process name: ephemeral, unique per check
Cleanup verified: yes

## Commands

```bash
python tools/service_check.py --yes
```

The checker:

1. creates a temporary 15-minute `service_autostart` grant and token file;
2. writes reviewable Startup and PowerShell runner files under a temp config;
3. starts `pando.coordinator` on a free high port;
4. polls `/pairing` and requires `{"required": true}`;
5. reads process status through the generated PID file;
6. stops and verifies the endpoint is gone;
7. deletes the temporary Startup/runner files;
8. verifies the task/process is absent;
9. writes all lifecycle events to the temporary grant audit chain.

## Evidence

The real report was:

```json
{
  "platform": "windows",
  "steps": [
    {"step": "install", "code": 0},
    {"step": "start", "code": 0},
    {"step": "verify", "code": 0},
    {"step": "status", "code": 0},
    {"step": "stop", "code": 0},
    {"step": "stopped_verified", "code": 0},
    {"step": "uninstall", "code": 0},
    {"step": "uninstall_verified", "code": 0}
  ],
  "pairing": {"required": true},
  "status_output": "Id ProcessName StartTime ... python",
  "status": "OK"
}
```

## Important negative result

The first implementation used:

```text
schtasks /Create /SC ONLOGON /RL LIMITED
```

and failed with `Access is denied` in a normal user session. That made the
adapter violate PandoSI's zero-privilege onboarding rule. The final Windows
adapter therefore uses the user Startup folder and user-owned files; it does
not request administrator or Task Scheduler logon-trigger privileges.

## Remaining platform checks

- Linux `syworker --user`: needs a real Linux desktop/server session.
- Termux `sv`: needs a real Termux host.
