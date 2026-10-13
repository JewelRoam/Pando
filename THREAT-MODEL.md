# Threat model & responsible-use statement

PandoSI is **defensive research**: it studies how decision-model swarms form,
reach consensus, and replicate, so that defenders can detect and contain such
systems. It is NOT a deployment framework for autonomous self-replicating
software on third-party machines.

## Hard guarantees built into the code and protocols

1. **Sandbox binding**: Spore's SCAN phase only enumerates nodes via explicit
   configuration or mDNS inside the operator's LAN. There is no exploit code,
   no vulnerability scanning, and no credential use anywhere in this repo.
2. **Signed playbooks only**: a stem refuses any Spore payload whose manifest
   fails Ed25519 verification against a pinned public key.
3. **Fail-closed guard**: tool execution proposed by a playbook passes
   deterministic rules first; unresolved cases require quorum consensus;
   guard unavailability blocks execution rather than permitting it.
4. **No stealth**: all nodes announce themselves via mDNS and write local
   metrics logs. The system is deliberately loud, unlike malware.
5. **Human kill-switch**: rootd exposes /shutdown (authenticated) and stems
   honor a signed revocation list of playbook versions.
6. **Tokenized root surface**: CLI roots generate a pairing token;
   `/register`, `/nodes`, `/decide`, playbook reads, and root-served
   Spore endpoints fail closed unless it is presented.
7. **Expiring capability grants**: LAN listeners and lab replication require
   an operator-created, expiring `~/.pandosi/grants.toml` entry; missing or
   expired grants fail closed.
8. **Tamper-evident grant audit**: creation, revocation, allowed uses, and
   denials append to a local hash chain; an invalid audit chain blocks
   authorization instead of silently continuing.
9. **Review-only service planning**: service adapters output platform plans
   and require a separate expiring autostart grant; they do not install
   services and refuse cross-host generation to avoid embedding wrong paths.

## What this project does NOT do

- It does not scan, exploit, or authenticate to third-party systems.
- It does not exfiltrate data; metrics never leave the operator's machine.
- It does not hide from network monitoring (see Anthropic's blue-team notes:
  sustained device-to-device traffic and multi-GB payloads are detectable; our
  design keeps those signatures).
