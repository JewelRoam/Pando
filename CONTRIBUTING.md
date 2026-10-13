# Contributing to PandoSI

## Getting started

```bash
git clone https://github.com/JewelRoam/PandoSI.git
cd PandoSI
pip install pynacl zeroconf          # minimum deps
python tools/demo.py                  # verify everything works
python -m pandosi.doctor             # check your capabilities
```

## Development principles

1. **Minimal mocks**: only the decision backend is mocked; everything else
   (HTTP, registry, quorum, aggregation) runs real code.
2. **Smoke over unit tests**: one smoke file (tests/test_smoke.py, 2 cases).
   Run at milestones, not per-edit. Static analysis (py_compile/ruff) is
   the default check.
3. **Honest negatives**: if something doesn't work, document it with data.
   A refuted hypothesis is as valuable as a confirmed one.
4. **Evidence-backed claims**: every number in the docs should trace to a
   research round (docs/research/) or a measurable experiment.
5. **No telemetry**: metrics and rollouts stay on the operator's disk.

## Areas that need contribution

- **Termux/TV box testing**: run the graft suite on a real TV box and
  publish the numbers (first public Amlogic S905 data would be valuable)
- **NPU graph surgery**: IsNaN(30x) + other op-level fixes for the vpux
  compiler; requires ONNX graph manipulation skills
- **RLCD training**: run the actual training loop with accumulated
  disagreement data; laya-rs or laya.train both work
- **Dashboard**: lineage graph rendering, playbook A/B comparison view
- **Multi-bucket auto-discovery**: join --multi-eu already globs static
  exports; extend to multiple checkpoint families

## Code of conduct

- All replication experiments must run in sandboxes (see THREAT-MODEL.md)
- No telemetry, no data exfiltration
- Honest reporting: negative results are contributions
