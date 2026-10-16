"""Per-checkpoint probability calibration (blueprint 3.2.1).

Laya ships uncalibrated temperatures (round 24: independently verified).
All confidences must pass temperature scaling before tier gating;
uncalibrated probabilities would make the auto/verify boundary unreliable.

Two sources of calibration:
1. Checkpoint's rl_agent_config.json (ships with the model)
2. Empirical: computed from a held-out set (requires labels)

Without calibration data, use identity (temperature=1.0) but flag as
uncalibrated so the root knows to downgrade the vote weight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_calibration(checkpoint_path: str) -> dict:
    """Load calibration data from a checkpoint's config files.

    Returns {temperature: float, calibrated: bool, source: str}
    """
    cp = Path(checkpoint_path)

    # Try rl_agent_config.json
    for config_name in ["rl_agent_config.json", "config.json"]:
        config_path = cp / config_name if cp.is_dir() else cp.parent / config_name
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                temps = config.get("temperature") or config.get("temperatures")
                if temps and isinstance(temps, (int, float)):
                    t = float(temps)
                    if 0.1 <= t <= 10.0:
                        return {"temperature": t, "calibrated": True,
                                "source": config_name}
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # Try the project's calibration directory
    cal_dir = Path(__file__).resolve().parents[1] / "assets" / "calibration"
    cal_file = cal_dir / f"{cp.stem if cp.suffix else cp.name}.json"
    if not cal_file.exists():
        # try by checkpoint name (e.g. "english", "multilingual")
        for f in cal_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("checkpoint") == cp.stem or data.get("checkpoint") == cp.name:
                cal_file = f
                break

    if cal_file.exists():
        try:
            data = json.loads(cal_file.read_text(encoding="utf-8"))
            t = float(data.get("temperature", 1.0))
            if 0.1 <= t <= 10.0:
                return {"temperature": t, "calibrated": True,
                        "source": f"assets/calibration/{cal_file.name}"}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # No calibration found
    return {"temperature": 1.0, "calibrated": False, "source": "none"}


def apply_temperature(probability: float, temperature: float) -> float:
    """Apply temperature scaling to a probability.

    p_calibrated = sigmoid(logit(p) / temperature)
    where logit(p) = ln(p / (1 - p))

    Temperature > 1 softens (more uncertain), < 1 sharpens (more confident).
    """
    p = min(max(probability, 1e-7), 1.0 - 1e-7)
    logit = math.log(p / (1.0 - p))
    scaled_logit = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled_logit))


class Calibrator:
    """Per-checkpoint calibrator with temperature cache."""

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def calibrate(self, checkpoint: str, probability: float) -> tuple[float, bool]:
        """Return (calibrated_probability, is_calibrated)."""
        if checkpoint not in self._cache:
            self._cache[checkpoint] = load_calibration(checkpoint)
        cal = self._cache[checkpoint]
        calibrated_p = apply_temperature(probability, cal["temperature"])
        return round(calibrated_p, 4), cal["calibrated"]

    def calibrate_batch(self, checkpoint: str,
                        probabilities: list[float]) -> list[float]:
        cal = self.calibrate(checkpoint, 0.5)  # load cache
        return [round(apply_temperature(p, cal["temperature"]), 4)
                for p in probabilities]
