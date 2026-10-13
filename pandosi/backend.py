"""Decision backends. MockBackend enables the full protocol without model weights."""

from __future__ import annotations

import hashlib

import numpy as np

from .common import DecisionRequest, Vote


class MockBackend:
    """Deterministic pseudo-decision backend for protocol testing.

    The choice is derived from a stable hash of (state, question). When
    ``correct_choice`` is set, the backend answers it with ``agree_rate``
    probability; otherwise it answers the hashed choice. This lets tests
    control agreement across replicas.
    """

    def __init__(self, node_id: str, checkpoint: str = "mock",
                 correct_choice: str | None = None, agree_rate: float = 1.0):
        self.node_id = node_id
        self.checkpoint = checkpoint
        self.correct_choice = correct_choice
        self.agree_rate = agree_rate

    def decide(self, req: DecisionRequest) -> Vote:
        if not req.candidates:
            raise ValueError("candidates must be non-empty")
        digest = hashlib.sha256(f"{req.state}|{req.question}".encode()).digest()
        hashed_choice = req.candidates[digest[0] % len(req.candidates)]

        choice = hashed_choice
        probability = 0.55 + (digest[1] % 40) / 100.0  # 0.55 - 0.94
        if self.correct_choice is not None:
            agrees = (digest[2] % 100) / 100.0 < self.agree_rate
            choice = self.correct_choice if agrees else hashed_choice
            probability = 0.9 if agrees else 0.4
        return Vote(self.node_id, self.checkpoint, choice, round(probability, 4))


class LayaBackend:
    """Laya decision-model backend (convaiinnovations/laya, Apache-2.0).

    Requires ``transformers`` and a local checkpoint. Not auto-imported so
    the protocol stack runs dependency-free in mock mode.
    """

    def __init__(self, node_id: str, checkpoint: str = "english"):
        try:
            from laya import load  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pip install laya (or transformers), then retry"
            ) from exc
        self.node_id = node_id
        self.checkpoint = checkpoint
        self._model = load(f"convaiinnovations/laya-{checkpoint}")

    def decide(self, req: DecisionRequest) -> Vote:
        result = self._model.decide(req.state, req.question, req.candidates)
        return Vote(self.node_id, self.checkpoint,
                    result.choice, float(result.probability))


class OnnxLayaBackend:
    """Real Laya decisions via the official ONNXAgent (onnxruntime CPU).

    Maps a DecisionRequest to a single choice question. Probabilities come
    from the checkpoint's answer_confidence - see blueprint calibration note:
    treat as uncalibrated until the per-checkpoint temperature layer exists.
    """

    def __init__(self, node_id: str, onnx_path: str,
                 repo: str = "convaiinnovations/laya", device: str = "CPU"):
        from laya.onnx_agent import ONNXAgent  # type: ignore

        self.node_id = node_id
        self.checkpoint = "onnx" if device == "CPU" else f"onnx-{device.lower()}"
        self._agent = ONNXAgent(repo, onnx_path=onnx_path)
        if device != "CPU":
            self._agent.session = _make_openvino_session(
                onnx_path, device, self._agent.session)
            # Warmup: trigger lazy compilation now, not on the first vote.
            warm = DecisionRequest(state="warmup", question="warmup",
                                   candidates=["a", "b"])
            self.decide(warm)

    def decide(self, req: DecisionRequest) -> Vote:
        if not req.candidates:
            raise ValueError("candidates must be non-empty")
        qid = req.question or "decision"
        questions = {qid: {"type": "choice",
                           "instructions": qid,
                           "criteria": req.candidates}}
        result = self._agent.predict(req.state, questions)
        answer = result["answers"][qid]
        choice = answer.get("choice")
        probability = float(answer.get("answer_confidence")
                            or answer.get("confidence") or 0.5)
        return Vote(self.node_id, self.checkpoint, choice, round(probability, 4))


class _OVLazySession:
    """Duck-types ort.InferenceSession with OpenVINO, compiled lazily per shape.

    The Laya graph has dynamic batch/seq/option dims that OV cannot infer
    correctly (round-8/13 findings). We compile per concrete input-shape tuple
    on first sight and cache; later identical shapes reuse the compiled model.
    """

    def __init__(self, onnx_path: str, device: str, ort_session):
        self._onnx_path = onnx_path
        self._device = device
        self._ort = ort_session  # used only to source output names/order
        self._cache: dict[tuple, tuple] = {}

    def _compile(self, key, input_feed):
        import openvino as ov  # type: ignore

        core = ov.Core()
        model = core.read_model(self._onnx_path)
        reshape_map = {}
        for inp in model.inputs:
            shape = list(input_feed[inp.any_name].shape)
            reshape_map[inp.any_name] = ov.PartialShape(
                [ov.Dimension(d) for d in shape])
        model.reshape(reshape_map)
        compiled = core.compile_model(model, self._device)
        out_order = [o.any_name for o in compiled.outputs]
        self._cache[key] = (compiled.create_infer_request(), out_order)

    def run(self, output_names, input_feed, *args, **kwargs):
        key = tuple(sorted((n, tuple(input_feed[n].shape))
                           for n in input_feed))
        if key not in self._cache:
            print(f"[ov-lazy] compiling {self._device} for shapes {key}...",
                  flush=True)
            self._compile(key, input_feed)
        infer, out_order = self._cache[key]
        res = infer.infer(input_feed)
        names = output_names or out_order
        return [np.asarray(res[name]) for name in names]


def _make_openvino_session(onnx_path: str, device: str, ort_session):
    return _OVLazySession(onnx_path, device, ort_session)
