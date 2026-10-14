"""Decision backends. MockBackend enables the full protocol without model weights."""

from __future__ import annotations

import hashlib

try:
    import numpy as np
except ImportError:  # MockBackend is a dependency-free Ring 0 path.
    np = None

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
                 repo: str = "convaiinnovations/laya", device: str = "CPU",
                 pad_to: int | None = None,
                 onnx_buckets: list[str] | None = None):
        if np is None:
            raise RuntimeError(
                "ONNX backend requires numpy; install numpy and onnxruntime")
        from laya.onnx_agent import ONNXAgent  # type: ignore

        self.node_id = node_id
        self.checkpoint = "onnx" if device == "CPU" else f"onnx-{device.lower()}"
        self._agent = ONNXAgent(repo, onnx_path=onnx_path)
        if device != "CPU" and onnx_buckets:
            def _seq_of(path: str) -> tuple[int, int, str]:
                import onnx  # type: ignore

                m = onnx.load(path, load_external_data=False)
                shapes = {}
                for inp in m.graph.input:
                    dims = [d.dim_value for d in
                            inp.type.tensor_type.shape.dim]
                    shapes[inp.name] = dims
                return shapes["input_ids"][1], shapes["marker_pos"][1], path
            buckets = [_seq_of(b) for b in onnx_buckets]
            self._agent.session = _make_openvino_session(
                onnx_path, device, self._agent.session, buckets=buckets)
            return
        if device != "CPU":
            pad_id = 0
            tok = getattr(self._agent, "tok", None)
            if tok is not None and getattr(tok, "pad_token_id", None) is not None:
                pad_id = int(tok.pad_token_id)
            self._agent.session = _make_openvino_session(
                onnx_path, device, self._agent.session,
                pad_to=pad_to, pad_token_id=pad_id)
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
    """Duck-types ort.InferenceSession with OpenVINO.

    Two modes:
    - pad_to=None: compile per exact shape (round-13 behaviour; ~8s compile
      per new shape - a cliff for varied-length workloads, round 30).
    - pad_to=N: bucket every dynamic dim up to the next multiple of N,
      padding inputs (input_ids with pad_token_id, others with 0) and
      slicing outputs back to the true batch. One compile per bucket.
    """

    def __init__(self, onnx_path: str, device: str, ort_session,
                 pad_to: int | None = None, pad_token_id: int = 0):
        self._onnx_path = onnx_path
        self._device = device
        self._ort = ort_session
        self._pad_to = pad_to
        self._pad_token_id = pad_token_id
        self._cache: dict[tuple, tuple] = {}

    @staticmethod
    def _bucket(dim: int, pad_to: int) -> int:
        return ((dim + pad_to - 1) // pad_to) * pad_to

    def _target_dim(self, name: str, axis: int, dim: int) -> int:
        """Per-dim-position bucket sizes for the Laya graph.

        dim 0 = question rows (batch): exact - the set of batch values is
        small and exactness avoids wasted rows.
        dim 1 = sequence length for input_ids/attention_mask (bucket 128),
        option-marker count for marker inputs (bucket 8).
        """
        if self._pad_to is None or axis == 0:
            return dim  # axis 0 = question rows: exact, no padding
        if name in ("input_ids", "attention_mask"):
            return self._bucket(dim, self._pad_to)
        return self._bucket(dim, min(self._pad_to, 8))

    def _pad_feed(self, input_feed: dict) -> tuple[dict, int]:
        padded = {}
        true_batch = 0
        for name, arr in input_feed.items():
            a = np.asarray(arr)
            if a.ndim == 0:
                padded[name] = a
                continue
            true_batch = a.shape[0]
            target = list(a.shape)
            changed = False
            for i, d in enumerate(a.shape):
                bucketed = self._target_dim(name, i, d)
                if bucketed != d:
                    target[i] = bucketed
                    changed = True
            if not changed:
                padded[name] = a
                continue
            fill = self._pad_token_id if name == "input_ids" else 0
            out = np.full(target, fill, dtype=a.dtype)
            out[tuple(slice(0, d) for d in a.shape)] = a
            padded[name] = out
        return padded, true_batch

    def _compile(self, key, input_feed):
        import openvino as ov  # type: ignore

        core = ov.Core()
        model = core.read_model(self._onnx_path)
        needs_reshape = any(
            not inp.get_partial_shape().is_static for inp in model.inputs)
        if needs_reshape:
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
        if self._pad_to:
            padded_feed, true_batch = self._pad_feed(input_feed)
            key = tuple(sorted((n, tuple(padded_feed[n].shape))
                               for n in padded_feed))
        else:
            padded_feed = input_feed
            true_batch = max(
                (np.asarray(v).shape[0] for v in input_feed.values()
                 if np.asarray(v).ndim > 0), default=1)
            key = tuple(sorted((n, tuple(padded_feed[n].shape))
                               for n in padded_feed))
        if key not in self._cache:
            print(f"[ov-lazy] compiling {self._device} for shapes {key}...",
                  flush=True)
            self._compile(key, padded_feed)
        infer, out_order = self._cache[key]
        res = infer.infer(padded_feed)
        names = output_names or out_order
        outs = []
        for name in names:
            arr = np.asarray(res[name])
            if true_batch and arr.shape and arr.shape[0] > true_batch:
                arr = arr[:true_batch]
            # Padded-marker lanes can produce NaN in the act head on GPU;
            # consumers sanitize to 0 (act_probability absent -> no action).
            arr = np.nan_to_num(arr, nan=0.0)
            outs.append(arr)
        return outs


class _MultiBucketSession:
    """Serves variable-length inputs from several static-shape OV models.

    Buckets are (seq, markers) static exports sorted ascending; run() picks
    the smallest bucket that fits the input, pads, and delegates.
    """

    def __init__(self, buckets: list[tuple[int, int, str]], device: str):
        import openvino as ov  # type: ignore

        self._device = device
        self._buckets = sorted(buckets)  # (seq, markers, path)
        self._compiled: dict[tuple, tuple] = {}
        core = ov.Core()
        self._models = []
        for seq, markers, path in self._buckets:
            model = core.read_model(path)
            compiled = core.compile_model(model, device)
            out_order = [o.any_name for o in compiled.outputs]
            self._models.append((seq, markers, compiled.create_infer_request(),
                                 out_order))

    @staticmethod
    def _pad(arr, pad_val, shape):
        import numpy as np
        out = np.full(shape, pad_val, dtype=arr.dtype)
        slices = tuple(slice(0, d) for d in arr.shape)
        out[slices] = arr
        return out

    def run(self, output_names, input_feed, *args, **kwargs):
        import numpy as np

        seq_needed = int(input_feed["input_ids"].shape[1])
        markers_needed = int(input_feed["marker_pos"].shape[1])
        for seq, markers, infer, out_order in self._models:
            if seq >= seq_needed and markers >= markers_needed:
                break
        else:
            raise ValueError(
                f"input (seq={seq_needed}, markers={markers_needed}) exceeds "
                f"largest bucket {self._buckets[-1][:2]}")

        padded = {}
        for name, arr in input_feed.items():
            a = np.asarray(arr)
            if name == "input_ids":
                padded[name] = self._pad(a, 1, (a.shape[0], seq))
            elif name == "attention_mask":
                padded[name] = self._pad(a, 0, (a.shape[0], seq))
            elif name in ("marker_pos", "marker_mask"):
                padded[name] = self._pad(a, 0, (a.shape[0], markers))
            else:
                padded[name] = a
        res = infer.infer(padded)
        names = output_names or out_order
        return [np.asarray(res[name]) for name in names]


def _make_openvino_session(onnx_path: str, device: str, ort_session,
                           pad_to: int | None = None,
                           pad_token_id: int = 0,
                           buckets: list[tuple[int, int, str]] | None = None):
    if buckets:
        return _MultiBucketSession(buckets, device)
    return _OVLazySession(onnx_path, device, ort_session,
                          pad_to=pad_to, pad_token_id=pad_token_id)
