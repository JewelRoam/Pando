"""Decision backends.

Every member that answers a :class:`~pando.common.DecisionRequest` is a
backend.  The protocol stack only ever calls ``decide(req) -> Vote`` and
reads ``.checkpoint``; everything else (chat, calibration, artifact
attestation) is optional duck-typed surface discovered with ``getattr``.

New backends register a factory once, at module import::

    @register_backend("my-backend")
    def _make_my_backend(node_id: str, **cfg):
        return MyBackend(node_id, **cfg)

and are reachable from the ``--backend`` flag without touching the dispatch
sites in :mod:`pando.stemd` or :mod:`pando.join`.  That keeps the if/elif
chains from multiplying every time a new runtime is wired in — the same
regression vLLM hit when adding platform backends ad hoc.
"""

from __future__ import annotations

import hashlib

try:
    import numpy as np
except ImportError:  # MockBackend is a dependency-free Ring 0 path.
    np = None

from .common import DecisionRequest, Vote, canonical_question


class MockBackend:
    """Deterministic pseudo-decision backend for protocol testing.

    The choice is derived from a stable hash of (state, question). When
    ``correct_choice`` is set, the backend answers it with ``agree_rate``
    probability; otherwise it answers the hashed choice. This lets tests
    control agreement across replicas.
    """

    def __init__(self, node_id: str, checkpoint: str = "mock",
                 correct_choice: str | None = None, agree_rate: float = 1.0,
                 delay_ms: float = 0.0, serialize: bool = False):
        self.node_id = node_id
        self.checkpoint = checkpoint
        self.correct_choice = correct_choice
        self.agree_rate = agree_rate
        # A simulated device speed, so scaling behaviour can be measured
        # without loading a real model per replica (round 109).
        self.delay_ms = float(delay_ms)
        # A real device serves one decision at a time; an unsynchronised
        # sleep is infinitely parallel and hides the capacity limit entirely.
        self.serialize = bool(serialize)
        self._gate = None

    def decide(self, req: DecisionRequest) -> Vote:
        if not req.candidates:
            raise ValueError("candidates must be non-empty")
        if self.delay_ms:
            import time
            if self.serialize:
                import threading
                if self._gate is None:
                    self._gate = threading.Lock()
                with self._gate:
                    time.sleep(self.delay_ms / 1000.0)
            else:
                time.sleep(self.delay_ms / 1000.0)
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
    """Real Laya decisions via the official ONNXAgent.

    Maps a DecisionRequest to a single choice question. Probabilities come
    from the checkpoint's answer_confidence - see blueprint calibration note:
    treat as uncalibrated until the per-checkpoint temperature layer exists.
    ``device='CUDA'`` requires ``onnxruntime-gpu`` and fails closed unless
    CUDA is the selected execution provider; OpenVINO devices are handled
    separately below.
    """

    def __init__(self, node_id: str, onnx_path: str,
                 repo: str = "convaiinnovations/laya", device: str = "CPU",
                 pad_to: int | None = None,
                 onnx_buckets: list[str] | None = None,
                 checkpoint: str | None = None):
        if np is None:
            raise RuntimeError(
                "ONNX backend requires numpy; install numpy and onnxruntime")
        from laya.onnx_agent import ONNXAgent  # type: ignore

        self.node_id = node_id
        normalized_device = device.strip().upper()
        default_checkpoint = "onnx" if normalized_device == "CPU" else (
            "onnx-cuda" if normalized_device == "CUDA"
            else f"onnx-{normalized_device.lower()}")
        self.checkpoint = checkpoint or default_checkpoint
        self._agent = ONNXAgent(repo, onnx_path=onnx_path)
        if normalized_device == "CUDA":
            providers = self._agent.session.get_providers()
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    "CUDA requested but ONNX Runtime selected "
                    f"{providers or ['none']}; install onnxruntime-gpu and "
                    "verify CUDA/cuDNN libraries before starting this stem")
            self._device = "CUDA"
            return
        if normalized_device != "CPU" and onnx_buckets:
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
        if normalized_device != "CPU":
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
        qid, options = canonical_question(req.question, req.candidates)
        questions = {qid: {"type": "choice",
                           "instructions": f"Classify: "
                                           f"{req.question or 'decision'}.",
                           "criteria": options}}
        result = self._agent.predict(req.state, questions)
        answer = result["answers"][qid]
        choice = answer.get("choice")
        probability = float(answer.get("answer_confidence")
                            or answer.get("confidence") or 0.5)
        return Vote(self.node_id, self.checkpoint, choice, round(probability, 4))


class TorchLayaBackend:
    """Native PyTorch Laya backend (transformers + torch), not OpenVINO.

    Runs the same decision model under the torch ecosystem on any torch
    device. Fails closed when CUDA is requested but torch cannot see it, so a
    misconfigured host cannot silently answer under a CUDA checkpoint label.
    Inference is serialized: one torch model is not safe for concurrent
    forward passes from the threaded stem server.
    """

    def __init__(self, node_id: str, model: str = "convaiinnovations/laya",
                 device: str = "CPU", checkpoint: str | None = None):
        try:
            import threading
            import torch
            from laya import load as laya_load
        except ImportError as exc:
            raise RuntimeError(
                "torch backend requires torch and laya; install them in this "
                "environment and retry") from exc

        requested = (device or "CPU").strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False; "
                "install a CUDA torch build or select a different --device")

        self.node_id = node_id
        self.checkpoint = checkpoint or (
            "torch-cuda" if requested == "cuda" else f"torch-{requested}")
        self._lock = threading.Lock()
        self._agent = laya_load(model, device=requested)
        resolved = getattr(self._agent, "device", None)
        if requested == "cuda" and getattr(resolved, "type", None) != "cuda":
            raise RuntimeError(
                f"CUDA requested but Laya resolved device={resolved}; "
                "refusing to run a CPU worker under a CUDA label")

    def decide(self, req: DecisionRequest) -> Vote:
        if not req.candidates:
            raise ValueError("candidates must be non-empty")
        qid, options = canonical_question(req.question, req.candidates)
        questions = {qid: {"type": "choice",
                           "instructions": f"Classify: "
                                           f"{req.question or 'decision'}.",
                           "criteria": options}}
        with self._lock:
            result = self._agent.predict(req.state, questions)
        answer = result["answers"][qid]
        choice = answer.get("choice")
        probability = float(answer.get("answer_confidence")
                            or answer.get("confidence") or 0.5)
        return Vote(self.node_id, self.checkpoint, choice, round(probability, 4))


def parse_openai_choice(text: str, candidates: list[str]) -> str:
    """Map an OpenAI-compatible completion to exactly one candidate.

    Exact matches ignore surrounding whitespace and case. When no candidate
    is an exact match, a candidate that appears as a substring of the
    response is accepted only if it is the unique such candidate; ambiguity
    is an error so a broken deployment cannot silently vote for the wrong
    option.
    """
    stripped = (text or "").strip()
    lowered = stripped.lower()
    for cand in candidates:
        if lowered == cand.strip().lower():
            return cand
    substring_hits = [c for c in candidates if c.lower() in lowered]
    if len(substring_hits) == 1:
        return substring_hits[0]
    raise ValueError(
        f"could not map completion {stripped!r} to exactly one candidate")


def _openai_completion_text(response: dict) -> str:
    """Extract the message text from a /chat/completions response."""
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("no choices in OpenAI response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict))
    return content or ""


class OpenAIChoiceBackend:
    """Decision backend for an OpenAI-compatible chat-completions endpoint.

    This is the bridge for a Windows-hosted stemd to use a Mac LLM worker over
    LAN: the process runs on Windows, the request goes to the Mac API. The
    prompt is Task 004's variant B, validated on the held-out suite: ask for
    the candidate word only, no JSON and no confidence, which cut latency and
    parse errors while improving accuracy. There is no calibrated probability
    in that response. Votes are marked neutral; the configurable probability
    remains on the wire only for compatibility with older readers and is
    excluded from confidence aggregation.
    """

    SYSTEM_PROMPT = (
        "只输出候选词本身，不要 JSON、不要置信度，不要任何其他文字。"
    )

    def __init__(self, node_id: str, base_url: str, api_key: str = "",
                 model: str = "qwen3-4b-instruct-2507",
                 checkpoint: str = "openai", prompt_variant: str = "b",
                 timeout: float = 60.0, probability: float = 0.5,
                 transport=None):
        if (prompt_variant or "b").strip().lower() != "b":
            raise ValueError("OpenAIChoiceBackend only implements prompt variant b")
        if not base_url:
            raise ValueError("base_url must be non-empty")
        self.node_id = node_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.checkpoint = checkpoint or "openai"
        self.prompt_variant = "b"
        self.timeout = float(timeout)
        self.probability = float(probability)
        self._transport = transport or self._default_transport

    @staticmethod
    def _default_transport(url: str, payload: dict, api_key: str,
                           timeout: float) -> dict:
        import json
        import urllib.error
        import urllib.request

        from .common import bypass_proxy_for_lan

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=data, headers=headers,
                                         method="POST")
        # LAN/loopback traffic must not be surrendered to a system proxy.
        bypass_proxy_for_lan()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"unreachable: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSON response: {body[:500]!r}") from exc

    def _user_prompt(self, req: DecisionRequest) -> str:
        return (
            f"状态：{req.state}\n"
            f"问题：{req.question}\n"
            "候选：\n"
            + "\n".join(f"- {candidate}" for candidate in req.candidates)
            + "\n只从以上候选中选择一个。"
        )

    def decide(self, req: DecisionRequest) -> Vote:
        if not req.candidates:
            raise ValueError("candidates must be non-empty")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._user_prompt(req)},
            ],
            "temperature": 0.0,
            "max_tokens": 16,
            "stream": False,
        }
        response = self._transport(self.base_url + "/chat/completions",
                                   payload, self.api_key, self.timeout)
        choice = parse_openai_choice(
            _openai_completion_text(response), req.candidates)
        return Vote(self.node_id, self.checkpoint, choice,
                    round(self.probability, 4), neutral=True)

    def chat_descriptor(self) -> dict:
        """What the node record stores. Presence of this is the chat grant."""
        return {"model": self.model}

    def complete(self, messages: list[dict], max_tokens: int = 400,
                 temperature: float = 0.2, timeout: float | None = None) -> str:
        """One chat completion. This is not a vote and carries no probability."""
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        response = self._transport(
            self.base_url + "/chat/completions", payload, self.api_key,
            self.timeout if timeout is None else float(timeout))
        text = _openai_completion_text(response).strip()
        if not text:
            raise RuntimeError("model returned an empty message")
        return text


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
        try:
            import openvino as ov  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO device requested but openvino is not installed; "
                "pip install openvino and retry") from exc
        core = ov.Core()
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": "0"})
        # The NPU plugin enumerates on import; probing devices at __init__
        # (instead of on the first decide) keeps failure messages inside the
        # stem's startup log rather than its first vote.
        self._devices = [d for d in core.available_devices
                         if d != "NPU" or "NPU" in device]
        self._core = core
        self._onnx_path = onnx_path
        self._device = device
        self._compiled: dict = {}
        self._lock = threading.Lock()
        self._ort = ort_session
        self._pad_to = pad_to
        self._pad_token_id = pad_token_id
        self.input_names = [i.name for i in ort_session.get_inputs()]
        self.output_names = [o.name for o in ort_session.get_outputs()]

    @staticmethod
    def _bucket(dim: int, pad_to: int) -> int:
        return ((dim + pad_to - 1) // pad_to) * pad_to

    def _target_dim(self, name: str, axis: int, dim: int) -> int:
        if self._pad_to is None or dim == 0:
            return dim
        if axis == 1 and name == "input_ids":
            return self._bucket(dim, self._pad_to)
        if axis == 1:
            return self._bucket(dim, self._pad_to)
        return dim

    def _pad_feed(self, input_feed: dict) -> tuple[dict, int]:
        if self._pad_to is None:
            return input_feed, 0
        import numpy as np
        padded = {}
        seq_dim = 0
        for name, arr in input_feed.items():
            target = list(arr.shape)
            for axis in range(1, len(target)):
                target[axis] = self._target_dim(name, axis, target[axis])
            if name == "input_ids" and len(target) > 1:
                seq_dim = target[1]
            pad_val = (self._pad_token_id if name == "input_ids" else 0)
            padded[name] = self._pad(arr, pad_val, tuple(target))
        return padded, seq_dim

    def _compile(self, key, input_feed):
        import openvino as ov  # type: ignore

        with self._lock:
            if key in self._compiled:
                return
            model = self._core.read_model(self._onnx_path)
            if self._pad_to is not None:
                for inp in model.inputs:
                    partial_shape = list(inp.get_partial_shape())
                    for axis in range(1, len(partial_shape)):
                        dim = partial_shape[axis]
                        if dim.is_dynamic:
                            target = self._target_dim(
                                inp.get_any_name(), axis,
                                int(input_feed[inp.get_any_name()].shape[axis]))
                            partial_shape[axis] = target
                model.reshape(dict(zip((i.get_any_name() for i in model.inputs),
                                       map(ov.PartialShape, partial_shape))))
            compiled = self._core.compile_model(model, self._device)
            self._compiled[key] = compiled

    def run(self, output_names, input_feed, *args, **kwargs):
        padded, seq_dim = self._pad_feed(input_feed)
        key = tuple((name, tuple(arr.shape)) for name, arr in padded.items())
        if key not in self._compiled:
            self._compile(key, padded)
        compiled = self._compiled[key]
        results = compiled(padded)
        outputs = [results[name] for name in output_names]
        if self._pad_to is not None and seq_dim:
            sliced = []
            for name, out in zip(output_names, outputs):
                sliced.append(out[:, :seq_dim] if out.ndim > 1 else out)
            return sliced
        return outputs


class _MultiBucketSession:
    """Serves several static-shape OpenVINO graphs as one dynamic session."""

    def __init__(self, buckets: list[tuple[int, int, str]], device: str):
        try:
            import openvino as ov  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO device requested but openvino is not installed; "
                "pip install openvino and retry") from exc
        core = ov.Core()
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": "0"})
        self._compiled: dict[tuple[int, int], object] = {}
        self._device = device
        self._lock = threading.Lock()
        for seq, marker, path in buckets:
            model = core.read_model(path)
            compiled = core.compile_model(model, device)
            self._compiled[(seq, marker)] = compiled

    @staticmethod
    def _pad(arr, pad_val, shape):
        import numpy as np
        out = np.full(shape, pad_val, dtype=arr.dtype)
        slices = tuple(slice(0, s) for s in arr.shape)
        out[slices] = arr
        return out

    def run(self, output_names, input_feed, *args, **kwargs):
        import numpy as np
        seq = int(input_feed["input_ids"].shape[1])
        marker = int(input_feed["marker_pos"][0][0]) \
            if input_feed["marker_pos"].ndim == 2 else int(input_feed["marker_pos"][0])
        key = None
        for (seq_len, mark_len) in sorted(self._compiled, reverse=True):
            if seq <= seq_len and marker <= mark_len:
                key = (seq_len, mark_len)
                break
        if key is None:
            raise RuntimeError(
                f"no bucket fits input_ids seq={seq} marker={marker}; "
                f"available {sorted(self._compiled)}")
        compiled = self._compiled[key]
        pad_id = 0
        feeds = {}
        for name, arr in input_feed.items():
            target = list(arr.shape)
            target[1] = key[0] if name == "input_ids" else key[1] \
                if name == "marker_pos" else target[1]
            pad_val = pad_id if name == "input_ids" else 0
            feeds[name] = self._pad(arr, pad_val, tuple(target))
        results = compiled(feeds)
        return [results[name] for name in output_names]


def _make_openvino_session(onnx_path: str, device: str, ort_session,
                           pad_to: int | None = None,
                           pad_token_id: int = 0,
                           buckets=None):
    if buckets:
        return _MultiBucketSession(buckets, device)
    return _OVLazySession(onnx_path, device, ort_session, pad_to=pad_to,
                          pad_token_id=pad_token_id)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

from typing import Callable, Dict  # noqa: E402  (kept near use site on purpose)

#: Factory signature every backend builder follows: (node_id, **config) -> backend.
BackendFactory = Callable[..., object]

_BACKENDS: Dict[str, BackendFactory] = {}


def register_backend(name: str) -> Callable[[BackendFactory], BackendFactory]:
    """Class decorator: add a backend factory under ``name``.

    The name is what ``--backend`` and ``pick_backend`` use, so it must be
    unique; a duplicate means the dispatch table has drifted and is a bug.
    """
    def _wrap(factory: BackendFactory) -> BackendFactory:
        if name in _BACKENDS:
            raise ValueError(f"backend {name!r} is already registered")
        _BACKENDS[name] = factory
        return factory
    return _wrap


def backend_names() -> list[str]:
    """Every registered backend name, sorted for stable help text."""
    return sorted(_BACKENDS)


def create_backend(name: str, node_id: str, **cfg):
    """Instantiate the backend registered under ``name``.

    Raises ``KeyError`` on an unknown name so the CLI can turn that into a
    clean refusal instead of a silent fallback to mock.
    """
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; registered: "
            + ", ".join(backend_names())) from None
    return factory(node_id, **cfg)


@register_backend("mock")
def _make_mock(node_id: str, **cfg):
    return MockBackend(node_id, **cfg)


@register_backend("onnx")
def _make_onnx(node_id: str, **cfg):
    return OnnxLayaBackend(node_id, **cfg)


@register_backend("laya")
def _make_laya(node_id: str, **cfg):
    return LayaBackend(node_id, **cfg)


@register_backend("torch")
def _make_torch(node_id: str, **cfg):
    return TorchLayaBackend(node_id, **cfg)


@register_backend("text")
def _make_text(node_id: str, **cfg):
    from .textmodel import TextModel
    model_path = cfg.pop("text_model", None)
    checkpoint = cfg.pop("checkpoint", None)
    if model_path is None:
        from .resources import asset_path
        from pathlib import Path as _Path
        packaged = asset_path("textmodel.json")
        model_path = str(packaged if packaged.exists()
                         else _Path("assets/textmodel.json"))
    backend = TextModel.load(model_path)
    if checkpoint:
        backend.checkpoint = checkpoint
    return backend


@register_backend("openai")
def _make_openai(node_id: str, **cfg):
    api_key = cfg.pop("api_key", "")
    base_url = cfg.pop("base_url", None)
    if base_url is None:
        raise ValueError("openai backend requires base_url")
    return OpenAIChoiceBackend(node_id, base_url, api_key=api_key, **cfg)
