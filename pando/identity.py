"""Per-node Ed25519 identity: durable membership, separate from the invite token.

The pairing token answers "may this device attempt to join". An identity answers
"which node joined, and can it prove that forever". node_id stays a human label;
the public key is the durable claim, so a token holder cannot re-register an
existing member's id with a fresh key, and a root started with
--require-identity refuses any unsigned registration at all.

The identity lives in ~/.pando/identity.json (override with PANDOSI_IDENTITY).
PyNaCl already ships as a hard dependency, so this adds no new requirements.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey

DEFAULT_IDENTITY_PATH = "~/.pando/identity.json"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def canonical(payload: dict) -> bytes:
    """Deterministic bytes for signing: sorted keys, no whitespace, UTF-8."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")


class Identity:
    """A node's signing key plus the public fingerprint others can pin."""

    def __init__(self, signing_key: SigningKey):
        self._key = signing_key
        self.public_key = _b64(self._key.verify_key.encode())
        self.node_fingerprint = hashlib.sha256(
            self._key.verify_key.encode()).hexdigest()[:16]

    def sign(self, payload: dict) -> str:
        return _b64(self._key.sign(canonical(payload)).signature)

    def proof(self, payload: dict) -> dict:
        """A transferable proof: the public key plus a signature over payload."""
        return {"pubkey": self.public_key, "sig": self.sign(payload)}

    def to_json(self) -> dict:
        return {
            "version": 1,
            "node_fingerprint": self.node_fingerprint,
            "public_key": self.public_key,
            "secret_key": _b64(self._key.encode()),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Identity":
        return cls(SigningKey(_unb64(data["secret_key"])))

    @classmethod
    def generate(cls) -> "Identity":
        return cls(SigningKey.generate())


def verify(pubkey_b64: str, payload: dict, sig_b64: str) -> bool:
    """True when sig_b64 is a valid signature of payload under pubkey_b64."""
    try:
        VerifyKey(_unb64(pubkey_b64)).verify(canonical(payload),
                                             _unb64(sig_b64))
        return True
    except Exception:
        return False


def default_path() -> Path:
    return Path(os.environ.get("PANDOSI_IDENTITY",
                               DEFAULT_IDENTITY_PATH)).expanduser()


def load_or_create(path: str | Path | None = None) -> Identity:
    """Load the node identity, creating one on first use (chmod 600)."""
    target = Path(path).expanduser() if path is not None else default_path()
    if target.exists():
        return Identity.from_json(
            json.loads(target.read_text(encoding="utf-8")))
    identity = Identity.generate()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(identity.to_json(), indent=2),
                      encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    return identity
