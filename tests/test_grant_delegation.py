"""Tests for pando/grant_delegation.py.

Focus: envelope shape validation, signature pinning, TTL enforcement,
delegated grant installation, and refresh_and_install semantics.
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pando.grant_delegation import (
    ACTION,
    AUDIENCE,
    DelegationError,
    ENVELOPE_FIELDS,
    MAX_TTL,
    ORIGIN,
    RESOURCE,
    SCOPE,
    install_envelope,
    issue_envelope,
    operator_status,
    refresh_and_install,
    verify_envelope,
)
from pando.grants import grant_status, save_grant
from pando.identity import Identity

# A real Ed25519 keypair that stands in for the pinned Mac operator.
_OPERATOR = Identity.generate()
_NOW = dt.datetime(2026, 10, 29, 12, 0, 0, tzinfo=dt.timezone.utc)


def _make_envelope(**overrides) -> dict:
    base = {
        "version": 1,
        "audience": AUDIENCE,
        "action": ACTION,
        "resource": RESOURCE,
        "scope": SCOPE,
        "approved_by": "test-operator",
        "approved_at": "2026-10-29T11:00:00+00:00",
        "expires_at": "2026-10-30T11:00:00+00:00",
        "evidence": "smoke-test",
        "signature": "placeholder",
    }
    base.update(overrides)
    return base


def _signed_envelope(identity: Identity | None = None, **overrides) -> dict:
    signer = identity or _OPERATOR
    env = _make_envelope(**overrides)
    payload = {k: v for k, v in env.items() if k != "signature"}
    env["signature"] = signer.sign(payload)
    return env


class _PatchedKeyTestCase(unittest.TestCase):
    """Passes the operator pin explicitly; the module carries no constant."""


class TestVerifyEnvelope(_PatchedKeyTestCase):

    def test_valid_envelope_passes(self):
        env = _signed_envelope()
        payload = verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)
        self.assertEqual(payload["action"], ACTION)

    def test_extra_field_rejected(self):
        env = _signed_envelope()
        env["rogue"] = "inject"
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_missing_field_rejected(self):
        env = _signed_envelope()
        del env["evidence"]
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_wrong_version_rejected(self):
        env = _signed_envelope(version=2)
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_wrong_action_rejected(self):
        env = _signed_envelope(action="not_deploy")
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_expired_envelope_rejected(self):
        past = _NOW - dt.timedelta(days=2)
        env = _signed_envelope(
            approved_at=past.isoformat(),
            expires_at=(past + dt.timedelta(hours=1)).isoformat(),
        )
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_ttl_exceeded_rejected(self):
        env = _signed_envelope(
            approved_at=_NOW.isoformat(),
            expires_at=(_NOW + dt.timedelta(days=8)).isoformat(),
        )
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_future_approved_at_rejected(self):
        future = _NOW + dt.timedelta(hours=1)
        env = _signed_envelope(
            approved_at=future.isoformat(),
            expires_at=(future + dt.timedelta(hours=1)).isoformat(),
        )
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)

    def test_wrong_signer_rejected(self):
        impostor = Identity.generate()
        env = _signed_envelope(identity=impostor)
        with self.assertRaises(DelegationError):
            verify_envelope(env, expected_pubkey=_OPERATOR.public_key, now=_NOW)


class TestIssueEnvelope(_PatchedKeyTestCase):

    def _make_active_grant(self, tmp: Path, now: dt.datetime) -> Path:
        grants = tmp / "grants.toml"
        # Back-date approved_at so issue_envelope's internal verify sees
        # approved_at <= now (save_grant stamps approved_at = _now()).
        past = now - dt.timedelta(hours=1)
        save_grant(
            ACTION, RESOURCE, SCOPE, "test-op",
            (now + dt.timedelta(hours=24)).isoformat(),
            evidence="local-test",
            path=grants,
        )
        # Overwrite approved_at to a past time so verify_envelope passes.
        import pando.grants as _g
        data = _g.load_grants(grants)
        data[ACTION]["approved_at"] = past.isoformat()
        _g.save_grants(data, grants)
        return grants

    def test_issue_and_verify_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            now = _NOW
            grants = self._make_active_grant(Path(td), now)
            env = issue_envelope(grant_path=grants, identity=_OPERATOR,
                                 expected_pubkey=_OPERATOR.public_key, now=now)
            payload = verify_envelope(
                env, expected_pubkey=_OPERATOR.public_key, now=now)
            self.assertEqual(payload["approved_by"], "test-op")

    def test_issue_fails_without_grant(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(DelegationError):
                issue_envelope(
                    grant_path=Path(td) / "empty.toml",
                    identity=_OPERATOR,
                    expected_pubkey=_OPERATOR.public_key,
                    now=_NOW,
                )

    def test_issue_fails_with_wrong_signer(self):
        impostor = Identity.generate()
        with tempfile.TemporaryDirectory() as td:
            now = _NOW
            grants = self._make_active_grant(Path(td), now)
            with self.assertRaises(DelegationError):
                issue_envelope(grant_path=grants, identity=impostor,
                               expected_pubkey=_OPERATOR.public_key, now=now)


class TestInstallEnvelope(_PatchedKeyTestCase):

    def test_install_and_idempotent_reinstall(self):
        env = _signed_envelope()
        with tempfile.TemporaryDirectory() as td:
            grants = Path(td) / "grants.toml"
            result = install_envelope(env, grants, expected_pubkey=_OPERATOR.public_key, now=_NOW)
            self.assertEqual(result["status"], "installed")
            again = install_envelope(env, grants, expected_pubkey=_OPERATOR.public_key, now=_NOW)
            self.assertEqual(again["status"], "unchanged")

    def test_manual_grant_not_overwritten(self):
        env = _signed_envelope()
        with tempfile.TemporaryDirectory() as td:
            grants = Path(td) / "grants.toml"
            save_grant(
                ACTION, RESOURCE, SCOPE, "human-op",
                (_NOW + dt.timedelta(hours=24)).isoformat(),
                evidence="human-installed",
                path=grants,
            )
            result = install_envelope(env, grants, expected_pubkey=_OPERATOR.public_key, now=_NOW)
            self.assertEqual(result["status"], "manual-preserved")
            status = grant_status(ACTION, grants, now=_NOW)
            self.assertEqual(status["grant"]["approved_by"], "human-op")


class TestRefreshAndInstall(_PatchedKeyTestCase):

    def _write_token(self, tmp: Path) -> Path:
        token = tmp / "token.txt"
        token.write_text("secret-token", encoding="utf-8")
        return token

    def test_refresh_installs_valid_envelope(self):
        env = _signed_envelope()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            token = self._write_token(tmp)
            grants = tmp / "grants.toml"

            class FakeResponse:
                def read(self, n=-1):
                    return json.dumps(env).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            with patch(
                "pando.grant_delegation.open_url",
                return_value=FakeResponse(),
            ):
                result = refresh_and_install(
                    ["http://shelf.local/grant"],
                    token_file=token,
                    grants_path=grants,
                    expected_pubkey=_OPERATOR.public_key,
                    now=_NOW,
                )
            self.assertEqual(result["status"], "installed")
            self.assertTrue(result["changed"])

    def test_refresh_404_removes_delegated_grant(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            token = self._write_token(tmp)
            grants = tmp / "grants.toml"
            env = _signed_envelope()
            install_envelope(env, grants, expected_pubkey=_OPERATOR.public_key, now=_NOW)

            import urllib.error

            def fake_open_url(request, timeout):
                raise urllib.error.HTTPError(
                    request.full_url, 404, "not found", {}, None
                )

            with patch(
                "pando.grant_delegation.open_url", side_effect=fake_open_url
            ):
                result = refresh_and_install(
                    ["http://shelf.local/grant"],
                    token_file=token,
                    grants_path=grants,
                    expected_pubkey=_OPERATOR.public_key,
                    now=_NOW,
                )
            self.assertEqual(result["status"], "revoked")

    def test_refresh_unreachable_keeps_grant(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            token = self._write_token(tmp)
            grants = tmp / "grants.toml"
            env = _signed_envelope()
            install_envelope(env, grants, expected_pubkey=_OPERATOR.public_key, now=_NOW)

            import urllib.error

            def fake_open_url(request, timeout):
                raise urllib.error.URLError("connection refused")

            with patch(
                "pando.grant_delegation.open_url", side_effect=fake_open_url
            ):
                result = refresh_and_install(
                    ["http://shelf.local/grant"],
                    token_file=token,
                    grants_path=grants,
                    expected_pubkey=_OPERATOR.public_key,
                    now=_NOW,
                )
            self.assertEqual(result["status"], "unreachable")
            status = grant_status(ACTION, grants, now=_NOW)
            self.assertEqual(status["status"], "active")

    def test_refresh_empty_urls_returns_disabled(self):
        result = refresh_and_install(
            [],
            token_file=Path("/dev/null"),
            grants_path=Path("/dev/null"),
            now=_NOW,
        )
        self.assertEqual(result["status"], "disabled")


class TestOperatorStatus(_PatchedKeyTestCase):

    def test_authorized_true_when_grant_active(self):
        env = _signed_envelope()
        with tempfile.TemporaryDirectory() as td:
            grants = Path(td) / "grants.toml"
            install_envelope(env, grants, expected_pubkey=_OPERATOR.public_key, now=_NOW)
            result = operator_status(grants, now=_NOW)
            self.assertTrue(result["authorized"])

    def test_authorized_false_when_no_grant(self):
        with tempfile.TemporaryDirectory() as td:
            grants = Path(td) / "grants.toml"
            result = operator_status(grants, now=_NOW)
            self.assertFalse(result["authorized"])


if __name__ == "__main__":
    unittest.main()
