"""Unit tests for rixs_app.agent.auth — API key resolution and connection testing."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from rixs_app.agent.auth import (
    CBORG_BASE_URL,
    CBORG_DEFAULT_MODEL,
    _auth_env_path,
    _find_project_root,
    fetch_model_list,
    resolve_api_key,
    save_api_key,
    verify_connection,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are correct."""

    def test_base_url(self):
        assert CBORG_BASE_URL == "https://api.cborg.lbl.gov/v1"

    def test_default_model(self):
        assert CBORG_DEFAULT_MODEL == "lbl/cborg-deepthought"


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    """Verify the project root locator finds the correct directory."""

    def test_finds_root(self):
        root = _find_project_root()
        assert (root / "run.py").exists() or (root / "requirements.txt").exists()

    def test_root_contains_rixs_app(self):
        root = _find_project_root()
        assert (root / "rixs_app").is_dir()


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    """Test the 3-tier API key resolution hierarchy."""

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        """Environment variable should win over dotenv file."""
        monkeypatch.setenv("CBORG_API_KEY", "env-key-123")
        key = resolve_api_key()
        assert key == "env-key-123"

    def test_dotenv_fallback(self, tmp_path, monkeypatch):
        """When env var is absent, the dotenv file should be used."""
        monkeypatch.delenv("CBORG_API_KEY", raising=False)
        # Create a temporary dotenv file
        env_path = tmp_path / "cborg-auth" / ".env"
        env_path.parent.mkdir()
        env_path.write_text("CBORG_API_KEY=dotenv-key-456\n")
        # Patch the auth env path to point to our temp file
        monkeypatch.setattr(
            "rixs_app.agent.auth._auth_env_path", lambda: env_path
        )
        key = resolve_api_key()
        assert key == "dotenv-key-456"

    def test_returns_none_when_no_key(self, monkeypatch):
        """When no key is available anywhere, should return None."""
        monkeypatch.delenv("CBORG_API_KEY", raising=False)
        monkeypatch.setattr(
            "rixs_app.agent.auth._auth_env_path",
            lambda: Path("/nonexistent/path/.env"),
        )
        key = resolve_api_key()
        assert key is None

    def test_strips_whitespace(self, monkeypatch):
        """Keys with leading/trailing whitespace should be stripped."""
        monkeypatch.setenv("CBORG_API_KEY", "  spaced-key  \n")
        key = resolve_api_key()
        assert key == "spaced-key"

    def test_empty_env_var_falls_through(self, monkeypatch):
        """An empty string env var should not be treated as a valid key."""
        monkeypatch.setenv("CBORG_API_KEY", "")
        monkeypatch.setattr(
            "rixs_app.agent.auth._auth_env_path",
            lambda: Path("/nonexistent/path/.env"),
        )
        key = resolve_api_key()
        assert key is None


# ---------------------------------------------------------------------------
# save_api_key
# ---------------------------------------------------------------------------


class TestSaveApiKey:
    """Test API key persistence to dotenv file."""

    def test_creates_directory_and_file(self, tmp_path, monkeypatch):
        env_path = tmp_path / "cborg-auth" / ".env"
        monkeypatch.setattr(
            "rixs_app.agent.auth._auth_env_path", lambda: env_path
        )
        result = save_api_key("test-key-789")
        assert result == env_path
        assert env_path.exists()
        content = env_path.read_text()
        assert "CBORG_API_KEY=test-key-789" in content

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        env_path = tmp_path / "cborg-auth" / ".env"
        env_path.parent.mkdir()
        env_path.write_text("CBORG_API_KEY=old-key\n")
        monkeypatch.setattr(
            "rixs_app.agent.auth._auth_env_path", lambda: env_path
        )
        save_api_key("new-key-abc")
        content = env_path.read_text()
        assert "new-key-abc" in content
        assert "old-key" not in content


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------


class TestVerifyConnection:
    """Test CBORG connection testing with mocked HTTP responses."""

    def _mock_urlopen(self, monkeypatch, *, status=200, body=None, error=None):
        """Helper to mock urllib.request.urlopen."""
        import urllib.error
        import urllib.request

        if error:
            monkeypatch.setattr(
                "urllib.request.urlopen",
                mock.Mock(side_effect=error),
            )
        else:
            response_mock = mock.MagicMock()
            response_mock.read.return_value = (body or "{}").encode("utf-8")
            response_mock.__enter__ = mock.Mock(return_value=response_mock)
            response_mock.__exit__ = mock.Mock(return_value=False)
            monkeypatch.setattr(
                "urllib.request.urlopen",
                mock.Mock(return_value=response_mock),
            )

    def test_success(self, monkeypatch):
        import json

        body = json.dumps(
            {
                "data": [
                    {"id": "lbl/cborg-deepthought"},
                    {"id": "lbl/gemma-4"},
                    {"id": "openai/gpt-4"},
                ]
            }
        )
        self._mock_urlopen(monkeypatch, body=body)
        ok, msg = verify_connection("test-key")
        assert ok is True
        assert "2 LBL models" in msg

    def test_401_error(self, monkeypatch):
        import urllib.error

        error = urllib.error.HTTPError(
            url="", code=401, msg="Unauthorized", hdrs=None, fp=None
        )
        self._mock_urlopen(monkeypatch, error=error)
        ok, msg = verify_connection("bad-key")
        assert ok is False
        assert "401" in msg
        assert "invalid" in msg.lower()

    def test_403_error(self, monkeypatch):
        import urllib.error

        error = urllib.error.HTTPError(
            url="", code=403, msg="Forbidden", hdrs=None, fp=None
        )
        self._mock_urlopen(monkeypatch, error=error)
        ok, msg = verify_connection("blocked-key")
        assert ok is False
        assert "403" in msg
        assert "IP" in msg or "VPN" in msg

    def test_timeout_error(self, monkeypatch):
        import urllib.error

        error = urllib.error.URLError(reason="timed out")
        self._mock_urlopen(monkeypatch, error=error)
        ok, msg = verify_connection("test-key")
        assert ok is False
        assert "timed out" in msg.lower()

    def test_network_error(self, monkeypatch):
        import urllib.error

        error = urllib.error.URLError(reason="Name resolution failed")
        self._mock_urlopen(monkeypatch, error=error)
        ok, msg = verify_connection("test-key")
        assert ok is False
        assert "Network error" in msg or "network" in msg.lower()


# ---------------------------------------------------------------------------
# fetch_model_list
# ---------------------------------------------------------------------------


class TestFetchModelList:
    """Test model list fetching with fallback behavior."""

    def _mock_urlopen(self, monkeypatch, *, body=None, error=None):
        """Helper to mock urllib.request.urlopen."""
        if error:
            monkeypatch.setattr(
                "urllib.request.urlopen",
                mock.Mock(side_effect=error),
            )
        else:
            response_mock = mock.MagicMock()
            response_mock.read.return_value = (body or "{}").encode("utf-8")
            response_mock.__enter__ = mock.Mock(return_value=response_mock)
            response_mock.__exit__ = mock.Mock(return_value=False)
            monkeypatch.setattr(
                "urllib.request.urlopen",
                mock.Mock(return_value=response_mock),
            )

    def test_returns_lbl_models_sorted(self, monkeypatch):
        import json

        body = json.dumps(
            {
                "data": [
                    {"id": "lbl/gemma-4"},
                    {"id": "openai/gpt-4"},
                    {"id": "lbl/cborg-deepthought"},
                    {"id": "lbl/cborg-coder"},
                ]
            }
        )
        self._mock_urlopen(monkeypatch, body=body)
        models = fetch_model_list("test-key")
        assert models == [
            "lbl/cborg-coder",
            "lbl/cborg-deepthought",
            "lbl/gemma-4",
        ]

    def test_fallback_on_error(self, monkeypatch):
        self._mock_urlopen(monkeypatch, error=Exception("boom"))
        models = fetch_model_list("test-key")
        assert "lbl/cborg-deepthought" in models
        assert len(models) == 4  # hardcoded fallback list

    def test_fallback_on_empty_lbl_list(self, monkeypatch):
        import json

        body = json.dumps({"data": [{"id": "openai/gpt-4"}]})
        self._mock_urlopen(monkeypatch, body=body)
        models = fetch_model_list("test-key")
        assert "lbl/cborg-deepthought" in models
