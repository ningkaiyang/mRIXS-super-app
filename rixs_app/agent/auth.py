"""CBORG API key resolution and connection testing.

Key resolution order:
  1. ``os.environ["CBORG_API_KEY"]`` (supports CI/CD overrides).
  2. ``{project_root}/cborg-auth/.env`` via ``python-dotenv``.
  3. ``None`` — triggers the first-time setup wizard.

Connection testing issues ``GET /v1/models`` against the CBORG endpoint
and provides specific diagnostics for common failure modes (401, 403,
timeout, network unreachable).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Default CBORG API endpoint
CBORG_BASE_URL = "https://api.cborg.lbl.gov/v1"
CBORG_DEFAULT_MODEL = "lbl/cborg-deepthought"

# Relative path from the project root to the auth directory
_AUTH_DIR_NAME = "cborg-auth"
_ENV_FILE_NAME = ".env"


def _find_project_root() -> Path:
    """Walk upward from this file to find the project root.

    The project root is identified as the directory containing ``run.py``
    (the primary application entry point) or ``requirements.txt``.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If no recognisable project root is found.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # safety bound
        if (current / "run.py").exists() or (current / "requirements.txt").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        "Could not locate project root (expected run.py or requirements.txt)"
    )


def _auth_env_path() -> Path:
    """Return the absolute path to ``cborg-auth/.env``."""
    return _find_project_root() / _AUTH_DIR_NAME / _ENV_FILE_NAME


def resolve_api_key() -> str | None:
    """Resolve the CBORG API key using the priority hierarchy.

    Resolution order:
      1. Environment variable ``CBORG_API_KEY``.
      2. Dotenv file at ``{project_root}/cborg-auth/.env``.
      3. ``None`` if no key is found.

    Returns:
        The API key string, or ``None`` if unavailable.
    """
    # 1. Environment variable (highest priority — CI/CD, shell exports)
    env_key = os.environ.get("CBORG_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Dotenv file
    try:
        env_path = _auth_env_path()
        if env_path.is_file():
            from dotenv import dotenv_values

            values = dotenv_values(env_path)
            dotenv_key = values.get("CBORG_API_KEY")
            if dotenv_key and dotenv_key.strip():
                return dotenv_key.strip()
    except (FileNotFoundError, ImportError):
        pass

    # 3. Not found
    return None


def save_api_key(key: str) -> Path:
    """Persist the API key to ``cborg-auth/.env``.

    Creates the ``cborg-auth/`` directory if it does not exist.

    Args:
        key: The CBORG API key to save.

    Returns:
        The path to the written ``.env`` file.
    """
    env_path = _auth_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(f"CBORG_API_KEY={key}\n", encoding="utf-8")
    return env_path


def verify_connection(
    api_key: str,
    base_url: str = CBORG_BASE_URL,
    timeout: float = 15.0,
) -> tuple[bool, str]:
    """Test connectivity to the CBORG API with the given key.

    Issues a synchronous ``GET /v1/models`` request and inspects the
    HTTP response to provide user-friendly diagnostics.

    Args:
        api_key: The CBORG API key to authenticate with.
        base_url: The CBORG API base URL (default: ``https://api.cborg.lbl.gov/v1``).
        timeout: Request timeout in seconds.

    Returns:
        A ``(success, message)`` tuple.  ``success`` is ``True`` when the
        API returned a valid models list; ``message`` provides details.
    """
    import urllib.request
    import urllib.error
    import json

    url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            model_ids = [m.get("id", "?") for m in data.get("data", [])]
            lbl_models = [m for m in model_ids if m.startswith("lbl/")]
            return (
                True,
                f"Connected successfully. {len(lbl_models)} LBL models available.",
            )

    except urllib.error.HTTPError as exc:
        status = exc.code
        if status == 401:
            return (
                False,
                "Authentication failed (HTTP 401). "
                "The API key is invalid or has been deleted. "
                "Visit https://api.cborg.lbl.gov/key/manage to verify your key.",
            )
        if status == 403:
            return (
                False,
                "Access denied (HTTP 403). "
                "Your IP address may not be authorized. "
                "Ensure you are on the LBNL network or connected via VPN, "
                "then visit https://api.cborg.lbl.gov/key/manage to check IP restrictions.",
            )
        return False, f"HTTP error {status}: {exc.reason}"

    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if "timed out" in reason.lower() or "timeout" in reason.lower():
            return (
                False,
                "Connection timed out. "
                "Check your network connection and VPN status. "
                "The CBORG API requires LBNL network access.",
            )
        return (
            False,
            f"Network error: {reason}. "
            "Verify your internet connection and VPN status.",
        )

    except TimeoutError:
        return (
            False,
            "Connection timed out. "
            "Check your network connection and VPN status.",
        )

    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}"


async def verify_connection_async(
    api_key: str,
    base_url: str = CBORG_BASE_URL,
    timeout: float = 15.0,
) -> tuple[bool, str]:
    """Async wrapper around :func:`verify_connection`.

    Runs the synchronous HTTP request in a thread-pool executor so it can
    be awaited from an ``asyncio`` event loop (e.g. the agent bridge worker
    thread) without blocking.

    Args:
        api_key: The CBORG API key to authenticate with.
        base_url: The CBORG API base URL.
        timeout: Request timeout in seconds.

    Returns:
        A ``(success, message)`` tuple.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, verify_connection, api_key, base_url, timeout
    )


def fetch_model_list(
    api_key: str,
    base_url: str = CBORG_BASE_URL,
    timeout: float = 15.0,
) -> list[str]:
    """Fetch available LBL model IDs from the CBORG API.

    Issues ``GET /v1/models`` and filters to models whose ``id`` starts
    with ``lbl/``.  Returns a sorted list of model IDs, or a hardcoded
    fallback list if the request fails.

    Args:
        api_key: The CBORG API key.
        base_url: The CBORG API base URL.
        timeout: Request timeout in seconds.

    Returns:
        Sorted list of LBL model ID strings.
    """
    import urllib.request
    import urllib.error
    import json

    _FALLBACK_MODELS = [
        "lbl/cborg-deepthought",
        "lbl/gemma-4",
        "lbl/gpt-oss-120b",
        "lbl/cborg-coder",
    ]

    url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            lbl_models = sorted(m for m in model_ids if m.startswith("lbl/"))
            return lbl_models if lbl_models else _FALLBACK_MODELS
    except Exception:  # noqa: BLE001
        return _FALLBACK_MODELS
