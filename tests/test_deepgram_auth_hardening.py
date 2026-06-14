"""Hardening for Deepgram connect failures (follow-up to issue #6).

Two small, safe improvements so credential problems are friendly instead of
cryptic:

1. The API key is stripped, so a trailing newline/space pasted into .env does
   not silently corrupt the auth header (a common cause of HTTP 401).
2. A 401/403 handshake rejection raises a clear, actionable ConnectionError
   instead of a raw ``websockets`` ``InvalidStatus`` traceback.
"""

import types

import pytest
import websockets

from app.ai.deepgram_agent import DeepgramAgentClient


class TestApiKeyStripping:
    def test_api_key_whitespace_is_stripped(self) -> None:
        client = DeepgramAgentClient(api_key="  dg_secret_key\n")
        assert client._api_key == "dg_secret_key"

    def test_whitespace_only_key_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DeepgramAgentClient(api_key="   \n")


class _AuthRejectingConnect:
    """Spy that mimics websockets rejecting the handshake with HTTP <status>."""

    def __init__(self, status: int) -> None:
        self._status = status

    async def __call__(self, *args, **kwargs):
        response = types.SimpleNamespace(status_code=self._status)
        raise websockets.exceptions.InvalidStatus(response)


class TestAuthFailureMessage:
    @pytest.mark.asyncio
    async def test_http_401_raises_clear_auth_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.ai.deepgram_agent.websockets.connect",
            _AuthRejectingConnect(401),
        )
        client = DeepgramAgentClient(api_key="dg_bad_key")

        with pytest.raises(ConnectionError, match="authentication failed"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_http_403_raises_clear_auth_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.ai.deepgram_agent.websockets.connect",
            _AuthRejectingConnect(403),
        )
        client = DeepgramAgentClient(api_key="dg_bad_key")

        with pytest.raises(ConnectionError, match="HTTP 403"):
            await client.connect()
