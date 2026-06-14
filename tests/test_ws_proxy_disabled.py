"""All AI WebSocket clients must disable proxy auto-detection.

websockets>=15 defaults to ``proxy=True``, which reads a proxy from the
environment (HTTPS_PROXY / ALL_PROXY / socks_proxy). When that is a SOCKS proxy
and ``python-socks`` is not installed, ``websockets.connect`` raises
``ImportError: connecting through a SOCKS proxy requires python-socks`` and the
AI client never connects — killing all audio (issue #6 in deployments behind a
proxy). Passing ``proxy=None`` disables this and connects directly.
"""

import pytest


class _StopConnect(Exception):
    """Raised by the spy to short-circuit connect() after capturing kwargs."""


class _ConnectSpy:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        raise _StopConnect()


def _make_client(module_path: str):
    if module_path == "app.ai.deepgram_agent":
        from app.ai.deepgram_agent import DeepgramAgentClient
        return DeepgramAgentClient(api_key="dg")
    if module_path == "app.ai.sixtydb_tts":
        from app.ai.sixtydb_tts import SixtyDBTTSClient
        return SixtyDBTTSClient(api_key="k")
    if module_path == "app.ai.openai_realtime":
        from app.ai.openai_realtime import OpenAIRealtimeClient
        return OpenAIRealtimeClient(api_key="k")
    if module_path == "app.ai.gemini_live":
        from app.ai.gemini_live import GeminiLiveClient
        return GeminiLiveClient(api_key="k")
    if module_path == "app.ai.grok_voice":
        from app.ai.grok_voice import GrokVoiceClient
        return GrokVoiceClient(api_key="k")
    raise AssertionError(module_path)


@pytest.mark.parametrize(
    "module_path",
    [
        "app.ai.deepgram_agent",
        "app.ai.sixtydb_tts",
        "app.ai.openai_realtime",
        "app.ai.gemini_live",
        "app.ai.grok_voice",
    ],
)
@pytest.mark.asyncio
async def test_connect_passes_proxy_none(module_path, monkeypatch) -> None:
    spy = _ConnectSpy()
    monkeypatch.setattr(f"{module_path}.websockets.connect", spy)

    client = _make_client(module_path)
    with pytest.raises(BaseException):
        await client.connect()

    assert spy.kwargs is not None, "websockets.connect was never called"
    _MISSING = object()
    proxy_arg = spy.kwargs.get("proxy", _MISSING)
    assert proxy_arg is None, (
        f"{module_path} must call websockets.connect(proxy=None) to avoid "
        f"SOCKS-proxy auto-detection, got proxy="
        f"{'<not passed>' if proxy_arg is _MISSING else repr(proxy_arg)}"
    )
