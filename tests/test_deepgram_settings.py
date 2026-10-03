"""Deepgram Voice Agent Settings payload and barge-in gating (issue #17).

- Flux listen/speak models carry ``version: "v2"``; legacy models ``"v1"``.
- ``language`` is never sent on speak, and not on Flux listen (live API rejects
  it with UNPARSABLE_CLIENT_MESSAGE); legacy listen keeps it.
- Flux forwards caller frames while the agent speaks (server-side barge-in);
  legacy models and SPEAK_PROVIDER=60db stay half-duplex.
"""

import json
import time
from typing import Any, Dict

import pytest

from app.ai.deepgram_agent import DeepgramAgentClient, build_settings
from app.ai.duplex_base import AiEventType, is_barge_in_event
from tests.fake_ws import FakeWebSocket, patch_connect

PCM_FRAME = b"\x10\x00" * 160  # 20 ms PCM16 @ 8 kHz
LEGACY = dict(
    listen_model="nova-2", speak_model="aura-asteria-en", llm_model="gpt-4o-mini"
)


def _settings(**overrides: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        audio_format="mulaw",
        sample_rate=8000,
        listen_model="flux-general-en",
        speak_model="flux-kit-en",
        llm_model="gpt-4o-mini",
        instructions="Be brief.",
    )
    kwargs.update(overrides)
    return build_settings(**kwargs)


class TestSettingsPayload:
    def test_flux_defaults(self) -> None:
        s = _settings(greeting="Hi")
        agent = s["agent"]
        assert agent["listen"]["provider"] == {
            "type": "deepgram", "version": "v2", "model": "flux-general-en",
        }
        assert agent["think"] == {
            "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
            "prompt": "Be brief.",
        }
        assert agent["speak"]["provider"] == {
            "type": "deepgram", "version": "v2", "model": "flux-kit-en",
        }
        assert agent["greeting"] == "Hi"
        assert s["audio"]["input"] == {"encoding": "mulaw", "sample_rate": 8000}
        assert s["audio"]["output"]["container"] == "none"

    def test_legacy_models_use_v1(self) -> None:
        agent = _settings(**LEGACY)["agent"]
        assert agent["listen"]["provider"] == {
            "type": "deepgram", "version": "v1", "model": "nova-2", "language": "en",
        }
        assert agent["speak"]["provider"] == {
            "type": "deepgram", "version": "v1", "model": "aura-asteria-en",
        }
        assert agent["think"]["provider"]["model"] == "gpt-4o-mini"

    def test_mixed_flux_listen_legacy_speak(self) -> None:
        agent = _settings(speak_model="aura-2-thalia-en")["agent"]
        assert agent["listen"]["provider"]["version"] == "v2"
        assert agent["speak"]["provider"]["version"] == "v1"

    @pytest.mark.parametrize("overrides", [{}, LEGACY, {"speak_provider": "60db"}])
    def test_speak_never_sends_language(self, overrides: Dict[str, Any]) -> None:
        assert "language" not in _settings(**overrides)["agent"]["speak"]["provider"]

    def test_flux_listen_has_no_language(self) -> None:
        assert "language" not in _settings()["agent"]["listen"]["provider"]

    def test_60db_omits_deepgram_greeting(self) -> None:
        agent = _settings(speak_provider="60db", greeting="Hi")["agent"]
        assert "greeting" not in agent
        # Deepgram still needs a valid speak provider block
        assert agent["speak"]["provider"]["model"] == "flux-kit-en"

    def test_optional_tuning_only_sent_when_set(self) -> None:
        plain = _settings()["agent"]
        assert set(plain["listen"]["provider"]) == {"type", "version", "model"}
        assert "speed" not in plain["speak"]["provider"]

        tuned = _settings(
            eot_threshold=0.8, eager_eot_threshold=0.5, eot_timeout_ms=4000,
            speak_speed=1.1,
        )["agent"]
        listen = tuned["listen"]["provider"]
        assert (listen["eot_threshold"], listen["eager_eot_threshold"],
                listen["eot_timeout_ms"]) == (0.8, 0.5, 4000)
        assert tuned["speak"]["provider"]["speed"] == 1.1

    def test_eot_settings_ignored_for_legacy_listen(self) -> None:
        listen = _settings(**LEGACY, eot_threshold=0.8)["agent"]["listen"]["provider"]
        assert "eot_threshold" not in listen

    async def test_client_sends_built_settings(self, monkeypatch) -> None:
        fake = FakeWebSocket()
        patch_connect(monkeypatch, "app.ai.deepgram_agent", fake)
        client = DeepgramAgentClient(api_key="dg", instructions="Be brief.")
        await client.connect()
        try:
            sent = json.loads(fake.sent_text[0])
            assert sent == _settings()
        finally:
            await client.close()


def _gated_client(**kwargs: Any) -> DeepgramAgentClient:
    """Client past the Settings/first-audio gates, with the agent mid-speech."""
    client = DeepgramAgentClient(api_key="dg", **kwargs)
    client._ws = FakeWebSocket()  # type: ignore[assignment]
    client._settings_ready.set()
    client._received_first_audio = True
    client._agent_speaking = True
    client._last_agent_audio_time = time.time()
    client._greeting_pending = False  # greeting phase covered separately below
    return client


class TestBargeInGating:
    async def test_flux_forwards_caller_frames_while_agent_speaks(self) -> None:
        client = _gated_client()
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_forwarded == 1
        assert client._caller_frames_dropped == 0
        assert len(client._ws.sent_binary) == 1  # type: ignore[union-attr]

    async def test_legacy_drops_caller_frames_while_agent_speaks(self) -> None:
        client = _gated_client(**LEGACY)
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_forwarded == 0
        assert client._caller_frames_dropped == 1
        assert client._ws.sent_binary == []  # type: ignore[union-attr]

    async def test_legacy_drops_during_post_audio_tail_window(self) -> None:
        client = _gated_client(**LEGACY)
        client._agent_speaking = False  # AgentAudioDone, but <2 s ago
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_dropped == 1

    async def test_60db_stays_half_duplex_even_with_flux_listen(self) -> None:
        client = _gated_client(speak_provider="60db", sixtydb_api_key="k")
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_dropped == 1
        assert client._caller_frames_forwarded == 0

    async def test_user_started_speaking_stops_agent_and_signals_barge_in(self) -> None:
        client = _gated_client()
        client._connected = True
        await client._handle_json_message(json.dumps({"type": "UserStartedSpeaking"}))
        assert client._agent_speaking is False
        event = client._event_queue.get_nowait()
        assert event.type == AiEventType.TRANSCRIPT_PARTIAL
        assert is_barge_in_event(event)

    async def test_barge_in_flush_drops_queued_agent_audio(self) -> None:
        client = _gated_client()
        client._connected = True
        await client._enqueue_agent_audio(b"\xff" * 960)
        await client._enqueue_agent_audio(b"\xff" * 960)
        assert client.clear_audio_queue() == 2
        assert client._audio_queue.empty()

    async def test_barge_in_false_is_half_duplex_with_flux(self) -> None:
        client = _gated_client(barge_in=False)
        assert client._half_duplex is True
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_dropped == 1
        assert client._caller_frames_forwarded == 0

    def test_half_duplex_property(self) -> None:
        assert DeepgramAgentClient(api_key="dg")._half_duplex is False
        assert DeepgramAgentClient(api_key="dg", barge_in=False)._half_duplex is True


def _greeting_client(**kwargs: Any) -> DeepgramAgentClient:
    client = DeepgramAgentClient(api_key="dg", **kwargs)
    client._ws = FakeWebSocket()  # type: ignore[assignment]
    client._connected = True
    client._settings_ready.set()
    client._received_first_audio = True  # first greeting audio already arrived
    client._agent_speaking = True
    client._last_agent_audio_time = time.time()
    return client


class TestGreetingGuard:
    async def test_drops_during_greeting_with_flux(self) -> None:
        client = _greeting_client(greeting="Hello there")
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_dropped == 1
        assert client._caller_frames_forwarded == 0

    async def test_user_started_speaking_does_not_end_guard(self) -> None:
        client = _greeting_client(greeting="Hello there")
        await client._handle_json_message(json.dumps({"type": "UserStartedSpeaking"}))
        assert client._agent_speaking is False
        client._last_agent_audio_time = time.time() - 10  # even long after audio
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_dropped == 1
        assert client._greeting_pending is True

    async def test_forwards_after_first_agent_audio_done_and_tail(self) -> None:
        client = _greeting_client(greeting="Hello there")
        await client._handle_json_message(json.dumps({"type": "AgentAudioDone"}))
        assert client._greeting_pending is False
        await client.send_pcm16_8k(PCM_FRAME)  # inside the 2 s tail
        assert client._caller_frames_dropped == 1
        client._greeting_guard_until = time.time() - 0.1  # tail elapsed
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_forwarded == 1
        # Later agent speech is full-duplex (no new guard)
        client._agent_speaking = True
        client._last_agent_audio_time = time.time()
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_forwarded == 2

    async def test_no_greeting_no_guard(self) -> None:
        client = _greeting_client()
        assert client._greeting_pending is False
        await client.send_pcm16_8k(PCM_FRAME)
        assert client._caller_frames_forwarded == 1

    def test_60db_has_no_deepgram_greeting_guard(self) -> None:
        client = DeepgramAgentClient(
            api_key="dg", greeting="Hi", speak_provider="60db", sixtydb_api_key="k"
        )
        assert client._greeting_pending is False


class TestBargeInConfig:
    def test_default_true(self, monkeypatch) -> None:
        from app.config import Config

        monkeypatch.delenv("DEEPGRAM_BARGE_IN", raising=False)
        cfg = Config()
        assert cfg.ai.deepgram_barge_in is True
        assert cfg.ai.deepgram_llm_model == "gpt-4o-mini"

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "No"])
    def test_false_values(self, monkeypatch, value: str) -> None:
        from app.config import Config

        monkeypatch.setenv("DEEPGRAM_BARGE_IN", value)
        assert Config().ai.deepgram_barge_in is False

    @pytest.mark.parametrize("value", ["true", "True", "1", "YES"])
    def test_true_values(self, monkeypatch, value: str) -> None:
        from app.config import Config

        monkeypatch.setenv("DEEPGRAM_BARGE_IN", value)
        assert Config().ai.deepgram_barge_in is True

    def test_invalid_raises(self, monkeypatch) -> None:
        from app.config import Config

        monkeypatch.setenv("DEEPGRAM_BARGE_IN", "maybe")
        with pytest.raises(ValueError, match="DEEPGRAM_BARGE_IN"):
            Config()
