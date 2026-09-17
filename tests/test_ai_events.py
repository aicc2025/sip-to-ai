"""AI event handling: non-blocking event queues and GA transcript event names."""

import asyncio

import pytest

from app.ai.deepgram_agent import DeepgramAgentClient
from app.ai.duplex_base import AiEvent, AiEventType, is_barge_in_event
from app.ai.gemini_live import GeminiLiveClient
from app.ai.grok_voice import GrokVoiceClient
from app.ai.openai_realtime import OpenAIRealtimeClient


def _clients() -> list:
    return [
        OpenAIRealtimeClient(api_key="sk-test"),
        GrokVoiceClient(api_key="xai-test"),
        GeminiLiveClient(api_key="gm-test"),
        DeepgramAgentClient(api_key="dg-test"),
    ]


class TestNonBlockingEventQueue:
    @pytest.mark.parametrize("client", _clients(), ids=lambda c: type(c).__name__)
    async def test_emit_event_never_blocks_and_drops_oldest(self, client) -> None:
        capacity = client._event_queue.maxsize
        for i in range(capacity + 5):
            client._emit_event(AiEvent(type=AiEventType.TRANSCRIPT_PARTIAL, data={"i": i}))

        assert client._event_queue.qsize() == capacity
        first = client._event_queue.get_nowait()
        assert first.data == {"i": 5}  # 5 oldest events were dropped
        assert client._events_dropped == 5

    async def test_openai_process_message_does_not_stall_without_consumer(self) -> None:
        client = OpenAIRealtimeClient(api_key="sk-test")

        async def feed() -> None:
            for _ in range(500):
                await client._process_message({"type": "input_audio_buffer.speech_started"})

        # With the old blocking put this hangs after 100 events
        await asyncio.wait_for(feed(), timeout=2.0)
        assert client._event_queue.full()

    async def test_grok_process_message_does_not_stall_without_consumer(self) -> None:
        client = GrokVoiceClient(api_key="xai-test")

        async def feed() -> None:
            for _ in range(500):
                await client._process_message({"type": "input_audio_buffer.speech_stopped"})

        await asyncio.wait_for(feed(), timeout=2.0)
        assert client._event_queue.full()

    async def test_deepgram_json_messages_do_not_stall_without_consumer(self) -> None:
        client = DeepgramAgentClient(api_key="dg-test")

        async def feed() -> None:
            for _ in range(500):
                await client._handle_json_message('{"type": "UserStartedSpeaking"}')

        await asyncio.wait_for(feed(), timeout=2.0)
        assert client._event_queue.full()


class TestBargeInEvents:
    async def test_openai_speech_started_is_barge_in(self) -> None:
        client = OpenAIRealtimeClient(api_key="sk-test")
        await client._process_message({"type": "input_audio_buffer.speech_started"})
        assert is_barge_in_event(client._event_queue.get_nowait())

    async def test_speech_stopped_is_not_barge_in(self) -> None:
        client = OpenAIRealtimeClient(api_key="sk-test")
        await client._process_message({"type": "input_audio_buffer.speech_stopped"})
        assert not is_barge_in_event(client._event_queue.get_nowait())

    async def test_deepgram_user_started_speaking_is_barge_in(self) -> None:
        client = DeepgramAgentClient(api_key="dg-test")
        await client._handle_json_message('{"type": "UserStartedSpeaking"}')
        assert is_barge_in_event(client._event_queue.get_nowait())

    async def test_gemini_interrupted_emits_barge_in(self) -> None:
        client = GeminiLiveClient(api_key="gm-test")
        await client._process_message({"serverContent": {"interrupted": True}})
        assert is_barge_in_event(client._event_queue.get_nowait())

    async def test_clear_audio_queue_keeps_end_marker_when_disconnected(self) -> None:
        client = OpenAIRealtimeClient(api_key="sk-test")
        client._audio_queue.put_nowait(b"")
        assert client.clear_audio_queue() == 0  # not connected: marker kept
        client._connected = True
        client._audio_queue.put_nowait(b"\x01\x00")
        assert client.clear_audio_queue() == 2
        assert client._audio_queue.empty()


class TestOpenAITranscriptEvents:
    @pytest.fixture
    def client_and_logs(self, monkeypatch):
        client = OpenAIRealtimeClient(api_key="sk-test")
        messages: list[str] = []

        class _Logger:
            def info(self, msg, *args, **kwargs):
                messages.append(msg)

            def debug(self, msg, *args, **kwargs):
                messages.append(f"DEBUG {msg}")

            warning = error = info

        monkeypatch.setattr(client, "_logger", _Logger())
        return client, messages

    @pytest.mark.parametrize("prefix", ["response.output_audio_transcript", "response.audio_transcript"])
    async def test_ai_transcript_delta_and_done_logged_at_info(self, client_and_logs, prefix) -> None:
        client, messages = client_and_logs
        await client._process_message({"type": f"{prefix}.delta", "delta": "Hel"})
        await client._process_message({"type": f"{prefix}.done", "transcript": "Hello"})

        assert "🤖 AI transcript delta: Hel" in messages
        assert "✅ AI transcript done: Hello" in messages
        assert not any(m.startswith("DEBUG Unhandled") for m in messages)

    async def test_session_updated_reads_ga_transcription_field(self, client_and_logs) -> None:
        client, messages = client_and_logs
        await client._process_message({
            "type": "session.updated",
            "session": {"audio": {"input": {"transcription": {"model": "gpt-live-transcribe"}}}},
        })
        assert "Session updated - input transcription: {'model': 'gpt-live-transcribe'}" in messages

    async def test_session_updated_falls_back_to_beta_field(self, client_and_logs) -> None:
        client, messages = client_and_logs
        await client._process_message({
            "type": "session.updated",
            "session": {"input_audio_transcription": {"model": "gpt-live-transcribe"}},
        })
        assert "Session updated - input transcription: {'model': 'gpt-live-transcribe'}" in messages
