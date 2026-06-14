"""Bidirectional audio-path tests for the SIP <-> AI bridge (issue #6).

These tests exercise the media path that the issue reports as broken:

  inbound:  caller PCM16  -> DeepgramAgentClient.send_pcm16_8k -> mu-law -> WS
  outbound: AI mu-law      -> PCM16 -> AudioAdapter -> 320-byte downlink frames

They use an in-memory FakeWebSocket so both directions are verified with no
network and no real SIP phone.
"""

import time

import pytest

from app.ai.deepgram_agent import DeepgramAgentClient
from app.bridge.audio_adapter import AudioAdapter
from app.utils.codec import Codec
from app.utils.constants import AudioConstants
from tests.fake_ws import FakeWebSocket, patch_connect

PCM16_FRAME = (b"\x10\x00") * 160  # 320 bytes, non-silent PCM16 @ 8kHz/20ms


def _ready_client(**kwargs) -> tuple[DeepgramAgentClient, FakeWebSocket]:
    """A DeepgramAgentClient wired to a FakeWebSocket with settings applied."""
    client = DeepgramAgentClient(api_key="dg", **kwargs)
    fake = FakeWebSocket()
    client._ws = fake  # type: ignore[assignment]
    client._connected = True
    client._settings_ready.set()
    return client, fake


# ---------------------------------------------------------------------------
# Inbound: caller audio must reach the AI (and never be dropped silently)
# ---------------------------------------------------------------------------
class TestInboundCallerAudio:
    @pytest.mark.asyncio
    async def test_caller_audio_forwarded_after_first_agent_audio(self) -> None:
        """Once the agent has spoken, caller PCM16 is mu-law encoded to the WS."""
        client, fake = _ready_client()

        # Agent audio arrived -> gate opens (also marks agent speaking).
        await client._enqueue_agent_audio(b"\xff" * 160)
        assert client._received_first_audio is True

        # Past the barge-in tail, agent not speaking.
        client._agent_speaking = False
        client._last_agent_audio_time = 0.0

        await client.send_pcm16_8k(PCM16_FRAME)

        assert fake.sent_binary == [Codec.pcm16_to_ulaw(PCM16_FRAME)]

    @pytest.mark.asyncio
    async def test_caller_audio_gate_opens_when_no_agent_audio_arrives(self) -> None:
        """Self-heal: if no agent audio ever arrives (e.g. greeting TTS failed),
        the inbound gate must open after a bounded time instead of deadlocking.
        This is the core issue-#6 fix: caller audio must not be dropped forever.
        """
        client, fake = _ready_client()
        assert client._received_first_audio is False

        # Pretend connect happened long ago and no agent audio ever came.
        client._connect_time = time.monotonic() - 999.0
        client._agent_speaking = False
        client._last_agent_audio_time = 0.0

        await client.send_pcm16_8k(PCM16_FRAME)

        assert fake.sent_binary == [Codec.pcm16_to_ulaw(PCM16_FRAME)]
        assert client._received_first_audio is True

    @pytest.mark.asyncio
    async def test_caller_audio_gated_briefly_before_first_audio(self) -> None:
        """Intended behavior preserved: right after connect, before any greeting
        audio, caller audio is still held back (don't trip VAD during greeting)."""
        client, fake = _ready_client()
        client._connect_time = time.monotonic()  # just connected
        client._agent_speaking = False
        client._last_agent_audio_time = 0.0

        await client.send_pcm16_8k(PCM16_FRAME)

        assert fake.sent_binary == []  # held, not forwarded yet

    @pytest.mark.asyncio
    async def test_drops_are_counted_for_diagnostics(self) -> None:
        """Every dropped caller frame must be counted so failures aren't silent."""
        client, _ = _ready_client()
        client._connect_time = time.monotonic()  # within gate window
        client._agent_speaking = False
        client._last_agent_audio_time = 0.0

        await client.send_pcm16_8k(PCM16_FRAME)

        assert client._caller_frames_dropped >= 1

    @pytest.mark.asyncio
    async def test_forwarded_frames_are_counted(self) -> None:
        client, _ = _ready_client()
        await client._enqueue_agent_audio(b"\xff" * 160)
        client._agent_speaking = False
        client._last_agent_audio_time = 0.0

        await client.send_pcm16_8k(PCM16_FRAME)

        assert client._caller_frames_forwarded >= 1


# ---------------------------------------------------------------------------
# Outbound: AI audio must become 320-byte PCM16 frames for the caller
# ---------------------------------------------------------------------------
class TestOutboundAiAudio:
    @pytest.mark.asyncio
    async def test_deepgram_mulaw_becomes_pcm16_frames(self) -> None:
        """Deepgram binary mu-law -> PCM16 chunk -> 320-byte downlink frames."""
        client, _ = _ready_client()
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=10)

        # 160 bytes mu-law = 20ms -> 320 bytes PCM16 = exactly one downlink frame.
        await client._handle_binary_audio(b"\x7f" * 160)
        chunk = client._audio_queue.get_nowait()
        assert chunk == Codec.ulaw_to_pcm16(b"\x7f" * 160)

        await adapter.feed_ai_audio(chunk)
        frame = await adapter.get_downlink_audio()
        assert len(frame) == AudioConstants.PCM16_FRAME_SIZE  # 320 bytes

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sixtydb_audio_reaches_bridge_as_pcm16(self) -> None:
        """60db mu-law (via on_audio) is queued as PCM16 for the bridge."""
        client, _ = _ready_client(
            speak_provider="60db",
            sixtydb_api_key="sk_live_test",
        )
        await client._enqueue_agent_audio(b"\xff\x7f\x00\x10")
        queued = client._audio_queue.get_nowait()
        assert queued == Codec.ulaw_to_pcm16(b"\xff\x7f\x00\x10")


# ---------------------------------------------------------------------------
# AudioAdapter is the logger/forward point for inbound caller audio
# ---------------------------------------------------------------------------
class TestAudioAdapterBothDirections:
    @pytest.mark.asyncio
    async def test_inbound_frame_logged_and_forwarded(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=10)

        adapter.on_rx_pcm16_8k(PCM16_FRAME)

        # Forwarded to the uplink (SIP -> AI) and counted.
        assert adapter.get_stats()["frames_received"] == 1
        got = await adapter.get_uplink_audio()
        assert got == PCM16_FRAME

        await adapter.close()

    @pytest.mark.asyncio
    async def test_outbound_chunk_split_into_frames(self) -> None:
        adapter = AudioAdapter(uplink_capacity=10, downlink_capacity=10)

        # 800 bytes -> 2 full 320-byte frames + 160 bytes pending.
        await adapter.feed_ai_audio(b"\x01\x02" * 400)

        f1 = await adapter.get_downlink_audio()
        f2 = await adapter.get_downlink_audio()
        assert len(f1) == 320 and len(f2) == 320

        await adapter.close()


# ---------------------------------------------------------------------------
# Full connect() flow over a FakeWebSocket (default Deepgram path)
# ---------------------------------------------------------------------------
class TestConnectOverFakeWebSocket:
    @pytest.mark.asyncio
    async def test_no_greeting_opens_gate_on_connect(self, monkeypatch) -> None:
        """With no greeting, nothing produces 'first audio', so the gate must be
        opened at connect — otherwise caller audio deadlocks (issue #6)."""
        fake = FakeWebSocket(
            responder=lambda m: (['{"type": "Welcome"}', '{"type": "SettingsApplied"}']
                                 if isinstance(m, str) and '"Settings"' in m else [])
        )
        patch_connect(monkeypatch, "app.ai.deepgram_agent", fake)

        client = DeepgramAgentClient(api_key="dg", greeting=None)
        await client.connect()

        assert client._received_first_audio is True
        await client.close()
