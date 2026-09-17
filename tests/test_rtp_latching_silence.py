"""Symmetric RTP latching, negotiated codec use, and uplink silence on missing RTP."""

import asyncio

import structlog

from app.bridge import AudioAdapter
from app.sip_async.audio_bridge import RTPAudioBridge
from app.sip_async.rtp_session import G711Codec, RTPConfig, RTPPacket, RTPProtocol, RTPSession

SDP_ADDR = ("198.18.0.1", 4000)
REAL_ADDR = ("127.0.0.1", 4000)


def _packet(ssrc: int = 1234, pt: int = 0, seq: int = 1) -> bytes:
    return RTPPacket.build(payload=b"\xff" * 160, seq=seq, timestamp=seq * 160, ssrc=ssrc, pt=pt)


class TestSymmetricRtp:
    async def test_latches_to_observed_source_after_valid_packets(self) -> None:
        session = RTPSession(local_port=40010, remote_addr=SDP_ADDR)
        protocol = RTPProtocol(session)

        protocol.datagram_received(_packet(seq=1), REAL_ADDR)
        assert session.remote_addr == SDP_ADDR  # one packet is not enough
        protocol.datagram_received(_packet(seq=2), REAL_ADDR)

        assert session.latched
        assert session.remote_addr == REAL_ADDR
        assert session.sdp_remote_addr == SDP_ADDR

    async def test_does_not_latch_on_unexpected_payload_type(self) -> None:
        session = RTPSession(local_port=40011, remote_addr=SDP_ADDR)
        protocol = RTPProtocol(session)

        for seq in range(5):
            protocol.datagram_received(_packet(pt=101, seq=seq), REAL_ADDR)  # telephone-event

        assert not session.latched
        assert session.remote_addr == SDP_ADDR
        assert session.rx_queue.empty()

    async def test_latches_once_and_ignores_later_sources(self) -> None:
        session = RTPSession(local_port=40012, remote_addr=SDP_ADDR)
        protocol = RTPProtocol(session)

        protocol.datagram_received(_packet(seq=1), REAL_ADDR)
        protocol.datagram_received(_packet(seq=2), REAL_ADDR)
        for seq in range(3, 10):
            protocol.datagram_received(_packet(ssrc=999, seq=seq), ("10.0.0.9", 6666))

        assert session.remote_addr == REAL_ADDR

    async def test_update_remote_resets_latch_and_changes_codec(self) -> None:
        session = RTPSession(local_port=40013, remote_addr=SDP_ADDR)
        protocol = RTPProtocol(session)
        protocol.datagram_received(_packet(seq=1), REAL_ADDR)
        protocol.datagram_received(_packet(seq=2), REAL_ADDR)

        session.update_remote(("127.0.0.1", 5000), payload_type=8)

        assert not session.latched
        assert session.remote_addr == ("127.0.0.1", 5000)
        assert session.config.payload_type == 8

    async def test_pcma_payload_is_decoded_with_alaw(self) -> None:
        session = RTPSession(
            local_port=40014, remote_addr=REAL_ADDR, config=RTPConfig(payload_type=8)
        )
        protocol = RTPProtocol(session)
        payload = b"\x55" * 160
        protocol.datagram_received(
            RTPPacket.build(payload=payload, seq=1, timestamp=0, ssrc=1, pt=8), REAL_ADDR
        )
        assert session.rx_queue.get_nowait() == G711Codec().decode_pcma(payload)

    async def test_send_loop_uses_latched_address(self) -> None:
        session = RTPSession(local_port=0, remote_addr=SDP_ADDR)
        sent: list[tuple] = []

        class _Transport:
            def sendto(self, data, addr):
                sent.append(addr)

            def close(self):
                pass

        session.transport = _Transport()  # type: ignore[assignment]
        session._running = True
        protocol = RTPProtocol(session)
        protocol.datagram_received(_packet(seq=1), REAL_ADDR)
        protocol.datagram_received(_packet(seq=2), REAL_ADDR)

        task = asyncio.create_task(session._send_loop())
        await asyncio.sleep(0.07)
        session._running = False
        await asyncio.wait_for(task, 1.0)

        assert sent and all(addr == REAL_ADDR for addr in sent)


class TestUplinkSilence:
    async def test_silence_fed_when_no_rtp_arrives(self) -> None:
        adapter = AudioAdapter(uplink_capacity=100, downlink_capacity=10)
        session = RTPSession(local_port=40015, remote_addr=REAL_ADDR)
        bridge = RTPAudioBridge(session, adapter)

        task = asyncio.create_task(bridge.run())
        await asyncio.sleep(0.3)  # no RTP at all
        await bridge.stop()
        await asyncio.wait_for(task, 1.0)

        frames = []
        while True:
            try:
                frames.append(adapter.uplink_stream.receive_nowait())
            except asyncio.QueueEmpty:
                break
        # ~300ms minus the 60ms detection gap at 20ms per frame
        assert 8 <= len(frames) <= 15
        assert all(f == b"\x00" * 320 for f in frames)

    async def test_no_silence_inserted_while_rtp_flows(self) -> None:
        adapter = AudioAdapter(uplink_capacity=100, downlink_capacity=10)
        session = RTPSession(local_port=40016, remote_addr=REAL_ADDR)
        bridge = RTPAudioBridge(session, adapter)
        frame = b"\x10\x00" * 160

        task = asyncio.create_task(bridge.run())
        for _ in range(10):
            session.rx_queue.put_nowait(frame)
            await asyncio.sleep(0.02)
        await bridge.stop()
        await asyncio.wait_for(task, 1.0)

        assert bridge._uplink_silence_frames == 0
        assert bridge._uplink_frames == 10


class TestLatchingRules:
    """Latch preference for the SDP address, stale sources and source filtering."""

    SPOOF = ("127.0.0.1", 4298)

    async def test_sdp_address_source_wins_over_earlier_unknown_source(self) -> None:
        session = RTPSession(local_port=40020, remote_addr=REAL_ADDR)
        now = 100.0
        # Unknown source sends first and latches (NAT rule)
        for i in range(15):
            session.observe_source(self.SPOOF, 777, now=now + i * 0.02)
        assert session.remote_addr == self.SPOOF

        # The SDP address starts sending: it takes the latch immediately
        assert session.observe_source(REAL_ADDR, 1234, now=now + 0.31)
        assert session.remote_addr == REAL_ADDR

        # Later packets from the unknown source are dropped
        assert not session.observe_source(self.SPOOF, 777, now=now + 0.32)
        assert session.remote_addr == REAL_ADDR

    async def test_non_latched_source_audio_is_dropped(self) -> None:
        session = RTPSession(local_port=40021, remote_addr=REAL_ADDR)
        protocol = RTPProtocol(session)
        protocol.datagram_received(_packet(seq=1), REAL_ADDR)
        protocol.datagram_received(_packet(seq=2), REAL_ADDR)
        assert session.rx_queue.qsize() == 2

        for seq in range(3, 40):
            protocol.datagram_received(_packet(ssrc=999, seq=seq), self.SPOOF)
            protocol.datagram_received(_packet(seq=seq), REAL_ADDR)

        assert session.remote_addr == REAL_ADDR
        assert session.rx_queue.qsize() == 39  # only the latched source's packets
        assert session.rejected_source_packets == 37

    async def test_latch_moves_only_after_silence_and_steady_new_source(self) -> None:
        session = RTPSession(local_port=40022, remote_addr=SDP_ADDR)
        new = ("127.0.0.1", 5002)
        t = 10.0
        session.observe_source(REAL_ADDR, 1, now=t)
        session.observe_source(REAL_ADDR, 1, now=t + 0.02)
        assert session.remote_addr == REAL_ADDR

        # Latched source still active: a steady new source does not take over
        for i in range(30):
            session.observe_source(REAL_ADDR, 1, now=t + 0.04 + i * 0.02)
            assert not session.observe_source(new, 2, now=t + 0.05 + i * 0.02)
        assert session.remote_addr == REAL_ADDR

        # Latched source silent for > LATCH_MOVE_SILENCE: new steady source takes over
        t2 = t + 0.64 + RTPSession.LATCH_MOVE_SILENCE
        results = [
            session.observe_source(new, 2, now=t2 + i * 0.02)
            for i in range(RTPSession.LATCH_MOVE_PACKETS)
        ]
        assert results[-1] and not any(results[:-1])
        assert session.remote_addr == new

    async def test_sporadic_packets_after_silence_do_not_move_latch(self) -> None:
        session = RTPSession(local_port=40023, remote_addr=SDP_ADDR)
        session.observe_source(REAL_ADDR, 1, now=1.0)
        session.observe_source(REAL_ADDR, 1, now=1.02)
        # One packet every 0.5s (not steady) long after the latched source stopped
        for i in range(30):
            session.observe_source(self.SPOOF, 3, now=5.0 + i * 0.5)
        assert session.remote_addr == REAL_ADDR

    async def test_old_port_packets_do_not_win_after_reinvite(self) -> None:
        old = ("127.0.0.1", 4140)
        new = ("127.0.0.1", 4142)
        session = RTPSession(local_port=40024, remote_addr=old)
        t = 50.0
        for i in range(5):
            session.observe_source(old, 42, now=t + i * 0.02)
        assert session.remote_addr == old

        session.update_remote(new, payload_type=0)
        # In-flight packets from the old port (same SSRC) arrive first
        for i in range(5):
            assert not session.observe_source(old, 42, now=t + 0.2 + i * 0.02)
        assert session.remote_addr == new
        assert session.observe_source(new, 42, now=t + 0.32)
        assert session.remote_addr == new
        assert not session.observe_source(old, 42, now=t + 0.34)

    async def test_nat_latching_after_reinvite_to_new_port(self) -> None:
        session = RTPSession(local_port=40025, remote_addr=("198.18.0.1", 4200))
        session.observe_source(("127.0.0.1", 4200), 7, now=1.0)
        session.observe_source(("127.0.0.1", 4200), 7, now=1.02)
        assert session.remote_addr == ("127.0.0.1", 4200)

        session.update_remote(("198.18.0.1", 4202), payload_type=0)
        session.observe_source(("127.0.0.1", 4202), 7, now=2.0)
        assert session.observe_source(("127.0.0.1", 4202), 7, now=2.02)
        assert session.remote_addr == ("127.0.0.1", 4202)

    async def test_garbage_datagrams_are_counted_not_logged_as_errors(self) -> None:
        session = RTPSession(local_port=40026, remote_addr=REAL_ADDR)
        protocol = RTPProtocol(session)
        with structlog.testing.capture_logs() as logs:
            protocol.datagram_received(b"\x00\x01garbage", self.SPOOF)
            protocol.datagram_received(b"\x80", self.SPOOF)

        assert session.invalid_packets == 2
        assert session.rx_queue.empty()
        assert [e["log_level"] for e in logs] == ["debug"]  # first one only, at debug


class TestHoldStopsSending:
    @staticmethod
    async def _run_send_loop(session: RTPSession, seconds: float) -> list[tuple]:
        sent: list[tuple] = []

        class _Transport:
            def sendto(self, data, addr):
                sent.append(addr)

            def close(self):
                pass

        session.transport = _Transport()  # type: ignore[assignment]
        session._running = True
        task = asyncio.create_task(session._send_loop())
        await asyncio.sleep(seconds)
        session._running = False
        await asyncio.wait_for(task, 1.0)
        return sent

    async def test_no_rtp_sent_while_send_disabled(self) -> None:
        session = RTPSession(local_port=0, remote_addr=REAL_ADDR)
        session.set_send_enabled(False)
        session.tx_queue.put_nowait(b"\x10\x00" * 160)
        ts_before = session.timestamp

        sent = await self._run_send_loop(session, 0.1)

        assert sent == []
        assert session.tx_queue.empty()  # queued audio discarded
        assert session.timestamp != ts_before  # RTP clock keeps running

    async def test_rtp_resumes_when_send_enabled(self) -> None:
        session = RTPSession(local_port=0, remote_addr=REAL_ADDR)
        session.set_send_enabled(False)
        session.set_send_enabled(True)
        sent = await self._run_send_loop(session, 0.07)
        assert sent
