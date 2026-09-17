"""SIP call lifecycle: re-INVITE, retransmissions, single-codec answer and teardown."""

import asyncio

from app.bridge import AudioAdapter, CallSession
from app.sip_async.async_call import AsyncCall
from app.sip_async.async_sip_server import AsyncSIPServer
from app.sip_async.sdp import get_answer_direction, parse_sdp
from app.sip_async.sip_protocol import SIPMessage
from tests.mock_ai_client import MockDuplexClient

CALLER = ("127.0.0.1", 5091)
CALL_ID = "call-1234@test"

OFFER = (
    "v=0\r\no=- 1 1 IN IP4 198.18.0.1\r\ns=pjmedia\r\nc=IN IP4 198.18.0.1\r\nt=0 0\r\n"
    "m=audio 4000 RTP/AVP 96 3 0 8 9 121\r\na=rtpmap:96 speex/16000\r\na=rtpmap:3 GSM/8000\r\n"
    "a=rtpmap:0 PCMU/8000\r\na=rtpmap:8 PCMA/8000\r\na=rtpmap:9 G722/8000\r\n"
    "a=rtpmap:121 telephone-event/8000\r\na=sendrecv\r\n"
)


def _invite(cseq: int = 1, to_tag: str = "", body: str = OFFER, call_id: str = CALL_ID) -> SIPMessage:
    to = "<sip:test@127.0.0.1:6060>" + (f";tag={to_tag}" if to_tag else "")
    raw = (
        "INVITE sip:test@127.0.0.1:6060 SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP 127.0.0.1:5091;rport;branch=z9hG4bK{cseq}\r\n"
        "From: <sip:caller@127.0.0.1>;tag=fromtag\r\n"
        f"To: {to}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq} INVITE\r\n"
        "Contact: <sip:caller@127.0.0.1:5091>\r\n"
        "Content-Type: application/sdp\r\n"
        f"Content-Length: {len(body)}\r\n\r\n{body}"
    )
    return SIPMessage(raw=raw.encode(), remote_addr=CALLER)


def _bye(to_tag: str, call_id: str = CALL_ID) -> SIPMessage:
    raw = (
        "BYE sip:127.0.0.1:6060 SIP/2.0\r\n"
        "Via: SIP/2.0/UDP 127.0.0.1:5091;rport;branch=z9hG4bKbye\r\n"
        "From: <sip:caller@127.0.0.1>;tag=fromtag\r\n"
        f"To: <sip:test@127.0.0.1:6060>;tag={to_tag}\r\n"
        f"Call-ID: {call_id}\r\nCSeq: 10 BYE\r\nContent-Length: 0\r\n\r\n"
    )
    return SIPMessage(raw=raw.encode(), remote_addr=CALLER)


class _RecordingServer(AsyncSIPServer):
    """SIP server that records outgoing messages instead of using a socket."""

    def __init__(self, call_callback=None) -> None:
        super().__init__(host="127.0.0.1", port=6060, call_callback=call_callback)
        self.sent: list[str] = []

    async def send_message(self, data: bytes, addr: tuple) -> None:
        self.sent.append(data.decode())


def _sdp_of(message: str) -> str:
    return message.split("\r\n\r\n", 1)[1]


def _m_line(message: str) -> str:
    return next(line for line in _sdp_of(message).split("\r\n") if line.startswith("m="))


def _to_tag(message: str) -> str:
    to_line = next(line for line in message.split("\r\n") if line.startswith("To:"))
    return to_line.split("tag=")[1]


def _responses(server: "_RecordingServer", status: str) -> list[str]:
    return [msg for msg in server.sent if msg.startswith(f"SIP/2.0 {status}")]


async def _invite_and_answer(server: "_RecordingServer", invite: SIPMessage | None = None) -> AsyncCall:
    """Deliver an INVITE to a server without call callback and answer it (200 OK)."""
    invite = invite or _invite()
    await server.handle_message(invite, CALLER)
    call = server.active_calls[invite.headers["Call-ID"]]
    await call.accept()
    return call


async def _wait_until(predicate, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class TestInviteHandling:
    async def test_invite_gets_trying_and_ringing_before_answer(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(), CALLER)

        assert [msg.split("\r\n", 1)[0] for msg in server.sent] == [
            "SIP/2.0 100 Trying",
            "SIP/2.0 180 Ringing",
        ]
        assert "tag=" not in next(line for line in server.sent[0].split("\r\n") if line.startswith("To:"))
        assert _to_tag(server.sent[1]) == server.active_calls[CALL_ID].dialog.local_tag
        assert not _responses(server, "200")

    async def test_answer_contains_single_negotiated_codec(self) -> None:
        server = _RecordingServer()
        call = await _invite_and_answer(server)

        ok = _responses(server, "200 OK")
        assert len(ok) == 1
        assert _to_tag(ok[0]) == _to_tag(server.sent[1])  # same tag as 180
        assert _m_line(ok[0]) == f"m=audio {call.local_rtp_port} RTP/AVP 0"
        assert "a=rtpmap:8" not in ok[0]
        assert "telephone-event" not in ok[0]

    async def test_pcma_only_offer_is_answered_with_pcma(self) -> None:
        server = _RecordingServer()
        body = OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 8 121")
        await _invite_and_answer(server, _invite(body=body))

        assert _m_line(_responses(server, "200 OK")[0]).endswith("RTP/AVP 8")
        assert server.active_calls[CALL_ID].payload_type == 8

    async def test_codec_follows_offer_order(self) -> None:
        server = _RecordingServer()
        body = OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 8 0 101")
        await _invite_and_answer(server, _invite(body=body))

        assert _m_line(_responses(server, "200 OK")[0]).endswith("RTP/AVP 8")
        assert server.active_calls[CALL_ID].payload_type == 8

    async def test_offer_without_g711_is_rejected_with_488(self) -> None:
        server = _RecordingServer()
        body = OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 96 9")
        await server.handle_message(_invite(body=body), CALLER)

        assert server.sent[0].startswith("SIP/2.0 488")
        assert "tag=" in next(line for line in server.sent[0].split("\r\n") if line.startswith("To:"))
        assert server.active_calls == {}
        assert server._allocated_ports == set()

    async def test_retransmitted_initial_invite_does_not_create_second_call(self) -> None:
        callbacks: list[AsyncCall] = []
        server = _RecordingServer(call_callback=lambda call: callbacks.append(call))
        server._run_call_callback = lambda call: asyncio.sleep(0)  # type: ignore[method-assign]

        await server.handle_message(_invite(), CALLER)
        await server.handle_message(_invite(), CALLER)

        assert len(server.active_calls) == 1
        assert len(server._allocated_ports) == 1
        assert len(server.sent) == 3
        assert server.sent[2] == server.sent[1]  # 180 Ringing resent while connecting

        await server.active_calls[CALL_ID].accept()
        await server.handle_message(_invite(), CALLER)
        ok = _responses(server, "200 OK")
        assert len(ok) == 2 and ok[0] == ok[1]  # identical 200 OK resent after answer

    async def test_concurrent_retransmission_while_call_is_being_set_up(self) -> None:
        server = _RecordingServer()
        await asyncio.gather(
            server.handle_message(_invite(), CALLER),
            server.handle_message(_invite(), CALLER),
        )
        assert len(server.active_calls) == 1
        assert len(server._allocated_ports) == 1

    async def test_reinvite_reuses_call_port_and_updates_remote(self) -> None:
        server = _RecordingServer()
        call = await _invite_and_answer(server)
        first_ok = _responses(server, "200 OK")[0]
        to_tag = _to_tag(first_ok)

        call.rtp_session = None  # not set up in this test
        reoffer = (
            OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 0")
            .replace("c=IN IP4 198.18.0.1", "c=IN IP4 127.0.0.1")
            .replace("a=sendrecv", "a=sendonly")
        )
        await server.handle_message(_invite(cseq=2, to_tag=to_tag, body=reoffer), CALLER)

        assert len(server.active_calls) == 1
        assert server.active_calls[CALL_ID] is call
        assert len(server._allocated_ports) == 1
        reinvite_ok = server.sent[-1]
        assert reinvite_ok.startswith("SIP/2.0 200 OK")
        assert "CSeq: 2 INVITE" in reinvite_ok
        assert _to_tag(reinvite_ok) == to_tag
        assert _m_line(reinvite_ok) == _m_line(first_ok)
        assert "a=recvonly" in reinvite_ok  # answer to a hold (sendonly) offer
        assert call.remote_rtp_addr == ("127.0.0.1", 4000)

        # o= session id stays, version increases after a change
        o_first = next(l for l in _sdp_of(first_ok).split("\r\n") if l.startswith("o="))
        o_second = next(l for l in _sdp_of(reinvite_ok).split("\r\n") if l.startswith("o="))
        assert o_first.split()[1] == o_second.split()[1]
        assert int(o_second.split()[2]) == int(o_first.split()[2]) + 1

    async def test_reinvite_updates_running_rtp_session(self) -> None:
        server = _RecordingServer()
        call = await _invite_and_answer(server)
        await call.setup(AudioAdapter(), CallSession(AudioAdapter(), MockDuplexClient(sample_rate=8000)))
        to_tag = _to_tag(server.sent[-1])

        reoffer = OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 8").replace(
            "m=audio 4000", "m=audio 4002"
        )
        await server.handle_message(_invite(cseq=2, to_tag=to_tag, body=reoffer), CALLER)

        assert call.rtp_session.remote_addr == ("198.18.0.1", 4002)
        assert call.rtp_session.config.payload_type == 8

    async def test_reinvite_for_unknown_call_gets_481(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(to_tag="stale"), CALLER)

        assert server.sent[0].startswith("SIP/2.0 481")
        assert server.active_calls == {}

    def test_answer_direction_mapping(self) -> None:
        assert get_answer_direction(parse_sdp(OFFER)) == "sendrecv"
        assert get_answer_direction(parse_sdp(OFFER.replace("a=sendrecv", "a=sendonly"))) == "recvonly"
        assert get_answer_direction(parse_sdp(OFFER.replace("a=sendrecv", "a=inactive"))) == "inactive"


class TestCallTeardown:
    @staticmethod
    def _server_with_mock_ai() -> tuple[_RecordingServer, dict]:
        created: dict = {}

        async def on_call(call: AsyncCall) -> None:
            adapter = AudioAdapter(uplink_capacity=100, downlink_capacity=100)
            client = MockDuplexClient(sample_rate=8000)
            created["client"] = client
            await call.setup(adapter, CallSession(audio_adapter=adapter, ai_client=client))

        return _RecordingServer(call_callback=on_call), created

    async def test_bye_stops_bridge_ai_rtp_and_removes_call(self) -> None:
        server, created = self._server_with_mock_ai()
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await _wait_until(lambda: call._running and len(call.audio_bridge._tasks) == 2
                          and created["client"].is_connected)
        bridge_tasks = list(call.audio_bridge._tasks)
        to_tag = _to_tag(_responses(server, "200 OK")[0])

        await server.handle_message(_bye(to_tag), CALLER)

        assert server.sent[-1].startswith("SIP/2.0 200 OK")
        assert all(task.done() for task in bridge_tasks)
        assert not created["client"].is_connected
        assert call.rtp_session.transport is None
        assert CALL_ID not in server.active_calls
        assert server._allocated_ports == set()
        assert call._finished.is_set()
        assert not any(msg.startswith("BYE ") for msg in server.sent)  # caller hung up

    async def test_ai_disconnect_ends_call_and_sends_bye(self) -> None:
        server, created = self._server_with_mock_ai()
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await _wait_until(lambda: call._running and created["client"].is_connected)
        bridge_tasks = list(call.audio_bridge._tasks)

        # Simulate the AI WebSocket closing: the audio stream ends
        client = created["client"]
        client._connected = False
        client._audio_queue.put_nowait(b"")

        await asyncio.wait_for(call._finished.wait(), 3.0)

        bye = [msg for msg in server.sent if msg.startswith("BYE ")]
        assert len(bye) == 1
        assert "Via: SIP/2.0/UDP" in bye[0] and bye[0].endswith("\r\n\r\n")
        assert all(task.done() for task in bridge_tasks)
        assert call.rtp_session.transport is None
        assert CALL_ID not in server.active_calls
        assert server._allocated_ports == set()


class TestHoldMediaDirection:
    async def _answered_call_with_rtp(self) -> tuple[_RecordingServer, AsyncCall, str]:
        server = _RecordingServer()
        call = await _invite_and_answer(server)
        await call.setup(AudioAdapter(), CallSession(AudioAdapter(), MockDuplexClient(sample_rate=8000)))
        return server, call, _to_tag(_responses(server, "200 OK")[0])

    async def test_sendonly_and_inactive_hold_stop_sending_and_resume(self) -> None:
        server, call, to_tag = await self._answered_call_with_rtp()
        assert call.rtp_session.send_enabled

        await server.handle_message(_invite(cseq=2, to_tag=to_tag, body=OFFER.replace("a=sendrecv", "a=sendonly")), CALLER)
        assert "a=recvonly" in server.sent[-1]
        assert not call.rtp_session.send_enabled

        await server.handle_message(_invite(cseq=3, to_tag=to_tag), CALLER)
        assert "a=sendrecv" in server.sent[-1]
        assert call.rtp_session.send_enabled

        await server.handle_message(_invite(cseq=4, to_tag=to_tag, body=OFFER.replace("a=sendrecv", "a=inactive")), CALLER)
        assert "a=inactive" in server.sent[-1]
        assert not call.rtp_session.send_enabled

        await server.handle_message(_invite(cseq=5, to_tag=to_tag), CALLER)
        assert call.rtp_session.send_enabled

    async def test_connection_address_zero_hold_stops_sending(self) -> None:
        server, call, to_tag = await self._answered_call_with_rtp()
        hold = OFFER.replace("c=IN IP4 198.18.0.1", "c=IN IP4 0.0.0.0").replace("a=sendrecv\r\n", "")

        await server.handle_message(_invite(cseq=2, to_tag=to_tag, body=hold), CALLER)
        assert server.sent[-1].startswith("SIP/2.0 200 OK")
        assert call.remote_rtp_addr == ("198.18.0.1", 4000)  # previous address kept
        assert not call.rtp_session.send_enabled

        await server.handle_message(_invite(cseq=3, to_tag=to_tag), CALLER)
        assert call.rtp_session.send_enabled

    async def test_reinvite_before_answer_gets_491(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await server.handle_message(_invite(cseq=2, to_tag=call.dialog.local_tag), CALLER)
        assert server.sent[-1].startswith("SIP/2.0 491")


class TestCodecOrder:
    def test_first_supported_codec_in_offer_order(self) -> None:
        from app.sip_async.sdp import get_supported_codecs, select_codec

        assert select_codec(parse_sdp(OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 8 0 101"))) == 8
        assert select_codec(parse_sdp(OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 0 8 101"))) == 0
        assert get_supported_codecs(parse_sdp(OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 9 8 18 0"))) == [8, 0]
        assert select_codec(parse_sdp(OFFER.replace("RTP/AVP 96 3 0 8 9 121", "RTP/AVP 9 111"))) is None
