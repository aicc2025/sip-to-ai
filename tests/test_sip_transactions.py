"""SIP transactions: CANCEL, OPTIONS, unsupported methods, BYE matching,
200 OK retransmission / ACK timeout, answer after AI connect (503 on failure)
and graceful shutdown."""

import asyncio

import pytest

from app.bridge import AudioAdapter, CallSession
from app.sip_async import async_call as async_call_module
from app.sip_async.async_call import STATE_CONFIRMED, STATE_EARLY, AsyncCall
from app.sip_async.sip_protocol import SIPMessage
from tests.mock_ai_client import MockDuplexClient
from tests.test_sip_call_lifecycle import (
    CALL_ID,
    CALLER,
    _bye,
    _invite,
    _RecordingServer,
    _responses,
    _to_tag,
    _wait_until,
)


def _request(method: str, call_id: str = CALL_ID, cseq: int = 1, to_tag: str = "",
             branch: str = "z9hG4bK1", cseq_method: str | None = None) -> SIPMessage:
    to = "<sip:test@127.0.0.1:6060>" + (f";tag={to_tag}" if to_tag else "")
    raw = (
        f"{method} sip:test@127.0.0.1:6060 SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP 127.0.0.1:5091;rport;branch={branch}\r\n"
        "From: <sip:caller@127.0.0.1>;tag=fromtag\r\n"
        f"To: {to}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq} {cseq_method or method}\r\n"
        "Content-Length: 0\r\n\r\n"
    )
    return SIPMessage(raw=raw.encode(), remote_addr=CALLER)


def _header(message: str, name: str) -> str:
    return next(line for line in message.split("\r\n") if line.startswith(f"{name}:"))


class _SlowClient(MockDuplexClient):
    """Mock AI client whose connect() blocks until released (or forever)."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__(sample_rate=8000)
        self.release = asyncio.Event()
        self.fail = fail
        self.connect_cancelled = False

    async def connect(self) -> None:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.connect_cancelled = True
            raise
        if self.fail:
            raise ConnectionError("Failed to connect to ws://gateway: TimeoutError: timed out")
        await super().connect()


def _server_with_client(client_factory, ai_connect_timeout: float | None = None):
    created: dict = {}

    async def on_call(call: AsyncCall) -> None:
        adapter = AudioAdapter(uplink_capacity=100, downlink_capacity=100)
        client = client_factory()
        created["client"] = client
        await call.setup(adapter, CallSession(audio_adapter=adapter, ai_client=client))

    server = _RecordingServer(call_callback=on_call)
    server.ai_connect_timeout = ai_connect_timeout
    return server, created


@pytest.fixture
def fast_sip_timers(monkeypatch):
    monkeypatch.setattr(async_call_module, "SIP_T1", 0.02)
    monkeypatch.setattr(async_call_module, "SIP_T2", 0.08)
    monkeypatch.setattr(async_call_module, "SIP_TIMER_H", 0.5)


class TestOptionsAndUnsupportedMethods:
    async def test_options_gets_200_with_allow(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_request("OPTIONS", call_id="opt@test"), CALLER)

        assert server.sent[0].startswith("SIP/2.0 200 OK")
        assert _header(server.sent[0], "Allow") == "Allow: INVITE, ACK, BYE, CANCEL, OPTIONS"
        assert "tag=" in _header(server.sent[0], "To")

    async def test_info_gets_405_with_allow(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_request("INFO", call_id="info@test"), CALLER)

        assert server.sent[0].startswith("SIP/2.0 405 Method Not Allowed")
        assert "Allow: INVITE, ACK, BYE, CANCEL, OPTIONS" in server.sent[0]
        assert "CSeq: 1 INFO" in server.sent[0]

    async def test_unknown_method_gets_501(self) -> None:
        server = _RecordingServer()
        msg = _request("FOOBAR", call_id="foo@test")
        assert msg.method is None and msg.method_name == "FOOBAR"
        await server.handle_message(msg, CALLER)

        assert server.sent[0].startswith("SIP/2.0 501 Not Implemented")
        assert "Allow: INVITE, ACK, BYE, CANCEL, OPTIONS" in server.sent[0]


class TestByeMatching:
    async def test_bye_for_unknown_call_gets_481_with_to_tag(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_bye("sometag", call_id="unknown@test"), CALLER)

        assert server.sent[0].startswith("SIP/2.0 481")
        assert "tag=sometag" in _header(server.sent[0], "To")

    async def test_retransmitted_bye_of_ended_call_gets_200(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await call.accept()
        to_tag = _to_tag(_responses(server, "200 OK")[0])

        await server.handle_message(_bye(to_tag), CALLER)
        assert CALL_ID not in server.active_calls
        await server.handle_message(_bye(to_tag), CALLER)

        bye_answers = [m for m in server.sent if "CSeq: 10 BYE" in m]
        assert len(bye_answers) == 2
        assert all(m.startswith("SIP/2.0 200 OK") for m in bye_answers)

    async def test_locally_generated_481_for_unknown_reinvite_has_to_tag(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(to_tag="stale"), CALLER)
        assert "tag=stale" in _header(server.sent[0], "To")


class TestCancel:
    async def test_cancel_before_answer_gets_200_and_invite_487(self) -> None:
        server, created = _server_with_client(_SlowClient)
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await _wait_until(lambda: "client" in created and call._run_started)

        await server.handle_message(_request("CANCEL", branch="z9hG4bK1", cseq=1), CALLER)

        cancel_ok = [m for m in server.sent if "CSeq: 1 CANCEL" in m]
        assert len(cancel_ok) == 1 and cancel_ok[0].startswith("SIP/2.0 200 OK")
        assert _to_tag(cancel_ok[0]) == call.dialog.local_tag
        invite_487 = [m for m in server.sent if m.startswith("SIP/2.0 487") and "CSeq: 1 INVITE" in m]
        assert len(invite_487) == 1
        assert _to_tag(invite_487[0]) == call.dialog.local_tag
        assert all("CSeq: 1 CANCEL" in m for m in _responses(server, "200"))
        assert not any(m.startswith("BYE ") for m in server.sent)

        await asyncio.wait_for(call._finished.wait(), 2.0)
        assert created["client"].connect_cancelled
        assert CALL_ID not in server.active_calls
        assert server._allocated_ports == set()

        # ACK for the 487 stops its retransmission
        await server.handle_message(_request("ACK", cseq=1), CALLER)
        assert call._ack_received.is_set()
        call.cancel_retransmissions()

    async def test_cancel_with_unknown_branch_gets_481(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(), CALLER)
        await server.handle_message(_request("CANCEL", branch="z9hG4bKother", cseq=1), CALLER)

        assert server.sent[-1].startswith("SIP/2.0 481")
        assert server.active_calls[CALL_ID].state == STATE_EARLY

    async def test_cancel_for_unknown_call_gets_481(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_request("CANCEL", call_id="nope@test"), CALLER)
        assert server.sent[0].startswith("SIP/2.0 481")

    async def test_cancel_after_answer_does_not_end_call(self) -> None:
        server = _RecordingServer()
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await call.accept()

        await server.handle_message(_request("CANCEL", branch="z9hG4bK1", cseq=1), CALLER)

        assert server.sent[-1].startswith("SIP/2.0 200 OK") and "CSeq: 1 CANCEL" in server.sent[-1]
        assert not [m for m in server.sent if m.startswith("SIP/2.0 487")]
        assert CALL_ID in server.active_calls


class TestAnswerAfterAiConnect:
    async def test_200_ok_only_after_ai_connected(self) -> None:
        server, created = _server_with_client(_SlowClient)
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await _wait_until(lambda: "client" in created and call._run_started)
        await asyncio.sleep(0.05)
        assert not _responses(server, "200 OK")

        created["client"].release.set()
        await _wait_until(lambda: bool(_responses(server, "200 OK")))
        first_lines = [m.split("\r\n", 1)[0] for m in server.sent]
        assert first_lines.index("SIP/2.0 180 Ringing") < first_lines.index("SIP/2.0 200 OK")

        await server.handle_message(_request("ACK", cseq=1, to_tag=call.dialog.local_tag), CALLER)
        assert call.state == STATE_CONFIRMED
        await call.stop()

    async def test_ai_connect_failure_rejects_with_503_without_bye(self) -> None:
        server, created = _server_with_client(lambda: _SlowClient(fail=True))
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await _wait_until(lambda: "client" in created)
        created["client"].release.set()

        await asyncio.wait_for(call._finished.wait(), 2.0)
        assert _responses(server, "503 Service Unavailable")
        assert _to_tag(_responses(server, "503")[0]) == call.dialog.local_tag
        assert not _responses(server, "200 OK")
        assert not any(m.startswith("BYE ") for m in server.sent)
        assert CALL_ID not in server.active_calls
        assert server._allocated_ports == set()
        call.cancel_retransmissions()

    async def test_ai_connect_timeout_rejects_with_503(self) -> None:
        server, created = _server_with_client(_SlowClient, ai_connect_timeout=0.1)
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]

        await asyncio.wait_for(call._finished.wait(), 2.0)
        assert created["client"].connect_cancelled
        assert _responses(server, "503 Service Unavailable")
        assert not _responses(server, "200 OK")
        assert not any(m.startswith("BYE ") for m in server.sent)

        # Retransmitted INVITE after the rejection gets the same 503, no new call
        await server.handle_message(_invite(), CALLER)
        assert len(_responses(server, "503")) >= 2
        assert CALL_ID not in server.active_calls
        call.cancel_retransmissions()

    async def test_retransmitted_invite_while_connecting_resends_180(self) -> None:
        server, _ = _server_with_client(_SlowClient)
        await server.handle_message(_invite(), CALLER)
        await server.handle_message(_invite(), CALLER)

        assert len(_responses(server, "180 Ringing")) == 2
        assert len(server.active_calls) == 1
        await server.active_calls[CALL_ID].stop()


class TestAckHandling:
    async def test_200_ok_retransmitted_until_ack(self, fast_sip_timers) -> None:
        server, _ = _server_with_client(lambda: MockDuplexClient(sample_rate=8000))
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]
        await _wait_until(lambda: len(_responses(server, "200 OK")) >= 3)

        await server.handle_message(_request("ACK", cseq=1, to_tag=call.dialog.local_tag), CALLER)
        count = len(_responses(server, "200 OK"))
        await asyncio.sleep(0.2)
        assert len(_responses(server, "200 OK")) == count
        assert call.state == STATE_CONFIRMED
        assert not any(m.startswith("BYE ") for m in server.sent)
        await call.stop()

    async def test_no_ack_ends_call_with_bye(self, fast_sip_timers) -> None:
        server, created = _server_with_client(lambda: MockDuplexClient(sample_rate=8000))
        await server.handle_message(_invite(), CALLER)
        call = server.active_calls[CALL_ID]

        await asyncio.wait_for(call._finished.wait(), 3.0)
        assert len(_responses(server, "200 OK")) >= 3
        assert len([m for m in server.sent if m.startswith("BYE ")]) == 1
        assert CALL_ID not in server.active_calls
        assert not created["client"].is_connected


class TestShutdown:
    async def test_stop_sends_bye_to_answered_calls_and_503_to_pending(self) -> None:
        server, _ = _server_with_client(lambda: MockDuplexClient(sample_rate=8000))
        closed: list[bool] = []

        class _Transport:
            def close(self) -> None:
                closed.append(True)

        server.transport = _Transport()  # type: ignore[assignment]
        await server.handle_message(_invite(), CALLER)
        answered = server.active_calls[CALL_ID]
        await _wait_until(lambda: answered._running)

        pending_server_client = _SlowClient()
        server.call_callback = None
        await server.handle_message(_invite(call_id="pending@test"), CALLER)
        pending = server.active_calls["pending@test"]
        pending.call_session = CallSession(AudioAdapter(), pending_server_client)

        await server.stop()
        await server.stop()  # idempotent

        byes = [m for m in server.sent if m.startswith("BYE ")]
        assert len(byes) == 1 and f"Call-ID: {CALL_ID}" in byes[0]
        assert [m for m in _responses(server, "503") if "pending@test" in m]
        assert answered._finished.is_set()
        assert server.active_calls == {}
        assert closed == [True]
