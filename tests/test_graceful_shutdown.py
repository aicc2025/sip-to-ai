"""Graceful shutdown: SIGINT/SIGTERM set a stop event; calls are hung up before exit."""

import asyncio
import os
import signal
from typing import ClassVar

import app.main as main_module


class _FakeServer:
    instances: ClassVar[list["_FakeServer"]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.stop_calls = 0
        self.run_cancelled = False
        _FakeServer.instances.append(self)

    async def run(self) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.run_cancelled = True
            raise

    async def stop(self) -> None:
        self.stop_calls += 1


async def test_run_real_mode_stops_server_when_stop_event_set(monkeypatch) -> None:
    _FakeServer.instances.clear()
    monkeypatch.setattr(main_module, "AsyncSIPServer", _FakeServer)
    stop_event = asyncio.Event()

    task = asyncio.create_task(main_module.run_real_mode(stop_event))
    await asyncio.sleep(0.05)
    assert not task.done()
    stop_event.set()
    await asyncio.wait_for(task, 1.0)

    server = _FakeServer.instances[0]
    assert server.stop_calls == 1
    assert server.run_cancelled
    assert server.kwargs["ai_connect_timeout"] > 0


async def test_sigterm_triggers_graceful_shutdown(monkeypatch) -> None:
    seen: list[bool] = []

    async def fake_run_real_mode(stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        seen.append(True)

    monkeypatch.setattr(main_module, "run_real_mode", fake_run_real_mode)
    task = asyncio.create_task(main_module.main())
    await asyncio.sleep(0.05)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, 2.0)

    assert seen == [True]
