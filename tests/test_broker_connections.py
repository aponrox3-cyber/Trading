import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import ml_auto_server as server_module


class FakeSocket:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.ended = asyncio.Event()
        self.sent = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.messages.get()
        if message is None:
            self.ended.set()
            raise StopAsyncIteration
        return message

    async def close(self):
        # Let tests control when the listener observes the close.
        self.closed = True

    async def send(self, message):
        await self.sent.put(message)


class FakeEngine:
    def __init__(self):
        self.candles = {"EURUSD_otc": []}
        self.ticks = []
        self.tick_received = asyncio.Event()

    def ingest_tick(self, pair, price, ts_ms, source):
        self.ticks.append((pair, price, ts_ms, source))
        self.tick_received.set()
        return []


class BrokerConnectionTests(unittest.IsolatedAsyncioTestCase):
    def make_server(self):
        server = server_module.QuantexServer.__new__(server_module.QuantexServer)
        server.engine = FakeEngine()
        server.broker_ws = None
        server._broker_lock = asyncio.Lock()
        server.live_mode = False
        server.broker = "pocket_option"
        server.ssid = ""
        server.broadcast = AsyncMock()
        return server

    async def test_ssid_is_forwarded_to_broker_without_broadcast(self):
        for broker in ("pocket_option", "quotex"):
            with self.subTest(broker=broker):
                server = self.make_server()
                socket = FakeSocket()
                token = 'synthetic-"session"'
                with patch.object(server_module.websockets, "connect", new=AsyncMock(return_value=socket)):
                    await server.connect_broker(broker, token)
                    await socket.messages.put('0{"sid":"synthetic"}')
                    self.assertEqual(await asyncio.wait_for(socket.sent.get(), 2), "40")
                    frame = await asyncio.wait_for(socket.sent.get(), 2)
                    self.assertTrue(frame.startswith("42"))
                    event, payload = json.loads(frame[2:])
                    self.assertEqual(event, "auth")
                    self.assertEqual(payload["session"], token)
                    self.assertNotIn(token, repr(server.broadcast.await_args_list))
                    await server.disconnect_broker()
                    await socket.messages.put(None)
                    await asyncio.wait_for(socket.ended.wait(), 3)

    async def test_connected_status_requires_valid_tick(self):
        for broker in ("pocket_option", "quotex"):
            with self.subTest(broker=broker):
                server = self.make_server()
                socket = FakeSocket()
                with patch.object(server_module.websockets, "connect", new=AsyncMock(return_value=socket)):
                    await server.connect_broker(broker, "synthetic-session")
                    self.assertEqual(server.broadcast.await_args_list[0].args[0]["status"], "connecting")
                    await socket.messages.put('42["EURUSD_otc",1700000000,"NaN"]')
                    await socket.messages.put('42["UNKNOWN",1700000000,1.2]')
                    await socket.messages.put('42["EURUSD_otc",1700000000,1.2]')
                    await asyncio.wait_for(server.engine.tick_received.wait(), 1)
                    await asyncio.sleep(0)
                    self.assertEqual(server.engine.ticks, [("EURUSD_otc", 1.2, 1700000000000, "live_ssid")])
                    statuses = [call.args[0].get("status") for call in server.broadcast.await_args_list]
                    self.assertEqual(statuses, ["connecting", "connected"])
                    await server.disconnect_broker()
                    await socket.messages.put(None)
                    await asyncio.wait_for(socket.ended.wait(), 1)

    async def test_old_listener_cannot_disconnect_replacement(self):
        server = self.make_server()
        old, new = FakeSocket(), FakeSocket()
        with patch.object(server_module.websockets, "connect", new=AsyncMock(side_effect=[old, new])):
            await server.connect_broker("pocket_option", "first")
            await asyncio.sleep(0)
            await server.connect_broker("pocket_option", "second")
            self.assertTrue(old.closed)
            await old.messages.put(None)
            await asyncio.wait_for(old.ended.wait(), 1)
            await asyncio.sleep(0)
            self.assertIs(server.broker_ws, new)
            self.assertTrue(server.live_mode)
            self.assertNotIn("disconnected", [call.args[0].get("status") for call in server.broadcast.await_args_list])
            await server.disconnect_broker()
            await new.messages.put(None)
            await asyncio.wait_for(new.ended.wait(), 1)

    async def test_connect_requests_are_serialized(self):
        server = self.make_server()
        first, second = FakeSocket(), FakeSocket()
        started, release = asyncio.Event(), asyncio.Event()
        sockets = iter((first, second))

        async def connect(*args, **kwargs):
            socket = next(sockets)
            if socket is first:
                started.set()
                await release.wait()
            return socket

        with patch.object(server_module.websockets, "connect", new=AsyncMock(side_effect=connect)) as dial:
            task1 = asyncio.create_task(server.connect_broker("pocket_option", "first"))
            await asyncio.wait_for(started.wait(), 1)
            task2 = asyncio.create_task(server.connect_broker("quotex", "second"))
            await asyncio.sleep(0)
            self.assertEqual(dial.await_count, 1)
            release.set()
            await asyncio.gather(task1, task2)
            self.assertIs(server.broker_ws, second)
            self.assertTrue(first.closed)
            await first.messages.put(None)
            await second.messages.put(None)
            await asyncio.wait_for(first.ended.wait(), 1)
            await asyncio.wait_for(second.ended.wait(), 1)

    async def test_reconnect_waits_for_disconnected_status(self):
        server = self.make_server()
        old, new = FakeSocket(), FakeSocket()
        close_started, release_close = asyncio.Event(), asyncio.Event()

        async def delayed_close():
            close_started.set()
            await release_close.wait()
            old.closed = True

        old.close = delayed_close
        with patch.object(server_module.websockets, "connect", new=AsyncMock(side_effect=[old, new])) as dial:
            await server.connect_broker("pocket_option", "first")
            await old.messages.put(None)
            await asyncio.wait_for(close_started.wait(), 1)
            reconnect = asyncio.create_task(server.connect_broker("quotex", "second"))
            await asyncio.sleep(0)
            self.assertEqual(dial.await_count, 1)
            release_close.set()
            await reconnect
            statuses = [call.args[0].get("status") for call in server.broadcast.await_args_list]
            self.assertEqual(statuses, ["connecting", "disconnected", "connecting"])
            await server.disconnect_broker()
            await new.messages.put(None)
            await asyncio.wait_for(new.ended.wait(), 1)


if __name__ == "__main__":
    unittest.main()
