import asyncio
import logging
import threading
from typing import List
from fastapi import WebSocket

LOGGER = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._loop = asyncio.get_running_loop()
        with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        with self._lock:
            connections = list(self.active_connections)
        stale_connections: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale_connections.append(connection)
        for connection in stale_connections:
            self.disconnect(connection)

    def publish(self, message: dict) -> None:
        loop = self._loop
        if not loop or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            future.add_done_callback(self._handle_future_result)
        except Exception as exc:
            LOGGER.warning("Websocket publish failed: %s", exc)

    @staticmethod
    def _handle_future_result(future):
        try:
            future.result()
        except Exception as exc:
            LOGGER.warning("Websocket broadcast failed: %s", exc)


manager = ConnectionManager()
