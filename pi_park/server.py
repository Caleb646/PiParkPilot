from typing import Any, Dict, Tuple, Union, List, Iterable
import socket
import json
import time

import asyncio
from collections import deque
from websockets.server import serve
from websockets.sync.client import connect

class WebSocketServer:
    def __init__(
            self, host=socket.gethostbyname(socket.gethostname()), port=8000
            ) -> None:
        self.host = host
        self.port = port
        self.connections_ = set()
        self.messages_: deque[Dict[str, Any]] = deque()

    async def send(self, message: Dict[str, Any]):
        encoded_data = json.dumps(message)
        for ws in self.connections_:
            await ws.send(encoded_data)

    def has_connections(self) -> bool:
        return len(self.connections_) > 0
    
    def get_message(self) -> Dict[str, Any]:
        if self.messages_:
            return self.messages_.popleft()
        return {}
    
    async def shutdown(self):
        for conn in self.connections_:
            await conn.close()

    async def recv_(self, ws):
        async for message in ws:
            print(f"Received Message: {message}")
            decoded_data = message
            if isinstance(message, bytes):
                decoded_data = message.decode()
            self.messages_.append(json.loads(decoded_data))
            await asyncio.sleep(0.05)

    async def handle(self, ws):
        self.connections_.add(ws)
        print(f"Adding Connection: {ws}")
        consumer_task = asyncio.create_task(self.recv_(ws))
        done, pending = await asyncio.wait(
            [consumer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    async def run(self):
        async with serve(lambda ws : self.handle(ws), self.host, self.port):
            print(f"Serving at: ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever