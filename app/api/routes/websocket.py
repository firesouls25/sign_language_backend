from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import base64
import numpy as np
import cv2
from app.services.ai_service import get_ai_service


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    ai_service = get_ai_service()
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                result = await ai_service.process_frame(message.get("data", ""))
                await manager.send_message(result, client_id)
            
            elif message.get("type") == "ping":
                await manager.send_message({"type": "pong"}, client_id)
                
            elif message.get("type") == "reset":
                ai_service.reset()
                await manager.send_message({"type": "reset", "status": "ok"}, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        await manager.send_message({
            "type": "error",
            "message": str(e),
            "code": "CONNECTION_ERROR"
        }, client_id)
        manager.disconnect(client_id)
