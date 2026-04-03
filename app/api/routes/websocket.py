from fastapi import WebSocket, WebSocketDisconnect, Query
from typing import Dict, Optional
import json
from app.services.ai_service import get_ai_service
from app.utils.security import decode_token
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_ids: Dict[str, str] = {}

    async def connect(
        self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None
    ):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if user_id:
            self.user_ids[client_id] = user_id
        logger.info(
            f"Client {client_id} connected" + (f" (user: {user_id})" if user_id else "")
        )

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.user_ids:
            del self.user_ids[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")

    def get_user_id(self, client_id: str) -> Optional[str]:
        return self.user_ids.get(client_id)


manager = ConnectionManager()


def validate_token(token: str) -> Optional[str]:
    """Validate JWT token and return user_id"""
    if not token:
        return None
    payload = decode_token(token)
    if payload and payload.get("type") == "access":
        return payload.get("sub")
    return None


async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
    client_id = None
    user_id = None

    if token:
        user_id = validate_token(token)
        if not user_id:
            await websocket.close(code=4001, reason="Invalid token")
            return

    ai_service = get_ai_service()
    client_id = str(id(websocket))
    await manager.connect(websocket, client_id, user_id)

    try:
        while True:
            try:
                data = await websocket.receive()

                if data["type"] == "text":
                    message = json.loads(data["text"])

                    if message.get("type") == "ping":
                        await manager.send_message({"type": "pong"}, client_id)

                    elif message.get("type") == "reset":
                        ai_service.reset()
                        await manager.send_message(
                            {"type": "reset", "status": "ok"}, client_id
                        )

                    elif message.get("type") == "landmarks":
                        logger.info(
                            f"[WebSocket] Received landmarks message from client {client_id}"
                        )
                        current_user_id = manager.get_user_id(client_id)
                        result = await ai_service.process_landmarks(
                            message.get("data", {}), current_user_id
                        )
                        logger.info(
                            f"[WebSocket] Sending response: {result.get('text', '')}"
                        )
                        await manager.send_message(result, client_id)

            except json.JSONDecodeError:
                await manager.send_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "code": "INVALID_JSON",
                    },
                    client_id,
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_message(
            {"type": "error", "message": str(e), "code": "CONNECTION_ERROR"}, client_id
        )
        manager.disconnect(client_id)
