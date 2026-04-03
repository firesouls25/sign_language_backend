from fastapi import WebSocket, WebSocketDisconnect, Query
from typing import Dict, Optional
import json
import asyncio
from datetime import datetime
from app.services.ai_service import get_ai_service
from app.utils.security import decode_token
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_ids: Dict[str, str] = {}
        self.last_activity: Dict[str, datetime] = {}

    async def connect(
        self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None
    ):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if user_id:
            self.user_ids[client_id] = user_id
        self.last_activity[client_id] = datetime.now()
        logger.info(
            f"Client {client_id} connected" + (f" (user: {user_id})" if user_id else "")
        )

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.user_ids:
            del self.user_ids[client_id]
        if client_id in self.last_activity:
            del self.last_activity[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
                self.last_activity[client_id] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    def get_user_id(self, client_id: str) -> Optional[str]:
        return self.user_ids.get(client_id)

    def update_activity(self, client_id: str):
        self.last_activity[client_id] = datetime.now()


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
    logger.info(f"[WebSocket] Connection established for client {client_id}")

    try:
        while True:
            try:
                data = await websocket.receive_text()
                manager.update_activity(client_id)

                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "")

                    logger.info(
                        f"[WebSocket] Received message type: {msg_type} from {client_id}"
                    )

                    if msg_type == "ping":
                        logger.info(f"[WebSocket] Sending pong to {client_id}")
                        await manager.send_message({"type": "pong"}, client_id)

                    elif msg_type == "reset":
                        logger.info(f"[WebSocket] Reset requested by {client_id}")
                        ai_service.reset()
                        await manager.send_message(
                            {"type": "reset", "status": "ok"}, client_id
                        )

                    elif msg_type == "landmarks":
                        landmarks_data = message.get("data", {})
                        logger.info(
                            f"[WebSocket] Received landmarks from {client_id}: "
                            f"left={len(landmarks_data.get('left_hand', []))} points, "
                            f"right={len(landmarks_data.get('right_hand', []))} points"
                        )
                        current_user_id = manager.get_user_id(client_id)

                        result = await ai_service.process_landmarks(
                            landmarks_data, current_user_id
                        )

                        logger.info(
                            f"[WebSocket] Recognition result for {client_id}: "
                            f"text='{result.get('text', '')}', "
                            f"confidence={result.get('confidence', 0.0):.2f}"
                        )
                        await manager.send_message(result, client_id)

                    else:
                        logger.warning(f"[WebSocket] Unknown message type: {msg_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"[WebSocket] Invalid JSON from {client_id}: {e}")
                    await manager.send_message(
                        {
                            "type": "error",
                            "message": "Invalid JSON format",
                            "code": "INVALID_JSON",
                        },
                        client_id,
                    )
                except Exception as e:
                    logger.error(
                        f"[WebSocket] Error processing message from {client_id}: {e}"
                    )
                    await manager.send_message(
                        {
                            "type": "error",
                            "message": f"Processing error: {str(e)}",
                            "code": "PROCESSING_ERROR",
                        },
                        client_id,
                    )

            except Exception as e:
                logger.error(f"[WebSocket] Error receiving data from {client_id}: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client {client_id} disconnected normally")
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"[WebSocket] Unexpected error for client {client_id}: {e}")
        manager.disconnect(client_id)
