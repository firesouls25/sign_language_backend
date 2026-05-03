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
        self.client_modes: Dict[str, str] = {}

    async def connect(
        self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None
    ):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if user_id:
            self.user_ids[client_id] = user_id
        self.last_activity[client_id] = datetime.now()
        self.client_modes[client_id] = "handshape"
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
        if client_id in self.client_modes:
            del self.client_modes[client_id]
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

    def get_mode(self, client_id: str) -> str:
        return self.client_modes.get(client_id, "handshape")

    def set_mode(self, client_id: str, mode: str):
        if mode in ["handshape", "fingerspelling"]:
            self.client_modes[client_id] = mode
            logger.info(f"[ConnectionManager] Client {client_id} mode set to: {mode}")
        else:
            logger.warning(f"[ConnectionManager] Unknown mode: {mode}")

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

                    elif msg_type == "finalize":
                        logger.warning(
                            f"[WebSocket] ================= FINALIZE RECEIVED ================="
                        )
                        logger.warning(f"[WebSocket] Finalize requested by {client_id}")
                        current_mode = manager.get_mode(client_id)
                        logger.warning(f"[WebSocket] Current mode: {current_mode}")

                        result = await ai_service.finalize(
                            mode=current_mode, user_id=manager.get_user_id(client_id)
                        )

                        logger.warning(
                            f"[WebSocket] Finalize result for {client_id}: "
                            f"text='{result.get('text', '')}', "
                            f"text='{result.get('text', '')}', "
                            f"mode={current_mode}"
                        )
                        await manager.send_message(result, client_id)
                        logger.warning(
                            f"[WebSocket] ================= FINALIZE SENT ================="
                        )

                    elif msg_type == "set_mode":
                        mode = message.get("mode", "handshape")
                        manager.set_mode(client_id, mode)
                        ai_service.set_mode(mode)
                        logger.info(
                            f"[WebSocket] Mode set to '{mode}' for client {client_id}"
                        )
                        await manager.send_message(
                            {"type": "mode_set", "mode": mode, "status": "ok"},
                            client_id,
                        )

                    elif msg_type == "landmarks":
                        landmarks_data = message.get("data", {})
                        current_mode = message.get("mode", manager.get_mode(client_id))

                        if current_mode != manager.get_mode(client_id):
                            manager.set_mode(client_id, current_mode)
                            ai_service.set_mode(current_mode)

                        logger.info(
                            f"[WebSocket] Received landmarks from {client_id}: "
                            f"left={len(landmarks_data.get('left_hand', []))} points, "
                            f"right={len(landmarks_data.get('right_hand', []))} points, "
                            f"mode={current_mode}"
                        )
                        current_user_id = manager.get_user_id(client_id)

                        result = await ai_service.process_landmarks(
                            landmarks_data, mode=current_mode, user_id=current_user_id
                        )

                        logger.info(
                            f"[WebSocket] Recognition result for {client_id}: "
                            f"text='{result.get('text', '')}', "
                            f"confidence={result.get('confidence', 0.0):.2f}"
                        )
                        await manager.send_message(result, client_id)

                    elif msg_type == "frame":
                        frame_data = message.get("data", [])
                        width = message.get("width", 0)
                        height = message.get("height", 0)
                        current_mode = message.get("mode", manager.get_mode(client_id))

                        if current_mode != manager.get_mode(client_id):
                            manager.set_mode(client_id, current_mode)
                            ai_service.set_mode(current_mode)

                        logger.info(
                            f"[WebSocket] Received frame from {client_id}: "
                            f"size={len(frame_data)} bytes, {width}x{height}, mode={current_mode}"
                        )
                        current_user_id = manager.get_user_id(client_id)

                        result = await ai_service.process_frame(
                            frame_data,
                            width=width,
                            height=height,
                            mode=current_mode,
                            user_id=current_user_id,
                        )

                        logger.info(
                            f"[WebSocket] Frame recognition result for {client_id}: "
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
