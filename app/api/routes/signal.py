from fastapi import WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import json
import logging

from app.database import get_db, AsyncSessionLocal, DB_AVAILABLE
from app.models.chat import Conversation
from app.services.chat_service import ChatService
from app.services.room_manager import room_manager
from app.utils.security import decode_token

logger = logging.getLogger(__name__)


async def handle_signal_websocket(
    websocket: WebSocket, conversation_id: str, token: Optional[str] = Query(None)
):
    user_id = None
    username = None

    if token:
        payload = decode_token(token)
        if payload and payload.get("type") == "access":
            user_id = payload.get("sub")

    if not user_id:
        await websocket.close(code=4001, reason="Invalid or missing token")
        return

    if not DB_AVAILABLE:
        await websocket.close(code=4001, reason="Database not available")
        return

    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        from app.models.user import User

        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = conv_result.scalar_one_or_none()

        if not conv or not conv.is_participant(user_id):
            await websocket.close(code=4003, reason="Not a participant in this conversation")
            return

        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()
        if user:
            username = user.username

    await websocket.accept()

    participants = await room_manager.join_room(conversation_id, user_id, websocket)

    await websocket.send_json({
        "type": "joined",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "username": username,
        "participants": participants,
    })

    try:
        while True:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "offer":
                    target_id = data.get("target_id")
                    if not target_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing target_id",
                        })
                        continue
                    await room_manager.relay_to(
                        conversation_id, user_id, target_id,
                        {"type": "offer", "from_id": user_id, "from_username": username, "sdp": data.get("sdp")},
                    )

                elif msg_type == "answer":
                    target_id = data.get("target_id")
                    if not target_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing target_id",
                        })
                        continue
                    await room_manager.relay_to(
                        conversation_id, user_id, target_id,
                        {"type": "answer", "from_id": user_id, "sdp": data.get("sdp")},
                    )

                elif msg_type == "ice_candidate":
                    target_id = data.get("target_id")
                    if not target_id:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing target_id",
                        })
                        continue
                    await room_manager.relay_to(
                        conversation_id, user_id, target_id,
                        {"type": "ice_candidate", "from_id": user_id, "candidate": data.get("candidate")},
                    )

                elif msg_type == "translation":
                    text = data.get("text", "")
                    if not text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing text",
                        })
                        continue

                    async with AsyncSessionLocal() as db:
                        msg = await ChatService.send_message(
                            db,
                            conversation_id,
                            user_id,
                            text=text,
                            video_url=data.get("video_url"),
                            audio_url=data.get("audio_url"),
                            confidence_score=data.get("confidence_score"),
                            message_type="translation",
                        )

                    if msg:
                        await room_manager.broadcast(
                            conversation_id,
                            {
                                "type": "translation",
                                "from_id": user_id,
                                "from_username": username,
                                "text": msg.text,
                                "video_url": msg.video_url,
                                "audio_url": msg.audio_url,
                                "confidence_score": msg.confidence_score,
                                "message_id": msg.id,
                                "created_at": msg.created_at.isoformat(),
                            },
                            exclude_user_id=user_id,
                        )
                        await websocket.send_json({
                            "type": "translation_sent",
                            "message_id": msg.id,
                            "created_at": msg.created_at.isoformat(),
                        })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                    "code": "INVALID_JSON",
                })
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(f"[SignalWS] Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                })

    except WebSocketDisconnect:
        logger.info(
            f"[SignalWS] User {user_id} disconnected from conversation {conversation_id}"
        )
    except Exception as e:
        logger.error(f"[SignalWS] Unexpected error for {user_id}: {e}")
    finally:
        await room_manager.leave_room(conversation_id, user_id)
