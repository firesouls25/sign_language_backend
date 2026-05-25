from fastapi import WebSocket
from typing import Dict, Optional, List
import asyncio
import logging

logger = logging.getLogger(__name__)


class RoomManager:
    def __init__(self):
        self._rooms: Dict[str, Dict[str, WebSocket]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self._locks:
            self._locks[conversation_id] = asyncio.Lock()
        return self._locks[conversation_id]

    async def join_room(
        self, conversation_id: str, user_id: str, websocket: WebSocket
    ):
        async with self._get_lock(conversation_id):
            if conversation_id not in self._rooms:
                self._rooms[conversation_id] = {}
            self._rooms[conversation_id][user_id] = websocket

        participants = list(self._rooms.get(conversation_id, {}).keys())

        await self.broadcast(
            conversation_id,
            {
                "type": "user_joined",
                "user_id": user_id,
            },
            exclude_user_id=user_id,
        )

        logger.info(
            f"[RoomManager] User {user_id} joined room {conversation_id}. "
            f"Participants: {participants}"
        )

        return participants

    async def leave_room(self, conversation_id: str, user_id: str):
        async with self._get_lock(conversation_id):
            if conversation_id in self._rooms:
                self._rooms[conversation_id].pop(user_id, None)
                if not self._rooms[conversation_id]:
                    del self._rooms[conversation_id]
                    self._locks.pop(conversation_id, None)

        await self.broadcast(
            conversation_id,
            {
                "type": "user_left",
                "user_id": user_id,
            },
        )

        logger.info(
            f"[RoomManager] User {user_id} left room {conversation_id}"
        )

    async def relay_to(
        self,
        conversation_id: str,
        from_user_id: str,
        target_user_id: str,
        message: dict,
    ):
        async with self._get_lock(conversation_id):
            room = self._rooms.get(conversation_id)
            if not room:
                return
            target_ws = room.get(target_user_id)
            if not target_ws:
                logger.warning(
                    f"[RoomManager] Target {target_user_id} not in room {conversation_id}"
                )
                return

        try:
            message["from_id"] = from_user_id
            await target_ws.send_json(message)
        except Exception as e:
            logger.error(
                f"[RoomManager] Error relaying to {target_user_id}: {e}"
            )
            await self.leave_room(conversation_id, target_user_id)

    async def broadcast(
        self,
        conversation_id: str,
        message: dict,
        exclude_user_id: Optional[str] = None,
    ):
        async with self._get_lock(conversation_id):
            room = self._rooms.get(conversation_id)
            if not room:
                return
            targets = list(room.items())

        for uid, ws in targets:
            if uid == exclude_user_id:
                continue
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(
                    f"[RoomManager] Error broadcasting to {uid}: {e}"
                )
                await self.leave_room(conversation_id, uid)

    async def get_participants(self, conversation_id: str) -> List[str]:
        async with self._get_lock(conversation_id):
            room = self._rooms.get(conversation_id, {})
            return list(room.keys())


room_manager = RoomManager()
