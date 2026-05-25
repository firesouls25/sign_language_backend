from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.schemas.chat import (
    ConversationCreate,
    ConversationWithUserResponse,
    MessageCreate,
    MessageResponse,
    MessageHistoryResponse,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional

router = APIRouter(prefix="/api/conversations", tags=["conversations"])
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    user_id = await AuthService.get_user_from_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return user_id


@router.get("/", response_model=List[ConversationWithUserResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    return await ChatService.get_conversations_with_users(db, user_id)


@router.post(
    "/", response_model=ConversationWithUserResponse, status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    data: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    conv = await ChatService.create_conversation(db, user_id, data.participant_id)

    from app.models.user import User
    from sqlalchemy import select
    from app.schemas.chat import UserBrief

    other_id = conv.other_user_id(user_id)
    user_result = await db.execute(select(User).where(User.id == other_id))
    other_user = user_result.scalar_one_or_none()

    return {
        "id": conv.id,
        "other_user": UserBrief(
            id=other_user.id,
            username=other_user.username,
            full_name=other_user.full_name,
            avatar_url=other_user.avatar_url,
        ),
        "is_self": conv.creator_id == conv.participant_id,
        "last_message_text": conv.last_message_text,
        "last_message_at": conv.last_message_at,
        "created_at": conv.created_at,
    }


@router.get("/{conversation_id}/messages", response_model=MessageHistoryResponse)
async def get_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=100),
):
    messages, total = await ChatService.get_messages(
        db, conversation_id, user_id, page, size
    )
    pages = (total + size - 1) // size if size > 0 else 0

    return {
        "items": messages,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages,
    }


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    conversation_id: str,
    data: MessageCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    msg = await ChatService.send_message(
        db,
        conversation_id,
        user_id,
        text=data.text,
        video_url=data.video_url,
        audio_url=data.audio_url,
        confidence_score=data.confidence_score,
        message_type=data.message_type,
    )
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a participant in this conversation",
        )
    return msg


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    success = await ChatService.delete_conversation(db, conversation_id, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or not a participant",
        )
