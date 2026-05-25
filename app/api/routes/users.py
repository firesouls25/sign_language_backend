from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from app.database import get_db
from app.models.user import User
from app.schemas.chat import UserBrief
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional

router = APIRouter(prefix="/api/users", tags=["users"])
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


@router.get("/me/id")
async def get_my_id(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
    }


@router.get("/search", response_model=List[UserBrief])
async def search_users(
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    users = await ChatService.search_users(db, q, user_id)
    return [
        UserBrief(
            id=u.id,
            username=u.username,
            full_name=u.full_name,
            avatar_url=u.avatar_url,
        )
        for u in users
    ]


@router.get("/{target_id}/profile", response_model=UserBrief)
async def get_user_profile(
    target_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(select(User).where(User.id == target_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserBrief(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
    )
