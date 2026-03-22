from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models.translation import Translation
from app.schemas.translation import TranslationResponse
from app.services.auth_service import AuthService
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List

router = APIRouter(prefix="/api/translation", tags=["translation"])
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    user_id = await AuthService.get_user_from_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return user_id


@router.get("/{translation_id}", response_model=TranslationResponse)
async def get_translation(
    translation_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    result = await db.execute(
        select(Translation).where(
            Translation.id == translation_id,
            Translation.user_id == user_id
        )
    )
    translation = result.scalar_one_or_none()
    if not translation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Translation not found"
        )
    return translation


@router.get("/", response_model=List[TranslationResponse])
async def get_translation_history(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
    limit: int = 20,
    offset: int = 0
):
    result = await db.execute(
        select(Translation)
        .where(Translation.user_id == user_id)
        .order_by(Translation.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    translations = result.scalars().all()
    return translations
