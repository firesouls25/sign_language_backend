from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models.translation import Translation
from app.schemas.translation import TranslationResponse, TranslationHistoryResponse
from app.services.auth_service import AuthService
from app.services.storage_service import get_storage_service
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from datetime import datetime

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


@router.get("/", response_model=TranslationHistoryResponse)
async def get_translation_history(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
    page: int = 1,
    size: int = 20,
    search: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    # Base query
    query = select(Translation).where(Translation.user_id == user_id)

    # Filters
    if search:
        query = query.where(Translation.text_result.ilike(f"%{search}%"))
    if start_date:
        query = query.where(Translation.created_at >= start_date)
    if end_date:
        query = query.where(Translation.created_at <= end_date)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Pagination
    offset = (page - 1) * size
    result = await db.execute(
        query.order_by(Translation.created_at.desc()).limit(size).offset(offset)
    )
    translations = result.scalars().all()

    pages = (total + size - 1) // size if size > 0 else 0

    return {
        "items": translations,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages,
    }


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    storage = Depends(get_storage_service)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    file_data = await file.read()
    video_url = await storage.upload_file(file_data, file.filename, file.content_type)
    
    if not video_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not upload video"
        )
        
    return {"video_url": video_url}
