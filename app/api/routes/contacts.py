from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.schemas.chat import ContactCreate, ContactWithUserResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List

router = APIRouter(prefix="/api/contacts", tags=["contacts"])
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


@router.get("/", response_model=List[ContactWithUserResponse])
async def list_contacts(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    return await ChatService.get_contacts_with_users(db, user_id)


@router.post("/", response_model=ContactWithUserResponse, status_code=status.HTTP_201_CREATED)
async def add_contact(
    data: ContactCreate,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    try:
        contact = await ChatService.add_contact(
            db, user_id, data.contact_id, data.display_name
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )

    from app.models.user import User
    from sqlalchemy import select
    from app.schemas.chat import UserBrief

    user_result = await db.execute(select(User).where(User.id == contact.contact_id))
    user = user_result.scalar_one_or_none()

    return {
        "id": contact.id,
        "user_id": contact.user_id,
        "contact": UserBrief(
            id=user.id,
            username=user.username,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
        ),
        "display_name": contact.display_name,
        "created_at": contact.created_at,
    }


@router.delete("/{contact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_contact(
    contact_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    success = await ChatService.remove_contact(db, user_id, contact_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Contact not found"
        )
