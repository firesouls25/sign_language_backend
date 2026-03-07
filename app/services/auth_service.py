from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, Token
from app.utils.security import verify_password, get_password_hash, create_access_token, create_refresh_token, decode_token
from app.database import DB_AVAILABLE
from typing import Optional
import uuid


class AuthService:
    @staticmethod
    async def register(db: AsyncSession, user_data: UserCreate) -> Optional[User]:
        if not DB_AVAILABLE:
            return None
        
        result = await db.execute(select(User).where(User.email == user_data.email))
        existing_user = result.scalar_one_or_none()
        if existing_user:
            return None
        
        result = await db.execute(select(User).where(User.username == user_data.username))
        existing_username = result.scalar_one_or_none()
        if existing_username:
            return None
        
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            id=str(uuid.uuid4()),
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
        )
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        return db_user

    @staticmethod
    async def login(db: AsyncSession, credentials: UserLogin) -> Optional[Token]:
        if not DB_AVAILABLE:
            return None
        
        result = await db.execute(select(User).where(User.username == credentials.username))
        user = result.scalar_one_or_none()
        
        if not user or not verify_password(credentials.password, user.hashed_password):
            return None
        
        access_token = create_access_token(data={"sub": user.id})
        refresh_token = create_refresh_token(data={"sub": user.id})
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        if not DB_AVAILABLE:
            return None
        
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    def get_user_from_token(token: str) -> Optional[str]:
        payload = decode_token(token)
        if payload and payload.get("type") == "access":
            return payload.get("sub")
        return None

    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[str]:
        payload = decode_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            return create_access_token(data={"sub": payload.get("sub")})
        return None
