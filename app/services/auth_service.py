from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, Token
from app.utils.security import verify_password, get_password_hash, create_access_token, create_refresh_token, decode_token
from app.utils.redis_client import redis_client
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
    async def get_user_from_token(token: str) -> Optional[str]:
        payload = decode_token(token)
        if payload and payload.get("type") == "access":
            jti = payload.get("jti")
            if jti and await redis_client.is_token_blacklisted(jti):
                return None
            return payload.get("sub")
        return None

    @staticmethod
    async def refresh_access_token(refresh_token: str) -> Optional[str]:
        payload = decode_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            jti = payload.get("jti")
            if jti and await redis_client.is_token_blacklisted(jti):
                return None
            return create_access_token(data={"sub": payload.get("sub")})
        return None

    @staticmethod
    async def logout(access_token: str, refresh_token: Optional[str] = None):
        # Blacklist access token
        payload = decode_token(access_token)
        if payload:
            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti and exp:
                # Calculate remaining time
                import datetime
                remaining = int(exp - datetime.datetime.utcnow().timestamp())
                if remaining > 0:
                    await redis_client.blacklist_token(jti, remaining)

        # Blacklist refresh token if provided
        if refresh_token:
            payload = decode_token(refresh_token)
            if payload:
                jti = payload.get("jti")
                exp = payload.get("exp")
                if jti and exp:
                    import datetime
                    remaining = int(exp - datetime.datetime.utcnow().timestamp())
                    if remaining > 0:
                        await redis_client.blacklist_token(jti, remaining)
