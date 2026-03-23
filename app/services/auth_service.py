from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.utils.exceptions import AppException
from app.schemas.user import (
    UserCreate,
    UserLogin,
    Token,
    UserUpdate,
    ChangePasswordRequest,
)
from app.utils.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.utils.redis_client import redis_client
from app.services.email_service import email_service
from app.database import DB_AVAILABLE
from typing import Optional
import uuid
from datetime import datetime, timezone


class AuthService:
    @staticmethod
    async def register(db: AsyncSession, user_data: UserCreate) -> Optional[User]:
        if not DB_AVAILABLE:
            return None

        result = await db.execute(select(User).where(User.email == user_data.email))
        existing_user = result.scalar_one_or_none()
        if existing_user:
            return None

        result = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        existing_username = result.scalar_one_or_none()
        if existing_username:
            return None

        try:
            hashed_password = get_password_hash(user_data.password)
        except ValueError as e:
            raise AppException(str(e), status_code=400)
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

        try:
            import secrets

            token = secrets.token_urlsafe(32)
            await redis_client.set_value(f"verify:{token}", db_user.id, 86400)
            await email_service.send_verification_email(db_user.email, token)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Post-registration email/redis failed: {e}"
            )

        return db_user

    @staticmethod
    async def login(db: AsyncSession, credentials: UserLogin) -> Optional[Token]:
        if not DB_AVAILABLE:
            return None

        result = await db.execute(
            select(User).where(User.username == credentials.username)
        )
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
                remaining = int(exp - datetime.now(timezone.utc).timestamp())
                if remaining > 0:
                    await redis_client.blacklist_token(jti, remaining)

        # Blacklist refresh token if provided
        if refresh_token:
            payload = decode_token(refresh_token)
            if payload:
                jti = payload.get("jti")
                exp = payload.get("exp")
                if jti and exp:
                    remaining = int(exp - datetime.now(timezone.utc).timestamp())
                    if remaining > 0:
                        await redis_client.blacklist_token(jti, remaining)

    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        if not DB_AVAILABLE:
            return None
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    @staticmethod
    async def forgot_password(db: AsyncSession, email: str) -> bool:
        user = await AuthService.get_user_by_email(db, email)
        if not user:
            # For security, we might want to return True anyway, but for a dev flow, False is fine
            return False

        # Generate random token
        import secrets

        token = secrets.token_urlsafe(32)

        # Store in Redis (1 hour expiration)
        await redis_client.set_value(f"reset:{token}", user.id, 3600)

        # Send email
        await email_service.send_reset_password_email(user.email, token)
        return True

    @staticmethod
    async def reset_password(db: AsyncSession, token: str, new_password: str) -> bool:
        user_id = await redis_client.get_value(f"reset:{token}")
        if not user_id:
            return False

        user = await AuthService.get_user_by_id(db, user_id)
        if not user:
            return False

        # Update password
        user.hashed_password = get_password_hash(new_password)
        await db.commit()

        # Delete token from Redis
        await redis_client.delete_value(f"reset:{token}")
        return True

    @staticmethod
    async def verify_email(db: AsyncSession, token: str) -> bool:
        user_id = await redis_client.get_value(f"verify:{token}")
        if not user_id:
            return False

        user = await AuthService.get_user_by_id(db, user_id)
        if not user:
            return False

        # Update verification status
        user.is_verified = True
        await db.commit()

        # Delete token from Redis
        await redis_client.delete_value(f"verify:{token}")
        return True

    @staticmethod
    async def update_user(
        db: AsyncSession, user_id: str, update_data: UserUpdate
    ) -> Optional[User]:
        user = await AuthService.get_user_by_id(db, user_id)
        if not user:
            return None

        # Only update provided fields
        if update_data.full_name is not None:
            user.full_name = update_data.full_name
        if update_data.avatar_url is not None:
            user.avatar_url = update_data.avatar_url

        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def change_password(
        db: AsyncSession, user_id: str, data: ChangePasswordRequest
    ) -> bool:
        user = await AuthService.get_user_by_id(db, user_id)
        if not user:
            return False

        # Verify old password
        if user.is_oauth:
            # Maybe allow setting password for OAuth users for the first time
            # For now, if no hashed_password, just verify if it matches None/empty? nah
            if not user.hashed_password:
                # First time setting password
                pass
            elif not verify_password(data.old_password, user.hashed_password):
                return False
        elif not verify_password(data.old_password, user.hashed_password):
            return False

        # Update password
        user.hashed_password = get_password_hash(data.new_password)
        await db.commit()
        return True

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: str) -> bool:
        user = await AuthService.get_user_by_id(db, user_id)
        if not user:
            return False

        # Optional: Delete all related translations first if no cascade exists
        # In SQLAlchemy, it depends on relationship configuration

        await db.delete(user)
        await db.commit()
        return True
