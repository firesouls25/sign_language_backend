from typing import Optional
from authlib.integrations.starlette_client import OAuth
from authlib.jose import jwt
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.config import settings
from app.models.user import User
from app.utils.security import create_access_token, create_refresh_token
import httpx

oauth = OAuth()

if settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if settings.APPLE_CLIENT_ID:
    oauth.register(
        name="apple",
        client_id=settings.APPLE_CLIENT_ID,
        client_secret=settings.APPLE_CLIENT_SECRET,
        server_metadata_url="https://appleid.apple.com/.well-known/openid-configuration",
        client_kwargs={"scope": "name email"},
    )


async def get_google_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Google access token",
            )
        return response.json()


async def verify_google_id_token(id_token: str) -> dict:
    """Verify Google ID token and return user info"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://oauth2.googleapis.googleapis.com/tokeninfo",
                params={"id_token": id_token},
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Google ID token",
                )
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to verify Google token: {str(e)}",
        )


async def verify_google_token_and_get_user(
    db: AsyncSession,
    id_token: str,
) -> Optional[User]:
    """Verify Google ID token and find or create user"""
    token_info = await verify_google_id_token(id_token)

    aud = token_info.get("aud")
    if aud != settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token audience mismatch",
        )

    sub = token_info.get("sub")
    email = token_info.get("email")

    if not sub or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token payload",
        )

    return await find_or_create_oauth_user(
        db=db,
        provider="google",
        provider_id=sub,
        email=email,
    )


async def get_apple_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://appleid.apple.com/auth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": settings.APPLE_CLIENT_ID,
                "client_secret": settings.APPLE_CLIENT_SECRET,
            },
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Apple access token",
            )

        id_token = response.json().get("id_token")
        if not id_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Apple response"
            )

        claims = jwt.decode(id_token, settings.APPLE_CLIENT_SECRET)
        return {
            "sub": claims.get("sub"),
            "email": claims.get("email"),
            "name": claims.get("name", {}),
        }


async def find_or_create_oauth_user(
    db: AsyncSession,
    provider: str,
    provider_id: str,
    email: str,
    full_name: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> User:
    result = await db.execute(
        select(User).where(
            User.oauth_provider == provider,
            User.oauth_provider_id == provider_id,
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        result = await db.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            existing_user.oauth_provider = provider
            existing_user.oauth_provider_id = provider_id
            existing_user.avatar_url = avatar_url or existing_user.avatar_url
            if not existing_user.full_name and full_name:
                existing_user.full_name = full_name
            await db.commit()
            await db.refresh(existing_user)
            return existing_user

        username = email.split("@")[0]
        result = await db.execute(select(User).where(User.username == username))
        base_username = username
        counter = 1
        while result.scalar_one_or_none():
            username = f"{base_username}{counter}"
            counter += 1
            result = await db.execute(select(User).where(User.username == username))

        user = User(
            email=email,
            username=username,
            full_name=full_name,
            avatar_url=avatar_url,
            is_oauth=True,
            oauth_provider=provider,
            oauth_provider_id=provider_id,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return user


async def create_tokens_for_user(user: User) -> dict:
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


def get_oauth_redirect_url(provider: str) -> str:
    backend_url = (
        settings.BACKEND_URL
        if hasattr(settings, "BACKEND_URL")
        else f"http://{settings.HOST}:{settings.PORT}"
    )

    if settings.DEBUG:
        callback_path = "/callback-dev"
    else:
        callback_path = "/callback-deep-link"
    
    url = f"{backend_url}/api/auth{callback_path}/{provider}"
    print(f"DEBUG redirect_uri: {url}")  # <-- agrega esto
    return url
