from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime, timezone


class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


class UserCreate(UserBase):
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password cannot be longer than 72 bytes")
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(UserBase):
    id: str
    is_oauth: bool
    is_verified: bool
    oauth_provider: Optional[str] = None
    translation_count: int
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None


class OAuthProviders(BaseModel):
    google: bool = False
    apple: bool = False


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class GoogleTokenRequest(BaseModel):
    id_token: str
