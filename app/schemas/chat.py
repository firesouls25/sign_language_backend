from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UserBrief(BaseModel):
    id: str
    username: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


class ContactCreate(BaseModel):
    contact_id: str
    display_name: Optional[str] = None


class ContactResponse(BaseModel):
    id: str
    user_id: str
    contact_id: str
    display_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ContactWithUserResponse(BaseModel):
    id: str
    user_id: str
    contact: UserBrief
    display_name: Optional[str] = None
    created_at: datetime


class ConversationCreate(BaseModel):
    participant_id: str


class MessageBrief(BaseModel):
    id: str
    text: str
    created_at: datetime


class ConversationResponse(BaseModel):
    id: str
    creator_id: str
    participant_id: str
    last_message_text: Optional[str] = None
    last_message_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationWithUserResponse(BaseModel):
    id: str
    other_user: UserBrief
    is_self: bool
    last_message_text: Optional[str] = None
    last_message_at: Optional[datetime] = None
    created_at: datetime


class MessageCreate(BaseModel):
    text: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    confidence_score: Optional[float] = None
    message_type: str = "translation"


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    sender_id: str
    text: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    confidence_score: Optional[float] = None
    message_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageHistoryResponse(BaseModel):
    items: List[MessageResponse]
    total: int
    page: int
    size: int
    pages: int
