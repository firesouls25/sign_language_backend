from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class TranslationBase(BaseModel):
    text_result: str


class TranslationCreate(TranslationBase):
    pass


class TranslationResponse(TranslationBase):
    id: str
    user_id: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TranslationWebSocket(BaseModel):
    type: str
    data: Optional[dict] = None
    message: Optional[str] = None
    code: Optional[str] = None


class TranslationHistoryResponse(BaseModel):
    items: List[TranslationResponse]
    total: int
    page: int
    size: int
    pages: int
