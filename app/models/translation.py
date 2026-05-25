from sqlalchemy import Column, String, DateTime, Float, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class Translation(Base):
    __tablename__ = "translations"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True)
    video_url = Column(String, nullable=True)
    text_result = Column(String, nullable=False)
    audio_url = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
