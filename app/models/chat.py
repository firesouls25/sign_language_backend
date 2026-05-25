from sqlalchemy import Column, String, DateTime, Float, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    contact_id = Column(String, ForeignKey("users.id"), nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "contact_id", name="uq_user_contact"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    creator_id = Column(String, ForeignKey("users.id"), nullable=False)
    participant_id = Column(String, ForeignKey("users.id"), nullable=False)
    last_message_text = Column(String, nullable=True)
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def is_participant(self, user_id: str) -> bool:
        return self.creator_id == user_id or self.participant_id == user_id

    def other_user_id(self, user_id: str) -> str:
        if self.creator_id == user_id:
            return self.participant_id
        return self.creator_id


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String, ForeignKey("conversations.id"), nullable=False, index=True
    )
    sender_id = Column(String, ForeignKey("users.id"), nullable=False)
    text = Column(String, nullable=False)
    video_url = Column(String, nullable=True)
    audio_url = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    message_type = Column(String, default="translation")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
