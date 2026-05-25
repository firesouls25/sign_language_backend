from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from app.models.chat import Contact, Conversation, Message
from app.models.user import User
from app.schemas.chat import (
    ContactWithUserResponse,
    ConversationWithUserResponse,
    UserBrief,
    MessageResponse,
)
from typing import Optional, List, Tuple
from datetime import datetime, timezone


class ChatService:

    @staticmethod
    async def add_contact(
        db: AsyncSession, user_id: str, contact_id: str, display_name: Optional[str] = None
    ) -> Contact:
        if user_id == contact_id:
            raise ValueError("Cannot add yourself as a contact")

        existing = await db.execute(
            select(Contact).where(
                Contact.user_id == user_id, Contact.contact_id == contact_id
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError("Contact already exists")

        contact = Contact(
            user_id=user_id, contact_id=contact_id, display_name=display_name
        )
        db.add(contact)
        await db.commit()
        await db.refresh(contact)
        return contact

    @staticmethod
    async def remove_contact(db: AsyncSession, user_id: str, contact_id: str) -> bool:
        result = await db.execute(
            select(Contact).where(
                Contact.user_id == user_id, Contact.id == contact_id
            )
        )
        contact = result.scalar_one_or_none()
        if not contact:
            return False
        await db.delete(contact)
        await db.commit()
        return True

    @staticmethod
    async def get_contacts_with_users(
        db: AsyncSession, user_id: str
    ) -> List[dict]:
        query = (
            select(Contact, User)
            .join(User, Contact.contact_id == User.id)
            .where(Contact.user_id == user_id)
            .order_by(Contact.created_at.desc())
        )
        result = await db.execute(query)
        rows = result.all()
        items = []
        for contact, user in rows:
            items.append({
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
            })
        return items

    @staticmethod
    async def create_conversation(
        db: AsyncSession, creator_id: str, participant_id: str
    ) -> Conversation:
        existing = await db.execute(
            select(Conversation).where(
                or_(
                    and_(
                        Conversation.creator_id == creator_id,
                        Conversation.participant_id == participant_id,
                    ),
                    and_(
                        Conversation.creator_id == participant_id,
                        Conversation.participant_id == creator_id,
                    ),
                )
            )
        )
        conv = existing.scalar_one_or_none()
        if conv:
            return conv

        conv = Conversation(creator_id=creator_id, participant_id=participant_id)
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    @staticmethod
    async def get_conversations_with_users(
        db: AsyncSession, user_id: str
    ) -> List[dict]:
        query = (
            select(Conversation)
            .where(
                or_(
                    Conversation.creator_id == user_id,
                    Conversation.participant_id == user_id,
                )
            )
            .order_by(Conversation.last_message_at.desc().nullslast())
        )
        result = await db.execute(query)
        conversations = result.scalars().all()

        items = []
        for conv in conversations:
            other_id = conv.other_user_id(user_id)
            user_result = await db.execute(
                select(User).where(User.id == other_id)
            )
            other_user = user_result.scalar_one_or_none()
            if not other_user:
                continue

            items.append({
                "id": conv.id,
                "other_user": UserBrief(
                    id=other_user.id,
                    username=other_user.username,
                    full_name=other_user.full_name,
                    avatar_url=other_user.avatar_url,
                ),
                "is_self": conv.creator_id == conv.participant_id,
                "last_message_text": conv.last_message_text,
                "last_message_at": conv.last_message_at,
                "created_at": conv.created_at,
            })
        return items

    @staticmethod
    async def get_messages(
        db: AsyncSession, conversation_id: str, user_id: str, page: int = 1, size: int = 50
    ) -> Tuple[List[Message], int]:
        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = conv_result.scalar_one_or_none()
        if not conv or not conv.is_participant(user_id):
            return [], 0

        count_query = select(func.count()).select_from(
            select(Message).where(Message.conversation_id == conversation_id).subquery()
        )
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        offset = (page - 1) * size
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(size)
            .offset(offset)
        )
        result = await db.execute(query)
        messages = result.scalars().all()

        return list(reversed(messages)), total

    @staticmethod
    async def send_message(
        db: AsyncSession, conversation_id: str, sender_id: str, text: str,
        video_url: Optional[str] = None, audio_url: Optional[str] = None,
        confidence_score: Optional[float] = None,
        message_type: str = "translation",
    ) -> Optional[Message]:
        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = conv_result.scalar_one_or_none()
        if not conv or not conv.is_participant(sender_id):
            return None

        msg = Message(
            conversation_id=conversation_id,
            sender_id=sender_id,
            text=text,
            video_url=video_url,
            audio_url=audio_url,
            confidence_score=confidence_score,
            message_type=message_type,
        )
        db.add(msg)

        conv.last_message_text = text
        conv.last_message_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(msg)
        return msg

    @staticmethod
    async def search_users(
        db: AsyncSession, query_str: str, current_user_id: str, limit: int = 20
    ) -> List[User]:
        stmt = (
            select(User)
            .where(
                and_(
                    User.id != current_user_id,
                    or_(
                        User.username.ilike(f"%{query_str}%"),
                        User.full_name.ilike(f"%{query_str}%"),
                    ),
                )
            )
            .limit(limit)
        )
        result = await db.execute(stmt)
        return result.scalars().all()
