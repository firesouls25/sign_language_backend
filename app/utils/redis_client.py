import redis.asyncio as redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self):
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            # Fallback if connect() hasn't been called or failed
            self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._client

    async def connect(self):
        try:
            self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            await self._client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {settings.REDIS_URL}: {e}")
            self._client = None

    async def close(self):
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")

    async def blacklist_token(self, token_jti: str, expire_seconds: int):
        if self._client:
            await self._client.set(f"blacklist:{token_jti}", "1", ex=expire_seconds)

    async def is_token_blacklisted(self, token_jti: str) -> bool:
        if not self._client:
            return False
        return await self._client.exists(f"blacklist:{token_jti}") > 0

    async def set_value(self, key: str, value: str, expire_seconds: int):
        if self._client:
            await self._client.set(key, value, ex=expire_seconds)

    async def get_value(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        return await self._client.get(key)

    async def delete_value(self, key: str):
        if self._client:
            await self._client.delete(key)


redis_client = RedisClient()

# For type hinting
from typing import Optional
