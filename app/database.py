from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings
import logging

logger = logging.getLogger(__name__)

engine = None
AsyncSessionLocal = None
DB_AVAILABLE = False

def init_db_engine():
    global engine, AsyncSessionLocal, DB_AVAILABLE
    try:
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            pool_pre_ping=True,
        )
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        DB_AVAILABLE = True
        logger.info("Database connection established")
    except Exception as e:
        logger.warning(f"Database not available: {e}")
        engine = None
        AsyncSessionLocal = None
        DB_AVAILABLE = False

init_db_engine()

Base = declarative_base()


async def get_db():
    if not DB_AVAILABLE:
        yield None
        return
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    if not DB_AVAILABLE:
        logger.warning("Database not available, skipping initialization")
        return
    if engine is not None:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")
