from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.database import init_db
from app.config import settings
from app.api.routes import auth, translation
from app.api.routes.websocket import websocket_endpoint
from app.utils.redis_client import redis_client
from app.utils.logging_config import setup_logging, LoggingMiddleware
import uuid

# Initialize logging
setup_logging()

app = FastAPI(
    title="LSC Translator API",
    description="Colombian Sign Language Translation Backend",
    version="1.0.0",
    debug=settings.DEBUG,
)

# Add Structured Logging Middleware
app.add_middleware(LoggingMiddleware)

# Static files for local storage
import os
from app.config import settings
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await init_db()
    await redis_client.connect()


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()


app.include_router(auth.router)
app.include_router(translation.router)

if settings.ENABLE_DEV_ROUTES:
    from app.api.routes import dev

    app.include_router(dev.router)


@app.websocket("/ws/translate")
async def websocket_route(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await websocket_endpoint(websocket, client_id)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.APP_ENV}


@app.get("/")
async def root():
    return {"message": "LSC Translator API", "docs": "/docs"}
