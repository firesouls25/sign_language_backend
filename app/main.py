import os
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.api.routes import auth, translation, contacts, conversations, users
from app.api.routes.websocket import websocket_endpoint
from app.config import settings
from app.database import init_db

app = FastAPI(
    title="LSC Translator API",
    description="Colombian Sign Language Translation Backend",
    version="1.0.0",
    debug=settings.DEBUG,
)

# Serve uploaded files
if os.path.exists(settings.UPLOAD_DIR):
    app.mount(
        "/api/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads"
    )

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

    from app.services.ai_service import get_ai_service

    ai_service = get_ai_service()
    detector = ai_service.detector

    print(f"SignDetectorManager initialized: {detector._initialized}")
    print(
        f"Handshape model loaded: {detector._handshape_recognizer is not None if detector._handshape_recognizer else False}"
    )
    print(
        f"Fingerspelling model loaded: {detector._fingerspelling_recognizer is not None if detector._fingerspelling_recognizer else False}"
    )


app.include_router(auth.router)
app.include_router(translation.router)
app.include_router(contacts.router)
app.include_router(conversations.router)
app.include_router(users.router)

if settings.ENABLE_DEV_ROUTES:
    from app.api.routes import dev

    app.include_router(dev.router)


@app.websocket("/ws/translate")
async def websocket_route(websocket: WebSocket, token: str = Query(None)):
    await websocket_endpoint(websocket, token)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.APP_ENV}


@app.get("/")
async def root():
    return {"message": "LSC Translator API", "docs": "/docs"}
