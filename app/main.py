from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.database import init_db
from app.config import settings
from app.api.routes import auth, translation
from app.api.routes.websocket import websocket_endpoint
import uuid

app = FastAPI(
    title="LSC Translator API",
    description="Colombian Sign Language Translation Backend",
    version="1.0.0",
    debug=settings.DEBUG,
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
