# Sign Language Backend

API FastAPI para traducción de Lengua de Señas Colombiana (LSC).

## Requisitos

- Python 3.11+
- PostgreSQL
- Redis (opcional)
- GPU (opcional, para mejor rendimiento)

## Estructura del Proyecto

```
app/
├── main.py                 # App FastAPI
├── config.py              # Configuración
├── database.py            # SQLAlchemy
├── models/                # Modelos DB
│   ├── user.py
│   └── translation.py
├── schemas/               # Pydantic
│   ├── user.py
│   └── translation.py
├── api/
│   └── routes/
│       ├── auth.py        # Autenticación
│       ├── translation.py # REST translation
│       ├── websocket.py  # WebSocket
│       └── dev.py        # Rutas dev
├── services/
│   ├── ai_service.py    # ML
│   ├── auth_service.py
│   ├── oauth_service.py
│   └── tts_service.py
└── utils/
    ├── security.py
    └── redis_client.py

ml/
├── sign_detector_manager.py  # Carga de modelos
├── text_normalizer.py       # Normalización LLM
└── sign_language_model/
    ├── models/
    │   ├── recognizers/
    │   │   ├── handshape_model.py
    │   │   └── fingerspelling_model.py
    │   └── mediapipe/
    │       └── hand_detector.py
    └── utils/
        └── dtw.py
```

## Configuración

Variables de entorno en `.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
SECRET_KEY=tu-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

GROQ_API_KEY=
LITELLM_MODEL=groq/llama-3.1-8b-instant

FRONTEND_URL=http://localhost:5173
BACKEND_URL=http://localhost:8000

APP_ENV=development
DEBUG=true
ENABLE_DEV_ROUTES=true
```

## Modelos ML

### Handshape

- Ubicación: `ml/sign_language_model/data/reference/handshape`
-_input: Landmarks de mano (MediaPipe)

### Fingerspelling

- Ubicación: `ml/sign_language_model/data/reference/fingerspelling`
- Input: Landmarks de mano

### TextNormalizer

- Usado para normalizar texto reconocido
- Provider: Groq (LLM)

## WebSocket

### Conexión

```
ws://host:port/ws/translate?token=JWT_TOKEN
```

### Mensajes del Cliente

**Landmarks:**
```json
{
  "type": "landmarks",
  "data": {
    "left_hand": [[x, y, z], ...],
    "right_hand": [[x, y, z], ...]
  },
  "mode": "handshape|fingerspelling"
}
```

**Frame:**
```json
{
  "type": "frame",
  "data": [R, G, B, ...],
  "width": 640,
  "height": 480,
  "mode": "handshape|fingerspelling"
}
```

**Set Mode:**
```json
{
  "type": "set_mode",
  "mode": "handshape|fingerspelling"
}
```

**Reset:**
```json
{"type": "reset"}
```

**Finalize:**
```json
{"type": "finalize"}
```

### Mensajes del Servidor

**Traducción:**
```json
{
  "type": "translation",
  "text": "A",
  "confidence": 0.95,
  "mode": "handshape",
  "candidate": "B",
  "candidate_confidence": 0.7,
  "is_finalized": false
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Error description",
  "code": "ERROR_CODE"
}
```

**Pong:**
```json
{"type": "pong"}
```

## API REST

### Autenticación

```
POST /api/auth/register
POST /api/auth/login
POST /api/auth/refresh
POST /api/auth/logout
POST /api/auth/callback/<provider>
```

### Traducciones

```
GET  /api/translations
POST /api/translations
GET  /api/translations/{id}
PUT  /api/translations/{id}
DELETE /api/translations/{id}
```

## Ejecutar

```bash
# Desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Producción
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Dependencias Principales

- `fastapi` - Framework
- `uvicorn` - Servidor
- `sqlalchemy` - ORM
- `asyncpg` - PostgreSQL
- `pydantic` - Validación
- `python-jose` - JWT
- `python-multipart` - Form data
- `google-auth` - OAuth
- `groq` - LLM