import os
import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import settings from app
try:
    # Add app directory to path
    # ml/text_normalizer.py -> ml -> project root -> app
    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    app_dir = os.path.abspath(app_dir)

    # Ensure in path at beginning
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    else:
        sys.path.remove(app_dir)
        sys.path.insert(0, app_dir)

    logger.warning(f"[TextNormalizer] Trying to load settings from: {app_dir}")

    # Import the SETTINGS CLASS, not the instance
    from config import Settings

    _settings = Settings()
    LITELLM_MODEL = _settings.LITELLM_MODEL
    GROQ_API_KEY = _settings.GROQ_API_KEY

    logger.warning(
        f"[TextNormalizer] Settings loaded - GROQ_API_KEY: {bool(GROQ_API_KEY)}"
    )

    # Verificar que se cargó correctamente
    if not GROQ_API_KEY:
        logger.warning(
            "[TextNormalizer] GROQ_API_KEY empty in settings, trying os.getenv"
        )
        GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        LITELLM_MODEL = os.getenv("LITELLM_MODEL", LITELLM_MODEL)
except Exception as e:
    logger.warning(f"[TextNormalizer] Could not load settings: {e}, using env vars")
    LITELLM_MODEL = os.getenv("LITELLM_MODEL", "groq/llama-3.1-8b-instant")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

logger.warning(f"[TextNormalizer] GROQ_API_KEY loaded: {bool(GROQ_API_KEY)}")

logger.warning(f"[TextNormalizer] GROQ_API_KEY loaded: {bool(GROQ_API_KEY)}")

SYSTEM_PROMPT = """Eres el procesador de SignText, una app móvil de comunicación
para personas con dificultades del habla. Recibes texto crudo generado por
dos modos de entrada gestual y debes convertirlo en español natural y fluido.

## Modos de entrada

### MODO 1 — Handshapes (conceptos)
Recibes una lista de conceptos en mayúsculas separados por espacios,
en el orden en que el usuario los gestualizó.
Ejemplo de entrada: "AGUA FAVOR"
Ejemplo de salida: "¿me puedes dar agua, por favor?"

Reglas para modo 1:
- Infiere la intención comunicativa más probable dado el contexto
- Agrega artículos, preposiciones y conjugaciones necesarias
- Si parece una pregunta,órmulala como pregunta
- Si parece una necesidad urgente, usa tono directo
- Mantén la frase corta (máximo 12 palabras) para que TTS suene natural

### MODO 2 — Fingerspelling (letras continuas)
Recibes texto en español escrito sin espacios ni puntuación, con posibles
errores de reconocimiento de caracteres, letras duplicadas, o letras faltantes.
El usuario fingerspellea letra por letra, y el modelo de visión puede:
- Repetir la misma letra muchas veces (ej: "hhhooollaaa" = "hola")
- Omitir letras (ej: "hoollaa" = "hola")
- Equivocar letras por outras similares (ej: "hhhooyy" = "hola")
- Mezclar letras de dos palabras sin pausa visible (ej: "hhoolllioiuaaaa" = "hola")

Ejemplos de entrada → salida:
- "hhhooollaaa" → "hola"
- "hhhlllaaa" → "hola"
- "aaaagggttyyyuuuaaa ffffaaavvvooooorrr" → "agua por favor"
- "hoollaa" → "hola"
- "bbuenooddiaa" → "buenos días"
- "mmmuyy bbbieen" → "muy bien"
- "hhoolllioiuaaaaacoccoooommmommomoessstttaass" → "hola, ¿cómo estás?"
- "bbyeenn" → "bien"
- "eesstoooyyy" → "esto"

Reglas para modo 2:
- Las letras repetidas consecutivamente cuentan como UNA sola letra
  (elimina duplicados consecutivos: "hhhooollaaa" → "hola")
- Si faltan letras pero se forma una palabra conocida del español, complétala:
  - "hll" o "hhl" o "hlllaaa" → "hola"
  - "bnn" o "bbnn" → "bien"
  - "yy"→"ll" cuando tiene sentido
- Detecta transiciones de palabras aunque no haya espacio:
  - Si hay cambios rápidos de letras (de 'l' a otra), agrega espacio
  - "holacomo" → "hola como", "buenoye" → "bueno ye"
  - Patrones como "hola" + "como" = "hola como"
- Corrige errores comunes de reconocimiento:
  - "nn"→"n", "yy"→"ll", "uu"→"u"
  - "ooo" puede ser "o" o mantener si es otra palabra
- Segmenta las palabras correctamente usando contexto del español
- Aplica concordancia de género y número ("bueno dia" → "buenos días")
- Agrega puntuación y acentos
- Si hay ambigüedad en la segmentación, elige la interpretación más común
- SIEMPRE responde con una interpretacion logica aunque falten letras

## Reglas generales para ambos modos

- Responde ÚNICAMENTE con el texto corregido, sin explicaciones ni comentarios
- No agregues comillas ni formato especial
- El texto debe sonar natural cuando lo lee un TTS en voz alta
- Usa español neutro latinoamericano (no voseo, no jerga regional)
- Si el input está completamente vacío o es incomprensible, responde exactamente: [entrada no reconocida]
- Máximo 20 palabras en la respuesta — es una app de comunicación, no un chat

## Formato de entrada del sistema

Recibirás un JSON con este formato:
{
  "mode": "handshape" | "fingerspelling",
  "input": "texto o CONCEPTOS aquí",
  "context": "última frase dicha (puede estar vacío)"
}

El campo "context" contiene la frase anterior del usuario si existe,
úsalo para mantener coherencia conversacional cuando sea relevante."""


class TextNormalizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = os.getenv("LITELLM_MODEL", LITELLM_MODEL)
        self._initialized = True
        logger.warning(f"[TextNormalizer] Initialized with model: {self.model}")
        logger.warning(f"[TextNormalizer] GROQ_API_KEY set: {bool(GROQ_API_KEY)}")

    def normalize(self, text: str, mode: str, context: str = "") -> str:
        """Normalize raw text to natural Spanish based on mode."""
        logger.warning(
            f"[TextNormalizer] normalize() called with text='{text}', mode='{mode}'"
        )

        if not text or not text.strip():
            logger.warning(
                f"[TextNormalizer] Empty text, returning '[entrada no reconocidas]'"
            )
            return "[entrada no reconocida]"

        try:
            import litellm
            from litellm import completion

            logger.warning(
                f"[TextNormalizer] About to call completion with model={self.model}"
            )

            # Verificar API key
            if not GROQ_API_KEY:
                logger.error("[TextNormalizer] GROQ_API_KEY is EMPTY or NOT SET!")
                return "[error: GROQ_API_KEY not configured]"

            # Set API key para litellm
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
            logger.warning(
                f"[TextNormalizer] GROQ_API_KEY configured: {GROQ_API_KEY[:10]}..."
            )

            user_prompt = f'{{"mode": "{mode}", "input": "{text.strip()}", "context": "{context}"}}'
            logger.warning(f"[TextNormalizer] Sending to Groq: {user_prompt}")

            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
                temperature=0.1,
            )

            logger.warning(f"[TextNormalizer] Response received: {response}")
            result = response.choices[0].message.content.strip()
            logger.warning(f"[TextNormalizer] Groq response: '{result}'")

            logger.warning(
                f"[TextNormalizer] Normalized '{text}' ({mode}) -> '{result}'"
            )

            return result

        except Exception as e:
            logger.error(f"[TextNormalizer] Error normalizing text: {e}")
            import traceback

            logger.error(f"[TextNormalizer] Traceback: {traceback.format_exc()}")
            return f"[error: {str(e)[:50]}]"


text_normalizer = None


def get_text_normalizer() -> TextNormalizer:
    global text_normalizer
    if text_normalizer is None:
        text_normalizer = TextNormalizer()
    return text_normalizer
