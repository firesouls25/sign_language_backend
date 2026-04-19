import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
errores de reconocimiento de caracteres.
Ejemplo de entrada: "buenodiacomoesta"
Ejemplo de salida: "buenos días, ¿cómo estás?"

Reglas para modo 2:
- Segmenta las palabras correctamente usando contexto del español
- Corrige errores tipográficos menores (letras cambiadas, faltantes)
- Aplica concordancia de género y número ("bueno dia" → "buenos días")
- Agrega puntuación y acentos
- Si hay ambigüedad en la segmentación, elige la interpretación más común
- NO inventes palabras que no estaban en el input original

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

        self.model = os.getenv("LITELLM_MODEL", "groq/llama-3.1-8b-instant")
        self._initialized = True
        logger.info(f"[TextNormalizer] Initialized with model: {self.model}")

    def normalize(self, text: str, mode: str, context: str = "") -> str:
        """Normalize raw text to natural Spanish based on mode."""
        if not text or not text.strip():
            return "[entrada no reconocida]"

        try:
            import litellm
            from litellm import completion

            user_prompt = f'{{"mode": "{mode}", "input": "{text.strip()}", "context": "{context}"}}'

            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=100,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip()

            logger.info(f"[TextNormalizer] Normalized '{text}' ({mode}) -> '{result}'")

            return result

        except Exception as e:
            logger.error(f"[TextNormalizer] Error normalizing text: {e}")
            return f"[error: {str(e)[:50]}]"


text_normalizer = None


def get_text_normalizer() -> TextNormalizer:
    global text_normalizer
    if text_normalizer is None:
        text_normalizer = TextNormalizer()
    return text_normalizer
