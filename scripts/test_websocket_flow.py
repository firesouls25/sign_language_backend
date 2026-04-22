#!/usr/bin/env python3
"""
Test script que simula el flujo completo del Frontend Flutter:
1. Envía landmarks (formato que usa el frontend: 21 puntos × 3 = 63 valores)
2. Usa WebSocket para conectar al backend
3. Accumula letras mientras el usuario fingerspellea
4. Envía "finalize" para obtener texto normalizado de Groq
"""

import asyncio
import json
import os
import sys
import numpy as np
from datetime import datetime
from collections import deque

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "sign_language_model")
sys.path.insert(0, ML_DIR)
os.chdir(ML_DIR)

import websocket


# Configuración
WS_URL = "ws://localhost:8000/ws/translate"
BACKEND_URL = "http://localhost:8000"


def generate_fake_hand_landmarks(letter: str = "H") -> list:
    """
    Genera landmarks simulados para una letra específica.
    El formato es exactamente igual que el frontendFlutter:
    21 puntos × 3 coordenadas (x, y, z) = 63 valores
    """
    # Semilla basada en la letra para variación consistente
    np.random.seed(ord(letter) % 26)

    landmarks = []
    for i in range(21):
        # Simular variaciones menores por frame
        base_x = np.random.uniform(0.3, 0.7)
        base_y = np.random.uniform(0.3, 0.7)
        base_z = np.random.uniform(-0.1, 0.1)

        landmarks.append([base_x, base_y, base_z])

    return landmarks


class FrontendSimulator:
    """Simula el comportamiento del Frontend Flutter"""

    def __init__(self, token: str = None):
        self.ws = None
        self.token = token
        self.is_connected = False
        self.landmarks_sent = 0
        self.mode = "fingerspelling"
        self.is_translating = False

    async def connect(self):
        """Conectar al WebSocket del backend"""
        print(f"\n=== Conectando al backend: {WS_URL} ===")

        # Agregar token si existe
        ws_url = WS_URL
        if self.token:
            ws_url = f"{WS_URL}?token={self.token}"

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_open=self.on_open,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        # Run en thread separada
        import threading

        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Esperar conexión
        for _ in range(10):
            await asyncio.sleep(0.2)
            if self.is_connected:
                print("✓ Conectado al WebSocket")
                return True

        print("✗ No se pudo conectar")
        return False

    def on_open(self, ws):
        print("[WS] Conexión abierta")
        self.is_connected = True

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            print(f"\n[WS] Mensaje recibido: {msg_type}")
            print(f"  Data: {data}")

            if msg_type == "translation":
                text = data.get("text", "")
                confidence = data.get("confidence", 0)
                is_finalized = data.get("is_finalized", False)
                mode = data.get("mode", "")

                print(f"\n{'=' * 50}")
                print(f"RESULTADO:")
                print(f"  Texto: '{text}'")
                print(f"  Confidence: {confidence}")
                print(f"  Finalized: {is_finalized}")
                print(f"  Modo: {mode}")
                print(f"{'=' * 50}\n")

        except Exception as e:
            print(f"[WS] Error parseando mensaje: {e}")

    def on_error(self, ws, error):
        print(f"[WS] Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[WS] Conexión cerrada: {close_status_code} - {close_msg}")
        self.is_connected = False

    def send_landmarks(self, landmarks: list):
        """Envía landmarks al backend (igual que Flutter)"""
        if not self.is_connected or not self.ws:
            return

        # Formato exacto que usa Flutter:
        # {"left_hand": [[x,y,z], ...], "right_hand": [[x,y,z], ...]}
        message = {
            "type": "landmarks",
            "data": {"left_hand": landmarks, "right_hand": [], "pose": []},
            "mode": self.mode,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }

        self.ws.send(json.dumps(message))
        self.landmarks_sent += 1
        print(
            f"[WS] Landmarks enviados #{self.landmarks_sent}: {len(landmarks)} puntos"
        )

    def send_finalize(self):
        """Envía señal de finalizar (igual que Flutter botón 'Listo')"""
        if not self.is_connected or not self.ws:
            return

        message = {"type": "finalize"}
        self.ws.send(json.dumps(message))
        print(f"\n[WS] finalize enviado!")

    def send_reset(self):
        """Reinicia la secuencia"""
        if not self.is_connected or not self.ws:
            return

        message = {"type": "reset"}
        self.ws.send(json.dumps(message))
        print("[WS] Reset enviado")

    def send_mode(self, mode: str):
        """Envía el modo al backend"""
        if not self.is_connected or not self.ws:
            return

        message = {"type": "set_mode", "mode": mode}
        self.ws.send(json.dumps(message))
        print(f"[WS] Modo enviado: {mode}")
        self.mode = mode

    def close(self):
        """Cierra la conexión"""
        if self.ws:
            self.ws.close()
        self.is_connected = False


async def test_flow():
    """Test completo del flujo fingerspelling"""
    print("=" * 60)
    print("TEST: Simulación de Frontend Flutter")
    print("=" * 60)

    simulator = FrontendSimulator()

    # 1. Conectar
    connected = await simulator.connect()
    if not connected:
        print("✗ No se pudo conectar al backend")
        return

    # 2. Configurar modo fingerspelling
    simulator.send_mode("fingerspelling")
    await asyncio.sleep(0.5)

    # 3. Simular fingerspelling: escribir "HOLA"
    test_letters = ["H", "O", "L", "A"]

    print("\n" + "=" * 60)
    print("ENVIANDO LETRAS: 'HOLA'")
    print("=" * 60 + "\n")

    for letter in test_letters:
        # Enviar varios frames para cada letra (igual que el flujo real)
        for frame in range(8):
            landmarks = generate_fake_hand_landmarks(letter)
            simulator.send_landmarks(landmarks)
            await asyncio.sleep(0.05)  # 50ms entre frames

        print(f"Letra '{letter}' enviada ({8} frames)")
        await asyncio.sleep(0.3)  # Pausa entre letras

    # 4. Finalizar (presionar botón "Listo")
    print("\n" + "=" * 60)
    print("PRESIONANDO BOTÓN 'LISTO' (FINALIZE)")
    print("=" * 60 + "\n")

    simulator.send_finalize()

    # Esperar respuesta
    await asyncio.sleep(5)

    # 5. Cerrar
    simulator.close()
    print("\nTest completado")


async def test_simple_letter():
    """Test simple de una sola letra"""
    print("=" * 60)
    print("TEST: Letra simple 'A'")
    print("=" * 60)

    simulator = FrontendSimulator()

    connected = await simulator.connect()
    if not connected:
        print("✗ No se pudo conectar")
        return

    simulator.send_mode("fingerspelling")
    await asyncio.sleep(0.5)

    # Enviar 10 frames de la letra "A"
    for i in range(10):
        landmarks = generate_fake_hand_landmarks("A")
        simulator.send_landmarks(landmarks)
        await asyncio.sleep(0.05)

    print("\nEnviando finalize...")
    simulator.send_finalize()

    await asyncio.sleep(5)
    simulator.close()


async def test_no_hand():
    """Test sin mano detectada"""
    print("=" * 60)
    print("TEST: Sin mano (landmarks vacíos)")
    print("=" * 60)

    simulator = FrontendSimulator()

    connected = await simulator.connect()
    if not connected:
        print("✗ No se pudo conectar")
        return

    simulator.send_mode("fingerspelling")
    await asyncio.sleep(0.5)

    # Enviar landmarks vacíos
    empty_landmarks = []
    for _ in range(5):
        simulator.send_landmarks(empty_landmarks)
        await asyncio.sleep(0.05)

    print("\nEnviando finalize sin letras...")
    simulator.send_finalize()

    await asyncio.sleep(3)
    simulator.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test flujo fingerspelling")
    parser.add_argument(
        "--test",
        choices=["full", "letter", "empty"],
        default="full",
        help="Tipo de test",
    )
    args = parser.parse_args()

    if args.test == "full":
        asyncio.run(test_flow())
    elif args.test == "letter":
        asyncio.run(test_simple_letter())
    elif args.test == "empty":
        asyncio.run(test_no_hand())


if __name__ == "__main__":
    main()
