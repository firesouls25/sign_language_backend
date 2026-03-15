#!/usr/bin/env python3
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np

async def test_websocket():
    uri = "ws://localhost:8000/ws/translate"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Send multiple frames to trigger recognition
            for i in range(10):
                # Create a test frame (gradient)
                test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                test_frame[:, :, 0] = i * 25  # Blue gradient
                test_frame[:, :, 1] = 100
                test_frame[:, :, 2] = 150
                
                _, buffer = cv2.imencode('.jpg', test_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send(json.dumps({
                    "type": "frame",
                    "data": frame_base64
                }))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    data = json.loads(response)
                    print(f"Frame {i+1}: text='{data.get('text')}', confidence={data.get('confidence')}")
                except asyncio.TimeoutError:
                    print(f"Frame {i+1}: Timeout")
                
                await asyncio.sleep(0.1)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
