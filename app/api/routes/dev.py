from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse
from app.config import settings
from app.services.ai_service import get_ai_service
import json

router = APIRouter(prefix="/dev", tags=["dev"])


def require_dev_mode():
    if not settings.ENABLE_DEV_ROUTES:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Dev routes not available in production")


@router.get("/test/hand-gesture", dependencies=[Depends(require_dev_mode)])
async def hand_gesture_test_page():
    return HTMLResponse("""<!DOCTYPE html>
<html>
<head>
    <title>LSC Hand Gesture Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #video, #canvas { display: inline-block; margin: 10px; }
        #result { padding: 20px; background: #f0f0f0; margin: 20px 0; border-radius: 5px; }
        .connected { color: green; }
        .disconnected { color: red; }
    </style>
</head>
<body>
    <h1>🖐️ Hand Gesture Recognition Test</h1>
    <p>Status: <span id="status" class="disconnected">Disconnected</span></p>
    
    <video id="video" width="320" height="240" autoplay></video>
    <canvas id="canvas" width="320" height="240"></canvas>
    
    <div id="result">
        <h3>Recognition Result:</h3>
        <p><strong>Text:</strong> <span id="text">-</span></p>
        <p><strong>Confidence:</strong> <span id="confidence">-</span></p>
        <p><strong>Keypoints:</strong> <span id="keypoints">No data</span></p>
    </div>
    
    <button id="connect">Connect to WebSocket</button>
    <button id="disconnect" disabled>Disconnect</button>
    <button id="snapshot">Send Frame</button>
    
    <script>
        let ws = null;
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        async function startCamera() {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        }
        
        startCamera();
        
        document.getElementById('connect').onclick = () => {
            ws = new WebSocket('ws://localhost:8000/ws/translate');
            
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
                document.getElementById('connect').disabled = true;
                document.getElementById('disconnect').disabled = false;
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'translation') {
                    document.getElementById('text').textContent = data.text || '(No text)';
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('keypoints').textContent = data.has_keypoints ? 'Detected ✓' : 'Not detected';
                } else if (data.type === 'error') {
                    console.error('Error:', data.message);
                }
            };
            
            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'disconnected';
                document.getElementById('connect').disabled = false;
                document.getElementById('disconnect').disabled = true;
            };
        };
        
        document.getElementById('disconnect').onclick = () => {
            if (ws) ws.close();
        };
        
        document.getElementById('snapshot').onclick = () => {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            
            ctx.drawImage(video, 0, 0, 320, 240);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            const base64 = dataUrl.split(',')[1];
            
            ws.send(JSON.stringify({ type: 'frame', data: base64 }));
        };
    </script>
</body>
</html>""")


@router.get("/test/models", dependencies=[Depends(require_dev_mode)])
async def list_models():
    return {
        "models": [
            {"name": "mediapipe-hands", "status": "loaded"},
            {"name": "mediapipe-pose", "status": "loaded"},
            {"name": "mediapipe-face", "status": "loaded"},
            {"name": "lsc-classifier", "status": "not_trained"}
        ],
        "note": "LSC classifier needs to be trained with Colombian Sign Language data"
    }


@router.get("/test/health", dependencies=[Depends(require_dev_mode)])
async def dev_health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "dev_mode": settings.ENABLE_DEV_ROUTES,
        "database": "available" if settings.APP_ENV != "production" else "N/A"
    }
