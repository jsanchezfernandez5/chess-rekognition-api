import asyncio
import base64
import os
import traceback
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, WebSocket, WebSocketDisconnect

from core.config import settings
from core.dependencies import get_current_user
from core.security import decode_token
from services import training
from services.vision import VisionService, _rectificar

# Router para los endpoints del dataset.
router = APIRouter(prefix="/dataset", tags=["Dataset"])

# Función para asegurar que los directorios del dataset existen.
def _ensure_dirs():
    for cls in settings.CLASSES:
        os.makedirs(os.path.join(settings.DATASET_DIR, cls), exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

_ensure_dirs()

# Endpoint para capturar imágenes y recortar el tablero.
@router.post(
    "/capture", 
    summary="Captura y recorta el tablero en 64 casillas"
)
async def capture(
    file: UploadFile = File(...), 
    coords: Optional[str] = Form(None),
    user=Depends(get_current_user)
):
    """
    Detecta el tablero en la imagen y devuelve las 64 casillas recortadas con sugerencia de etiqueta Heurística.
    """
    try:
        # Lee el archivo de imagen.
        data  = await file.read()

        # Decodifica la imagen.
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "No se pudo decodificar la imagen"}

        # Detecta el tablero en la imagen.
        found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
        if not found:
            return {"success": False, "error": "Tablero no detectado. Asegúrate de que los bordes interiores (7×7) sean visibles o usa calibración manual."}

        # Rectifica la imagen.
        warped = _rectificar(frame, exterior)

        # Convierte la imagen a escala de grises.
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Itera sobre las 64 casillas del tablero.
        squares = []
        for row in range(8):
            for col in range(8):
                x, y   = col * settings.CELL_SIZE, row * settings.CELL_SIZE
                cell   = warped[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]
                cell_g = gray[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]

                h, w = cell_g.shape
                ch, cw = int(h * 0.65), int(w * 0.65)
                yo, xo = (h - ch) // 2, (w - cw) // 2

                # Calcula la desviación estándar de la parte central de la casilla.
                std_inner = float(np.std(cell_g[yo:yo + ch, xo:xo + cw]))

                # Convierte la casilla a base64.
                _, buf = cv2.imencode(".jpg", cell, [cv2.IMWRITE_JPEG_QUALITY, 92])
                b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                # Añade la casilla a la lista con su información.
                squares.append({
                    "id":              f"{settings.COLS[col]}{8 - row}",
                    "row":             row,
                    "col":             col,
                    "image":           b64,
                    "std":             round(std_inner, 2),
                    "suggested_label": "empty" if std_inner < 45 else "?",
                })

        return {"success": True, "squares": squares}

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# Endpoint para guardar las casillas etiquetadas.
@router.post(
    "/save", 
    summary="Guarda recortes etiquetados en el dataset"
)
async def save_squares(payload: dict, user=Depends(get_current_user)):
    """
    Almacena las casillas etiquetadas en el directorio de su clase con nombre UUID único.
    """
    try:
        saved = 0
        # Itera sobre las casillas etiquetadas.
        for item in payload.get("squares", []):
            label    = item.get("label")
            b64_data = item.get("image", "")
            if not label or label not in settings.CLASSES:
                continue
            b64_content = b64_data.split(",")[1] if "," in b64_data else b64_data
            path = os.path.join(settings.DATASET_DIR, label, f"{uuid.uuid4().hex}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_content))
            saved += 1
        return {"success": True, "saved": saved}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Endpoint para obtener las estadísticas del dataset.
@router.get(
    "/stats", 
    summary="Estadísticas del dataset"
)
def get_stats(user=Depends(get_current_user)):
    """
    Devuelve el número de imágenes por clase y el total.
    """
    stats = {
        # Itera sobre las clases y cuenta el número de imágenes por clase.
        cls: len([f for f in os.listdir(os.path.join(settings.DATASET_DIR, cls)) if f.lower().endswith(".jpg")])
        for cls in settings.CLASSES
    }
    return {"classes": stats, "total": sum(stats.values())}

# Endpoint para iniciar el entrenamiento.
@router.post(
    "/train", 
    summary="Inicia el entrenamiento del modelo"
)
async def start_training(user=Depends(get_current_user)):
    """
    Lanza el entrenamiento de MobileNetV2 en un hilo secundario.
    """
    # Validamos que no haya un entrenamiento en curso.
    if training.is_running():
        return {"success": False, "error": "Ya hay un entrenamiento en curso."}

    # Iniciamos el entrenamiento en un hilo secundario.
    training.start(asyncio.get_running_loop())
    return {"success": True, "message": "Entrenamiento iniciado."}

# Endpoint para obtener el estado del entrenamiento.
@router.get(
    "/train/status", 
    summary="Estado del entrenamiento (polling)"
)
def get_train_status(user=Depends(get_current_user)):
    """
    Devuelve el estado actual del entrenamiento.
    """
    return training.get_state()

# Endpoint WebSocket para recibir actualizaciones del entrenamiento en tiempo real.
@router.websocket("/train/ws")
async def train_websocket(
    websocket: WebSocket, 
    token: str = Query(...)
):
    """
    WebSocket de progreso del entrenamiento. El token JWT se pasa como query param.
    """
    try:
        decode_token(token, expected_type="access")
    except ValueError:
        await websocket.close(code=1008)
        return

    # Aceptamos la conexión WebSocket.
    await websocket.accept()

    # Agregamos el cliente WebSocket al entrenamiento.
    training.add_ws_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        training.remove_ws_client(websocket)
