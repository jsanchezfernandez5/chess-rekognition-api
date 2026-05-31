"""
Router del módulo de Dataset y Entrenamiento del modelo MobileNetV2.

Endpoints:
    POST /dataset/capture         | Detecta el tablero, rectifica la homografía y devuelve las 64 casillas recortadas (crops) con etiquetado sugerido por el STD.
    POST /dataset/save            | Recibe las casillas (crops) etiquetadas y las guarda en el directorio de su clase con nombre UUID único.
    GET  /dataset/stats           | Devuelve el número de imágenes por clase y el total acumulado.
    POST /dataset/train           | Lanza el entrenamiento de MobileNetV2 en un hilo secundario que generar el archivo del modelo.
    GET  /dataset/train/status    | Devuelve el estado actual del entrenamiento (polling).
    WS   /dataset/train/ws        | WebSocket de progreso del entrenamiento en tiempo real para consola. El token JWT se pasa como query param: ?token=<access_token>
"""
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

# Función interna para asegurar que los directorios del dataset existen y los crea.
def _ensure_dirs():
    # ["b_B", "b_K", "b_N", "b_P", "b_Q", "b_R", "empty", "w_B", "w_K", "w_N", "w_P", "w_Q", "w_R"]
    for cls in settings.CLASSES:        
        os.makedirs(os.path.join(settings.DATASET_DIR, cls), exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

# Asegura que los directorios del dataset existen al importar el módulo.
_ensure_dirs()

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /dataset/capture
# Detecta el tablero, rectifica la homografía y devuelve las 64 casillas recortadas (crops) con etiquetado sugerido por el STD.
# -------------------------------------------------------------------------------
@router.post(
    "/capture", 
    summary="Detecta el tablero, rectifica la homografía y devuelve las 64 casillas recortadas (crops) con etiquetado sugerido por el STD."
)
async def capture(
    file: UploadFile = File(...), 
    coords: Optional[str] = Form(None),
    rotation: int = Form(0),
    user=Depends(get_current_user)
):
    """
    Detecta el tablero, rectifica la homografía y devuelve las 64 casillas recortadas (crops) con etiquetado sugerido por el STD.
    """
    try:
        # Lee el archivo de imagen.
        data  = await file.read()

        # Decodifica la imagen y convierte los bytes a un array NumPy y luego a imagen OpenCV (matriz BGR)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "No se pudo decodificar la imagen"}

         # Intenta detectar automáticamente el tablero buscando la cuadrícula interior (7×7 esquinas) o usando las coordenadas manuales si se proporcionan.
        found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
        if not found:
            return {"success": False, "error": "Tablero no detectado. Asegúrate de que los bordes interiores (7x7) sean visibles o usa calibración manual."}

        # Aplica la transformación homográfica de OpenCV a vista cenital 2D (400x400)
        warped = _rectificar(frame, exterior)

       # Aplica la rotación solicitada sobre la imagen ya rectificada para que el modelo aprenda a reconocer el tablero en distintas orientaciones.
        try:
            rot_val = int(rotation) if rotation else 0
        except (ValueError, TypeError):
            rot_val = 0
        if rot_val == 90:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif rot_val == 180:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        elif rot_val == 270:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convierte la imagen rectificada a escala de grises para calcular la desviación estándar Standard Deviation (STD) de cada casilla (empty si std < 45).
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Recorre las 8 filas y 8 columnas del tablero (64 casillas en total)
        squares = []
        for row in range(8):
            for col in range(8):
                # Calcula las coordenadas píxel de la esquina superior izquierda de la casilla
                x, y   = col * settings.CELL_SIZE, row * settings.CELL_SIZE

                # Recorta la casilla de la imagen en color y de la imagen en gris
                cell   = warped[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]
                cell_g = gray[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]
                h, w = cell_g.shape

                # Recorta solo el 65% central de la casilla para ignorar los bordes para el STD que pueden contener partes de otras piezas o del tablero.
                ch, cw = int(h * 0.65), int(w * 0.65)
                yo, xo = (h - ch) // 2, (w - cw) // 2

                # Calcula la desviación estándar de la parte central de la casilla Standard Deviation (STD).
                std_inner = float(np.std(cell_g[yo:yo + ch, xo:xo + cw]))

                # Convierte la casilla a base64.
                _, buf = cv2.imencode(".jpg", cell, [cv2.IMWRITE_JPEG_QUALITY, 92])
                b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                # Construye el diccionario de la casilla con todos sus datos y lo agrega a la lista de casillas.
                squares.append({
                    "id":              f"{settings.COLS[col]}{8 - row}",
                    "row":             row,
                    "col":             col,
                    "image":           b64,
                    "std":             round(std_inner, 2),
                    "suggested_label": "empty" if std_inner < 45 else "?",
                })

        # Retorna la lista de casillas con su imagen en base64, su desviación estándar y una etiqueta sugerida basada en un umbral de desviación estándar (empty si std < 45).
        return {"success": True, "squares": squares}

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /dataset/save
# Recibe las casillas (crops) etiquetadas y las guarda en el directorio de su clase con nombre UUID único.
# -------------------------------------------------------------------------------
@router.post(
    "/save", 
    summary="Recibe las casillas (crops) etiquetadas y las guarda en el directorio de su clase con nombre UUID único."
)
async def save_squares(payload: dict, user=Depends(get_current_user)):
    """
    Recibe las casillas (crops) etiquetadas y las guarda en el directorio de su clase con nombre UUID único.
    """
    try:
        saved = 0
        # Itera sobre las casillas etiquetadas.
        for item in payload.get("squares", []):
            # Clase asignada por el usuario: "w_P", "b_K", "empty", etc.
            label    = item.get("label")
            # Imagen en base64 de la casilla recortada.
            b64_data = item.get("image", "")

            # Ignora la casilla si no tiene etiqueta o si la etiqueta no es una clase válida del modelo
            if not label or label not in settings.CLASSES:
                continue

            # Elimina el prefijo "data:image/jpeg;base64," si está presente, dejando solo los datos Base64.
            b64_content = b64_data.split(",")[1] if "," in b64_data else b64_data

            # Genera un nombre de fichero único con UUID para evitar colisiones entre imágenes
            path = os.path.join(settings.DATASET_DIR, label, f"{uuid.uuid4().hex}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_content))
            saved += 1

        # Devuelve cuántas imágenes se guardaron correctamente
        return {"success": True, "saved": saved}
    except Exception as e:
        return {"success": False, "error": str(e)}

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /dataset/stats
# Devuelve el número de imágenes por clase y el total acumulado.
# -------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /dataset/train
# Lanza el entrenamiento de MobileNetV2 en un hilo secundario que generar el archivo del modelo.
# -------------------------------------------------------------------------------
@router.post(
    "/train", 
    summary="Inicia el entrenamiento del modelo MobileNetV2 con el dataset actual"
)
async def start_training(user=Depends(get_current_user)):
    """
    Lanza el entrenamiento de MobileNetV2 en un hilo secundario que generar el archivo del modelo.
    """
    # Validamos que no haya un entrenamiento en curso.
    if training.is_running():
        return {"success": False, "error": "Ya hay un entrenamiento en curso."}

    # Iniciamos el entrenamiento en un hilo secundario.
    training.start(asyncio.get_running_loop())
    return {"success": True, "message": "Entrenamiento iniciado."}

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /dataset/train/status
# Devuelve el estado actual del entrenamiento (polling).
# -------------------------------------------------------------------------------
@router.get(
    "/train/status", 
    summary="Estado del entrenamiento (polling)"
)
def get_train_status(user=Depends(get_current_user)):
    """
    Devuelve el estado actual del entrenamiento (polling).
    """
    return training.get_state()

# -------------------------------------------------------------------------------
# [ENDPOINT] - WS /dataset/train/ws
# WebSocket de progreso del entrenamiento en tiempo real para consola. El token JWT se pasa como query param: ?token=<access_token>
# -------------------------------------------------------------------------------
@router.websocket("/train/ws")
async def train_websocket(
    websocket: WebSocket, 
    token: str = Query(...)
):
    """
    WebSocket de progreso del entrenamiento en tiempo real para consola. El token JWT se pasa como query param: ?token=<access_token>
    """
    # Validamos el token JWT de acceso antes de aceptar la conexión WebSocket. 
    # Al no ser endpoint HTTP, no podemos usar Depends() y get_current_user, así que validamos manualmente el token JWT.    
    try:
        decode_token(token, expected_type="access")
    except ValueError:
        # Si el token no es válido o ha expirado, cerramos la conexión WebSocket con código 1008 (Policy Violation).
        await websocket.close(code=1008)
        return

    # Aceptamos la conexión WebSocket.
    await websocket.accept()

    # Agregamos el cliente WebSocket al entrenamiento.
    training.add_ws_client(websocket)

    # Iniciamos un bucle para mantener la conexión WebSocket abierta (solo enviamos).
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        training.remove_ws_client(websocket)
