"""
Router del módulo de Dataset y Entrenamiento del detector YOLO26.

Dataset de DETECCIÓN de objetos: cada imagen del tablero rectificado 400x400 va acompañada
de un fichero .txt con las bounding boxes en formato YOLO ("class_id x_center y_center width
height", normalizado 0-1). Las anotaciones se crean de forma semi-automática en la herramienta
/yolo-dataset: cuando el modelo YOLO está entrenado, pre-rellena cajas con sus predicciones;
si el modelo no está disponible, el usuario anota manualmente desde cero.

Endpoints:
    POST /yolo-dataset/save         | Guarda la imagen anotada y su .txt de etiquetas YOLO con nombre UUID único.
    GET  /yolo-dataset/stats        | Nº de imágenes anotadas, cajas por clase, total y desglose por origen de la caja.
    POST /yolo-dataset/train        | Lanza el entrenamiento de YOLO26 en un hilo secundario.
    GET  /yolo-dataset/train/status | Devuelve el estado actual del entrenamiento (polling).
    WS   /yolo-dataset/train/ws     | WebSocket de progreso del entrenamiento en tiempo real para consola. El token JWT se pasa como query param: ?token=<access_token>
"""
import asyncio
import base64
import json
import os
import traceback
import uuid

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from core.config import settings
from core.dependencies import get_current_user
from core.security import decode_token
from services import yolo_training

# Router para los endpoints del dataset YOLO.
router = APIRouter(prefix="/yolo-dataset", tags=["Dataset YOLO"])

# Fichero donde se acumulan los contadores de origen de las cajas (prefill/corregida/manual).
# Cada guardado envía cuántas cajas aceptó tal cual de la predicción YOLO (cuando el modelo
# está disponible), cuántas corrigió a mano y cuántas añadió de forma manual.
_STATS_META_FILE = "stats_meta.json"


def _ensure_dirs():
    """Crea la estructura de directorios del dataset YOLO si no existe."""
    os.makedirs(os.path.join(settings.YOLO_DATASET_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(settings.YOLO_DATASET_DIR, "labels"), exist_ok=True)
    os.makedirs(os.path.join(settings.YOLO_DATASET_DIR, "fuentes"), exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)


# Asegura que los directorios existen al importar el módulo.
_ensure_dirs()


def _leer_stats_meta() -> dict:
    """Lee los contadores acumulados de origen de cajas; devuelve valores a cero si no existen."""
    path = os.path.join(settings.YOLO_DATASET_DIR, _STATS_META_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"prefilled": 0, "corrected": 0, "manual": 0}


def _guardar_stats_meta(meta: dict):
    """Persiste los contadores acumulados de origen de cajas."""
    path = os.path.join(settings.YOLO_DATASET_DIR, _STATS_META_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /yolo-dataset/save
# Guarda la imagen del tablero rectificado 400x400 y su fichero de etiquetas YOLO asociado.
# -------------------------------------------------------------------------------
@router.post(
    "/save",
    summary="Guarda la imagen anotada y su fichero de etiquetas YOLO (.txt) con nombre UUID único."
)
async def save_annotation(payload: dict, user=Depends(get_current_user)):
    """
    Guarda una anotación completa del dataset YOLO.

    Body esperado (JSON):
        image:         data URI (o base64 crudo) de la imagen del tablero rectificado 400x400.
        boxes:         lista de cajas {class_id, x_center, y_center, width, height} normalizadas 0-1.
        origin_counts: opcional {prefilled, corrected, manual} con cuántas cajas vienen de cada origen.
    """
    try:
        b64_data = payload.get("image", "")
        boxes = payload.get("boxes", [])
        if not b64_data:
            return {"success": False, "error": "Falta la imagen a guardar."}
        if not isinstance(boxes, list) or len(boxes) == 0:
            return {"success": False, "error": "La anotación no contiene ninguna caja."}

        # Valida las cajas antes de escribir nada en disco.
        lineas = []
        num_clases = len(settings.YOLO_CLASSES)
        for box in boxes:
            try:
                class_id = int(box["class_id"])
                x_c = float(box["x_center"])
                y_c = float(box["y_center"])
                w = float(box["width"])
                h = float(box["height"])
            except (KeyError, TypeError, ValueError):
                return {"success": False, "error": f"Caja malformada: {box}"}

            if not (0 <= class_id < num_clases):
                return {"success": False, "error": f"class_id fuera de rango: {class_id} (0..{num_clases - 1})"}
            # Clamp defensivo a [0, 1]: evita cajas degeneradas por redondeos del canvas.
            x_c = min(max(x_c, 0.0), 1.0)
            y_c = min(max(y_c, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            lineas.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Decodifica la imagen (acepta data URI "data:image/jpeg;base64,..." o base64 crudo).
        b64_content = b64_data.split(",")[1] if "," in b64_data else b64_data
        img_bytes = base64.b64decode(b64_content)

        # Guarda imagen + etiquetas con el mismo nombre base UUID (convención Ultralytics).
        nombre = uuid.uuid4().hex
        img_path = os.path.join(settings.YOLO_DATASET_DIR, "images", f"{nombre}.jpg")
        lbl_path = os.path.join(settings.YOLO_DATASET_DIR, "labels", f"{nombre}.txt")
        src_path = os.path.join(settings.YOLO_DATASET_DIR, "fuentes", f"{nombre}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
        # Guarda copia de la imagen fuente original para referencia futura.
        with open(src_path, "wb") as f:
            f.write(img_bytes)

        # Acumula los contadores de origen si el frontend los envía.
        meta = _leer_stats_meta()
        origenes = payload.get("origin_counts") or {}
        for clave in ("prefilled", "corrected", "manual"):
            try:
                meta[clave] += max(0, int(origenes.get(clave, 0)))
            except (TypeError, ValueError):
                pass
        _guardar_stats_meta(meta)

        return {"success": True, "saved": nombre, "boxes": len(lineas)}
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}


# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /yolo-dataset/stats
# Estadísticas del dataset YOLO: imágenes, cajas por clase y desglose por origen.
# -------------------------------------------------------------------------------
@router.get(
    "/stats",
    summary="Estadísticas del dataset YOLO (imágenes, cajas por clase y desglose por origen)."
)
def get_yolo_stats(user=Depends(get_current_user)):
    """Devuelve nº de imágenes anotadas, cajas por clase, total y desglose prefill/corregida/manual."""
    try:
        img_dir = os.path.join(settings.YOLO_DATASET_DIR, "images")
        lbl_dir = os.path.join(settings.YOLO_DATASET_DIR, "labels")

        imagenes = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        cajas_por_clase = {cls: 0 for cls in settings.YOLO_CLASSES}
        total_cajas = 0
        for lbl in os.listdir(lbl_dir):
            if not lbl.lower().endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, lbl), "r", encoding="utf-8") as f:
                for linea in f:
                    partes = linea.strip().split()
                    if not partes:
                        continue
                    try:
                        class_id = int(partes[0])
                        if 0 <= class_id < len(settings.YOLO_CLASSES):
                            cajas_por_clase[settings.YOLO_CLASSES[class_id]] += 1
                            total_cajas += 1
                    except ValueError:
                        continue

        return {
            "classes": cajas_por_clase,
            "total_boxes": total_cajas,
            "total_images": len(imagenes),
            "origins": _leer_stats_meta(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /yolo-dataset/train
# Lanza el entrenamiento de YOLO26 en un hilo secundario.
# -------------------------------------------------------------------------------
@router.post(
    "/train",
    summary="Inicia el entrenamiento del detector YOLO26 con el dataset anotado actual."
)
async def start_yolo_training(user=Depends(get_current_user)):
    """Lanza el entrenamiento de YOLO26 en un hilo secundario que genera api/data/models/yolo_chess.pt."""
    # Validamos que no haya un entrenamiento en curso.
    if yolo_training.is_running():
        return {"success": False, "error": "Ya hay un entrenamiento en curso."}

    # Iniciamos el entrenamiento en un hilo secundario.
    yolo_training.start(asyncio.get_running_loop())
    return {
        "success": True,
        "message": "Entrenamiento YOLO iniciado. Progreso en /yolo-dataset/train/status.",
    }


# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /yolo-dataset/train/status
# Estado actual del entrenamiento YOLO (polling), igual que /dataset/train/status.
# -------------------------------------------------------------------------------
@router.get(
    "/train/status",
    summary="Estado del entrenamiento YOLO (polling)."
)
def get_yolo_train_status(user=Depends(get_current_user)):
    """Devuelve el estado actual del entrenamiento YOLO."""
    return yolo_training.get_state()


# -------------------------------------------------------------------------------
# [ENDPOINT] - WS /yolo-dataset/train/ws
# WebSocket de progreso del entrenamiento YOLO26 en tiempo real para la consola de la
# herramienta /yolo-dataset. El token JWT se pasa como query param porque un WebSocket
# no pasa por Depends()/get_current_user.
# Conecta aquí la infraestructura add_ws_client/_broadcast de services/yolo_training.py.
# -------------------------------------------------------------------------------
@router.websocket("/train/ws")
async def train_websocket(websocket: WebSocket, token: str = Query(...)):
    """WebSocket de progreso del entrenamiento YOLO26. Autenticación por query param: ?token=<access_token>."""
    # Validamos el token JWT de acceso antes de aceptar la conexión WebSocket.
    try:
        decode_token(token, expected_type="access")
    except ValueError:
        # Token inválido o expirado: cierre 1008 (Policy Violation).
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # Registra el cliente para recibir los broadcasts del hilo de entrenamiento.
    yolo_training.add_ws_client(websocket)

    # Bucle que mantiene la conexión abierta (este canal solo envía).
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        yolo_training.remove_ws_client(websocket)
