"""Módulo de gestión del dataset y entrenamiento del modelo de clasificación.

Proporciona endpoints para capturar y etiquetar casillas de tableros
de ajedrez, así como para entrenar el modelo MobileNetV2 con los
datos recopilados. Incluye comunicación WebSocket para monitorizar
el progreso del entrenamiento en tiempo real.
"""

import os
import uuid
import base64
import json
import asyncio
import threading
import gc
import cv2
import numpy as np
import traceback
import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, Query

from core.config import settings
from core.dependencies import get_current_user
from core.security import decode_token
from services.vision import _detectar_tablero, _calcular_esquinas_exteriores, _rectificar

_main_loop: asyncio.AbstractEventLoop | None = None

router = APIRouter(
    prefix="/dataset", 
    tags=["Dataset"]
)

training_state = {
    "running": False,
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "val_loss": 0.0,
    "val_accuracy": 0.0,
    "status": "idle", # idle, loading, building, training, done, error
    "message": ""
}

ws_clients: list[WebSocket] = []

def ensure_dataset_dirs():
    """Crea las carpetas necesarias para el dataset y los modelos.

    Verifica que existan los directorios para cada clase del dataset
    y el directorio de modelos. Los crea si no existen.
    """
    for cls in settings.CLASSES:
        os.makedirs(os.path.join(settings.DATASET_DIR, cls), exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)

ensure_dataset_dirs()

@router.post("/capture", summary="Captura y recorta el tablero")
async def capture(file: UploadFile = File(...), user=Depends(get_current_user)):
    """Captura una imagen y extrae las 64 casillas del tablero.

    Recibe un frame de la cámara, detecta el tablero mediante visión
    clásica, lo rectifica y recorta cada una de las 64 casillas.
    Cada recorte se codifica en base64 e incluye una heurística
    de sugerencia de etiqueta basada en la desviación estándar.

    Args:
        file: Imagen subida con el tablero a procesar.
        user: Usuario autenticado (requerido).

    Returns:
        Dict con los 64 recortes, sus coordenadas y sugerencias de etiqueta.
    """
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "No se pudo decodificar la imagen"}

        # Pipeline de visión
        found, corners = _detectar_tablero(frame)
        if not found:
            return {"success": False, "error": "Tablero no detectado. Asegúrate de que los bordes interiores (7x7) sean visibles."}

        exterior = _calcular_esquinas_exteriores(corners)
        warped = _rectificar(frame, exterior)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        squares = []
        for row in range(8):
            for col in range(8):
                x, y = col * settings.CELL_SIZE, row * settings.CELL_SIZE
                cell = warped[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]
                cell_g = gray[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]

                # Heurística: si la desviación típica del centro de la casilla es baja → vacía
                h, w = cell_g.shape
                ch, cw = int(h * 0.65), int(w * 0.65)
                yo, xo = (h - ch) // 2, (w - cw) // 2
                std_inner = float(np.std(cell_g[yo:yo + ch, xo:xo + cw]))

                _, buf = cv2.imencode(".jpg", cell, [cv2.IMWRITE_JPEG_QUALITY, 92])
                b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                squares.append({
                    "id": f"{settings.COLS[col]}{8 - row}",
                    "row": row, "col": col,
                    "image": b64,
                    "std": round(std_inner, 2),
                    "suggested_label": "empty" if std_inner < 45 else "?"
                })

        return {"success": True, "squares": squares}
        
    except Exception as e:
        return {
            "success": False, 
            "error": str(e), 
            "detail": traceback.format_exc()
        }

@router.post("/save", summary="Guarda recortes etiquetados")
async def save_squares(payload: dict, user=Depends(get_current_user)):
    """Guarda en disco las casillas etiquetadas por el usuario.

    Almacena las imágenes de las casillas en el directorio
    correspondiente a su clase, usando un nombre de archivo
    UUID único para evitar colisiones.

    Args:
        payload: Diccionario con la lista de recortes etiquetados.
        user: Usuario autenticado (requerido).

    Returns:
        Dict con el número de imágenes guardadas correctamente.
    """
    try:
        saved = 0
        for item in payload.get("squares", []):
            label = item.get("label")
            b64_data = item.get("image", "")
            
            if not label or label not in settings.CLASSES:
                continue
                
            if "," in b64_data:
                b64_content = b64_data.split(",")[1]
            else:
                b64_content = b64_data

            # Guardar con UUID único
            filename = f"{uuid.uuid4().hex}.jpg"
            path = os.path.join(settings.DATASET_DIR, label, filename)
            
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_content))
            saved += 1
            
        return {"success": True, "saved": saved}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/stats", summary="Estadísticas del dataset")
def get_stats(user=Depends(get_current_user)):
    """Obtiene las estadísticas actuales del dataset de entrenamiento.

    Cuenta el número de imágenes almacenadas para cada clase
    y calcula el total de muestras disponibles.

    Args:
        user: Usuario autenticado (requerido).

    Returns:
        Dict con el recuento por clase y el total de imágenes.
    """
    stats = {}
    for cls in settings.CLASSES:
        path = os.path.join(settings.DATASET_DIR, cls)
        stats[cls] = len([f for f in os.listdir(path) if f.lower().endswith(".jpg")])
    
    return {
        "classes": stats,
        "total": sum(stats.values())
    }

@router.post("/train", summary="Inicia el entrenamiento")
async def start_training(user=Depends(get_current_user)):
    """Inicia el proceso de entrenamiento del modelo en un hilo separado.

    Lanza el entrenamiento del modelo MobileNetV2 en un hilo
    background, capturando el event loop del hilo principal
    para poder enviar actualizaciones por WebSocket.

    Args:
        user: Usuario autenticado (requerido).

    Returns:
        Dict indicando si el entrenamiento se inició correctamente.
    """
    if training_state["running"]:
        return {"success": False, "error": "Ya hay un entrenamiento en curso."}

    global _main_loop
    _main_loop = asyncio.get_running_loop()  # Captura el loop principal para broadcasting

    thread = threading.Thread(target=_run_training_wrapper, daemon=True)
    thread.start()
    
    return {"success": True, "message": "Entrenamiento iniciado."}

@router.get("/train/status", summary="Estado del entrenamiento")
def get_train_status(user=Depends(get_current_user)):
    """Obtiene el estado actual del entrenamiento mediante polling.

    Devuelve una copia del estado global del entrenamiento,
    incluyendo época actual, pérdida, precisión y mensajes
    de progreso.

    Args:
        user: Usuario autenticado (requerido).

    Returns:
        Dict con el estado completo del entrenamiento.
    """
    return training_state.copy()

@router.websocket("/train/ws")
async def train_websocket(websocket: WebSocket, token: str = Query(...)):
    """Canal WebSocket para recibir actualizaciones del entrenamiento.

    Establece una conexión WebSocket que envía actualizaciones
    periódicas sobre el progreso del entrenamiento. La autenticación
    se realiza mediante un query param con el token JWT.

    Args:
        websocket: Conexión WebSocket del cliente.
        token: Token JWT de acceso para autenticación.
    """
    try:
        # WebSockets nativos no envían headers→el token se pasa como query param
        decode_token(token, expected_type="access")
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            # Mantener conexión abierta
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in ws_clients:
            ws_clients.remove(websocket)

def _broadcast_training_update(data: dict):
    """Envía progreso del entrenamiento a todos los WebSocket conectados.

    Seguro para hilos secundarios: programa el envío en el event loop principal.
    """
    if _main_loop is None:
        return

    msg = json.dumps(data)
    for client in ws_clients[:]:
        asyncio.run_coroutine_threadsafe(client.send_text(msg), _main_loop)

def _run_training_wrapper():
    """Wrapper que captura errores del entrenamiento y notifica por WebSocket."""
    try:
        _run_training_logic()
    except Exception as e:
        print(f"Error en entrenamiento: {traceback.format_exc()}")
        training_state.update({
            "running": False,
            "status": "error",
            "message": str(e)
        })
        _broadcast_training_update(training_state.copy())

def _run_training_logic():
    """Pipeline completo de entrenamiento MobileNetV2 con transfer learning."""
    training_state.update({
        "running": True, "status": "loading",
        "epoch": 0, "message": "Preparando dataset..."
    })
    _broadcast_training_update(training_state.copy())

    # Carga con split 80/20 desde directorio estructurado por clases
    train_ds = tf.keras.utils.image_dataset_from_directory(
        settings.DATASET_DIR, validation_split=0.2, subset="training",
        seed=42, image_size=settings.IMG_SIZE, batch_size=32,
        label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        settings.DATASET_DIR, validation_split=0.2, subset="validation",
        seed=42, image_size=settings.IMG_SIZE, batch_size=32,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    with open(os.path.join(settings.MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Prefetch para solapar preprocessing y entrenamiento
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    training_state["status"] = "building"
    training_state["message"] = "Construyendo modelo MobileNetV2..."
    _broadcast_training_update(training_state.copy())

    # Transfer Learning: MobileNetV2 con pesos ImageNet, base congelada
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*settings.IMG_SIZE, 3),
        include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        # Data augmentation para generalización
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        # MobileNetV2 espera entrada en [-1, 1]
        tf.keras.layers.Rescaling(1. / 127.5, offset=-1),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy", metrics=["accuracy"]
    )

    EPOCHS = 20
    training_state.update({
        "status": "training", "total_epochs": EPOCHS,
        "message": "Entrenando..."
    })
    _broadcast_training_update(training_state.copy())

    class WSUpdateCallback(tf.keras.callbacks.Callback):
        """Callback que transmite pérdida y precisión por WebSocket tras cada época."""
        def on_epoch_end(self, epoch, logs=None):
            training_state.update({
                "epoch": epoch + 1,
                "loss": round(float(logs.get("loss", 0)), 4),
                "accuracy": round(float(logs.get("accuracy", 0)), 4),
                "val_loss": round(float(logs.get("val_loss", 0)), 4),
                "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4)
            })
            _broadcast_training_update(training_state.copy())

    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        callbacks=[
            WSUpdateCallback(),
            # Early stopping con tolerancia 4 épocas sin mejora significativa
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, min_delta=0.005),
            # Reduce LR a la mitad si 3 épocas sin mejora
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )

    model_path = os.path.join(settings.MODELS_DIR, "chess_model.h5")
    model.save(model_path)

    training_state.update({
        "running": False, "status": "done",
        "message": "Entrenamiento completado y modelo guardado."
    })
    _broadcast_training_update(training_state.copy())

    tf.keras.backend.clear_session()
    gc.collect()
