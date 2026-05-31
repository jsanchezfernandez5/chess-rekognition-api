"""
Servicio de entrenamiento del modelo MobileNetV2.

    - Gestiona el estado global del entrenamiento, 
    - Broadcasting de progreso por WebSocket y 
    - el pipeline completo de transfer learning con Keras.
"""
import asyncio
import gc
import json
import os
import threading
from typing import List

import tensorflow as tf
from fastapi import WebSocket

from core.config import settings

# Estado global del entrenamiento (un único entrenamiento a la vez)
training_state: dict = {
    "running":       False,
    "epoch":         0,
    "total_epochs":  0,
    "loss":          0.0,
    "accuracy":      0.0,
    "val_loss":      0.0,
    "val_accuracy":  0.0,
    "status":        "idle",   # idle | loading | building | training | done | error
    "message":       "",
}

state_lock = threading.RLock()

# Lista de clientes WebSocket conectados
ws_clients: List[WebSocket] = []

# Event loop principal capturado desde FastAPI
_main_loop: asyncio.AbstractEventLoop | None = None

# Función que devuelve el estado actual del entrenamiento
def get_state() -> dict:
    with state_lock:
        return training_state.copy()

# Función que devuelve si el entrenamiento está corriendo
def is_running() -> bool:
    with state_lock:
        return training_state["running"]

# Función que añade un cliente WebSocket a la lista de clientes conectados
def add_ws_client(ws: WebSocket):
    ws_clients.append(ws)

# Función que elimina un cliente WebSocket de la lista de clientes conectados
def remove_ws_client(ws: WebSocket):
    if ws in ws_clients:
        ws_clients.remove(ws)

# Función que inicia el entrenamiento en un hilo separado
def start(loop: asyncio.AbstractEventLoop):
    """
    Lanza el entrenamiento en un hilo separado capturando el event loop principal.
    """
    global _main_loop
    _main_loop = loop
    thread = threading.Thread(target=_run_training_wrapper, daemon=True)
    thread.start()

# Función interna que envía el estado a todos los WebSocket conectados desde el hilo de entrenamiento
def _broadcast(data: dict):
    """Envía el estado a todos los WebSocket conectados desde el hilo de entrenamiento."""
    if _main_loop is None:
        return
    msg = json.dumps(data)
    for client in ws_clients[:]:
        asyncio.run_coroutine_threadsafe(client.send_text(msg), _main_loop)

# Función wrapper que ejecuta el entrenamiento
def _run_training_wrapper():
    try:
        _run_training_logic()
    except Exception as e:
        with state_lock:
            training_state.update({"running": False, "status": "error", "message": str(e)})
            current_state = training_state.copy()
        _broadcast(current_state)

# Función interna que ejecuta el entrenamiento
def _run_training_logic():
    # Actualiza el estado del entrenamiento a loading
    with state_lock:
        training_state.update({
            "running": True, 
            "status": "loading", 
            "epoch": 0, 
            "message": "Preparando dataset..."
        })
        current_state = training_state.copy()

    # Envía el estado a todos los WebSocket conectados
    _broadcast(current_state)

    # Carga el dataset de imágenes
    train_ds = tf.keras.utils.image_dataset_from_directory(
        settings.DATASET_DIR, 
        validation_split=0.2, 
        subset="training",
        seed=42, 
        image_size=settings.IMG_SIZE, 
        batch_size=32, 
        label_mode="categorical",
    )
    # Carga el dataset de validación
    val_ds = tf.keras.utils.image_dataset_from_directory(
        settings.DATASET_DIR, 
        validation_split=0.2, 
        subset="validation",
        seed=42, 
        image_size=settings.IMG_SIZE, 
        batch_size=32, 
        label_mode="categorical",
    )

    # Obtiene los nombres de las clases
    class_names = train_ds.class_names
    with open(os.path.join(settings.MODELS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Prepara el dataset para el entrenamiento
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Actualiza el estado del entrenamiento a building
    with state_lock:
        training_state.update({"status": "building", "message": "Construyendo modelo MobileNetV2..."})
        current_state = training_state.copy()
    
    # Envía el estado a todos los WebSocket conectados
    _broadcast(current_state)

    # Crea el modelo base MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*settings.IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    # Congela las capas del modelo base
    base_model.trainable = False

    # Construye el modelo con el modelo base preentrenado
    model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])

    # Compila el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Número de Epochs para el entrenamiento (con EarlyStopping se detendrá antes si no hay mejora para no reentrenar).
    EPOCHS = 20
    with state_lock:
        training_state.update({"status": "training", "total_epochs": EPOCHS, "message": "Entrenando..."})
        current_state = training_state.copy()
    
    # Envía el estado a todos los WebSocket conectados
    _broadcast(current_state)

    # Callback para enviar el estado del entrenamiento a los WebSocket
    class _WSCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with state_lock:
                training_state.update({
                    "epoch":        epoch + 1,
                    "loss":         round(float(logs.get("loss", 0)), 4),
                    "accuracy":     round(float(logs.get("accuracy", 0)), 4),
                    "val_loss":     round(float(logs.get("val_loss", 0)), 4),
                    "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
                })
                current_state = training_state.copy()
            _broadcast(current_state)

    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        callbacks=[
            _WSCallback(),
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, min_delta=0.005),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ],
    )

    # Guardamos el modelo entrenado en disco.
    model.save(os.path.join(settings.MODELS_DIR, "chess_model.keras"))

    # Actualiza el estado del entrenamiento a done
    with state_lock:
        training_state.update({"running": False, "status": "done", "message": "Entrenamiento completado y modelo guardado."})
        current_state = training_state.copy()

    # Envía el estado a todos los WebSocket conectados
    _broadcast(current_state)

    # Libera memoria (esencial para evitar errores de GPU en procesos consecutivos)
    tf.keras.backend.clear_session()

    # Recolecta la basura
    gc.collect()
