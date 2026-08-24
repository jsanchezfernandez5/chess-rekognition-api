"""
Servicio de entrenamiento del detector YOLO26 (Ultralytics).

Entrena el modelo de detección de objetos yolo_chess.pt sobre el dataset anotado en YOLO_DATASET_DIR,
siguiendo el mismo patrón que services/training.py (entrenamiento de MobileNetV2):
    - Estado global protegido con RLock consultable por polling.
    - Ejecución en un hilo daemon lanzado desde el router.
    - Broadcasting del progreso a los clientes WebSocket conectados.

Funciones principales:
    - get_state()             | Devuelve el estado actual del entrenamiento.
    - is_running()            | Indica si hay un entrenamiento en curso.
    - add_ws_client()         | Añade un cliente WebSocket a la lista de clientes conectados.
    - remove_ws_client()      | Elimina un cliente WebSocket de la lista de clientes conectados.
    - start()                 | Lanza el entrenamiento en un hilo separado capturando el event loop principal.
    - _preparar_split_y_yaml()| Genera el split train/val (reservando la validación, que NUNCA entrena) y el data.yaml de Ultralytics.
    - _broadcast()            | Envía el estado a todos los WebSocket conectados desde el hilo de entrenamiento.
    - _run_training_wrapper() | Wrapper que captura excepciones no controladas y actualiza el estado a "error".
    - _run_training_logic()   | Pipeline completo de entrenamiento YOLO26.

NOTA sobre el tamaño del modelo: se usa por defecto "yolo26n" (nano) por velocidad en CPU
(despliegue Railway sin GPU). Para más precisión a costa de velocidad, cambiar YOLO_BASE_WEIGHTS
a "yolo26s.pt" (small). Ver https://docs.ultralytics.com/models/yolo26/
"""
import asyncio
import hashlib
import json
import os
import shutil
import threading
from typing import List

from fastapi import WebSocket

from core.config import settings

# Peso base preentrenado de YOLO26 para transfer learning ("n" = nano, el más rápido en CPU).
# Cambiar a "yolo26s.pt", "yolo26m.pt"... para más precisión a costa de velocidad.
YOLO_BASE_WEIGHTS = "yolo26n.pt"

# Proporción de imágenes reservadas para validación (nunca se usan para entrenar).
VAL_RATIO = 0.2

# Número de épocas de entrenamiento. Ultralytics aplica early stopping interno (patience) si no hay mejora.
EPOCHS = 50

# Estado global del entrenamiento (un único entrenamiento a la vez), mismo patrón que services/training.py
training_state: dict = {
    "running":       False,
    "epoch":         0,
    "total_epochs":  0,
    "loss":          0.0,
    "map50":         0.0,
    "status":        "idle",   # idle | preparing | training | done | error
    "message":       "",
}

state_lock = threading.RLock()

# Lista de clientes WebSocket conectados
ws_clients: List[WebSocket] = []

# Event loop principal capturado desde FastAPI
_main_loop: asyncio.AbstractEventLoop | None = None


def get_state() -> dict:
    """Devuelve una copia del estado actual del entrenamiento."""
    with state_lock:
        return training_state.copy()


def is_running() -> bool:
    """Indica si hay un entrenamiento en curso."""
    with state_lock:
        return training_state["running"]


def add_ws_client(ws: WebSocket):
    """Añade un cliente WebSocket a la lista de clientes conectados."""
    ws_clients.append(ws)


def remove_ws_client(ws: WebSocket):
    """Elimina un cliente WebSocket de la lista de clientes conectados."""
    if ws in ws_clients:
        ws_clients.remove(ws)


def start(loop: asyncio.AbstractEventLoop):
    """Lanza el entrenamiento en un hilo separado capturando el event loop principal."""
    global _main_loop
    _main_loop = loop
    thread = threading.Thread(target=_run_training_wrapper, daemon=True)
    thread.start()


def _broadcast(data: dict):
    """Envía el estado a todos los WebSocket conectados desde el hilo de entrenamiento."""
    if _main_loop is None:
        return
    msg = json.dumps(data)
    for client in ws_clients[:]:
        try:
            asyncio.run_coroutine_threadsafe(client.send_text(msg), _main_loop)
        except Exception:
            pass


def _run_training_wrapper():
    """Ejecuta el entrenamiento capturando cualquier excepción no controlada."""
    try:
        _run_training_logic()
    except Exception as e:
        with state_lock:
            training_state.update({"running": False, "status": "error", "message": str(e)})
            current_state = training_state.copy()
        _broadcast(current_state)


def _hash_val(name: str, total_buckets: int = 100) -> int:
    """
    Hash determinista del nombre de fichero a un bucket 0..99.

    Se usa para repartir train/val de forma ESTABLE entre ejecuciones: aunque luego se añadan
    más imágenes al dataset y se reentrene, cada imagen sigue asignada al mismo conjunto.
    Así el conjunto de validación queda reservado y NUNCA participa en el entrenamiento,
    lo que permite comparar métricas de TensorFlow/YOLO/fusión de forma limpia (criterio 1.D).
    """
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % total_buckets


def _preparar_split_y_yaml() -> tuple:
    """
    Genera automáticamente el split train/val y el data.yaml de Ultralytics.

    Estructura generada dentro de YOLO_DATASET_DIR (formato esperado por Ultralytics):
        images/train/*.jpg   images/val/*.jpg
        labels/train/*.txt   labels/val/*.txt

    Returns:
        Tupla (ruta_data_yaml, num_train, num_val).

    Raises:
        RuntimeError: Si no hay suficientes imágenes etiquetadas para cubrir ambos conjuntos.
    """
    dataset_dir = os.path.abspath(settings.YOLO_DATASET_DIR)
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    # Limpia splits anteriores para no acumular ficheros obsoletos.
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        shutil.rmtree(os.path.join(dataset_dir, sub), ignore_errors=True)
        os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)

    n_train = n_val = 0
    # Solo imágenes que tienen su .txt de etiquetas asociado (mismo nombre base).
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base, _ = os.path.splitext(fname)
        if not os.path.exists(os.path.join(lbl_dir, f"{base}.txt")):
            continue

        es_val = _hash_val(base) < int(VAL_RATIO * 100)
        destino_img = os.path.join("images", "val" if es_val else "train")
        destino_lbl = os.path.join("labels", "val" if es_val else "train")
        shutil.copy2(
            os.path.join(img_dir, fname),
            os.path.join(dataset_dir, destino_img, fname),
        )
        shutil.copy2(
            os.path.join(lbl_dir, f"{base}.txt"),
            os.path.join(dataset_dir, destino_lbl, f"{base}.txt"),
        )
        if es_val:
            n_val += 1
        else:
            n_train += 1

    if n_train == 0 or n_val == 0:
        raise RuntimeError(
            f"Dataset insuficiente para el split: {n_train} train / {n_val} val. "
            "Anota más imágenes en /yolo-dataset antes de entrenar."
        )

    # data.yaml de Ultralytics: rutas + nombres de clase indexados por class_id.
    data_yaml = {
        "path": dataset_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(settings.YOLO_CLASSES)},
    }
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        # Volcado YAML manual para no añadir dependencia extra (pyyaml ya viene con ultralytics,
        # pero así el formato queda bajo control y documentado).
        f.write(f"# Generado automáticamente por services/yolo_training.py\n")
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write("names:\n")
        for idx, name in data_yaml["names"].items():
            f.write(f"  {idx}: {name}\n")

    return yaml_path, n_train, n_val


def _run_training_logic():
    """Pipeline completo de entrenamiento YOLO26 sobre el dataset anotado."""
    # Import diferido: 'ultralytics' arrastra PyTorch (~2 GB instalado). Si no está instalado,
    # el resto de la API debe seguir arrancando y funcionando con normalidad.
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "El paquete 'ultralytics' no está instalado. "
            "Instálalo con: pip install -r requirements.txt"
        ) from e

    # Fase 1: preparar split y data.yaml
    with state_lock:
        training_state.update({
            "running": True,
            "status": "preparing",
            "epoch": 0,
            "total_epochs": 0,
            "message": "Generando split train/val y data.yaml...",
        })
        current_state = training_state.copy()
    _broadcast(current_state)

    yaml_path, n_train, n_val = _preparar_split_y_yaml()

    # Fase 2: cargar modelo base preentrenado y lanzar el fine-tuning.
    with state_lock:
        training_state.update({
            "status": "training",
            "total_epochs": EPOCHS,
            "message": f"Entrenando YOLO ({n_train} train / {n_val} val)...",
        })
        current_state = training_state.copy()
    _broadcast(current_state)

    model = YOLO(YOLO_BASE_WEIGHTS)

    def _on_fit_epoch_end(trainer):
        """Callback de Ultralytics: se ejecuta al terminar cada época; actualiza estado y broadcast."""
        try:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", EPOCHS))
            loss_val = 0.0
            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                try:
                    loss_val = round(float(tloss.sum() if hasattr(tloss, "sum") else tloss), 4)
                except (TypeError, ValueError):
                    loss_val = 0.0
            metrics = getattr(trainer, "metrics", {}) or {}
            map50 = round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4)
            with state_lock:
                training_state.update({
                    "epoch": min(epoch, total),
                    "total_epochs": total,
                    "loss": loss_val,
                    "map50": map50,
                    "status": "training",
                })
                current_state = training_state.copy()
            _broadcast(current_state)
        except Exception:
            pass  # El progreso nunca debe romper el entrenamiento.

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        patience=10,
        batch=8,               # Batch pequeño pensado para CPU / memoria limitada en Railway.
        imgsz=settings.YOLO_IMG_SIZE,  # Múltiplo de 32 cercano al tablero rectificado 400x400.
        device="cpu",          # Sin GPU en producción (Railway): fuerza CPU explícitamente.
        verbose=False,
    )

    # Fase 3: localizar los mejores pesos (best.pt) y copiarlos a MODELS_DIR/yolo_chess.pt.
    best_path = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "best", None):
        best_path = str(trainer.best)
    if not best_path or not os.path.exists(best_path):
        # Fallback: buscar en runs/detect/train*/weights/best.pt (ubicación por defecto de Ultralytics).
        import glob
        candidatos = sorted(glob.glob("runs/detect/train*/weights/best.pt"), key=os.path.getmtime)
        if candidatos:
            best_path = candidatos[-1]
    if not best_path or not os.path.exists(best_path):
        raise RuntimeError("No se encontró best.pt tras el entrenamiento.")

    destino = os.path.join(settings.MODELS_DIR, "yolo_chess.pt")
    shutil.copy2(best_path, destino)

    with state_lock:
        training_state.update({
            "running": False,
            "status": "done",
            "message": f"Entrenamiento completado. Modelo guardado en {destino}.",
        })
        current_state = training_state.copy()
    _broadcast(current_state)
