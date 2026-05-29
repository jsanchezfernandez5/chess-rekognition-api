"""
Módulo de clasificación de piezas de ajedrez mediante un modelo TensorFlow (MobileNetV2).

    - Proporciona la clase ChessClassifier que carga un modelo preentrenado desde disco.
    - Clasifica las 64 casillas de un tablero rectificado en un solo batch
    - Devuelve la etiqueta y confianza para cada casilla.
"""
import cv2
import json
import os
import threading
import numpy as np
import tensorflow as tf
from core.config import settings

# Clase que carga el modelo y clasifica las piezas
class ChessClassifier:
    """
    Clasificador de piezas de ajedrez basado en un modelo MobileNetV2 de TensorFlow.
    """
    # Método constructor que carga el modelo
    def __init__(self):
        """
        Inicializa el clasificador cargando el modelo y los nombres de clase desde el disco.
        """
        self.model = None
        self.class_names = []
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        """
        Carga el modelo TensorFlow y nombres de clase desde MODELS_DIR.

        Si los archivos no existen, el clasificador queda en estado no listo.
        Usa _lock para evitar inferencias durante la recarga.
        """
        model_path = os.path.join(settings.MODELS_DIR, "chess_model.h5")
        names_path = os.path.join(settings.MODELS_DIR, "class_names.json")

        # Bloqueo para evitar inferencias durante la recarga
        with self._lock:
            if os.path.exists(model_path) and os.path.exists(names_path):
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    with open(names_path, "r") as f:
                        self.class_names = json.load(f)
                    print(f"Modelo cargado desde {model_path}")
                except Exception as e:
                    print(f"Error cargando el modelo: {e}")
            else:
                print("No hay modelo entrenado. Usa el endpoint /dataset/train primero.")

    # Método para recargar el modelo desde disco en caliente
    def reload(self):
        """
        Recarga el modelo desde disco en caliente (sin reiniciar servidor).
        """
        self._load()

    # Método para verificar si el modelo está cargado y listo para inferencia
    def is_ready(self) -> bool:
        """
        Verifica si el modelo está cargado y listo para inferencia.
        """
        with self._lock:
            return self.model is not None and len(self.class_names) > 0

    # Método para clasificar las 64 casillas de un tablero rectificado usando MobileNetV2
    def classify_board(self, warped: np.ndarray) -> dict:
        """
        Clasifica las 64 casillas de un tablero rectificado usando MobileNetV2.

        Args:
            warped: Tablero rectificado 400x400 en vista cenital.

        Returns:
            Dict con 64 entradas: clave = notación algebraica (ej: "e4"),
            valor = {"label": str, "confidence": float}.

        Raises:
            RuntimeError: Si el modelo no está cargado.
        """
        # Bloqueo para evitar inferencias durante la recarga
        with self._lock:
            if not self.is_ready():
                raise RuntimeError("El clasificador no está listo. Entrena el modelo primero.")

            # Prepara el lote de imágenes
            batch = []
            sq_ids = []

            # Itera sobre cada casilla del tablero rectificado
            for row in range(8):
                for col in range(8):
                    # Extraer cada casilla de 50x50 del tablero rectificado
                    x, y = col * settings.CELL_SIZE, row * settings.CELL_SIZE
                    cell = warped[y:y + settings.CELL_SIZE, x:x + settings.CELL_SIZE]

                    # Redimensionar a 96x96 (input esperado por MobileNetV2) y convertir a RGB
                    cell_resized = cv2.resize(cell, settings.IMG_SIZE)
                    cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)

                    # Añade la casilla procesada al lote y su identificador
                    batch.append(cell_rgb)
                    sq_ids.append(f"{settings.COLS[col]}{8 - row}")

            # Inferencia por lotes: el modelo procesa las 64 casillas simultáneamente
            preds = self.model.predict(np.array(batch, dtype=np.float32), verbose=0)

            # Construye el resultado
            result = {}
            for i, sq_id in enumerate(sq_ids):
                idx = int(np.argmax(preds[i]))
                result[sq_id] = {
                    "label": self.class_names[idx],
                    "confidence": round(float(preds[i][idx]), 3)
                }

            # Devuelve el resultado
            return result

# Instancia global del clasificador
classifier = ChessClassifier()
