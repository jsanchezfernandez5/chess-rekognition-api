import cv2
import json
import os
import threading
import numpy as np
import tensorflow as tf
from core.config import settings

class ChessClassifier:
    """
    Servicio de clasificación de piezas usando un modelo de TensorFlow (MobileNetV2).
    """
    def __init__(self):
        self.model = None
        self.class_names = []
        self._lock = threading.RLock() # Sincronización para evitar race conditions
        self._load()

    def _load(self):
        """Carga el modelo y los nombres de las clases desde el disco."""
        model_path = os.path.join(settings.MODELS_DIR, "chess_model.h5")
        names_path = os.path.join(settings.MODELS_DIR, "class_names.json")
        
        with self._lock: # Asegura que nadie está prediciendo mientras cargamos
            if os.path.exists(model_path) and os.path.exists(names_path):
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    with open(names_path, "r") as f:
                        self.class_names = json.load(f)
                    print(f"Modelo cargado correctamente desde {model_path}")
                except Exception as e:
                    print(f"Error cargando el modelo: {e}")
            else:
                print("No se encontró un modelo entrenado. Por favor, entrena el modelo primero.")

    def reload(self):
        """Recarga el modelo en memoria sin reiniciar el servidor."""
        self._load()

    def is_ready(self) -> bool:
        """Indica si el modelo está listo para realizar predicciones."""
        with self._lock:
            return self.model is not None and len(self.class_names) > 0

    def classify_board(self, warped: np.ndarray) -> dict:
        """
        Clasifica las 64 casillas de un tablero rectificado en un solo batch.
        """
        with self._lock: # Bloquea el acceso al modelo durante la predicción
            if not self.is_ready():
                raise RuntimeError("El clasificador no está listo. Asegúrate de que el modelo esté entrenado.")

            batch = []
            sq_ids = []
            
            for row in range(8):
                for col in range(8):
                    x, y = col * settings.CELL_SIZE, row * settings.CELL_SIZE
                    cell = warped[y:y+settings.CELL_SIZE, x:x+settings.CELL_SIZE]
                    
                    cell_resized = cv2.resize(cell, settings.IMG_SIZE)
                    cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)
                    
                    batch.append(cell_rgb)
                    sq_ids.append(f"{settings.COLS[col]}{8 - row}")

            preds = self.model.predict(np.array(batch, dtype=np.float32), verbose=0)
            
            result = {}
            for i, sq_id in enumerate(sq_ids):
                idx = int(np.argmax(preds[i]))
                label = self.class_names[idx]
                confidence = float(preds[i][idx])
                
                result[sq_id] = {
                    "label": label,
                    "confidence": round(confidence, 3)
                }
                
            return result

# Instancia global del clasificador
classifier = ChessClassifier()
