import cv2
import json
import os
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
        self._load()

    def _load(self):
        """Carga el modelo y los nombres de las clases desde el disco."""
        model_path = os.path.join(settings.MODELS_DIR, "chess_model.h5")
        names_path = os.path.join(settings.MODELS_DIR, "class_names.json")
        
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
        return self.model is not None and len(self.class_names) > 0

    def classify_board(self, warped: np.ndarray) -> dict:
        """
        Clasifica las 64 casillas de un tablero rectificado en un solo batch.
        
        Args:
            warped: Imagen de 400x400 px del tablero rectificado.
            
        Returns:
            Diccionario con el resultado por casilla: {"e2": {"label": "w_P", "confidence": 0.99}, ...}
        """
        if not self.is_ready():
            raise RuntimeError("El clasificador no está listo. Asegúrate de que el modelo esté entrenado.")

        batch = []
        sq_ids = []
        
        # Extraer cada casilla, redimensionar y preparar para el batch
        for row in range(8):
            for col in range(8):
                x, y = col * settings.CELL_SIZE, row * settings.CELL_SIZE
                cell = warped[y:y+settings.CELL_SIZE, x:x+settings.CELL_SIZE]
                
                # Preprocesamiento: resize a 96x96 y conversión a RGB (MobileNetV2 espera RGB)
                cell_resized = cv2.resize(cell, settings.IMG_SIZE)
                cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)
                
                batch.append(cell_rgb)
                sq_ids.append(f"{settings.COLS[col]}{8 - row}")

        # Ejecutar predicción en batch
        # np.array(batch) tiene forma (64, 96, 96, 3)
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
