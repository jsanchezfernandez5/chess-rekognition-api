"""Módulo de clasificación de piezas de ajedrez mediante un modelo TensorFlow (MobileNetV2).

Proporciona la clase ChessClassifier que carga un modelo preentrenado desde disco y
clasifica las 64 casillas de un tablero rectificado en un solo batch, devolviendo
la etiqueta y confianza para cada casilla."""
import cv2
import json
import os
import threading
import numpy as np
import tensorflow as tf
from core.config import settings

class ChessClassifier:
    """Clasificador de piezas de ajedrez basado en un modelo MobileNetV2 de TensorFlow.

    Gestiona la carga del modelo y los nombres de clases desde el disco, y proporciona
    un método para clasificar las 64 casillas de un tablero rectificado en un solo batch.
    Incluye sincronización con RLock para evitar condiciones de carrera durante la carga
    y la inferencia simultánea.
    """
    def __init__(self):
        """Inicializa el clasificador cargando el modelo y los nombres de clase desde el disco."""
        self.model = None
        self.class_names = []
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        """Carga el modelo TensorFlow y el archivo de nombres de clase desde el directorio de modelos.

        Busca los archivos chess_model.h5 y class_names.json en MODELS_DIR.
        Si no existen, el clasificador queda en estado no listo. Utiliza el candado
        _lock para garantizar que no se realicen predicciones durante la carga.

        Raises:
            Exception: Si el archivo del modelo existe pero no puede cargarse correctamente.
                       El error se captura y se muestra en consola sin propagarse.
        """
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
        """Recarga el modelo y los nombres de clase desde el disco sin necesidad de reiniciar el servidor.

        Útil para actualizar el modelo tras un reentrenamiento en caliente.
        """
        self._load()

    def is_ready(self) -> bool:
        """Verifica si el modelo y los nombres de clase están cargados y listos para inferencia.

        Returns:
            True si el modelo no es None y la lista de nombres de clase no está vacía.
            False en caso contrario.
        """
        with self._lock:
            return self.model is not None and len(self.class_names) > 0

    def classify_board(self, warped: np.ndarray) -> dict:
        """Clasifica las 64 casillas de un tablero rectificado usando el modelo MobileNetV2.

        Procesa todas las casillas en un solo batch para maximizar el rendimiento.
        Cada casilla se redimensiona a IMG_SIZE y se convierte a RGB antes de la
        inferencia. Los resultados incluyen la etiqueta y la confianza para cada
        casilla identificada por su notación algebraica.

        Args:
            warped: Imagen del tablero rectificado en vista cenital (400x400).

        Returns:
            Diccionario con una entrada por cada casilla. La clave es la notación
            algebraica (ej: "e4") y el valor es otro diccionario con las claves
            "label" (str) y "confidence" (float entre 0 y 1).

        Raises:
            RuntimeError: Si el clasificador no está listo (modelo no cargado).
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
