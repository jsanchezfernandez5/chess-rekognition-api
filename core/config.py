"""
Configuración de la aplicación.

Carga variables de entorno mediante Pydantic Settings y provee una instancia singleton accesible desde cualquier módulo.
"""
import os

from pydantic_settings import BaseSettings

# Decorador de la librería estándar que cachea el resultado de una función (evita ejecutarla más de una vez)
from functools import lru_cache

# Clase para la centralización de las variables de la aplicación
class Settings(BaseSettings):
    """Configuración centralizada de la aplicación.

    Lee las variables del archivo .env y expone propiedades como la URL de conexión a la base de datos y parámetros JWT.
    """
    # BASE DE DATOS
    DB_HOST: str
    DB_PORT: int = 3306
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # MAIL
    RESEND_API_KEY: str
    RESEND_FROM: str = "onboarding@resend.dev"

    # MODELOS
    MODELS_DIR: str = "./data/models"

    # DATASET Y MODELO YOLO26 (detección de objetos)
    # Directorio donde se guardan las imágenes anotadas y sus etiquetas en formato YOLO:
    #   yolo_dataset/images/  → imágenes del tablero rectificado 400x400 (.jpg)
    #   yolo_dataset/labels/  → un .txt por imagen con cajas "class_id x_center y_center width height" normalizadas 0-1
    YOLO_DATASET_DIR: str = "./data/yolo_dataset"

    # Clases del detector YOLO: detección de objetos con bounding boxes.
    # Solo se anotan objetos reales con su bounding box, así que "empty"
    # no aplica (una zona sin pieza simplemente no tiene caja), y se añade la clase "hand"
    # para detectar manos o dedos sobre el tablero.
    YOLO_CLASSES: list = [
        "w_P", "w_N", "w_B", "w_R", "w_Q", "w_K",
        "b_P", "b_N", "b_B", "b_R", "b_Q", "b_K",
        "hand"
    ]

    # Parámetros de inferencia del detector YOLO26:
    #   - Confianza mínima para aceptar una detección (valor por defecto razonable de Ultralytics).
    #   - Umbral IoU del postprocesado de cajas solapadas (Non-Maximum Suppression).
    #   - Tamaño de imagen de inferencia/entrenamiento en píxeles (múltiplo de 32; el tablero
    #     rectificado mide 400x400, así que 416 es el múltiplo de 32 inmediatamente superior).
    YOLO_CONF_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_IMG_SIZE: int = 416

    # Confianza mínima para que una detección de mano invalide la lectura de las casillas que ocupa.
    HAND_MIN_CONFIDENCE: float = 0.45

    # VARIABLES PARA LA VISIÓN POR COMPUTADORA
    BOARD_SIZE: int = 400       # Tamaño en píxeles del tablero rectificado.
    CELL_SIZE: int = 50         # Tamaño en píxeles de cada casilla (400 / 8 columnas = 50 px).
    COLS: str = "abcdefgh"      # Letras de las columnas del tablero en notación algebraica de ajedrez.

    # Propiedad autocalculada para construir la URL de conexión a la base de datos a partir de las variables individuales.
    @property
    def DATABASE_URL(self) -> str:
        """
        Construye la URL de conexión SQLAlchemy a partir de las variables individuales.

        Usa pymysql como driver para MySQL o MariaDB.
        """
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?ssl_disabled=true"
        )

    # Propiedad autocalculada con la ruta del modelo YOLO entrenado dentro de MODELS_DIR.
    @property
    def YOLO_MODEL_PATH(self) -> str:
        """Ruta absoluta/relativa del fichero .pt del detector YOLO entrenado."""
        return os.path.join(self.MODELS_DIR, "yolo_chess.pt")

    # Configuración interna de pydantic para decir a BaseSettings donde está el archivo .env y su codificación.
    class Config:
        """Configuración del archivo .env."""
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instancia Singleton de Settings
@lru_cache()
def get_settings() -> Settings:
    """Retorna una instancia singleton de Settings.

    Utiliza lru_cache para que la configuración se cargue una sola vez
    y se reutilice en toda la aplicación, evitando releer el .env en cada request.
    """
    return Settings()

# Carga la configuración al importar el módulo, garantizando que esté disponible globalmente.
settings = get_settings()
