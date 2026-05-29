"""
Configuración de la aplicación.

Carga variables de entorno mediante Pydantic Settings y provee una instancia singleton accesible desde cualquier módulo.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache

# Clase para la centralización de las variables de la aplicación
class Settings(BaseSettings):
    """Configuración centralizada de la aplicación.

    Lee las variables del archivo .env y expone propiedades
    como la URL de conexión a la base de datos y parámetros JWT.
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

    # DATASET Y MODELOS
    DATASET_DIR: str = "./data/dataset"
    MODELS_DIR: str = "./data/models"
    CLASSES: list = [
        "empty",
        "w_P", "w_N", "w_B", "w_R", "w_Q", "w_K",
        "b_P", "b_N", "b_B", "b_R", "b_Q", "b_K"
    ]
    BOARD_SIZE: int = 400
    CELL_SIZE: int = 50
    IMG_SIZE: tuple = (96, 96)
    COLS: str = "abcdefgh"

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

    # Configuración del archivo .env
    class Config:
        """Configuración del archivo .env."""
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instancia singleton de Settings
@lru_cache()
def get_settings() -> Settings:
    """Retorna una instancia singleton de Settings.

    Utiliza lru_cache para que la configuración se cargue una sola vez
    y se reutilice en toda la aplicación, evitando releer el .env en cada request.
    """
    return Settings()

# Carga la configuración al importar el módulo, garantizando que esté disponible globalmente.
settings = get_settings()
