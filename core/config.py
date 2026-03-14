# core/config.py
# Settings (carga .env y configuración de la aplicación)
from pydantic_settings import BaseSettings
from functools import lru_cache

# Configuración de la aplicación usando Pydantic BaseSettings
class Settings(BaseSettings):
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

    @property
    def DATABASE_URL(self) -> str:
        """
        Construye la URL de conexión SQLAlchemy a partir de las variables individuales. 
        Se usa pymysql como driver para MySQL/MariaDB.
        """
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?ssl_disabled=true"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Función para obtener la configuración de la aplicación. Patrón Singlenton usando lru_cache para evitar múltiples instancias.
@lru_cache()
def get_settings() -> Settings:
    """
    Singleton de configuración usando lru_cache.
    Se instancia una sola vez y se reutiliza en toda la aplicación evitando releer el .env en cada request.
    """
    return Settings()

# Instancia global de configuración que se puede importar en cualquier módulo de la aplicación
# Uso: from core.config import settings
settings = get_settings()