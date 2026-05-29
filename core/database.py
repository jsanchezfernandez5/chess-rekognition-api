"""
Configuración de la base de datos.

Configura la conexión a MySQL con SQLAlchemy, la fábrica de sesiones
y la clase base declarativa para los modelos ORM.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Importamos el settings
from core.config import settings

# Motor de base de datos
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, echo=False)

# Fábrica de sesiones
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Clase base declarativa
class Base(DeclarativeBase):
    pass

# Generador de sesiones
def get_db():
    """Generador de sesiones de base de datos para inyectar con Depends().

    El bloque try/finally garantiza que la sesión siempre se cierre,
    incluso ante una excepción, evitando fugas de conexiones.
    """
    db = SessionLocal()
    try:
        # Genera una sesión de base de datos
        yield db
    finally:
        # Cierra la sesión
        db.close()
