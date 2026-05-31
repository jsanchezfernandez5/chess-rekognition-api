"""
Configura la conexión a MySQL con SQLAlchemy.

Fábrica de sesiones, clase base declarativa y generador de sesiones para inyectar con Depends() en los endpoints.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
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
    """
    Generador de sesiones de base de datos para inyectar con Depends().
    """
    db = SessionLocal()
    try:        
        yield db # Genera una sesión de base de datos
    finally:        
        db.close() # Cierra la sesión
