"""Configuración de la base de datos.

Configura la conexión a MySQL con SQLAlchemy, la fábrica de sesiones
y la clase base declarativa para los modelos ORM.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, echo=False)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Clase base para todos los modelos ORM.

    SQLAlchemy utiliza sus subclases para mapear tablas de la base de datos.
    """
    pass

def get_db():
    """Generador de sesiones de base de datos para inyectar con Depends().

    El bloque try/finally garantiza que la sesión siempre se cierre,
    incluso ante una excepción, evitando fugas de conexiones.

    Uso:
        @router.get("/ejemplo")
        def ejemplo(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
