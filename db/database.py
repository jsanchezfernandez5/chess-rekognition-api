# db/database.py
# Configura la conexión a MySQL con SQLAlchemy.
# SQLAlchemy usa el patrón Session para gestionar transacciones y el patrón Declarative para definir modelos ORM.
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from core.config import settings

# engine: representa la conexión al motor de BD
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, echo=False)

# SessionLocal: fábrica de sesiones (una por request)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Base: clase base de la que heredan todos los modelos ORM
class Base(DeclarativeBase):
    """
    Clase base de la que heredan todos los modelos ORM.
    SQLAlchemy usa sus subclases para mapear tablas de la BD.
    """
    pass

# Función generadora de sesiones de BD para inyectar con Depends() en FastAPI.
def get_db():
    """
    Generador de sesiones de BD para inyectar con Depends().

    El bloque try/finally garantiza que la sesión siempre se cierra,
    aunque ocurra una excepción durante el request. Evita memory leaks
    y conexiones colgadas.

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
