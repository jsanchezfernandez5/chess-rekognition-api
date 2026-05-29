"""Configuración de la base de datos.

Configura la conexión a MySQL con SQLAlchemy, la fábrica de sesiones
y la clase base declarativa para los modelos ORM.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from core.config import settings

# Configuración de la conexión a la base de datos usando SQLAlchemy.
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, echo=False)

# Fábrica de sesiones para crear conexiones a la base de datos.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Clase base para los modelos ORM que se mapearán a las tablas de la base de datos.
class Base(DeclarativeBase):
    """Clase base para todos los modelos ORM.

    SQLAlchemy utiliza sus subclases para mapear tablas de la base de datos.
    """
    pass

# Generador de sesiones de base de datos para inyectar con Depends() en FastAPI.
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
