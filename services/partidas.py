# services/partidas.py
# Lógica de negocio para las partidas
"""Módulo de gestión de partidas de ajedrez.

Proporciona funciones CRUD para crear, listar, obtener, actualizar y eliminar
partidas, vinculadas al usuario autenticado y con soporte para filtrado por tipo."""
from sqlalchemy.orm import Session
from typing import List, Optional
from models.partidas import Partida
from schemas.partidas import PartidaCreate, PartidaUpdate

# Función para crear una nueva partida.
# Se llama desde el endpoint POST /partidas.
def create(partida_in: PartidaCreate, username: str, db: Session) -> Partida:
    """Crea una nueva partida en la base de datos asociada al usuario autenticado.

    Convierte el esquema de entrada en un modelo ORM, lo persiste en la base de
    datos y devuelve el objeto creado con los datos actualizados.

    Args:
        partida_in: Esquema PartidaCreate con los datos de la nueva partida.
        username: Nombre de usuario propietario de la partida.
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        Objeto Partida recién creado y persistido.
    """
    # Convertimos el esquema de entrada en un modelo de base de datos
    db_partida = Partida(**partida_in.model_dump(), username=username)
    db.add(db_partida)
    db.commit()
    db.refresh(db_partida)
    return db_partida

# Función para listar todas las partidas de un usuario.
# Se llama desde el endpoint GET /partidas.
def list_by_user(username: str, db: Session, tipo: Optional[str] = None) -> List[Partida]:
    """Obtiene todas las partidas de un usuario, ordenadas por fecha de registro descendente.

    Permite filtrar opcionalmente por tipo de partida (PI para partida introducida
    manualmente, PR para partida por retransmisión).

    Args:
        username: Nombre de usuario propietario de las partidas.
        db: Sesión de base de datos SQLAlchemy.
        tipo: Filtro opcional por tipo de partida ("PI" o "PR").

    Returns:
        Lista de objetos Partida ordenados del más reciente al más antiguo.
    """
    query = db.query(Partida).filter(Partida.username == username)
    if tipo:
        query = query.filter(Partida.tipo_partida == tipo)
    return query.order_by(Partida.fecha_registro.desc()).all()

# Función para obtener una partida concreta.
# Se llama desde el endpoint GET /partidas/{id_partida}.
def get_one(id_partida: int, username: str, db: Session) -> Optional[Partida]:
    """Obtiene una partida específica por su ID, verificando que pertenezca al usuario.

    Realiza una búsqueda combinada por identificador de partida y nombre de usuario
    para garantizar que solo el propietario pueda acceder a sus partidas.

    Args:
        id_partida: Identificador único de la partida.
        username: Nombre de usuario propietario.
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        Objeto Partida si existe y pertenece al usuario, None en caso contrario.
    """
    return db.query(Partida).filter(
        Partida.id_partida == id_partida, 
        Partida.username == username
    ).first()

# Función para actualizar una partida existente.
# Se llama desde el endpoint PUT /partidas/{id_partida}.
def update(id_partida: int, username: str, partida_in: PartidaUpdate, db: Session) -> Optional[Partida]:
    """Actualiza parcialmente los campos de una partida existente.

    Busca la partida por ID y propietario, y aplica únicamente los campos
    enviados en el esquema de actualización (excluyendo los no especificados).
    Persiste los cambios y devuelve la partida actualizada.

    Args:
        id_partida: Identificador único de la partida a actualizar.
        username: Nombre de usuario propietario.
        partida_in: Esquema PartidaUpdate con los campos a modificar.
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        Objeto Partida actualizado si existe y pertenece al usuario, None en caso contrario.
    """
    db_partida = get_one(id_partida, username, db)
    if not db_partida:
        return None
    
    # Solo actualizamos los campos que el usuario ha enviado realmente
    update_data = partida_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_partida, key, value)
    
    db.commit()
    db.refresh(db_partida)
    return db_partida

# Función para borrar una partida existente.
# Se llama desde el endpoint DELETE /partidas/{id_partida}.
def delete(id_partida: int, username: str, db: Session) -> bool:
    """Elimina una partida de la base de datos si existe y pertenece al usuario.

    Busca la partida por ID y propietario. Si existe, la elimina y confirma la
    transacción. Si no existe o no pertenece al usuario, retorna False sin errores.

    Args:
        id_partida: Identificador único de la partida a eliminar.
        username: Nombre de usuario propietario.
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        True si la partida se eliminó correctamente, False si no se encontró.
    """
    db_partida = get_one(id_partida, username, db)
    if not db_partida:
        return False
    
    db.delete(db_partida)
    db.commit()
    return True
