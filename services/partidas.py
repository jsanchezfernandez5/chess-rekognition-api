# services/partidas.py - Lógica de negocio para las partidas
# Aquí gestionamos las operaciones directas con la base de datos para las partidas.

from sqlalchemy.orm import Session
from typing import List, Optional
from models.partidas import Partida
from schemas.partidas import PartidaCreate, PartidaUpdate

def create(partida_in: PartidaCreate, username: str, db: Session) -> Partida:
    """
    Crea un nuevo registro de partida en la base de datos vinculado al usuario.
    """
    # Convertimos el esquema de entrada en un modelo de base de datos
    db_partida = Partida(**partida_in.model_dump(), username=username)
    db.add(db_partida)
    db.commit()
    db.refresh(db_partida)
    return db_partida

def list_by_user(username: str, db: Session, tipo: Optional[str] = None) -> List[Partida]:
    """
    Obtiene todas las partidas de un usuario. Permite filtrar por tipo (PI o PR).
    Las devuelve ordenadas por fecha de creación (de más nueva a más antigua).
    """
    query = db.query(Partida).filter(Partida.username == username)
    if tipo:
        query = query.filter(Partida.tipo_partida == tipo)
    return query.order_by(Partida.fecha_registro.desc()).all()

def get_one(id_partida: int, username: str, db: Session) -> Optional[Partida]:
    """
    Busca una partida concreta asegurando que pertenece al usuario que la solicita.
    """
    return db.query(Partida).filter(
        Partida.id_partida == id_partida, 
        Partida.username == username
    ).first()

def update(id_partida: int, username: str, partida_in: PartidaUpdate, db: Session) -> Optional[Partida]:
    """
    Modifica campos específicos de una partida (actualización parcial).
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

def delete(id_partida: int, username: str, db: Session) -> bool:
    """
    Borra una partida de la base de datos si existe y es propiedad del usuario.
    """
    db_partida = get_one(id_partida, username, db)
    if not db_partida:
        return False
    
    db.delete(db_partida)
    db.commit()
    return True
