# services/partidas.py
from sqlalchemy.orm import Session
from typing import List, Optional
from models.partidas import Partida
from schemas.partidas import PartidaCreate, PartidaUpdate

def create(data: PartidaCreate, username: str, db: Session) -> Partida:
    """
    Crea una nueva partida asociada a un usuario.
    """
    db_partida = Partida(**data.model_dump(), username=username)
    db.add(db_partida)
    db.commit()
    db.refresh(db_partida)
    return db_partida

def list_by_user(username: str, db: Session, tipo: Optional[str] = None) -> List[Partida]:
    """
    Lista las partidas de un usuario con filtrado opcional por tipo.
    Orden cronológico descendente por fecha de registro.
    """
    query = db.query(Partida).filter(Partida.username == username)
    if tipo:
        query = query.filter(Partida.tipo_partida == tipo)
    return query.order_by(Partida.fecha_registro.desc()).all()

def get_one(id_partida: int, username: str, db: Session) -> Optional[Partida]:
    """
    Obtiene una partida específica verificando que pertenezca al usuario.
    """
    return db.query(Partida).filter(
        Partida.id_partida == id_partida, 
        Partida.username == username
    ).first()

def update(id_partida: int, username: str, data: PartidaUpdate, db: Session) -> Optional[Partida]:
    """
    Actualiza datos de una partida de forma parcial.
    """
    db_partida = get_one(id_partida, username, db)
    if not db_partida:
        return None
    
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_partida, key, value)
    
    db.commit()
    db.refresh(db_partida)
    return db_partida

def delete(id_partida: int, username: str, db: Session) -> bool:
    """
    Elimina una partida perteneciente al usuario indicando el éxito de la operación.
    """
    db_partida = get_one(id_partida, username, db)
    if not db_partida:
        return False
    
    db.delete(db_partida)
    db.commit()
    return True
