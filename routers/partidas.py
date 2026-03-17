# routers/partidas.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from core.dependencies import get_current_user
from db.database import get_db
from models.usuarios import Usuario
from schemas.partidas import PartidaCreate, PartidaUpdate, PartidaResponse
from services import partidas as partidas_service

router = APIRouter(prefix="/partidas", tags=["Partidas"])

# POST: Crear nueva partida asociada al usuario autenticado.
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=PartidaResponse,
    summary="Añadir una partida",
    description="Crea una nueva partida en el historial del usuario autenticado.",
)
def create_partida(
    schema: PartidaCreate, 
    db: Session = Depends(get_db), 
    current_user: Usuario = Depends(get_current_user)
):
    # La lógica de creación se delega al servicio y se inyecta el username del token.
    return partidas_service.create(schema, current_user.username, db)

# GET: Listar partidas del usuario autenticado, con filtrado opcional por tipo.
@router.get(
    "/",
    response_model=List[PartidaResponse],
    summary="Listado de partidas",
    description="Devuelve el historial del usuario autenticado de forma cronológica descendente.",
)
def list_partidas(
    tipo: Optional[str] = Query(None, pattern="^(PI|PR)$", description="Filtrar por 'PI' o 'PR'."),
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user)
):
    return partidas_service.list_by_user(current_user.username, db, tipo)

# GET: Obtener detalles de una partida por su ID único.
@router.get(
    "/{id_partida}",
    response_model=PartidaResponse,
    summary="Detalles de una partida",
)
def get_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user)
):
    partida = partidas_service.get_one(id_partida, current_user.username, db)
    if not partida:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Partida no encontrada o no pertenece al usuario."
        )
    return partida

# PATCH: Actualizar de forma parcial los datos de una partida.
@router.patch(
    "/{id_partida}",
    response_model=PartidaResponse,
    summary="Editar una partida",
)
def update_partida(
    id_partida: int,
    schema: PartidaUpdate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user)
):
    db_partida = partidas_service.update(id_partida, current_user.username, schema, db)
    if not db_partida:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Partida no encontrada o sin permisos de edición."
        )
    return db_partida

# DELETE: Eliminar permanentemente una partida.
@router.delete(
    "/{id_partida}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Borrar una partida",
)
def delete_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user)
):
    if not partidas_service.delete(id_partida, current_user.username, db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Partida no encontrada o sin permisos para eliminarla."
        )
    return None
