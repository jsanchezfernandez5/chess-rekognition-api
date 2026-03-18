# routers/partidas.py - Endpoints para gestionar las partidas
# Aquí definimos las rutas para que el usuario pueda guardar, ver, editar o borrar sus partidas.

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from core.dependencies import get_current_user
from db.database import get_db
from models.usuarios import Usuario
from schemas.partidas import PartidaCreate, PartidaUpdate, PartidaResponse
from services import partidas as partidas_service

router = APIRouter(prefix="/partidas", tags=["Partidas"])

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=PartidaResponse,
    summary="Guardar una nueva partida",
)
def create_partida(
    partida_in: PartidaCreate, 
    db: Session = Depends(get_db), 
    usuario_actual: Usuario = Depends(get_current_user)
):
    # Delegamos la creación al servicio pasándole el autor (usuario logueado)
    return partidas_service.create(partida_in, usuario_actual.username, db)

@router.get(
    "/",
    response_model=List[PartidaResponse],
    summary="Listar mis partidas",
)
def list_partidas(
    tipo: Optional[str] = Query(None, pattern="^(PI|PR)$", description="Filtro opcional: PI (Manual) o PR (Retransmisión)"),
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user)
):
    # Recuperamos el historial del usuario, pudiendo filtrar por el tipo de entrada
    return partidas_service.list_by_user(usuario_actual.username, db, tipo)

@router.get(
    "/{id_partida}",
    response_model=PartidaResponse,
    summary="Ver detalles de una partida",
)
def get_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user)
):
    # Buscamos la partida. Si no existe o no es nuestra, lanzamos un 404.
    partida = partidas_service.get_one(id_partida, usuario_actual.username, db)
    if not partida:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No hemos encontrado la partida solicitada."
        )
    return partida

@router.patch(
    "/{id_partida}",
    response_model=PartidaResponse,
    summary="Actualizar datos de una partida",
)
def update_partida(
    id_partida: int,
    partida_in: PartidaUpdate,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user)
):
    # Actualizamos solo los campos que nos envía el usuario
    db_partida = partidas_service.update(id_partida, usuario_actual.username, partida_in, db)
    if not db_partida:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No se ha podido actualizar la partida (no existe o no tienes permiso)."
        )
    return db_partida

@router.delete(
    "/{id_partida}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar una partida",
)
def delete_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user)
):
    # Borrado definitivo de la partida del usuario
    if not partidas_service.delete(id_partida, usuario_actual.username, db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Error al intentar borrar: partida no encontrada."
        )
    return None
