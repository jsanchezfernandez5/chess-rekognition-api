"""Módulo de gestión de partidas de ajedrez.

Proporciona endpoints CRUD para crear, listar, obtener,
actualizar y eliminar partidas. Cada operación está vinculada
al usuario autenticado.
"""

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
    """Crea una nueva partida asociada al usuario autenticado.

    Almacena la partida en la base de datos con el username
    del usuario como autor.

    Args:
        partida_in: Datos de la partida a crear.
        db: Sesión de base de datos.
        usuario_actual: Usuario autenticado (autor de la partida).

    Returns:
        PartidaResponse con los datos de la partida creada.
    """
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
    """Lista las partidas del usuario autenticado, con filtro opcional.

    Recupera el historial completo de partidas del usuario. Puede
    filtrarse por tipo: 'PI' (partida manual) o 'PR' (retransmisión).

    Args:
        tipo: Filtro opcional por tipo de partida (PI o PR).
        db: Sesión de base de datos.
        usuario_actual: Usuario autenticado.

    Returns:
        List[PartidaResponse] con las partidas del usuario.
    """
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
    """Obtiene los detalles de una partida concreta del usuario.

    Busca la partida por su identificador. Solo devuelve la partida
    si pertenece al usuario autenticado.

    Args:
        id_partida: Identificador único de la partida.
        db: Sesión de base de datos.
        usuario_actual: Usuario autenticado (propietario de la partida).

    Returns:
        PartidaResponse con los datos completos de la partida.

    Raises:
        HTTPException 404: Si la partida no existe o no pertenece al usuario.
    """
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
    """Actualiza los datos de una partida existente.

    Modifica únicamente los campos proporcionados en la solicitud.
    La partida debe pertenecer al usuario autenticado.

    Args:
        id_partida: Identificador único de la partida a actualizar.
        partida_in: Campos a actualizar.
        db: Sesión de base de datos.
        usuario_actual: Usuario autenticado (propietario de la partida).

    Returns:
        PartidaResponse con los datos actualizados de la partida.

    Raises:
        HTTPException 404: Si la partida no existe o no pertenece al usuario.
    """
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
    """Elimina una partida de forma definitiva.

    Borra la partida de la base de datos. Solo permite la eliminación
    si la partida pertenece al usuario autenticado.

    Args:
        id_partida: Identificador único de la partida a eliminar.
        db: Sesión de base de datos.
        usuario_actual: Usuario autenticado (propietario de la partida).

    Returns:
        None si la eliminación es exitosa.

    Raises:
        HTTPException 404: Si la partida no existe o no pertenece al usuario.
    """
    if not partidas_service.delete(id_partida, usuario_actual.username, db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Error al intentar borrar: partida no encontrada."
        )
    return None
