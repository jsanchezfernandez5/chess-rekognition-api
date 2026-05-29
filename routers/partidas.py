"""
Endpoints CRUD de partidas de ajedrez.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.dependencies import get_current_user
from core.models import Partida, PartidaCreate, PartidaUpdate, PartidaResponse, Usuario

# Creamos el router para las partidas.
router = APIRouter(prefix="/partidas", tags=["Partidas"])

# Endpoint para crear una nueva partida.
@router.post(
    "/", 
    status_code=status.HTTP_201_CREATED, 
    response_model=PartidaResponse, 
    summary="Guardar una nueva partida"
)
def create_partida(
    partida_in: PartidaCreate,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Crea una nueva partida asociada al usuario autenticado.
    """
    # Creamos la partida.
    nueva = Partida(**partida_in.model_dump(), username=usuario_actual.username)
    db.add(nueva)
    db.commit()

    # Refrescamos la partida y la retornamos.
    db.refresh(nueva)
    return nueva

# Endpoint para listar las partidas del usuario.
@router.get(
    "/", 
    response_model=List[PartidaResponse], 
    summary="Listar mis partidas")
def list_partidas(
    tipo: Optional[str] = Query(None, pattern="^(PI|PR)$", description="Filtro: PI (Manual) o PR (Retransmisión)"),
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Lista las partidas del usuario autenticado, opcionalmente filtradas por tipo.
    """
    # Filtramos las partidas por usuario y tipo.
    q = db.query(Partida).filter(Partida.username == usuario_actual.username)
    if tipo:
        q = q.filter(Partida.tipo_partida == tipo)

    # Retornamos las partidas ordenadas por fecha de registro.
    return q.order_by(Partida.fecha_registro.desc()).all()

# Endpoint para obtener una partida por ID.
@router.get(
    "/{id_partida}", 
    response_model=PartidaResponse, 
    summary="Ver detalles de una partida"
)
def get_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Obtiene el detalle de una partida. 
    Solo devuelve la partida si pertenece al usuario.
    """
    partida = db.query(Partida).filter(
        Partida.id_partida == id_partida,
        Partida.username == usuario_actual.username,
    ).first()

    # Lanzamos excepción si no existe la partida.
    if not partida:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Partida no encontrada.")
    
    # Retornamos la partida si existe
    return partida

# Endpoint para actualizar una partida.
@router.patch(
    "/{id_partida}", 
    response_model=PartidaResponse, 
    summary="Actualizar datos de una partida"
)
def update_partida(
    id_partida: int,
    partida_in: PartidaUpdate,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Actualiza los campos de una partida del usuario autenticado.
    """
    # Buscamos la partida.
    partida = db.query(Partida).filter(
        Partida.id_partida == id_partida,
        Partida.username == usuario_actual.username,
    ).first()

    # Lanzamos excepción si no existe la partida.
    if not partida:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Partida no encontrada o sin permiso.")

    # Actualizamos los campos de la partida.
    for key, value in partida_in.model_dump(exclude_unset=True).items():
        setattr(partida, key, value)

    # Guardamos los cambios en la base de datos.
    db.commit()
    db.refresh(partida)

    # Retornamos la partida actualizada.
    return partida


# Endpoint para eliminar una partida. (Actualmente no se usa en la app)
@router.delete(
    "/{id_partida}", 
    status_code=status.HTTP_204_NO_CONTENT, 
    summary="Eliminar una partida"
)
def delete_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Elimina una partida del usuario autenticado.
    """
    # Buscamos la partida.
    partida = db.query(Partida).filter(
        Partida.id_partida == id_partida,
        Partida.username == usuario_actual.username,
    ).first()

    # Lanzamos excepción si no existe la partida.
    if not partida:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Partida no encontrada.")

    # Eliminamos la partida.
    db.delete(partida)
    db.commit()
