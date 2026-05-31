"""
Router CRUD de partidas de ajedrez.

Endpoints:
    POST   /partidas/                 | Crea una nueva partida asociada al usuario autenticado.
    GET    /partidas/                 | Lista todas las partidas del usuario autenticado, con filtro opcional por tipo (PI: Partida Introducida / PR: Partida Retransmitida).
    GET    /partidas/{id_partida}     | Obtiene el detalle de una partida por ID. Solo devuelve la partida si pertenece al usuario.
    PATCH  /partidas/{id_partida}     | Actualiza los campos enviados de una partida del usuario autenticado.
    DELETE /partidas/{id_partida}     | Elimina una partida del usuario autenticado. (Actualmente no se usa en la app).
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.dependencies import get_current_user
from core.models import Partida, PartidaCreate, PartidaUpdate, PartidaResponse, Usuario

# Creamos el router para las partidas.
router = APIRouter(prefix="/partidas", tags=["Partidas"])

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /partidas/
# Crea una nueva partida asociada al usuario autenticado.
# -------------------------------------------------------------------------------
@router.post(
    "/", 
    status_code=status.HTTP_201_CREATED, 
    response_model=PartidaResponse, 
    summary="Crea una nueva partida asociada al usuario autenticado."
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

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /partidas/
# Lista todas las partidas del usuario autenticado, con filtro opcional por tipo (PI: Partida Introducida / PR: Partida Retransmitida).
# -------------------------------------------------------------------------------
@router.get(
    "/", 
    response_model=List[PartidaResponse], 
    summary="Lista todas las partidas del usuario autenticado, con filtro opcional por tipo (PI: Partida Introducida / PR: Partida Retransmitida).")
def list_partidas(
    tipo: Optional[str] = Query(None, pattern="^(PI|PR)$", description="Filtro: PI (Partida Introducida) o PR (Partida Retransmitida)"),
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Lista todas las partidas del usuario autenticado, con filtro opcional por tipo (PI: Partida Introducida / PR: Partida Retransmitida).
    """
    # Filtramos las partidas por usuario y tipo.
    q = db.query(Partida).filter(Partida.username == usuario_actual.username)
    if tipo:
        q = q.filter(Partida.tipo_partida == tipo)

    # Retornamos las partidas ordenadas por fecha de registro.
    return q.order_by(Partida.fecha_registro.desc()).all()

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /partidas/{id_partida}
# Obtiene el detalle de una partida por ID. Solo devuelve la partida si pertenece al usuario.
# -------------------------------------------------------------------------------
@router.get(
    "/{id_partida}", 
    response_model=PartidaResponse, 
    summary="Obtiene el detalle de una partida por ID. Solo devuelve la partida si pertenece al usuario."
)
def get_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Obtiene el detalle de una partida por ID. Solo devuelve la partida si pertenece al usuario.
    """
    # Verificamos que la partida exista y pertenezca al usuario.
    partida = db.query(Partida).filter(
        Partida.id_partida == id_partida,
        Partida.username == usuario_actual.username,
    ).first()

    # Lanzamos excepción si no existe la partida.
    if not partida:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Partida no encontrada.")
    
    # Retornamos la partida si existe
    return partida

# -------------------------------------------------------------------------------
# [ENDPOINT] - PATCH /partidas/{id_partida}
# Actualiza los campos enviados de una partida del usuario autenticado.
# -------------------------------------------------------------------------------
@router.patch(
    "/{id_partida}", 
    response_model=PartidaResponse, 
    summary="Actualiza los campos enviados de una partida del usuario autenticado."
)
def update_partida(
    id_partida: int,
    partida_in: PartidaUpdate,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Actualiza los campos enviados de una partida del usuario autenticado.
    """
    # Verificamos que la partida exista y pertenezca al usuario.
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

# -------------------------------------------------------------------------------
# [ENDPOINT] - DELETE /partidas/{id_partida}
# Elimina una partida del usuario autenticado. (Actualmente no se usa en la app).
# -------------------------------------------------------------------------------
@router.delete(
    "/{id_partida}", 
    status_code=status.HTTP_204_NO_CONTENT, 
    summary="Elimina una partida del usuario autenticado"
)
def delete_partida(
    id_partida: int,
    db: Session = Depends(get_db),
    usuario_actual: Usuario = Depends(get_current_user),
):
    """
    Elimina una partida del usuario autenticado.
    """
    # Verificamos que la partida exista y pertenezca al usuario.
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
