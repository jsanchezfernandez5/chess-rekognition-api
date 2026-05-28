# routers/engine.py
# Endpoints para interactuar con Stockfish

"""Módulo de interacción con el motor de ajedrez Stockfish.

Proporciona endpoints para consultar el estado del motor,
obtener la mejor jugada desde una posición FEN y configurar
parámetros como el nivel ELO y la profundidad de búsqueda.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from services.engine import engine_service

# Creación del Router de Stockfish
router = APIRouter(prefix="/engine", tags=["Motor"])

class EngineRequest(BaseModel):
    """Modelo de solicitud para consultar al motor Stockfish.

    Contiene la posición en formato FEN y los parámetros
    opcionales de configuración del análisis.

    Attributes:
        fen: Posición actual del tablero en notación FEN.
        elo: Nivel ELO objetivo del motor (1320-3190).
        depth: Profundidad de búsqueda en semimovidas (1-30).
    """
    fen: str = Field(
        ..., 
        description="Posición en formato FEN (Forsyth-Edwards Notation)",
        examples=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
    )
    elo: Optional[int] = Field(
        None, 
        ge=1320, 
        le=3190, 
        description="Nivel de ELO (1320-3190)",
        examples=[1800]
    )
    depth: int = Field(
        15, 
        ge=1, 
        le=30, 
        description="Profundidad de búsqueda (1-30)",
        examples=[15]
    )

class EngineResponse(BaseModel):
    """Modelo de respuesta del motor Stockfish.

    Contiene el resultado del análisis junto con información
    detallada de la evaluación.

    Attributes:
        ok: Indica si la operación se completó correctamente.
        best_move: Mejor jugada encontrada en notación UCI.
        info: Información adicional del análisis (score, depth, pv, nodes).
        message: Mensaje descriptivo adicional.
    """
    ok: bool
    best_move: str
    info: Optional[dict] = Field(None, description="Información de análisis (score, depth, pv, nodes)")
    message: Optional[str] = None

# Endpoint de Estado del Motor
# GET /engine/status
@router.get(
    "/status",
    summary="Verificar salud y versión del motor",
)
def get_engine_status():
    """Verifica el estado y la versión del motor de ajedrez.

    Comprueba que el binario de Stockfish existe y responde
    correctamente a los comandos UCI, devolviendo información
    sobre su disponibilidad y versión.

    Returns:
        Dict con el estado del motor, versión y mensaje informativo.
    """
    return engine_service.check_status()

# Endpoint de Movimiento del Motor
# POST /engine/move
@router.post(
    "/move",
    response_model=EngineResponse,
    summary="Obtiene la mejor jugada dada una posición en FEN",
)
def get_move(request: EngineRequest):
    """Obtiene la mejor jugada desde una posición FEN mediante Stockfish.

    Envía la posición al motor de ajedrez, que la analiza con los
    parámetros especificados (ELO y profundidad) y devuelve la
    mejor jugada encontrada junto con información de evaluación.

    Args:
        request: Parámetros de la solicitud (FEN, ELO opcional, depth).

    Returns:
        EngineResponse con la mejor jugada y análisis asociado.

    Raises:
        HTTPException 503: Si el binario de Stockfish no está disponible.
        HTTPException 500: Si ocurre un error interno durante el análisis.
    """
    try:
        best_move, info = engine_service.get_best_move(
            fen=request.fen,
            elo=request.elo,
            depth=request.depth
        )
        
        if not best_move:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo obtener la mejor jugada del motor."
            )
            
        return EngineResponse(
            ok=True, 
            best_move=best_move, 
            info=info
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al consultar Stockfish: {str(e)}"
        )
