"""
Router del motor de ajedrez Stockfish v17.1.

Endpoints:
    GET  /engine/status    | Verifica el estado y la versión del motor Stockfish v17.1.
    POST /engine/move      | Obtiene la mejor jugada dada una posición en FEN. Parámetros opcionales: elo (1320-3190) y depth (1-30).
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

# Importamos el servicio del motor.
from services.engine import engine_service

# Creamos el router para el motor.
router = APIRouter(prefix="/engine", tags=["Motor"])

# -------------------------------------------------------------------------------
# SCHEMA DE REQUEST
# Define los datos que el cliente debe enviar en el body del POST /engine/move.
# Pydantic valida automáticamente los tipos y restricciones antes de ejecutar el endpoint.
# -------------------------------------------------------------------------------
class EngineRequest(BaseModel):
    """
    Modelo de solicitud para consultar al motor Stockfish.

    Contiene la posición en formato FEN y los parámetros opcionales de configuración del análisis.

    Attributes:
        fen: Posición actual del tablero en notación FEN.
        elo: Nivel ELO objetivo del motor (1320-3190).
        depth: Profundidad de búsqueda (1-30).
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

# -------------------------------------------------------------------------------
# SCHEMA DE RESPONSE
# Define la estructura del JSON que devuelve el endpoint POST /engine/move.
# -------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /engine/status
# Verifica el estado y la versión del motor Stockfish v17.1.
# -------------------------------------------------------------------------------
@router.get(
    "/status",
    summary="Verifica el estado y la versión del motor Stockfish v17.1.",
)
def get_engine_status():
    """
    Verifica el estado y la versión del motor Stockfish v17.1.

    Returns:
        Dict con el estado del motor, versión y mensaje informativo.
    """
    return engine_service.check_status()

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /engine/move
# Obtiene la mejor jugada dada una posición en FEN. Parámetros opcionales: elo (1320-3190) y depth (1-30).
# -------------------------------------------------------------------------------
@router.post(
    "/move",
    response_model=EngineResponse,
    summary="Obtiene la mejor jugada dada una posición en FEN. Parámetros opcionales: elo (1320-3190) y depth (1-30).",
)
def get_move(request: EngineRequest):
    """
    Obtiene la mejor jugada dada una posición en FEN. Parámetros opcionales: elo (1320-3190) y depth (1-30).

    Args:
        request: Parámetros de la solicitud (FEN, ELO opcional, depth).

    Returns:
        EngineResponse con la mejor jugada y análisis asociado.

    Raises. Gestiona 3 casos:
        - Jugada normal: Devuelve best_move en notación UCI (ej: "e2e4").
        - Partida terminada: Devuelve best_move="(none)" con el motivo.
        - Error del motor: Lanza HTTP 503 si Stockfish no está disponible, HTTP 500 si es otro error.
    """
    try:
        # Llama al servicio para obtener la mejor jugada.
        best_move, info = engine_service.get_best_move(
            fen=request.fen,
            elo=request.elo,
            depth=request.depth
        )

        # Partida ya terminada (jaque mate, tablas, etc.) → respuesta informativa
        if info.get("game_over"):
            return EngineResponse(
                ok=True,
                best_move="(none)",
                info=info,
                message=f"Partida terminada: {info.get('reason')} — {info.get('result')}",
            )

        # Si no se obtuvo jugada por otro motivo, se lanza una excepción.
        if not best_move:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo obtener la mejor jugada del motor."
            )

        # Retorna la mejor jugada y la información del análisis.
        return EngineResponse(
            ok=True,
            best_move=best_move,
            info=info
        )

    # Motor no disponible (binario ausente o fallo de inicialización).
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )

    # Cualquier otro error inesperado.
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al consultar Stockfish: {str(e)}"
        )
