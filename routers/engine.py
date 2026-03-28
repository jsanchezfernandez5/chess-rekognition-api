# routers/engine.py
# Endpoints para interactuar con Stockfish
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import Optional
from services.engine import engine_service

router = APIRouter(prefix="/engine", tags=["Motor"])

class EngineRequest(BaseModel):
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
    ok: bool
    best_move: str
    message: Optional[str] = None

@router.get(
    "/status",
    summary="Verificar salud y versión del motor",
)
def get_engine_status():
    """
    Confirma que el binario existe y responde correctamente a los comandos UCI.
    """
    return engine_service.check_status()

@router.post(
    "/move",
    response_model=EngineResponse,
    summary="Obtener la mejor jugada para una posición",
)
def get_move(request: EngineRequest):
    """
    Solicita al motor Stockfish que analice la posición y devuelva la mejor continuación.
    """
    try:
        best_move = engine_service.get_best_move(
            fen=request.fen,
            elo=request.elo,
            depth=request.depth
        )
        
        if not best_move:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo obtener la mejor jugada del motor."
            )
            
        return EngineResponse(ok=True, best_move=best_move)
        
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
