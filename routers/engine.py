# routers/engine.py
# Endpoints para interactuar con Stockfish
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import Optional
from services.engine import engine_service

router = APIRouter(prefix="/engine", tags=["Motor"])

class EngineRequest(BaseModel):
    fen: str = Field(..., description="Posición en formato FEN (Forsyth-Edwards Notation)")
    elo: Optional[int] = Field(None, ge=1320, le=3190, description="Nivel de ELO (1320 - 3190) para ajustar la fuerza del motor")
    depth: int = Field(15, ge=1, le=30, description="Profundidad de búsqueda (1 - 30)")

class EngineResponse(BaseModel):
    ok: bool
    best_move: str
    message: Optional[str] = None

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
