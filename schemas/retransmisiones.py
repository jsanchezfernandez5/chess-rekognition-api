from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class RetransmisionCreate(BaseModel):
    blancas: Optional[str] = None
    negras: Optional[str] = None
    resultado: Optional[str] = None
    evento: Optional[str] = None
    ronda: Optional[int] = None
    tablero: Optional[int] = None
    lugar: Optional[str] = None

class RetransmisionResponse(RetransmisionCreate):
    id_retransmision: int
    token: str
    username: str
    is_activa: bool
    fecha_creacion: datetime
    fecha_actualizacion: datetime

    class Config:
        from_attributes = True
