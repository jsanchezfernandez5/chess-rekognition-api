"""Schemas Pydantic para la validación de datos de retransmisiones.

Define los modelos de entrada y respuesta para las
operaciones relacionadas con retransmisiones en directo.
"""
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class RetransmisionCreate(BaseModel):
    """Schema para la creación de una nueva retransmisión (POST)."""
    blancas:       Optional[str] = None
    negras:        Optional[str] = None
    resultado:     Optional[str] = None
    evento:        Optional[str] = None
    ronda:         Optional[int] = None
    tablero:       Optional[int] = None
    lugar:         Optional[str] = None


class RetransmisionResponse(RetransmisionCreate):
    """Schema de respuesta al cliente con los datos completos de la retransmisión (GET)."""
    id_retransmision:    int
    token:               str
    username:            str
    is_activa:           bool
    fecha_creacion:      datetime
    fecha_actualizacion: datetime

    model_config = ConfigDict(from_attributes=True)
