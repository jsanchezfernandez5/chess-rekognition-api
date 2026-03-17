# schemas/partidas.py
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

# Esquema para la creación de una nueva partida (POST)
class PartidaCreate(BaseModel):
    evento: str = Field(..., max_length=250, description="Nombre del torneo o evento.")
    blancas: str = Field(..., max_length=250, description="Nombre del jugador con blancas.")
    negras: str = Field(..., max_length=250, description="Nombre del jugador con negras.")
    fecha: date = Field(..., description="Fecha de la partida (YYYY-MM-DD).")
    resultado: str = Field(..., max_length=7, description="Resultado final (1-0, 0-1, 1/2-1/2).")
    pgn: str = Field(..., description="Contenido PGN completo de la partida.")
    tipo_partida: Optional[str] = Field(None, max_length=2, pattern="^(PI|PR)$", description="PI: Introducida, PR: Retransmisión.")
    ronda: Optional[int] = Field(None, ge=1)
    tablero: Optional[int] = Field(None, ge=1)
    lugar: Optional[str] = Field(None, max_length=250)
    observaciones: Optional[str] = None

# Esquema para actualización parcial (PATCH/PUT)
class PartidaUpdate(BaseModel):
    evento: Optional[str] = Field(None, max_length=250)
    blancas: Optional[str] = Field(None, max_length=250)
    negras: Optional[str] = Field(None, max_length=250)
    fecha: Optional[date] = None
    resultado: Optional[str] = Field(None, max_length=7)
    pgn: Optional[str] = None
    tipo_partida: Optional[str] = Field(None, max_length=2, pattern="^(PI|PR)$")
    ronda: Optional[int] = Field(None, ge=1)
    tablero: Optional[int] = Field(None, ge=1)
    lugar: Optional[str] = Field(None, max_length=250)
    observaciones: Optional[str] = None

# Esquema para la respuesta al cliente (GET)
class PartidaResponse(BaseModel):
    id_partida: int
    username: str
    evento: str
    blancas: str
    negras: str
    fecha: date
    resultado: str
    pgn: str
    tipo_partida: Optional[str]
    ronda: Optional[int]
    tablero: Optional[int]
    lugar: Optional[str]
    observaciones: Optional[str]
    fecha_registro: datetime

    # Permite a Pydantic leer directamente de objetos ORM de SQLAlchemy.
    model_config = ConfigDict(from_attributes=True)
