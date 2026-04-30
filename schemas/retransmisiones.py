# schemas/retransmisiones.py
# Esquemas para retransmisiones
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

# Esquema para la creación de una nueva retransmisión (POST)
class RetransmisionCreate(BaseModel):
    blancas:       Optional[str] = None
    negras:        Optional[str] = None
    resultado:     Optional[str] = None
    evento:        Optional[str] = None
    ronda:         Optional[int] = None
    tablero:       Optional[int] = None
    lugar:         Optional[str] = None

# Esquema para la respuesta al cliente (GET)
class RetransmisionResponse(RetransmisionCreate):
    id_retransmision:    int
    token:               str
    username:            str
    is_activa:           bool
    fecha_creacion:      datetime
    fecha_actualizacion: datetime

    # Permite a Pydantic leer directamente de objetos ORM de SQLAlchemy.
    model_config = ConfigDict(from_attributes=True)
