"""
Modelos ORM y schemas de validación del sistema.

Unifica en un único módulo los modelos SQLAlchemy (ORM) y los schemas Pydantic (validación/serialización).
"""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, ConfigDict

from sqlalchemy import Boolean, Column, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database import Base

# ---------------------------------------------------------------------------
# ORM — tablas de la base de datos
# ---------------------------------------------------------------------------

# Clase tipo Tabla con los datos de los usuarios.
class Usuario(Base):
    """
    Tabla usuarios: Credenciales y datos personales del jugador.
    Relaciones con la tabla partidas (1:N) y retransmisiones (1:N).
    """
    __tablename__ = "usuarios"

    username  = Column(String(50),  primary_key=True, index=True)
    nombre    = Column(String(255), nullable=False)
    apellidos = Column(String(255), nullable=False)
    password  = Column(String(255), nullable=False)
    mail      = Column(String(255), nullable=False)

    # Relaciones con partidas y retransmisiones.
    partidas        = relationship("Partida",        back_populates="usuario", lazy="dynamic")
    retransmisiones = relationship("Retransmision",  back_populates="usuario", lazy="dynamic")

# Clase tipo Tabla con los datos de las partidas.
class Partida(Base):
    """
    Tabla partidas: Metadatos y PGN de una partida de ajedrez.
    Relación con la tabla usuarios (N:1).
    """
    __tablename__ = "partidas"

    id_partida     = Column(Integer,     primary_key=True, autoincrement=True)
    username       = Column(String(50),  ForeignKey("usuarios.username"), nullable=False, index=True)
    evento         = Column(String(250), nullable=False)
    blancas        = Column(String(250), nullable=False)
    negras         = Column(String(250), nullable=False)
    fecha          = Column(Date,        nullable=False)
    resultado      = Column(String(7),   nullable=False)
    pgn            = Column(Text,        nullable=False)
    tipo_partida   = Column(String(2),   default=None)
    ronda          = Column(Integer)
    tablero        = Column(Integer)
    lugar          = Column(String(250))
    observaciones  = Column(Text)
    fecha_registro = Column(DateTime,    nullable=False, server_default=func.now())

    # Relación con la tabla 'usuarios'.
    usuario = relationship("Usuario", back_populates="partidas")

# Clase tipo Tabla con los datos de las retransmisiones.
class Retransmision(Base):
    """
    Tabla retransmisiones: Retransmisión en directo de una partida.
    Relación con la tabla usuarios (N:1).
    """
    __tablename__ = "retransmisiones"

    id_retransmision    = Column(Integer,     primary_key=True, autoincrement=True, index=True)
    token               = Column(String(250), unique=True, index=True)
    username            = Column(String(100), ForeignKey("usuarios.username"), nullable=False)
    blancas             = Column(String(250))
    negras              = Column(String(250))
    resultado           = Column(String(50))
    evento              = Column(String(250))
    ronda               = Column(Integer)
    tablero             = Column(Integer)
    lugar               = Column(String(250))
    is_activa           = Column(Boolean,     nullable=False, default=False)
    fecha_creacion      = Column(DateTime,    server_default=func.now(), nullable=False)
    fecha_actualizacion = Column(DateTime,    server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relación con la tabla 'usuarios'.
    usuario = relationship("Usuario", back_populates="retransmisiones")


# ---------------------------------------------------------------------------
# Pydantic schemas - Validación de entrada y serialización de salida
# ---------------------------------------------------------------------------

# Clase tipo Pydantic para la creación de usuarios.
class UsuarioCreate(BaseModel):
    """Body para POST /usuarios/register."""
    username:  str      = Field(..., min_length=3, max_length=50,  examples=["chess_test01"])
    nombre:    str      = Field(..., max_length=255,               examples=["José Joaquín"])
    apellidos: str      = Field(..., max_length=255,               examples=["Sánchez Fernández"])
    password:  str      = Field(..., min_length=8,                 examples=["chess_test01"])
    mail:      EmailStr = Field(...,                               examples=["web@ajedrezcoimbra.com"])

# Clase tipo Pydantic para la respuesta de los usuarios.
class UsuarioResponse(BaseModel):
    """Respuesta pública del usuario (sin password)."""
    username:  str
    nombre:    str
    apellidos: str
    mail:      str

    # Configuración del modelo.
    model_config = ConfigDict(from_attributes=True)

# Clase tipo Pydantic para la petición de login.
class LoginRequest(BaseModel):
    """Body para POST /auth/login."""
    username: str = Field(..., examples=["chess_test01"])
    password: str = Field(..., examples=["chess_test01"])

# Clase tipo Pydantic para la respuesta de login y refresh.
class TokenResponse(BaseModel):
    """Respuesta de login y refresh con los dos tokens JWT."""
    access_token:  str = Field(..., description="Expira en 30 minutos.")
    refresh_token: str = Field(..., description="Expira en 7 días.")
    token_type:    str = Field(default="bearer")

# Clase tipo Pydantic para la petición de refresh.
class RefreshRequest(BaseModel):
    """Body para POST /auth/refresh."""
    refresh_token: str

# Clase tipo Pydantic para la creación de partidas.
class PartidaCreate(BaseModel):
    """Body para POST /partidas/."""
    evento:        str           = Field(..., max_length=250)
    blancas:       str           = Field(..., max_length=250)
    negras:        str           = Field(..., max_length=250)
    fecha:         date
    resultado:     str           = Field(..., max_length=7)
    pgn:           str
    tipo_partida:  Optional[str] = Field(None, max_length=2, pattern="^(PI|PR)$")
    ronda:         Optional[int] = Field(None, ge=1)
    tablero:       Optional[int] = Field(None, ge=1)
    lugar:         Optional[str] = Field(None, max_length=250)
    observaciones: Optional[str] = None

# Clase tipo Pydantic para la actualización de partidas.
class PartidaUpdate(BaseModel):
    """Body para PATCH /partidas/{id} — todos los campos opcionales."""
    evento:        Optional[str]  = Field(None, max_length=250)
    blancas:       Optional[str]  = Field(None, max_length=250)
    negras:        Optional[str]  = Field(None, max_length=250)
    fecha:         Optional[date] = None
    resultado:     Optional[str]  = Field(None, max_length=7)
    pgn:           Optional[str]  = None
    tipo_partida:  Optional[str]  = Field(None, max_length=2, pattern="^(PI|PR)$")
    ronda:         Optional[int]  = Field(None, ge=1)
    tablero:       Optional[int]  = Field(None, ge=1)
    lugar:         Optional[str]  = Field(None, max_length=250)
    observaciones: Optional[str]  = None

# Clase tipo Pydantic para la respuesta de las partidas.
class PartidaResponse(BaseModel):
    """Respuesta completa de una partida (GET)."""
    id_partida:     int
    username:       str
    evento:         str
    blancas:        str
    negras:         str
    fecha:          date
    resultado:      str
    pgn:            str
    tipo_partida:   Optional[str]
    ronda:          Optional[int]
    tablero:        Optional[int]
    lugar:          Optional[str]
    observaciones:  Optional[str]
    fecha_registro: datetime

    # Configuración del modelo.
    model_config = ConfigDict(from_attributes=True)

# Clase tipo Pydantic para la creación de retransmisiones.
class RetransmisionCreate(BaseModel):
    """Body para POST /retransmision/host y PATCH /retransmision/{id}."""
    blancas:   Optional[str] = None
    negras:    Optional[str] = None
    resultado: Optional[str] = None
    evento:    Optional[str] = None
    ronda:     Optional[int] = None
    tablero:   Optional[int] = None
    lugar:     Optional[str] = None

# Clase tipo Pydantic para la respuesta de las retransmisiones.
class RetransmisionResponse(RetransmisionCreate):
    """Respuesta completa de una retransmisión (GET)."""
    id_retransmision:    int
    token:               str
    username:            str
    is_activa:           bool
    fecha_creacion:      datetime
    fecha_actualizacion: datetime

    # Configuración del modelo.
    model_config = ConfigDict(from_attributes=True)
