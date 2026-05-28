"""Modelo ORM para la tabla de partidas.

Define la estructura y relaciones del modelo Partida
que se mapea a la tabla 'partidas' en la base de datos.
"""
from sqlalchemy import Column, Integer, String, Text, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from db.database import Base


class Partida(Base):
    """Representa una partida de ajedrez almacenada en la plataforma.

    Contiene los metadatos de la partida (evento, jugadores, resultado),
    el contenido PGN completo y la relación con el usuario propietario.
    """
    __tablename__ = "partidas"

    id_partida     = Column(Integer, primary_key=True, autoincrement=True)
    username       = Column(String(50), ForeignKey("usuarios.username"), nullable=False, index=True)
    evento         = Column(String(250), nullable=False)
    blancas        = Column(String(250), nullable=False)
    negras         = Column(String(250), nullable=False)
    fecha          = Column(Date, nullable=False)
    resultado      = Column(String(7), nullable=False)
    pgn            = Column(Text, nullable=False)
    tipo_partida   = Column(String(2), default=None)
    ronda          = Column(Integer)
    tablero        = Column(Integer)
    lugar          = Column(String(250))
    observaciones  = Column(Text)
    fecha_registro = Column(DateTime, nullable=False, server_default=func.now())

    usuario = relationship("Usuario", back_populates="partidas")

    def __repr__(self) -> str:
        """Representación legible de la partida para depuración y logging."""
        return (
            f"<Partida id={self.id_partida}"
            f" evento={self.evento!r}"
            f" blancas={self.blancas!r}"
            f" negras={self.negras!r}>"
        )
