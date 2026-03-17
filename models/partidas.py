# models/partidas.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Date, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from db.database import Base

class Partida(Base):
    """
    Modelo ORM que representa una partida de ajedrez en la base de datos.
    Sigue el esquema definido para la tabla 'partidas'.
    """
    __tablename__ = "partidas"

    id_partida     = Column(Integer, primary_key=True, autoincrement=True)
    username       = Column(String(50), ForeignKey("usuarios.username"), nullable=False, index=True)
    evento         = Column(String(250), nullable=False)
    blancas        = Column(String(250), nullable=False)
    negras         = Column(String(250), nullable=False)
    fecha          = Column(Date, nullable=False) # Fecha del Evento
    resultado      = Column(String(7), nullable=False) # Ej: "1-0", "0-1", "1/2-1/2"
    pgn            = Column(Text, nullable=False) # PGN (Portable Game Notation)
    tipo_partida   = Column(String(2), default=None) # 'PI' (Partida Introducida) o 'PR' (Partida Retransmitida)
    ronda          = Column(Integer)
    tablero        = Column(Integer)
    lugar          = Column(String(250))
    observaciones  = Column(Text)
    fecha_registro = Column(DateTime, nullable=False, server_default=func.now())

    # Relación inversa con el Usuario propietario de la partida.
    usuario = relationship("Usuario", back_populates="partidas")

    def __repr__(self) -> str:
        return f"<Partida id={self.id_partida} evento={self.evento!r} blancas={self.blancas!r} negras={self.negras!r}>"
