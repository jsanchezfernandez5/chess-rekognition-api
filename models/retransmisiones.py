from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.database import Base

class Retransmision(Base):
    __tablename__ = "retransmisiones"

    id_retransmision    = Column(Integer, primary_key=True, autoincrement=True, index=True)
    token               = Column(String(250), unique=True, index=True)
    username            = Column(String(100), ForeignKey("usuarios.username"), nullable=False)
    blancas             = Column(String(250))
    negras              = Column(String(250))
    resultado           = Column(String(50))
    evento              = Column(String(250))
    ronda               = Column(Integer)
    tablero             = Column(Integer)
    lugar               = Column(String(250))
    is_activa           = Column(Boolean, nullable=False, default=False)
    fecha_creacion      = Column(DateTime, server_default=func.now(), nullable=False)
    fecha_actualizacion = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    usuario = relationship("Usuario", back_populates="retransmisiones")
