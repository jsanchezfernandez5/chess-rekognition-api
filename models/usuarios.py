"""Modelo ORM para la tabla de usuarios.

Define la estructura y relaciones del modelo Usuario
que se mapea a la tabla 'usuarios' en la base de datos.
"""
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from db.database import Base


class Usuario(Base):
    """Representa un usuario registrado en la plataforma.

    Mapea la tabla 'usuarios' y contiene las credenciales,
    datos personales y relaciones con partidas y retransmisiones.
    """
    __tablename__ = "usuarios"

    username = Column(String(50), primary_key=True, index=True)

    nombre    = Column(String(255), nullable=False)
    apellidos = Column(String(255), nullable=False)
    password  = Column(String(255), nullable=False)
    mail      = Column(String(255), nullable=False)

    partidas         = relationship("Partida",         back_populates="usuario", lazy="dynamic")
    retransmisiones  = relationship("Retransmision",   back_populates="usuario", lazy="dynamic")

    def __repr__(self) -> str:
        """Representación legible del usuario para depuración y logging."""
        return f"<Usuario username={self.username!r} nombre={self.nombre!r}>"
