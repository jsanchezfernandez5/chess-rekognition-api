# models/usuarios.py
# Modelo ORM que mapea la tabla 'usuarios' de la base de datos.
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from db.database import Base

# Clase Usuario que hereda de Base, representando la tabla 'usuarios' en la BD.
class Usuario(Base):
    __tablename__ = "usuarios"

    # PRIMARY KEY
    username = Column(String(50), primary_key=True, index=True)

    # Datos personales
    nombre    = Column(String(255), nullable=False)
    apellidos = Column(String(255), nullable=False)
    password  = Column(String(255), nullable=False) # Almacenamos el hash de la contraseña, no el texto plano
    mail      = Column(String(255), nullable=False)

    # Relaciones con otras tablas (Partida, Retransmision)
    # back_populates crea la relación bidireccional entre modelos
    # lazy="dynamic": las partidas no se cargan hasta que se accede explícitamente
    # partidas         = relationship("Partida",         back_populates="usuario", lazy="dynamic")
    # retransmisiones  = relationship("Retransmision",   back_populates="usuario", lazy="dynamic")

    # Método de representación para facilitar debugging y logging
    def __repr__(self) -> str:
        return f"<Usuario username={self.username!r} nombre={self.nombre!r}>"
