# services/usuarios.py
# Lógica de negocio para gestión de usuarios
"""Módulo de gestión de usuarios.

Proporciona la función de registro de nuevos usuarios con validación de
unicidad de username y email, hash de contraseña con bcrypt, y envío de
correo de bienvenida mediante el servicio de email."""
from sqlalchemy.orm import Session
from core.security import hash_password
from models.usuarios import Usuario
from schemas.usuarios import UsuarioCreate
from services.email import send_welcome_email

# Función para registrar un nuevo usuario.
# Se llama desde el endpoint POST /usuarios/register.
async def register(data: UsuarioCreate, db: Session) -> Usuario:
    """Registra un nuevo usuario en la base de datos y envía el correo de bienvenida.

    El flujo de registro incluye:
    1. Verificar que el username y el email no estén ya registrados.
    2. Hashear la contraseña con bcrypt para almacenamiento seguro.
    3. Insertar el nuevo usuario en la base de datos.
    4. Enviar un correo de bienvenida de forma asíncrona.
    Si el envío del email falla, la transacción se revierte completamente.

    Args:
        data: Esquema UsuarioCreate con los datos del nuevo usuario (username,
              nombre, apellidos, password, mail).
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        Objeto Usuario recién creado con los datos persistidos.

    Raises:
        ValueError: Si el username o el email ya están registrados, o si ocurre
                    un error al enviar el correo de bienvenida.
    """
    # Verificar username duplicado
    if db.query(Usuario).filter(Usuario.username == data.username).first():
        raise ValueError(f"El username '{data.username}' ya está en uso")

    # Verificar email duplicado
    if db.query(Usuario).filter(Usuario.mail == data.mail).first():
        raise ValueError(f"El correo '{data.mail}' ya está registrado")

    # Objeto ORM Usuario. La contraseña se almacena hasheada, no en texto plano.
    nuevo_usuario = Usuario(
        username=data.username,
        nombre=data.nombre,
        apellidos=data.apellidos,
        password=hash_password(data.password),  # Hash bcrypt
        mail=data.mail,
    )

    db.add(nuevo_usuario)

    # Intentar enviar el email antes de hacer el commit final.
    # Si el email falla, se hace rollback y no se crea el usuario.
    try:
        await send_welcome_email(nombre=nuevo_usuario.nombre, mail=nuevo_usuario.mail)
        db.commit()
        db.refresh(nuevo_usuario)  # Recarga el objeto con los datos de BD
    except Exception as e:
        db.rollback()
        raise ValueError(f"No se pudo completar el registro por un error en el envío del email: {e}")

    return nuevo_usuario
