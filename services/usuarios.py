# services/usuarios.py
# Lógica de negocio para gestión de usuarios:
#   - register: crea usuario, hashea password, envía email de bienvenida
from sqlalchemy.orm import Session

from core.security import hash_password
from models.usuarios import Usuario
from schemas.usuarios import UsuarioCreate
from services.email import send_welcome_email

# Función principal para registrar un nuevo usuario. Se llama desde el endpoint POST /usuarios/register.
async def register(data: UsuarioCreate, db: Session) -> Usuario:
    """
    Registra un nuevo usuario en la BD.

    Flujo:
      1. Verifica que username y email no estén en uso
      2. Hashea la contraseña con bcrypt
      3. Inserta el usuario en BD
      4. Envía email de bienvenida (async, no bloquea)

    Returns:
        El objeto Usuario recién creado.

    Raises:
        ValueError: si el username o email ya están registrados.
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
    db.commit()
    db.refresh(nuevo_usuario)  # Recarga el objeto con los datos de BD

    # Enviar email de bienvenida de forma asíncrona. Si falla, no bloquea el registro del usuario.
    try:
        await send_welcome_email(nombre=nuevo_usuario.nombre, mail=nuevo_usuario.mail)
    except Exception:
        pass

    return nuevo_usuario
