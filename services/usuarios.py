# services/usuarios.py
# Lógica de negocio para gestión de usuarios
from sqlalchemy.orm import Session
from core.security import hash_password
from models.usuarios import Usuario
from schemas.usuarios import UsuarioCreate
from services.email import send_welcome_email

# Función para registrar un nuevo usuario.
# Se llama desde el endpoint POST /usuarios/register.
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
