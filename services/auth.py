# services/auth.py
# Lógica de negocio relacionada con autenticación (login, refresh token, whoami).
"""Módulo de autenticación de usuarios.

Proporciona funciones para el inicio de sesión con generación de tokens JWT,
renovación del access token mediante refresh token, y consulta del usuario
autenticado actual."""
from sqlalchemy.orm import Session
from core.security import (verify_password, create_access_token, create_refresh_token, decode_token)
from models.usuarios import Usuario

# Función principal para autenticar al usuario y generar tokens JWT.
# Se llama desde el endpoint POST /auth/login.
def login(username: str, password: str, db: Session) -> dict:
    """Autentica a un usuario mediante username y contraseña, y genera los tokens JWT.

    Busca el usuario en la base de datos y verifica la contraseña con bcrypt.
    Si las credenciales son correctas, genera un access token de corta duración
    (30 minutos) y un refresh token de larga duración (7 días).

    Args:
        username: Nombre de usuario.
        password: Contraseña en texto plano (se verifica contra el hash bcrypt).
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        Diccionario con access_token, refresh_token y token_type="bearer".

    Raises:
        ValueError: Si el usuario no existe o la contraseña es incorrecta.
    """
    # Buscamos el usuario en BD
    usuario = db.query(Usuario).filter(Usuario.username == username).first()
    if not usuario or not verify_password(password, usuario.password):
        raise ValueError("Credenciales incorrectas")

    return {
        "access_token":  create_access_token(usuario.username),
        "refresh_token": create_refresh_token(usuario.username),
        "token_type":    "bearer",
    }

# Función para renovar el access token usando el refresh token.
# No requiere autenticación con access token.
# Se llama desde el endpoint POST /auth/refresh.
def refresh(refresh_token: str, db: Session) -> dict:
    """Renueva el access token utilizando un refresh token válido sin requerir autenticación completa.

    El refresh token tiene una validez de 7 días. Cuando el access token caduca
    (tras 30 minutos), el cliente puede llamar a esta función para obtener un nuevo
    access token de forma transparente para el usuario. El refresh token original
    se reutiliza sin modificarse.

    Args:
        refresh_token: Token JWT de tipo "refresh" enviado por el cliente.
        db: Sesión de base de datos SQLAlchemy.

    Returns:
        Diccionario con el nuevo access_token, el mismo refresh_token y token_type="bearer".

    Raises:
        ValueError: Si el refresh token no es válido, ha expirado o el usuario
                    asociado ya no existe en la base de datos.
    """
    try:
        username = decode_token(refresh_token, expected_type="refresh")
    except ValueError:
        raise ValueError("Refresh token no válido o expirado")

    # Verificamos que el usuario sigue existiendo
    usuario = db.query(Usuario).filter(Usuario.username == username).first()
    if not usuario:
        raise ValueError("Usuario no encontrado")

    return {
        "access_token":  create_access_token(usuario.username),
        "refresh_token": refresh_token,   # Reutilizamos el mismo refresh token
        "token_type":    "bearer",
    }

# Función que devuelve los datos del usuario autenticado.
# Requiere un access token válido.
# Se llama desde el endpoint GET /auth/whoami.
def whoami(usuario: Usuario) -> Usuario:
    """Devuelve los datos del usuario autenticado actual.

    Es una función de paso que recibe el objeto Usuario extraído del token JWT
    por el sistema de dependencias y lo retorna directamente para ser serializado.

    Args:
        usuario: Objeto Usuario obtenido del token de acceso.

    Returns:
        El mismo objeto Usuario con todos sus datos.
    """
