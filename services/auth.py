# services/auth.py
# Logica de negocio relacionada con autenticación (login, refresh token, whoami).
from sqlalchemy.orm import Session

from core.security import (verify_password, create_access_token, create_refresh_token, decode_token)
from models.usuarios import Usuario

# Función principal para autenticar al usuario y generar tokens JWT. Se llama desde el endpoint POST /auth/login.
def login(username: str, password: str, db: Session) -> dict:
    """
    Autentica al usuario con username + password.

    Returns:
        dict con access_token, refresh_token y token_type.

    Raises:
        ValueError: si las credenciales son incorrectas.
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

# Función para renovar el access token usando el refresh token. No requiere autenticación con access token.
def refresh(refresh_token: str, db: Session) -> dict:
    """
    Renueva el access token usando un refresh token válido.

    El refresh token tiene vida larga (7 días). 
    Cuando el access token caduca (30 min), el cliente llama a este endpoint sin molestar al usuario.

    Returns:
        dict con el nuevo access_token (el refresh_token no cambia).

    Raises:
        ValueError: si el refresh token es no válido o el usuario no existe.
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

# Función que devuelve los datos del usuario autenticado. Requiere un access token válido.
def whoami(usuario: Usuario) -> Usuario:
    """
    Devuelve los datos del usuario autenticado.
    """
    return usuario
