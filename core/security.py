"""Seguridad de la aplicación.

Proporciona funciones para:
  - Hashing y verificación de contraseñas con bcrypt (passlib).
  - Creación y verificación de tokens JWT (access + refresh).
"""
from datetime import datetime, timedelta, timezone
from typing import Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Funciones para hashing de contraseñas y generación/validación de tokens JWT. 
# Se utilizan en los endpoints de autenticación y en la dependencia de seguridad para proteger rutas.
def hash_password(plain_password: str) -> str:
    """Genera el hash bcrypt de una contraseña en texto plano."""
    return pwd_context.hash(plain_password)

# Función para verificar que una contraseña en texto plano coincide con su hash almacenado.
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica que una contraseña en texto plano coincida con su hash almacenado."""
    return pwd_context.verify(plain_password, hashed_password)

# Función interna para crear un token JWT con el payload y la expiración indicados.
def _create_token(
    subject: str,
    token_type: Literal["access", "refresh"],
    expires_delta: timedelta,
) -> str:
    """Crea un token JWT con el payload y la expiración indicados.

    El payload incluye:
      - sub: identificador del usuario (username).
      - type: diferencia access de refresh para seguridad.
      - exp: timestamp de expiración.
      - iat: timestamp de emisión.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

# Funciones para crear tokens de acceso y refresh, y para decodificar y validar tokens JWT.
def create_access_token(username: str) -> str:
    """Crea un access token JWT de vida corta (por defecto 30 minutos).

    Se usa para autenticar cada request a endpoints protegidos.
    """
    return _create_token(
        subject=username,
        token_type="access",
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )

# Función para crear un refresh token JWT de vida larga (por defecto 7 días).
def create_refresh_token(username: str) -> str:
    """Crea un refresh token JWT de vida larga (por defecto 7 días).

    Se usa exclusivamente para obtener un nuevo access token cuando caduca.
    """
    return _create_token(
        subject=username,
        token_type="refresh",
        expires_delta=timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )

# Función para decodificar y validar un token JWT, retornando el username (sub) si es válido.
def decode_token(token: str, expected_type: Literal["access", "refresh"]) -> str:
    """Decodifica y valida un token JWT, retornando el username (sub) si es válido.

    Lanza ValueError en caso de:
      - Token expirado.
      - Firma no válida.
      - Tipo de token incorrecto.
      - Payload malformado o sin identificador de usuario.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        raise ValueError("Token no válido o expirado")

    if payload.get("type") != expected_type:
        raise ValueError(f"Se esperaba un token de tipo '{expected_type}'")

    username: str | None = payload.get("sub")
    if not username:
        raise ValueError("Token sin identificador de usuario")

    return username
