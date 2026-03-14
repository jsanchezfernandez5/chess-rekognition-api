# core/security.py
# Gestiona toda la seguridad de la aplicación:
#   - Hashing de contraseñas con bcrypt (passlib)
#   - Creación y verificación de tokens JWT (access + refresh)
#   - Diferenciamos access token (vida corta) y refresh token (vida larga)
from datetime import datetime, timedelta, timezone
from typing import Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from core.config import settings

# Hashing de contraseñas (algoritmo bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Función de hashing
def hash_password(plain_password: str) -> str:
    """Devuelve el hash bcrypt de la contraseña en texto plano."""
    return pwd_context.hash(plain_password)

# Función de verificación de contraseña
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compara contraseña en texto plano con su hash almacenado."""
    return pwd_context.verify(plain_password, hashed_password)


# Función interna genérica para crear tokens JWT
def _create_token(
    subject: str,
    token_type: Literal["access", "refresh"],
    expires_delta: timedelta,
) -> str:
    """
    Función interna genérica para crear tokens JWT.

    El payload incluye:
      - sub: identificador del usuario (username)
      - type: diferencia access de refresh (importante para seguridad)
      - exp: timestamp de expiración
      - iat: timestamp de emisión
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

# Función que crea un access token de vida corta (por defecto 30 minutos).
def create_access_token(username: str) -> str:
    """
    Función que crea un access token de vida corta.
    Se usa para autenticar cada request a endpoints protegidos.
    """
    return _create_token(
        subject=username,
        token_type="access",
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )


# Función que crea un refresh token de vida larga (por defecto 7 días).
def create_refresh_token(username: str) -> str:
    """
    Función que crea un refresh token de vida larga.
    Solo se usa para obtener un nuevo access token cuando caduca.
    """
    return _create_token(
        subject=username,
        token_type="refresh",
        expires_delta=timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )

# Función para decodificar y validar un token JWT (tanto access como refresh).
def decode_token(token: str, expected_type: Literal["access", "refresh"]) -> str:
    """
    Decodifica y valida un JWT. Devuelve el username (sub) si es válido.

    Lanza ValueError en caso de:
      - Token expirado
      - Firma inválida
      - Tipo de token incorrecto (ej: usar refresh token como access token)
      - Payload malformado
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        raise ValueError("Token no válido o expirado")

    # Verificamos que el tipo de token sea el esperado
    if payload.get("type") != expected_type:
        raise ValueError(f"Se esperaba un token de tipo '{expected_type}'")

    username: str | None = payload.get("sub")
    if not username:
        raise ValueError("Token sin identificador de usuario")

    return username
