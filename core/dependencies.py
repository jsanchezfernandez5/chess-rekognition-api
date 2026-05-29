"""
Dependencias reutilizables de FastAPI.

Define las dependencias que se inyectan en los endpoints mediante
Depends(), como la autenticación y acceso a la base de datos.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

# Importamos el generador de sesiones, el modelo Usuario y la función decode_token.
from core.database import get_db
from core.models import Usuario
from core.security import decode_token

# Esquema HTTP Bearer para la autenticación.
bearer_scheme = HTTPBearer(auto_error=False)

# Función para obtener el usuario actual.
def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Usuario:
    """
    Extrae y valida el Bearer token JWT, retornando el usuario autenticado.

    Flujo:
      1. Extrae el Bearer token del header Authorization.
      2. Decodifica y valida el JWT (tipo 'access').
      3. Busca el usuario en la base de datos.
      4. Retorna el objeto Usuario o lanza HTTP 401.
    """
    # Comprobamos si se ha proporcionado un token.
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No se proporcionó token de autenticación",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Intentamos decodificar el token y obtener el usuario.
    try:
        username = decode_token(credentials.credentials, expected_type="access")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Consultamos el usuario en la base de datos.
    usuario = db.query(Usuario).filter(Usuario.username == username).first()
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Retornamos el usuario autenticado.
    return usuario
