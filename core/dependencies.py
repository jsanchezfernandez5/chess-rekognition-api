"""Dependencias reutilizables de FastAPI.

Define las dependencias que se inyectan en los endpoints mediante
Depends(), como la autenticación y acceso a la base de datos.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from core.security import decode_token
from db.database import get_db
from models.usuarios import Usuario

bearer_scheme = HTTPBearer(auto_error=False)

# Dependencia para obtener el usuario autenticado a partir del token JWT en el header Authorization.
def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Usuario:
    """Dependencia de autenticación que retorna el usuario autenticado.

    Flujo:
      1. Extrae el Bearer token del header Authorization.
      2. Decodifica y valida el JWT (tipo 'access').
      3. Busca el usuario en la base de datos.
      4. Retorna el objeto Usuario o lanza HTTP 401.

    Uso en un endpoint protegido:
        @router.get("/perfil")
        def perfil(current_user: Usuario = Depends(get_current_user)):
            ...
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No se proporcionó token de autenticación",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        username = decode_token(credentials.credentials, expected_type="access")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    usuario = db.query(Usuario).filter(Usuario.username == username).first()
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return usuario
