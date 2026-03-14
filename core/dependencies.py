# core/dependencies.py
# Define las dependencias de FastAPI que se inyectan en los endpoints y que son reutilizables.
# El patrón Depends() de FastAPI permite reutilizar lógica común (auth, DB) de forma limpia y testeable.
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from core.security import decode_token
from db.database import get_db
from models.usuarios import Usuario

# Esquema HTTPBearer extrae automáticamente el token del header "Authorization: Bearer <token>"
bearer_scheme = HTTPBearer(auto_error=False)

# Dependencia de autenticación que se puede usar en cualquier endpoint protegido
def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Usuario:
    """
    Dependencia principal de autenticación.

    Uso en un endpoint protegido:
        @router.get("/perfil")
        def perfil(current_user: Usuario = Depends(get_current_user)):
            ...

    Flujo:
      1. Extrae el Bearer token del header Authorization
      2. Decodifica y valida el JWT (tipo 'access')
      3. Busca el usuario en la BD
      4. Devuelve el objeto Usuario o lanza 401
    """
    # Sin credenciales → no autenticado
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

    # Verificar que el usuario sigue existiendo en la BD
    usuario = db.query(Usuario).filter(Usuario.username == username).first()
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return usuario
