# routers/usuarios.py
# Endpoints de gestión de usuarios

"""Módulo de gestión de usuarios del sistema.

Proporciona los endpoints para el registro de nuevas cuentas
de usuario en la aplicación.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.usuarios import UsuarioCreate, UsuarioResponse
from services import usuarios

# Router de usuarios
router = APIRouter(prefix="/usuarios", tags=["Usuarios"])

# Endpoint de registro de nuevos usuarios
# POST /usuarios/register
@router.post(
    "/register",
    response_model=UsuarioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar nuevo usuario",
    description=(
        "Crea una cuenta nueva. La contraseña se almacena hasheada con **bcrypt**. "
        "Tras el registro, se envía un correo de bienvenida a la dirección indicada. "
        "El `username` y el `mail` deben ser únicos."
    ),
    responses={
        201: {"description": "Usuario creado correctamente."},
        409: {"description": "El username o email ya están en uso."},
        422: {"description": "Datos de entrada no válidos (validación Pydantic)."},
    },
)
# La función del endpoint llama al servicio de usuario para manejar la lógica de negocio.
async def register(body: UsuarioCreate, db: Session = Depends(get_db)):
    """Registra un nuevo usuario en el sistema.

    Crea una cuenta nueva con los datos proporcionados. La contraseña
    se almacena hasheada con bcrypt. El username y el email deben
    ser únicos en la base de datos.

    Args:
        body: Datos del nuevo usuario (username, email, password).
        db: Sesión de base de datos.

    Returns:
        UsuarioResponse con los datos del usuario creado.

    Raises:
        HTTPException 409: Si el username o el email ya existen.
        HTTPException 422: Si los datos de entrada no son válidos.
    """
    try:
        return await usuarios.register(body, db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
