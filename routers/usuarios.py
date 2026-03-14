# routers/usuarios.py
# Endpoints de gestión de usuarios
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.usuarios import UsuarioCreate, UsuarioResponse
from services import usuarios

# Router específico para endpoints relacionados con usuarios, con prefijo "/usuarios" y etiqueta "Usuarios" para la documentación.
router = APIRouter(prefix="/usuarios", tags=["Usuarios"])

# Endpoint de registro de nuevos usuarios
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
    try:
        return await usuarios.register(body, db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
