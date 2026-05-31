"""
Router de registro de nuevos usuarios.

Endpoints:
    POST /usuarios/register | Registra un nuevo usuario y envía un email de bienvenida via Resend.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.models import Usuario, UsuarioCreate, UsuarioResponse
from core.security import hash_password
from services.email import send_welcome_email

# Creación del router.
router = APIRouter(prefix="/usuarios", tags=["Usuarios"])

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /usuarios/register
# Registra un nuevo usuario y envía un email de bienvenida via Resend.
# -------------------------------------------------------------------------------
@router.post(
    "/register",
    response_model=UsuarioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registra un nuevo usuario y envía un email de bienvenida via Resend.",
    responses={
        201: {"description": "Usuario creado correctamente."},
        409: {"description": "El username o email ya están en uso."},
    },
)
async def register(body: UsuarioCreate, db: Session = Depends(get_db)):
    """
    Registra un nuevo usuario y envía un email de bienvenida via Resend.

    Valida unicidad del username y email, hashea la contraseña y envía email de bienvenida.
    """
    # Valida la unicidad del username.
    if db.query(Usuario).filter(Usuario.username == body.username).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"El username '{body.username}' ya está en uso")

    # Valida la unicidad del correo.
    if db.query(Usuario).filter(Usuario.mail == body.mail).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"El correo '{body.mail}' ya está registrado")

    # Crea el nuevo usuario.
    nuevo = Usuario(
        username=body.username,
        nombre=body.nombre,
        apellidos=body.apellidos,
        password=hash_password(body.password),
        mail=body.mail,
    )
    db.add(nuevo)

    # Intenta enviar el email de bienvenida a través de Resend. 
    # En caso de error, hacemos rollback para que no se guarde el usuario y lanzamos HTTPException.
    try:
        await send_welcome_email(nombre=nuevo.nombre, mail=nuevo.mail)
        db.commit()
        db.refresh(nuevo)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error al enviar el email de bienvenida: {e}"
        )

    # Retorna el nuevo usuario.
    return nuevo
