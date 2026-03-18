# routers/auth.py - Gestión de acceso y seguridad
# Aquí centralizamos el login, la renovación de tokens y la identificación del usuario.

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.dependencies import get_current_user
from db.database import get_db
from models.usuarios import Usuario
from schemas.usuarios import LoginRequest, RefreshRequest, TokenResponse, UsuarioResponse
from services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["Autenticación"])

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Iniciar sesión en el sistema",
)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    Recibe las credenciales (usuario/contraseña) y genera los tokens JWT 
    si todo es correcto. Devuelve un access_token y un refresh_token.
    """
    try:
        return auth_service.login(login_data.username, login_data.password, db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Renovar el token de acceso",
)
def refresh(token_in: RefreshRequest, db: Session = Depends(get_db)):
    """
    Permite obtener un nuevo access_token usando el refresh_token, 
    evitando que el usuario tenga que loguearse de nuevo constantemente.
    """
    try:
        return auth_service.refresh(token_in.refresh_token, db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get(
    "/whoami",
    response_model=UsuarioResponse,
    summary="Datos del usuario actual",
)
def whoami(usuario_actual: Usuario = Depends(get_current_user)):
    """
    Endpoint de utilidad para que el frontend sepa quién está logueado 
    y pueda mostrar su nombre en el perfil o dashboard.
    """
    return auth_service.whoami(usuario_actual)
