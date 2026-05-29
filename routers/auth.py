"""Módulo de autenticación del sistema.

Proporciona los endpoints necesarios para el inicio de sesión,
la renovación de tokens JWT y la obtención de la información
del usuario autenticado.
"""

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
    """Autentica al usuario con sus credenciales y genera tokens JWT.

    Verifica el nombre de usuario y la contraseña contra la base de datos.
    Si las credenciales son válidas, devuelve un access_token (30 min)
    y un refresh_token (7 días) para mantener la sesión.

    Args:
        login_data: Credenciales de acceso (username y password).
        db: Sesión de base de datos.

    Returns:
        TokenResponse con access_token, refresh_token y datos del usuario.

    Raises:
        HTTPException 401: Si las credenciales son incorrectas.
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
    """Renueva el token de acceso mediante un refresh_token válido.

    Permite al usuario obtener un nuevo access_token sin necesidad
    de volver a introducir sus credenciales, facilitando sesiones
    prolongadas de forma segura.

    Args:
        token_in: Objeto que contiene el refresh_token.
        db: Sesión de base de datos.

    Returns:
        TokenResponse con el nuevo access_token y refresh_token.

    Raises:
        HTTPException 401: Si el refresh_token no es válido o ha expirado.
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
    """Obtiene los datos del usuario autenticado.

    Endpoint de utilidad para que el frontend conozca la identidad
    del usuario actualmente logueado, permitiendo mostrar su perfil
    o personalizar la interfaz.

    Args:
        usuario_actual: Usuario autenticado extraído del token JWT.

    Returns:
        UsuarioResponse con los datos del usuario (username, email, etc.).
    """
    return auth_service.whoami(usuario_actual)
