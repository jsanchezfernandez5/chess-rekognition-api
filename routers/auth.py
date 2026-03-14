# routers/auth.py
# Endpoints relacionados con la autenticación de usuarios y gestión de tokens JWT.
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.dependencies import get_current_user
from db.database import get_db
from models.usuarios import Usuario
from schemas.usuarios import LoginRequest, RefreshRequest, TokenResponse, UsuarioResponse
from services import auth

# Router específico para endpoints relacionados con autenticación, con prefijo "/auth" y etiqueta "Autenticación" para la documentación.
router = APIRouter(prefix="/auth", tags=["Autenticación"])

# Endpoint de login que autentica al usuario y devuelve los tokens JWT.
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Iniciar sesión",
    description=(
        "Autentica al usuario con **username** y **password**. "
        "Devuelve un `access_token` (válido 30 min) y un `refresh_token` (válido 7 días). "
        "Incluye el `access_token` en el header `Authorization: Bearer <token>` "
        "para acceder a los endpoints protegidos."
    ),
    responses={
        200: {"description": "Login exitoso. Se devuelven los tokens JWT."},
        401: {"description": "Credenciales incorrectas."},
    },
)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    try:
        return auth.login(body.username, body.password, db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

# Endpoint para renovar el access token usando el refresh token. No requiere autenticación con access token.
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Renovar access token",
    description=(
        "Usa el `refresh_token` para obtener un nuevo `access_token` sin necesidad "
        "de que el usuario vuelva a introducir sus credenciales. "
        "El `refresh_token` no cambia."
    ),
    responses={
        200: {"description": "Nuevo access token generado."},
        401: {"description": "Refresh token no válido o expirado."},
    },
)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    try:
        return auth.refresh(body.refresh_token, db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

# Endpoint protegido que devuelve los datos del usuario autenticado. Requiere un access token válido.
@router.get(
    "/whoami",
    response_model=UsuarioResponse,
    summary="¿Quién soy?",
    description=("Devuelve los datos del usuario propietario del `access_token` enviado. "),
    responses={
        200: {"description": "Datos del usuario autenticado."},
        401: {"description": "Token no proporcionado, no válido o expirado."},
    },
)
def whoami(current_user: Usuario = Depends(get_current_user)):
    return auth.whoami(current_user)
