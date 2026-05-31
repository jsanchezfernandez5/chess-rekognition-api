"""
Router de autenticación de los endpoints de autenticación.

Endpoints:
    POST /auth/login      | Autentica con username + password y devuelve access token (30 min) y refresh token (7 días).
    POST /auth/refresh    | Emite un nuevo access token a partir de un refresh token válido.
    GET  /auth/whoami     | Devuelve los datos del usuario identificado por el access token.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.dependencies import get_current_user
from core.models import Usuario, LoginRequest, RefreshRequest, TokenResponse, UsuarioResponse
from core.security import verify_password, create_access_token, create_refresh_token, decode_token

# Router para los endpoints de Autenticación.
router = APIRouter(prefix="/auth", tags=["Autenticación"])

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /auth/login
# Autentica con username + password y devuelve access token (30 min) y refresh token (7 días).
# -------------------------------------------------------------------------------
@router.post(
    "/login", 
    response_model=TokenResponse, 
    summary="Iniciar sesión en el sistema"
)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """
    Autentica con username + password y devuelve access token (30 min) y refresh token (7 días).
    """
    usuario = db.query(Usuario).filter(Usuario.username == body.username).first()
    if not usuario or not verify_password(body.password, usuario.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token":  create_access_token(usuario.username),
        "refresh_token": create_refresh_token(usuario.username),
        "token_type":    "bearer",
    }

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /auth/refresh
# Emite un nuevo access token a partir de un refresh token válido.
# -------------------------------------------------------------------------------
@router.post(
    "/refresh", 
    response_model=TokenResponse, 
    summary="Renovar el token de acceso"
)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    """
    Emite un nuevo access token a partir de un refresh token válido.
    """
    try:
        username = decode_token(body.refresh_token, expected_type="refresh")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Comprueba que el usuario exista en la base de datos.
    usuario = db.query(Usuario).filter(Usuario.username == username).first()
    if not usuario:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado")

    return {
        "access_token":  create_access_token(usuario.username),
        "refresh_token": body.refresh_token,
        "token_type":    "bearer",
    }

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /auth/whoami
# Devuelve los datos del usuario identificado por el access token.
# -------------------------------------------------------------------------------
@router.get(
    "/whoami", 
    response_model=UsuarioResponse, 
    summary="Datos del usuario autenticado"
)
def whoami(usuario_actual: Usuario = Depends(get_current_user)):
    """
    Devuelve los datos del usuario identificado por el access token.
    """
    return usuario_actual
