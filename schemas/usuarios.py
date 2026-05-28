"""Schemas Pydantic para la validación de datos de usuarios.

Define los modelos de entrada y salida para el registro,
login y consulta de perfil de usuario.
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UsuarioCreate(BaseModel):
    """Schema para el registro de un nuevo usuario (POST /register)."""
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Nombre de usuario único. Entre 3 y 50 caracteres.",
        examples=["chess_test01"],
    )
    nombre: str = Field(
        ...,
        max_length=255,
        description="Nombre del usuario.",
        examples=["José Joaquín"],
    )
    apellidos: str = Field(
        ...,
        max_length=255,
        description="Apellidos del usuario.",
        examples=["Sánchez Fernández"],
    )
    password: str = Field(
        ...,
        min_length=8,
        description="Contraseña en texto plano. Se almacenará hasheada con bcrypt.",
        examples=["chess_test01"],
    )
    mail: EmailStr = Field(
        ...,
        description="Correo electrónico válido. Recibirá un email de bienvenida.",
        examples=["jsanchezfernandez5@uoc.edu"],
    )


class UsuarioResponse(BaseModel):
    """Schema de respuesta pública del usuario.

    Nunca incluye el password. Se usa en /whoami y en la respuesta del registro.
    """
    username:  str
    nombre:    str
    apellidos: str
    mail:      str

    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    """Credenciales para el inicio de sesión (POST /auth/login)."""
    username: str = Field(
        ...,
        description="Nombre de usuario registrado.",
        examples=["chess_test01"],
    )
    password: str = Field(
        ...,
        description="Contraseña en texto plano.",
        examples=["chess_test01"],
    )


class TokenResponse(BaseModel):
    """Respuesta del login y del refresh.

    Contiene ambos tokens JWT (access y refresh).
    """
    access_token:  str = Field(..., description="Token de acceso. Expira en 30 minutos.")
    refresh_token: str = Field(..., description="Token de refresco. Expira en 7 días.")
    token_type:    str = Field(default="bearer", description="Tipo de token. Siempre 'bearer'.")


class RefreshRequest(BaseModel):
    """Body para la renovación del token (POST /auth/refresh).

    Requiere el refresh token para obtener un nuevo access token.
    """
    refresh_token: str = Field(
        ...,
        description="Refresh token obtenido en el login.",
    )
