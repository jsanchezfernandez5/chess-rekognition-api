# schemas/usuarios.py
# Schemas Pydantic para validación de datos de entrada/salida relacionados con usuarios.
# Los field descriptions aparecen automáticamente en Swagger /docs.
from pydantic import BaseModel, EmailStr, Field, ConfigDict

# Esquema para el registro de un nuevo usuario (POST /register).
class UsuarioCreate(BaseModel):
    """
    Schema para el endpoint POST /register. Valida los datos del nuevo usuario.
    """
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

# Esquema de respuesta pública del usuario (GET /whoami y respuesta POST /register).
class UsuarioResponse(BaseModel):
    """
    Schema de respuesta. Nunca incluye el password.
    Se usa en /whoami y en la respuesta del registro.
    """
    username:  str
    nombre:    str
    apellidos: str
    mail:      str
    # Permite crear desde un objeto ORM (como el modelo Usuario de SQLAlchemy) sin necesidad de convertirlo a dict primero.
    model_config = ConfigDict(from_attributes=True)

# Esquema para el login (POST /auth/login).
class LoginRequest(BaseModel):
    """
    Credenciales para el endpoint POST /auth/login.
    """
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

# Esquema de respuesta del login y del refresh (POST /auth/login y POST /auth/refresh).
class TokenResponse(BaseModel):
    """
    Respuesta del login y del refresh.
    Contiene ambos tokens JWT.
    """
    access_token:  str = Field(..., description="Token de acceso. Expira en 30 minutos.")
    refresh_token: str = Field(..., description="Token de refresco. Expira en 7 días.")
    token_type:    str = Field(default="bearer", description="Tipo de token. Siempre 'bearer'.")

# Esquema para el refresh token (POST /auth/refresh).
class RefreshRequest(BaseModel):
    """
    Body para el endpoint POST /auth/refresh.
    Requiere solo el refresh token para obtener un nuevo access token.
    """
    refresh_token: str = Field(
        ...,
        description="Refresh token obtenido en el login.",
    )
