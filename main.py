# main.py
# Punto de entrada de la aplicación FastAPI.
# Aquí se configura la app, se registran los routers y se definen los metadatos para Swagger UI.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, usuarios

# Metadatos para Swagger UI. Aparecen en la documentación generada automáticamente en /docs y /redoc.
tags_metadata = [
    {
        "name": "Autenticación",
        "description": "Endpoints relacionados con autenticación de usuarios y gestión de tokens JWT.",
    },
    {
        "name": "Usuarios",
        "description": "Registro de nuevas cuentas de usuario.",
    }
]

# Instancia de la API
app = FastAPI(
    title="Chess Rekognition API",
    description=("## API REST para gestión y retransmisiones de partidas de ajedrez"),
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "José Joaquín Sánchez Fernández ",
        "email": "jsanchezfernandez5@uoc.edu",
    },
    license_info={"name": "TFG - Uso académico"},
)

# CORSMiddleware permite que el frontend (que corre en otro origen) pueda consumir esta API sin problemas de CORS.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Registro de routers
app.include_router(auth.router)
app.include_router(usuarios.router)

# Endpoint raíz para health check. No requiere autenticación, útil para monitorización.
@app.get(
    "/",
    tags=["Sistema"],
    summary="Health check",
    description="Verifica que la API está en funcionamiento.",
)
def root():
    return {"status": "ok", "message": "API funcionando correctamente"}
