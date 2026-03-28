# main.py - Punto de entrada del servidor FastAPI
# Este archivo centraliza la configuración de la API, rutas y documentación técnica.

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, usuarios, partidas, engine

# Configuración de etiquetas para que el Swagger se vea organizado por secciones
tags_metadata = [
    {
        "name": "Autenticación",
        "description": "Gestión de sesiones, login y tokens JWT para la seguridad de la app.",
    },
    {
        "name": "Usuarios",
        "description": "Registro de nuevos perfiles en la plataforma.",
    },
    {
        "name": "Partidas",
        "description": "Operaciones CRUD para gestionar el historial de partidas guardadas.",
    },
    {
        "name": "Motor",
        "description": "Integración con Stockfish para análisis y juego contra el ordenador.",
    }
]

# Inicialización de la app con metadatos personalizados para el TFG
app = FastAPI(
    title="Chess Rekognition API",
    description="### Sistema para el registro y retransmisión inteligente de partidas de ajedrez.",
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "José Joaquín Sánchez Fernández",
        "email": "jsanchezfernandez5@uoc.edu",
    },
    license_info={
        "name": "CC BY-SA 4.0", 
        "url": "https://creativecommons.org/licenses/by-sa/4.0/"
    },
    docs_url=None, # Desactivamos la URL por defecto para personalizarla abajo
)

# Servimos archivos estáticos (como el favicon o imágenes)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuración de CORS: Vital para que el frontend en React pueda hablar con este backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se podría restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Registro de las rutas del sistema
app.include_router(auth.router)
app.include_router(usuarios.router)
app.include_router(partidas.router)
app.include_router(engine.router)

# Ruta específica para el favicon del navegador
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")

# Personalización de la interfaz de Swagger para que use nuestro logo y título
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Documentación - Chess Rekognition API",
        swagger_favicon_url="/static/favicon.ico"
    )

# Endpoint de comprobación rápida para ver si el servidor responde
@app.get("/", tags=["Sistema"], summary="Estado de la API")
def root():
    return {
        "status": "online", 
        "message": "Servidor Chess Rekognition operando correctamente."
    }
