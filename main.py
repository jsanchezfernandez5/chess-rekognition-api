# main.py - Punto de entrada del servidor FastAPI
# Este archivo centraliza la configuración de la API, rutas y documentación técnica.
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, usuarios, partidas, engine, vision, retransmision, dataset

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
        "description": "Operaciones CRUD de partidas PGN.",
    },
    {
        "name": "Motor",
        "description": "Integración con Stockfish para jugar contra el ordenador.",
    },
    {
        "name": "Visión",
        "description": "Reconocimiento del tablero mediante OpenCV.",
    },
    {
        "name": "Retransmisión",
        "description": "Servicio para retransmitir partidas en directo.",
    },
    {
        "name": "Dataset",
        "description": "Herramientas para captura de imágenes y entrenamiento del modelo ML.",
    }
]

# Inicialización de la app con metadatos
app = FastAPI(
    title="Chess Rekognition API",
    description="### Reconocimiento visual de jugadas en partidas de ajedrez presencial.",
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

# Carpeta de archivos estáticos (favicon, imágenes, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Registro de rutas (router)
app.include_router(auth.router)
app.include_router(usuarios.router)
app.include_router(partidas.router)
app.include_router(engine.router)
app.include_router(vision.router)
app.include_router(retransmision.router)
app.include_router(dataset.router)

# Ruta del favicon del navegador
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")

# Ruta para pruebas de visión
@app.get("/opencv", include_in_schema=False)
def opencv():
    return FileResponse("static/opencv.html")

# Ruta para la herramienta de dataset
@app.get("/dataset", include_in_schema=False)
def dataset_tool():
    return FileResponse("static/dataset.html")

# Ruta DOC personalizada para la interfaz de Swagger
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
