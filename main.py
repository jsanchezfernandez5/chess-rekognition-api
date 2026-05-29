"""
Punto de entrada del servidor FastAPI.

Centraliza la configuración de la API, el registro de rutas, la documentación técnica y los endpoints auxiliares.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, usuarios, partidas, engine, vision, retransmision, dataset

# Metadatos para organizar la documentación de la API en categorías.
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
    docs_url=None,
)

# Monta el directorio de archivos estáticos para servir recursos como el favicon y las páginas HTML auxiliares.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configura CORS para permitir solicitudes desde cualquier origen, lo que es útil durante el desarrollo y para aplicaciones frontend que consumen esta API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Registra los routers de cada módulo para organizar las rutas de la API según su funcionalidad.
app.include_router(auth.router)
app.include_router(usuarios.router)
app.include_router(partidas.router)
app.include_router(engine.router)
app.include_router(vision.router)
app.include_router(retransmision.router)
app.include_router(dataset.router)

# Endpoint favorito para servir el favicon de la aplicación
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Devuelve el favicon de la aplicación para los navegadores."""
    return FileResponse("static/favicon.ico")

# Endpoint para servir la página de pruebas del módulo de visión por computadora, que incluye ejemplos de uso de OpenCV para el reconocimiento del tablero.
@app.get("/opencv", include_in_schema=False)
def opencv():
    """Sirve la página de pruebas del módulo de visión por computadora."""
    return FileResponse("static/opencv.html")

# Endpoint para servir la herramienta interactiva de captura de imágenes para el dataset, 
# que permite a los usuarios generar nuevas muestras para entrenar el modelo de reconocimiento visual.
@app.get("/dataset", include_in_schema=False)
def dataset_tool():
    """Sirve la herramienta interactiva de captura de imágenes para el dataset."""
    return FileResponse("static/dataset.html")

# Endpoint personalizado para servir la documentación de Swagger UI, que incluye un favicon personalizado y un título específico para la API.
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Genera la interfaz de Swagger UI personalizada con el favicon de la aplicación."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Documentación - Chess Rekognition API",
        swagger_favicon_url="/static/favicon.ico"
    )

# Endpoint raíz para comprobar el estado de la API y confirmar que el servidor está operativo.
@app.get("/", tags=["Sistema"], summary="Estado de la API")
def root():
    """Comprueba que el servidor está operativo y devuelve su estado actual."""
    return {
        "status": "online",
        "message": "Servidor Chess Rekognition operando correctamente."
    }
