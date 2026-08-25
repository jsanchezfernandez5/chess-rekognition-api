"""
Punto de entrada del servidor FastAPI.

Centraliza la configuración de la API, el registro de rutas, la documentación técnica y los endpoints auxiliares.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware

# Importamos los routers
from routers import auth, usuarios, partidas, engine, vision, dataset, retransmision, yolo_dataset, yolo_model

# Metadatos para organizar la documentación de la API en categorías.
tags_metadata = [
    {
        "name": "Autenticación",
        "description": "Gestión de sesiones, login y tokens JWT para la seguridad de la app.",
    },
    {
        "name": "Usuarios",
        "description": "Registro de nuevos usuarios.",
    },
    {
        "name": "Partidas",
        "description": "Operaciones CRUD de partidas de ajedrez en formato PGN.",
    },
    {
        "name": "Motor",
        "description": "Integración con Stockfish v17.1 para jugar contra el ordenador.",
    },
    {
        "name": "Visión",
        "description": "Reconocimiento mediante rectificación de la homografía del tablero mediante OpenCV.",
    },
    {
        "name": "Dataset",
        "description": "Herramientas para captura de imágenes (crops) y entrenamiento del modelo ML.",
    },
    {
        "name": "Dataset YOLO",
        "description": "Dataset de detección de objetos (bounding boxes) y entrenamiento del detector YOLO26, con anotación semi-automática desde las predicciones de MobileNetV2.",
    },
    {
        "name": "Modelo YOLO",
        "description": "Gestión del modelo YOLO26 entrenado: clases, bounding boxes del dataset y borrado del modelo.",
    },
    {
        "name": "Retransmisión",
        "description": "Servicio para retransmitir partidas en directo mediante WebSockets.",
    },
    {
        "name": "Sistema",
        "description": "Endpoints de estado y monitorización del servidor.",
    }
]

# -------------------------------------------------------------------------------------------
# ---------------- INICIALIZACIÓN API CON SWAGGER PERSONALIZADO -----------------------------
# -------------------------------------------------------------------------------------------
app = FastAPI(
    title="Chess Rekognition API",
    description=(
        "### Reconocimiento visual de jugadas en partidas de ajedrez presencial.\n\n"
        "**Herramientas web integradas:**\n"
        "* **Correción homográfica - De imagen 3D a vista cenital 2D con OpenCV:** [/opencv](/opencv)\n"
        "* **Módulo de Dataset - Captura recortes (crops) de la imagen y Etiquetado de clases para el Entrenamiento del Modelo MobileNetV2:** [/dataset](/dataset)\n"
        "* **Reconocimiento del tablero y las piezas, a través del Modelo MobileNetV2, obteniéndose el código de ajedrez FEN:** [/reconocimiento](/reconocimiento)\n"
        "* **Módulo de Dataset YOLO26 - Anotación semi-automática de bounding boxes (arrancada desde las predicciones de MobileNetV2) y entrenamiento del detector YOLO26:** [/yolo-dataset](/yolo-dataset)\n"
        "* **Gestión del Dataset YOLO26 - Consulta y borrado de clases, bounding boxes e imágenes:** [/yolo-manage](/yolo-manage)\n"
        "* **Detección en directo con YOLO26 (bounding boxes de piezas y manos) sobre el tablero rectificado:** [/yolo](/yolo)"
    ),
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

# -------------------------------------------------------------------------------------------
# ----------------------- DIRECTORIO STATIC y CORS ------------------------------------------
# -------------------------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# -------------------------------------------------------------------------------------------
# ---------------------------------- ROUTERS ------------------------------------------------
# -------------------------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(usuarios.router)
app.include_router(partidas.router)
app.include_router(engine.router)
app.include_router(vision.router)
app.include_router(dataset.router)
app.include_router(yolo_dataset.router)
app.include_router(yolo_model.router)
app.include_router(retransmision.router)

# -------------------------------------------------------------------------------------------
# --------------------- ENDPOINTS AUXILIARES ------------------------------------------------
# -------------------------------------------------------------------------------------------

# Endpoint para servir el favicon de la aplicación
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Devuelve el favicon de la aplicación para los navegadores."""
    return FileResponse("static/favicon.ico")

# Endpoint de pruebas de OpenCV para el reconocimiento del tablero rectificando la homografía a vista cenital.
@app.get("/opencv", include_in_schema=False)
def opencv():
    """Sirve la página de pruebas del módulo de visión por computadora."""
    return FileResponse("static/opencv.html")

# Endpoint para capturar imágenes para el dataset haciendo recortes (crops) y entrenar el modelo de reconocimiento neuronal MobileNetV2 de TensorFlow.
@app.get("/dataset", include_in_schema=False)
def dataset_tool():
    """Sirve la herramienta interactiva de captura de imágenes (crops) para el dataset."""
    return FileResponse("static/dataset.html")

# Endpoint herramienta de reconocimiento del tablero y las piezas de ajedrez a través del modelo de reconocimiento neuronal MobileNetV2 de TensorFlow.
@app.get("/reconocimiento", include_in_schema=False)
def reconocimiento():
    """Sirve la herramienta de reconocimiento visual en tiempo real."""
    return FileResponse("static/reconocimiento.html")

# Endpoint para la anotación semi-automática del dataset YOLO (bounding boxes arrancadas desde las predicciones de MobileNetV2) y el entrenamiento del detector YOLO26.
@app.get("/yolo-dataset", include_in_schema=False)
def yolo_dataset_tool():
    """Sirve la herramienta interactiva de anotación y entrenamiento del dataset YOLO26."""
    return FileResponse("static/yolo_dataset.html")

# Endpoint para probar el detector YOLO26 sobre el tablero rectificado (cajas de piezas y manos en directo).
@app.get("/yolo", include_in_schema=False)
def yolo_tool():
    """Sirve la herramienta de prueba del detector YOLO26 en tiempo real."""
    return FileResponse("static/yolo.html")

# Endpoint para gestionar el dataset YOLO: clases, bounding boxes e imágenes.
@app.get("/yolo-manage", include_in_schema=False)
def yolo_manage_tool():
    """Sirve la herramienta de gestión del dataset YOLO26 (clases, cajas, imágenes y modelo)."""
    return FileResponse("static/yolo_manage.html")

# Endpoint para servir la documentación de Swagger UI.
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
        "message": "Servidor Chess Rekognition operando correctamente.",
        "urls": {
            "opencv": "/opencv",
            "dataset": "/dataset",
            "reconocimiento": "/reconocimiento",
            "yolo_dataset": "/yolo-dataset",
            "yolo_manage": "/yolo-manage",
            "yolo": "/yolo",
            "docs": "/docs"
        }
    }
