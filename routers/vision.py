from fastapi import APIRouter, UploadFile, File
from services.vision import VisionService
import cv2

router = APIRouter(
    prefix="/vision",
    tags=["Visión"],
    responses={404: {"description": "No encontrado"}},
)

@router.post("/recognize-board", summary="Reconoce y rectifica un tablero de ajedrez")
async def recognize_board(file: UploadFile = File(...)):
    """
    Recibe una imagen (desde la cámara o archivo) y devuelve el tablero 
    rectificado en perspectiva cenital.
    """
    # 1. Leer los bytes de la imagen subida
    contents = await file.read()
    
    # 2. Procesar mediante el servicio de visión
    result = VisionService.detect_and_rectify(contents)
    
    # 3. Validar resultado
    if not result["success"]:
        # Se puede devolver un 200 con success: false o un 422 si hay problemas de reconocimiento
        return result
        
    return result

@router.get("/status", summary="Estado del motor de visión")
def vision_status():
    """Devuelve la versión de OpenCV para verificar que el módulo está cargado."""
    return {
        "estado": "operativo",
        "modulo": "OpenCV",
        "version": cv2.__version__
    }
