from fastapi import APIRouter, UploadFile, File
from services.vision import VisionService
import cv2
import numpy as np
import traceback

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
    try:
        contents = await file.read()

        # Verificar que la imagen llega correctamente antes de procesarla
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {
                "success": False,
                "error": "No se pudo decodificar la imagen recibida"
            }

        result = VisionService.detect_and_rectify(contents)
        return result

    except Exception as e:
        # Capturar cualquier excepción y devolverla como JSON
        # para que el frontend pueda parsearla correctamente
        return {
            "success": False,
            "error": str(e),
            "detail": traceback.format_exc()
        }


@router.get("/status", summary="Estado del motor de visión")
def vision_status():
    """Devuelve la versión de OpenCV para verificar que el módulo está cargado."""
    return {
        "estado": "operativo",
        "modulo": "OpenCV",
        "version": cv2.__version__
    }
