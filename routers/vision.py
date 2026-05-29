"""Módulo de visión por computador para el reconocimiento del tablero.

Proporciona endpoints para detectar y rectificar tableros de ajedrez,
clasificar piezas mediante un modelo ML y detectar movimientos entre
dos estados consecutivos del tablero.
"""

import cv2
import numpy as np
import traceback
from fastapi import APIRouter, UploadFile, File, Form

from services.vision import VisionService, _detectar_tablero, _calcular_esquinas_exteriores, _rectificar
from services.classifier import classifier
from services.move_detector import detect_move as _detect_move
from utils.chess_utils import board_state_to_fen_board

router = APIRouter(
    prefix="/vision",
    tags=["Visión"],
    responses={404: {"description": "No encontrado"}},
)

@router.post("/recognize-board", summary="Reconoce y rectifica un tablero de ajedrez")
async def recognize_board(file: UploadFile = File(...)):
    """Reconoce y rectifica un tablero de ajedrez desde una imagen.

    Recibe una imagen capturada desde la cámara o un archivo,
    detecta el tablero mediante procesamiento de visión clásica
    y devuelve una vista rectificada en perspectiva cenital (400x400).

    Args:
        file: Imagen subida en formato JPEG/PNG.

    Returns:
        Dict con el resultado de la detección y el tablero rectificado.
    """
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "No se pudo decodificar la imagen recibida"}

        result = VisionService.detect_and_rectify(contents)
        return result

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

@router.get("/status", summary="Estado del motor de visión")
def vision_status():
    """Obtiene el estado del módulo de visión por computador.

    Verifica que el módulo de OpenCV está correctamente cargado
    y devuelve información sobre su versión.

    Returns:
        Dict con el estado operativo y la versión de OpenCV.
    """
    return {
        "estado": "operativo",
        "modulo": "OpenCV",
        "version": cv2.__version__
    }

@router.post("/classify", summary="Clasifica las 64 casillas del tablero")
async def classify_board(file: UploadFile = File(...)):
    """Clasifica las 64 casillas del tablero de ajedrez mediante ML.

    Recibe una imagen, detecta y rectifica el tablero, luego utiliza
    el modelo de clasificación (MobileNetV2) para identificar el
    contenido de cada casilla: vacía ('empty') o pieza con color
    (ej. 'w_P', 'b_R', etc.). Finalmente, genera el FEN completo
    del estado del tablero.

    Args:
        file: Imagen subida con el tablero a clasificar.

    Returns:
        Dict con el estado de cada casilla, el FEN generado y flag de éxito.
    """
    try:
        if not classifier.is_ready():
            return {"success": False, "error": "El modelo de clasificación no está cargado. Entrena primero en /dataset."}

        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        found, corners = _detectar_tablero(frame)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}
            
        exterior = _calcular_esquinas_exteriores(corners)
        warped = _rectificar(frame, exterior)
        
        board_state = classifier.classify_board(warped)
        
        simple_state = {sq: v["label"] for sq, v in board_state.items()}
        
        return {
            "success": True,
            "board_state": board_state,
            "fen": board_state_to_fen_board(simple_state).fen()
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

@router.post("/classify/reload", summary="Recarga el modelo ML desde disco")
def reload_model():
    """Recarga el modelo de clasificación desde disco en caliente.

    Vuelve a cargar el modelo ML guardado en el sistema de archivos
    sin necesidad de reiniciar el servidor. Útil tras reentrenar
    el modelo desde el módulo de dataset.

    Returns:
        Dict indicando si la recarga fue exitosa y el estado del modelo.
    """
    classifier.reload()
    return {"success": True, "ready": classifier.is_ready()}

@router.post("/detect-move", summary="Detecta el movimiento entre dos posiciones")
async def detect_move_endpoint(
    file: UploadFile = File(...),
    prev_fen: str = Form(...)
):
    """Detecta el movimiento de ajedrez entre dos estados del tablero.

    Clasifica las piezas del nuevo estado mediante ML, compara con
    el estado previo en FEN y determina el movimiento legal realizado.

    Args:
        file: Imagen actual del tablero.
        prev_fen: FEN completo del estado anterior del tablero.

    Returns:
        Dict con el movimiento detectado (origen, destino, pieza) y éxito.
    """
    try:
        if not classifier.is_ready():
            return {"success": False, "error": "Modelo no cargado. Entrena primero en /dataset."}

        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Imagen no decodificable"}

        found, corners = _detectar_tablero(frame)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}

        exterior = _calcular_esquinas_exteriores(corners)
        warped   = _rectificar(frame, exterior)

        result = _detect_move(warped, prev_fen, classifier)
        return {"success": True, **result}

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}
