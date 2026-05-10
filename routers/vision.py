# routers/vision.py
# Visión por computador para el reconocimiento del tablero de ajedrez
import cv2
import numpy as np
import traceback
from fastapi import APIRouter, UploadFile, File, Form

# Importaciones de servicios
from services.vision import VisionService, _detectar_tablero, _calcular_esquinas_exteriores, _rectificar
from services.classifier import classifier
from services.move_detector import detect_move as _detect_move
from utils.chess_utils import board_state_to_fen_board

# Router para visión por computador
router = APIRouter(
    prefix="/vision",
    tags=["Visión"],
    responses={404: {"description": "No encontrado"}},
)

# Endpoint para reconocer el tablero
@router.post("/recognize-board", summary="Reconoce y rectifica un tablero de ajedrez")
async def recognize_board(file: UploadFile = File(...)):
    """
    Recibe una imagen (desde la cámara o archivo) 
    y devuelve el tablero rectificado en perspectiva cenital.
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

# Endpoint para obtener estado del motor de visión
@router.get("/status", summary="Estado del motor de visión")
def vision_status():
    """Devuelve la versión de OpenCV para verificar que el módulo está cargado."""
    return {
        "estado": "operativo",
        "modulo": "OpenCV",
        "version": cv2.__version__
    }

# Endpoint para clasificar las piezas del tablero
@router.post("/classify", summary="Clasifica las 64 casillas del tablero")
async def classify_board(file: UploadFile = File(...)):
    """
    Recibe un frame, detecta el tablero y usa el modelo ML para clasificar 
    cada una de las 64 casillas (empty, w_P, b_R, etc.).
    """
    try:
        if not classifier.is_ready():
            return {"success": False, "error": "El modelo de clasificación no está cargado. Entrena primero en /dataset."}

        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # 1. Visión clásica para rectificar
        found, corners = _detectar_tablero(frame)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}
            
        exterior = _calcular_esquinas_exteriores(corners)
        warped = _rectificar(frame, exterior)
        
        # 2. ML para clasificar
        board_state = classifier.classify_board(warped)
        
        # 3. Formato simple para el GameState
        simple_state = {sq: v["label"] for sq, v in board_state.items()}
        
        return {
            "success": True,
            "board_state": board_state,
            "fen": board_state_to_fen_board(simple_state).fen()
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# Endpoint para recargar el modelo ML desde disco
@router.post("/classify/reload", summary="Recarga el modelo ML desde disco")
def reload_model():
    """Recarga el modelo en memoria sin reiniciar el servidor."""
    classifier.reload()
    return {"success": True, "ready": classifier.is_ready()}

# --- DETECCIÓN DE MOVIMIENTOS ---

@router.post("/detect-move", summary="Detecta el movimiento entre dos posiciones")
async def detect_move_endpoint(
    file: UploadFile = File(...),
    prev_fen: str = Form(...)
):
    """
    Recibe un frame de la cámara y el FEN completo del estado anterior.
    Devuelve el movimiento legal que se ha producido (si existe).
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
