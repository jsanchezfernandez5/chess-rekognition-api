"""
Router del módulo de visión por computador (OpenCV + MobileNetV2).

Endpoints:
    GET  /vision/status          | Devuelve el estado operativo del módulo y la versión de OpenCV.
    POST /vision/recognize-board | Detecta y rectifica el tablero aplicando la homografía. Devuelve vista cenital 400x400.
    POST /vision/classify        | Clasifica las 64 casillas del tablero con MobileNetV2 y genera el FEN completo.
    POST /vision/classify/reload | Recarga el modelo ML desde disco sin reiniciar el servidor.
    POST /vision/detect-move     | Compara el estado actual del tablero con un FEN previo y detecta el movimiento realizado.
"""
from typing import Optional
import cv2
import numpy as np
import traceback
from fastapi import APIRouter, UploadFile, File, Form

from services.vision import VisionService, _rectificar, board_state_to_fen_board
from services.classifier import classifier
from services.move_detector import detect_move as _detect_move

# Creación del router.
router = APIRouter(prefix="/vision", tags=["Visión"], responses={404: {"description": "No encontrado"}})

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/recognize-board
# Reconoce y rectifica un tablero de ajedrez desde una imagen. Devuelve una vista rectificada en perspectiva cenital (400x400).
# -------------------------------------------------------------------------------
@router.post(
    "/recognize-board", 
    summary="Reconoce y rectifica un tablero de ajedrez desde una imagen."
)
async def recognize_board(
    file: UploadFile = File(...),
    coords: Optional[str] = Form(None),
    rotation: int = Form(0)
):
    """
    Reconoce y rectifica un tablero de ajedrez desde una imagen. 
    
    Devuelve una vista rectificada en perspectiva cenital (400x400).

    Args:
        file: Imagen subida en formato JPEG/PNG.
        coords: Coordenadas manuales del tablero en formato 'x1,y1,x2,y2,x3,y3,x4,y4'.
        rotation: Rotación del tablero (0, 90, 180, 270 grados).

    Returns:
        Dict con el resultado de la detección y el tablero rectificado.
    """
    try:
        # Lee todos los bytes del fichero subido de forma asíncrona
        contents = await file.read()

        # Convierte los bytes a array NumPy y luego a imagen OpenCV (matriz BGR)
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "No se pudo decodificar la imagen recibida"}

        # Delega toda la lógica de detección y rectificación al servicio de visión
        result = VisionService.detect_and_rectify(contents, coords, rotation)
        return result

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /vision/status
# Devuelve el estado operativo del módulo de visión y la versión de OpenCV.
# -------------------------------------------------------------------------------
@router.get(
    "/status", 
    summary="Devuelve el estado operativo del módulo de visión y la versión de OpenCV."
)
def vision_status():
    """
    Devuelve el estado operativo del módulo de visión y la versión de OpenCV.

    Returns:
        Dict con el estado operativo y la versión de OpenCV.
    """
    return {
        "estado": "operativo",
        "modulo": "OpenCV",
        "version": cv2.__version__
    }

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/classify
# Corazón del sistema de reconocimiento visual. Pipeline completo en 3 pasos:
#   1. Detecta y rectifica el tablero (homografía → vista cenital 400x400).
#   2. Clasifica las 64 casillas con MobileNetV2 (13 clases: 12 piezas + empty).
#   3. Genera el FEN completo del estado del tablero para usarlo con python-chess o Stockfish.
# -------------------------------------------------------------------------------
@router.post(
    "/classify", 
    summary="Clasifica las 64 casillas del tablero con MobileNetV2 y genera el FEN completo."
)
async def classify_board(
    file: UploadFile = File(...),
    coords: Optional[str] = Form(None),
    rotation: int = Form(0)
):
    """
    Clasifica las 64 casillas del tablero de ajedrez mediante ML.

    1. Recibe una imagen, detecta y rectifica el tablero.    
    2. Posteriormente utiliza el modelo de clasificación (MobileNetV2) para identificar las clases. Tipos de clases: 
        - White Pawn (w_P), Black Pawn (b_P), 
        - White Rook (w_R), Black Rook (b_R), 
        - White Knight (w_N), Black Knight (b_N), 
        - White Bishop (w_B), Black Bishop (b_B), 
        - White Queen (w_Q), Black Queen (b_Q), 
        - White King (w_K), Black King (b_K),
        - Casilla vacía (empty).    
    3. Finalmente, genera el FEN completo del estado del tablero.

    Args:
        file: Imagen subida con el tablero a clasificar.
        coords: Coordenadas manuales del tablero en formato 'x1,y1,x2,y2,x3,y3,x4,y4'.
        rotation: Rotación del tablero (0, 90, 180, 270 grados).

    Returns:
        Dict con el estado de cada casilla, el FEN generado y flag de éxito.
    """
    try:
        # Comprobamos si el modelo está cargado.
        if not classifier.is_ready():
            return {"success": False, "error": "El modelo de clasificación no está cargado. Entrena primero en /dataset."}

        # Lee y decodifica la imagen (mismo proceso que en /recognize-board)
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # Detectamos el tablero en la imagen con OpenCV.
        found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}
            
        # Rectificamos la imagen para obtener una vista cenital del tablero.
        warped = _rectificar(frame, exterior)

        # Aplicamos la rotación si corresponde
        try:
            rot_val = int(rotation) if rotation else 0
        except (ValueError, TypeError):
            rot_val = 0
        if rot_val == 90:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif rot_val == 180:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        elif rot_val == 270:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Clasificamos las 64 casillas del tablero.
        board_state = classifier.classify_board(warped)

        # Generamos el FEN completo del estado del tablero.
        simple_state = {sq: v["label"] for sq, v in board_state.items()}
        
        # Retornamos el estado de cada casilla y el FEN generado.
        return {
            "success": True,
            "board_state": board_state,
            "fen": board_state_to_fen_board(simple_state).fen()
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/classify/reload
# Recarga el modelo de clasificación desde disco sin reiniciar el servidor.
# -------------------------------------------------------------------------------
@router.post(
    "/classify/reload", 
    summary="Recarga el modelo de clasificación desde disco sin reiniciar el servidor."
)
def reload_model():
    """
    Recarga el modelo de clasificación desde disco sin reiniciar el servidor.

    Vuelve a cargar el modelo ML guardado en el sistema de archivos sin necesidad de reiniciar el servidor. 
    Útil tras reentrenar el modelo desde el módulo de dataset.

    Returns:
        Dict indicando si la recarga fue exitosa y el estado del modelo.
    """
    classifier.reload()
    return {"success": True, "ready": classifier.is_ready()}

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/detect-move
# Compara el estado actual del tablero con un FEN previo y detecta el movimiento realizado.
# -------------------------------------------------------------------------------
@router.post(
    "/detect-move", 
    summary="Detecta el movimiento entre dos posiciones"
)
async def detect_move_endpoint(
    file: UploadFile = File(...),
    prev_fen: str = Form(...),
    coords: Optional[str] = Form(None),
    rotation: int = Form(0)
):
    """
    Detecta el movimiento de ajedrez entre dos estados del tablero.

    1. Clasifica las piezas del nuevo estado mediante ML.
    2. Compara con el estado previo en FEN y determina el movimiento legal realizado.

    Args:
        file: Imagen actual del tablero.
        prev_fen: FEN completo del estado anterior del tablero.
        coords: Coordenadas manuales del tablero en formato 'x1,y1,x2,y2,x3,y3,x4,y4'.
        rotation: Rotación del tablero (0, 90, 180, 270 grados).

    Returns:
        Dict con el movimiento detectado (origen, destino, pieza) y éxito.
    """
    try:
        # Comprobamos si el modelo está cargado.
        if not classifier.is_ready():
            return {"success": False, "error": "Modelo no cargado. Entrena primero en /dataset."}
        
        # Leemos el archivo
        contents = await file.read()

        # Convertimos el archivo a array de numpy
        arr = np.frombuffer(contents, np.uint8)

        # Decodificamos el array de numpy a imagen
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Imagen no decodificable"}

        # Detectamos el tablero en la imagen
        found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}

        # Rectificamos la imagen para obtener una vista cenital
        warped   = _rectificar(frame, exterior)
        
        # Aplicamos la rotación si corresponde
        try:
            rot_val = int(rotation) if rotation else 0
        except (ValueError, TypeError):
            rot_val = 0
        if rot_val == 90:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif rot_val == 180:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        elif rot_val == 270:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Detectamos el movimiento
        result = _detect_move(warped, prev_fen, classifier)
        return {"success": True, **result}

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}
