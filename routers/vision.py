"""
Router del módulo de visión por computador (OpenCV + YOLO26).

Endpoints:
    GET  /vision/status          | Devuelve el estado operativo del módulo y la versión de OpenCV.
    POST /vision/recognize-board | Detecta y rectifica el tablero aplicando la homografía. Devuelve vista cenital 400x400.
    POST /vision/detect-yolo     | Detecta piezas y manos con YOLO26 sobre el tablero rectificado y, opcionalmente, el movimiento realizado.
    POST /vision/engine-stats    | Evalúa YOLO26 sobre la partición de validación (square_accuracy, fen_exact, precision/recall).
"""
from typing import Optional
import cv2
import numpy as np
import traceback
from fastapi import APIRouter, UploadFile, File, Form

from core.config import settings
from services.vision import VisionService, _rectificar, board_state_to_fen_board
from services.move_detector import detect_move_desde_estado as _detect_move_desde_estado
from services.yolo_detector import yolo_detector, boxes_a_board_state
from services.engine_stats import evaluar_motor

# Creación del router.
router = APIRouter(prefix="/vision", tags=["Visión"], responses={404: {"description": "No encontrado"}})

# Función auxiliar para aplicar la rotación a la vista cenital.
def _rotar_warped(warped: np.ndarray, rotation) -> np.ndarray:
    """
    Aplica la rotación seleccionada a la imagen ya rectificada.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.
        rotation: Rotación deseada (0, 90, 180 o 270 grados).

    Returns:
        La imagen rotada (o la original si la rotación es 0 o inválida).
    """
    try:
        rot_val = int(rotation) if rotation else 0
    except (ValueError, TypeError):
        rot_val = 0
    if rot_val == 90:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    elif rot_val == 180:
        return cv2.rotate(warped, cv2.ROTATE_180)
    elif rot_val == 270:
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return warped

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
# [ENDPOINT] - POST /vision/detect-yolo
# Motor principal de reconocimiento. Pipeline:
#   1. Detecta y rectifica el tablero (homografía → vista cenital 400x400). YOLO NUNCA ve el frame crudo.
#   2. Detecta piezas y manos con YOLO26 y mapea cada bounding box a su casilla por el centro de la caja.
#   3. Si se envía prev_fen, busca el movimiento legal que explica la transición (misma lógica
#      reutilizada de move_detector, sin duplicarla).
# -------------------------------------------------------------------------------
@router.post(
    "/detect-yolo",
    summary="Detecta piezas y manos con YOLO26 sobre el tablero rectificado y, opcionalmente, el movimiento realizado."
)
async def detect_yolo_endpoint(
    file: UploadFile = File(...),
    coords: Optional[str] = Form(None),
    rotation: int = Form(0),
    prev_fen: Optional[str] = Form(None)
):
    """
    Detecta piezas y manos sobre el tablero mediante YOLO26.

    1. Recibe una imagen, detecta y rectifica el tablero (mismo pipeline OpenCV que los demás endpoints).
    2. Ejecuta el detector YOLO sobre la vista cenital 400x400 y asigna cada caja a su casilla.
       Las detecciones de clase "hand" no cuentan como piezas: se devuelven aparte como señal de
       interferencia humana (mano/dedo sobre el tablero).
    3. Genera el FEN del estado detectado y, si se proporciona prev_fen, también el movimiento legal
       que explica la transición (reutilizando detect_move_desde_estado).

    Args:
        file: Imagen actual del tablero.
        coords: Coordenadas manuales del tablero en formato 'x1,y1,x2,y2,x3,y3,x4,y4' (porcentajes 0-1).
        rotation: Rotación del tablero (0, 90, 180, 270 grados).
        prev_fen: FEN opcional del estado anterior para detectar además el movimiento realizado.

    Returns:
        Dict con las detecciones, el estado de las casillas, el FEN, las manos detectadas,
        los conflictos de cajas solapadas y (si procede) el movimiento detectado.
    """
    try:
        # Comprobamos si el modelo YOLO está cargado (puede no estarlo si falta ultralytics o el .pt).
        if not yolo_detector.is_ready():
            return {"success": False, "error": "El modelo YOLO no está disponible. Entrena primero en /yolo-dataset."}

        # Leemos y decodificamos la imagen (mismo proceso que en los demás endpoints).
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Imagen no decodificable"}

        # Detectamos el tablero en la imagen y rectificamos SIEMPRE antes de pasar el frame a YOLO.
        found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}
        warped = _rectificar(frame, exterior)

        # Aplicamos la rotación si corresponde.
        warped = _rotar_warped(warped, rotation)

        # Inferencia YOLO y mapeo de cajas → casillas (por el centro de cada bbox).
        detecciones = yolo_detector.detect(warped)
        board_state, hand_boxes_todas, conflictos = boxes_a_board_state(detecciones)

        # Conflictos de cajas solapadas: aviso informativo, nunca un error.
        for conflicto in conflictos:
            print(f"[detect-yolo] Aviso: varias cajas en {conflicto['square']}: "
                  f"se mantiene {conflicto['kept']}, descartadas {conflicto['discarded']}")

        # FEN del estado detectado.
        simple_state = {sq: v["label"] for sq, v in board_state.items()}

        # Manos: solo cuentan como interferencia real si superan el umbral de confianza configurado.
        hand_boxes = [h for h in hand_boxes_todas if h["confidence"] >= settings.HAND_MIN_CONFIDENCE]
        hand_detected = len(hand_boxes) > 0

        response = {
            "success": True,
            "engine": "yolo",
            "detections": detecciones,
            "board_state": board_state,
            "fen": board_state_to_fen_board(simple_state).fen(),
            "hand_detected": hand_detected,
            "hand_boxes": hand_boxes,
            "box_conflicts": conflictos
        }

        # Si nos dan el FEN previo, buscamos además el movimiento legal que explica la transición.
        if prev_fen:
            resultado = _detect_move_desde_estado(board_state, prev_fen)
            # Aplanamos la respuesta con found/move/new_fen/error/board_state.
            response.update({k: v for k, v in resultado.items() if k != "board_state"})

        return response

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/engine-stats
# Evalúa YOLO26 sobre la partición de VALIDACIÓN del dataset YOLO (que nunca
# participa en el entrenamiento): precisión casilla a casilla, tableros exactos,
# precisión/recall de piezas y tiempo medio por imagen. SOLO INFORMA, nunca
# cambia configuración.
#
# ⚠️ CORRE SÍNCRONO en la petición HTTP con inferencia CPU: con max_images alto
# puede tardar minutos y el gateway de Railway cortar la conexión por timeout.
# Se recomienda 50-100 imágenes en producción.
# -------------------------------------------------------------------------------
@router.post(
    "/engine-stats",
    summary="Evalúa YOLO26 sobre el dataset de validación (square_accuracy, fen_exact, precision/recall)."
)
def engine_stats_endpoint(max_images: int = Form(100)):
    """
    Evalúa el motor YOLO26 sobre las imágenes de validación etiquetadas.

    Args:
        max_images: Número máximo de imágenes a evaluar (por defecto 100). Ejecución
                    síncrona sobre CPU: valores altos pueden exceder el timeout del
                    gateway en producción; se recomienda mantenerse entre 50 y 100.

    Returns:
        Dict con las métricas de YOLO26 (square_accuracy, fen_exact, piece_precision,
        piece_recall, avg_ms), la nota metodológica sobre el sesgo de la anotación
        semi-automática y nota_rendimiento con la recomendación de uso en producción.
    """
    try:
        # Acotamos el parámetro para que nadie pueda pedir una evaluación eterna en producción.
        max_images = max(1, min(int(max_images), 500))
        return evaluar_motor(max_images)
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}
