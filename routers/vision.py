"""
Router del módulo de visión por computador (OpenCV + MobileNetV2 + YOLO26).

Endpoints:
    GET  /vision/status          | Devuelve el estado operativo del módulo y la versión de OpenCV.
    POST /vision/recognize-board | Detecta y rectifica el tablero aplicando la homografía. Devuelve vista cenital 400x400.
    POST /vision/classify        | Clasifica las 64 casillas del tablero con MobileNetV2 y genera el FEN completo.
    POST /vision/classify/reload | Recarga el modelo ML desde disco sin reiniciar el servidor.
    POST /vision/detect-move     | Compara el estado actual del tablero con un FEN previo y detecta el movimiento realizado.
    POST /vision/detect-yolo     | Detecta piezas y manos con YOLO26 sobre el tablero rectificado y, opcionalmente, el movimiento realizado.
"""
from typing import Optional
import cv2
import numpy as np
import traceback
from fastapi import APIRouter, UploadFile, File, Form

from core.config import settings
from services.vision import VisionService, _rectificar, board_state_to_fen_board
from services.classifier import classifier
from services.move_detector import detect_move as _detect_move
from services.move_detector import detect_move_desde_estado as _detect_move_desde_estado
from services.yolo_detector import yolo_detector, boxes_a_board_state

# Creación del router.
router = APIRouter(prefix="/vision", tags=["Visión"], responses={404: {"description": "No encontrado"}})

# Función auxiliar para aplicar la rotación a la vista cenital.
def _rotar_warped(warped: np.ndarray, rotation) -> np.ndarray:
    """
    Aplica la rotación seleccionada a la imagen ya rectificada (helper para los endpoints nuevos;
    los endpoints históricos conservan su switch duplicado intacto).

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

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/detect-yolo
# Segundo motor de reconocimiento EN PARALELO con MobileNetV2. Pipeline:
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
            # Aplanamos la respuesta igual que hace /vision/detect-move (found/move/new_fen/error/...).
            response.update({k: v for k, v in resultado.items() if k != "board_state"})

        return response

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /vision/classify-fusion
# Tercer modo de reconocimiento: ejecuta AMBOS motores sobre el MISMO tablero rectificado
# y arbitra casilla a casilla según la confianza de cada uno (services/fusion.py).
# MobileNetV2 sigue siendo el motor por defecto: YOLO solo sobrescribe cuando sus
# detecciones superan sus umbrales. Las manos detectadas invalidan la lectura de las
# casillas que cubren y, si llega prev_fen, bloquean la búsqueda de movimiento mientras
# duren (la posición está tapada: cualquier FEN sería una suposición).
# -------------------------------------------------------------------------------
@router.post(
    "/classify-fusion",
    summary="Clasifica el tablero con AMBOS motores (MobileNetV2 + YOLO26), arbitra por confianza y opcionalmente detecta el movimiento."
)
async def classify_fusion_endpoint(
    file: UploadFile = File(...),
    coords: Optional[str] = Form(None),
    rotation: int = Form(0),
    prev_fen: Optional[str] = Form(None)
):
    """
    Fusión por arbitraje de confianza entre los dos motores de reconocimiento.

    1. Rectifica el tablero una sola vez (misma homografía para ambos motores).
    2. Clasifica las 64 casillas con MobileNetV2 y detecta objetos con YOLO26 en paralelo.
    3. Fusiona ambos board_states casilla a casilla (services/fusion.py): las detecciones de
       mano tapan su lectura; en desacuerdo gana la lectura más confiada.
    4. Genera el FEN final y, si se envía prev_fen y no hay mano sobre el tablero, busca el
       movimiento legal que explica la transición.

    Args:
        file: Imagen actual del tablero.
        coords: Coordenadas manuales del tablero en formato 'x1,y1,x2,y2,x3,y3,x4,y4' (porcentajes 0-1).
        rotation: Rotación del tablero (0, 90, 180, 270 grados).
        prev_fen: FEN opcional del estado anterior para detectar además el movimiento realizado.

    Returns:
        Dict con el estado fusionado, el FEN, las manos detectadas, el resumen de decisiones
        del arbitraje y (si procede) el movimiento detectado.
    """
    try:
        # La fusión necesita los DOS motores operativos; si falta alguno se avisa con claridad
        # para que el frontend pueda caer al modo TensorFlow a secas.
        if not classifier.is_ready():
            return {"success": False, "error": "El modelo TensorFlow no está cargado. Entrena primero en /dataset."}
        if not yolo_detector.is_ready():
            return {"success": False, "error": "El modelo YOLO no está disponible. Entrena primero en /yolo-dataset."}

        # Leemos y decodificamos la imagen.
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Imagen no decodificable"}

        # Homografía compartida: UNA rectificación para los dos motores.
        found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
        if not found:
            return {"success": False, "error": "Tablero no detectado"}
        warped = _rectificar(frame, exterior)

        # Aplicamos la rotación si corresponde (misma rotación para ambos motores).
        warped = _rotar_warped(warped, rotation)

        # Los dos motores leen exactamente la misma vista cenital.
        board_state_tf = classifier.classify_board(warped)
        detecciones = yolo_detector.detect(warped)
        board_state_yolo, hand_boxes_todas, conflictos = boxes_a_board_state(detecciones)

        # Arbitraje casilla a casilla.
        board_state_final, resumen_fusion = fusionar_board_states(board_state_tf, board_state_yolo, hand_boxes_todas)

        simple_final = {sq: v["label"] for sq, v in board_state_final.items()}
        hand_detected = len(resumen_fusion["manos_validas"]) > 0

        response = {
            "success": True,
            "engine": "fusion",
            "board_state": board_state_final,
            "fen": board_state_to_fen_board(simple_final).fen(),
            "hand_detected": hand_detected,
            "hand_boxes": resumen_fusion["manos_validas"],
            "hand_squares": resumen_fusion["casillas_con_mano"],
            "fusion_summary": resumen_fusion["contadores"],
            "fusion_decisions": resumen_fusion["decisiones"],
            "box_conflicts": conflictos
        }

        # Con una mano tapando casillas NO tiene sentido buscar movimiento: la posición está
        # incompleta y cualquier coincidencia sería casualidad. Se informa como estado incierto.
        if prev_fen:
            if hand_detected:
                response.update({"found": False, "error": "hand_over_board"})
            else:
                resultado = _detect_move_desde_estado(board_state_final, prev_fen)
                response.update({k: v for k, v in resultado.items() if k != "board_state"})

        return response

    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}
