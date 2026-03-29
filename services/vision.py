import cv2
import numpy as np
import base64
import traceback

BOARD_SIZE = 400
CELL_SIZE = BOARD_SIZE // 8
COLS = "abcdefgh"

INNER_CROP_PCT = 0.65  # Usar solo el 65% central de la casilla


def _encode_image(img: np.ndarray) -> str:
    """Convierte un array numpy a base64 para enviar al frontend."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def _detectar_tablero(frame: np.ndarray):
    """
    Detecta el tablero usando findChessboardCornersSB.
    Devuelve (found, corners) donde corners son las 49 esquinas internas.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Intento principal — más robusto
    found, corners = cv2.findChessboardCornersSB(
        gray, (7, 7),
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )

    if not found:
        # Fallback clásico
        found, corners = cv2.findChessboardCorners(
            gray, (7, 7),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    if not found:
        return False, None

    # Refinar a subpíxel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return True, corners


def _calcular_esquinas_exteriores(corners: np.ndarray) -> np.ndarray:
    """
    Calcula las 4 esquinas exteriores del tablero completo a partir
    de las 49 esquinas internas (7x7).
    Usa vectores locales por esquina para mayor precisión
    cuando el tablero está en perspectiva.
    """
    tl = corners[0][0]   # fila 0, col 0
    tr = corners[6][0]   # fila 0, col 6
    bl = corners[42][0]  # fila 6, col 0
    br = corners[48][0]  # fila 6, col 6

    # Vector de una casilla en cada dirección local
    # (las 7 esquinas cubren 6 intervalos → dividir entre 6)
    step_h_top    = (tr - tl) / 6.0   # horizontal arriba
    step_h_bottom = (br - bl) / 6.0   # horizontal abajo
    step_v_left   = (bl - tl) / 6.0   # vertical izquierda
    step_v_right  = (br - tr) / 6.0   # vertical derecha

    # Extrapolar una casilla hacia afuera desde cada esquina
    # usando los vectores locales de esa esquina concreta
    MARGIN = 1.12
    board_tl = tl - step_h_top    * MARGIN - step_v_left   * MARGIN
    board_tr = tr + step_h_top    * MARGIN - step_v_right  * MARGIN
    board_bl = bl - step_h_bottom * MARGIN + step_v_left   * MARGIN
    board_br = br + step_h_bottom * MARGIN + step_v_right  * MARGIN

    return np.array(
        [board_tl, board_tr, board_br, board_bl],
        dtype=np.float32
    )


def _rectificar(frame: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    """Aplica la homografía para obtener vista cenital 400x400."""
    dst = np.array([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(exterior, dst)
    warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))
    return warped


def _calibrar_umbrales(warped: np.ndarray) -> tuple:
    """
    Calcula umbrales dinámicamente a partir del tablero rectificado.
    Funciona con cualquier iluminación sin necesidad de ajuste manual.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    stds = []

    for row in range(8):
        for col in range(8):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cell = gray[y:y+CELL_SIZE, x:x+CELL_SIZE]
            # Recorte central para evitar ruido de bordes de casilla
            h, w = cell.shape
            ch, cw = int(h * INNER_CROP_PCT), int(w * INNER_CROP_PCT)
            yo, xo = (h - ch) // 2, (w - cw) // 2
            stds.append(float(np.std(cell[yo:yo+ch, xo:xo+cw])))

    stds_arr = np.array(stds)
    p25  = float(np.percentile(stds_arr, 25))
    p75  = float(np.percentile(stds_arr, 75))
    spread = p75 - p25

    if spread < 10:
        # Tablero uniforme (vacío o muy poca variación)
        # Umbral conservador: mediana + margen fijo
        std_thresh = float(np.median(stds_arr)) + max(spread * 1.5, 15)
    else:
        # Hay variación significativa: umbral entre grupo bajo y grupo alto
        std_thresh = p25 + spread * 0.8

    # Rango razonable: nunca menor de 45 ni mayor de 120
    std_thresh = float(np.clip(std_thresh, 45, 120))
    edge_thresh = std_thresh * 18

    return std_thresh, edge_thresh


def _analizar_casillas(warped: np.ndarray) -> tuple:
    """
    Analiza las 64 casillas del tablero rectificado.
    Autocalibra los umbrales según la iluminación actual.
    """
    std_thresh, edge_thresh = _calibrar_umbrales(warped)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    squares = []

    for row in range(8):
        for col in range(8):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cell_gray = gray[y:y+CELL_SIZE, x:x+CELL_SIZE]
            cell_edges = edges[y:y+CELL_SIZE, x:x+CELL_SIZE]

            h, w = cell_gray.shape
            ch, cw = int(h * INNER_CROP_PCT), int(w * INNER_CROP_PCT)
            y_off, x_off = (h - ch) // 2, (w - cw) // 2
            inner_gray = cell_gray[y_off:y_off+ch, x_off:x_off+cw]
            inner_edges = cell_edges[y_off:y_off+ch, x_off:x_off+cw]

            std = float(np.std(inner_gray))
            edge_count = int(np.sum(inner_edges > 0))
            occupied = std > std_thresh or edge_count > edge_thresh

            squares.append({
                "id": f"{COLS[col]}{8 - row}",
                "row": row,
                "col": col,
                "occupied": occupied,
                "std": round(std, 2),
                "edges": edge_count,
                "is_light": (row + col) % 2 == 0
            })

    return squares, std_thresh, edge_thresh


def _generar_vista_real(warped: np.ndarray, squares: list) -> np.ndarray:
    """
    Vista cenital real con overlay verde/rojo sobre las casillas.
    """
    output = warped.copy()

    for sq in squares:
        x, y = sq["col"] * CELL_SIZE, sq["row"] * CELL_SIZE
        color = (0, 80, 0) if not sq["occupied"] else (0, 0, 180)
        alpha = 0.18
        overlay = output.copy()
        cv2.rectangle(overlay, (x+2, y+2),
                      (x+CELL_SIZE-2, y+CELL_SIZE-2), color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        border_color = (0, 200, 80) if not sq["occupied"] else (60, 60, 220)
        cv2.rectangle(output, (x+2, y+2),
                      (x+CELL_SIZE-2, y+CELL_SIZE-2), border_color, 1)

    # Cuadrícula
    for i in range(9):
        cv2.line(output, (i*CELL_SIZE, 0),
                 (i*CELL_SIZE, BOARD_SIZE), (80, 80, 80), 1)
        cv2.line(output, (0, i*CELL_SIZE),
                 (BOARD_SIZE, i*CELL_SIZE), (80, 80, 80), 1)

    return output


def _generar_vista_2d(squares: list) -> np.ndarray:
    """
    Vista diagnóstico 2D: tablero sintético con colores
    de casilla clásicos y marcadores de ocupación.
    """
    output = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    COLOR_LIGHT = (210, 200, 180)
    COLOR_DARK  = (100, 70,  50)
    COLOR_OCC   = (60,  60,  220)
    COLOR_FREE  = (40,  180, 80)

    for sq in squares:
        x, y = sq["col"] * CELL_SIZE, sq["row"] * CELL_SIZE
        base = COLOR_LIGHT if sq["is_light"] else COLOR_DARK
        cv2.rectangle(output, (x, y),
                      (x+CELL_SIZE, y+CELL_SIZE), base, -1)

        if sq["occupied"]:
            cv2.rectangle(output, (x+4, y+4),
                          (x+CELL_SIZE-4, y+CELL_SIZE-4), COLOR_OCC, -1)
            cv2.putText(output, sq["id"], (x+6, y+CELL_SIZE-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        else:
            cv2.circle(output,
                       (x + CELL_SIZE//2, y + CELL_SIZE//2),
                       4, COLOR_FREE, -1)

    # Cuadrícula fina
    for i in range(9):
        cv2.line(output, (i*CELL_SIZE, 0),
                 (i*CELL_SIZE, BOARD_SIZE), (50, 50, 50), 1)
        cv2.line(output, (0, i*CELL_SIZE),
                 (BOARD_SIZE, i*CELL_SIZE), (50, 50, 50), 1)

    return output


def _generar_debug(frame: np.ndarray, corners: np.ndarray,
                   exterior: np.ndarray) -> np.ndarray:
    """
    Imagen de debug: frame original con los 49 puntos internos
    en rojo y el perímetro del tablero en verde.
    """
    debug = frame.copy()

    # Perímetro verde del tablero
    pts = exterior.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(debug, [pts], True, (0, 220, 80), 3)

    # 49 esquinas internas en rojo
    for pt in corners:
        cx, cy = int(pt[0][0]), int(pt[0][1])
        cv2.circle(debug, (cx, cy), 4, (0, 0, 220), -1)

    return debug


def _generar_collage(debug: np.ndarray, real: np.ndarray, diag: np.ndarray, 
                     occ: int, total: int, stdt: float, edget: float) -> np.ndarray:
    """
    Crea una imagen única combinando los 3 estados principales para exportación fácil.
    Incluye metadatos técnicos para diagnóstico profundo.
    """
    # Escalar todo a tamaños consistentes
    h_main = 450
    w_main = int(debug.shape[1] * (h_main / debug.shape[0]))
    debug_res = cv2.resize(debug, (w_main, h_main))

    # Tira lateral con las dos vistas (225x225 cada una)
    # Dejamos espacio para info técnica abajo
    real_res = cv2.resize(real, (225, 225))
    diag_res = cv2.resize(diag, (225, 225))
    side_strip = np.vstack([real_res, diag_res])
    
    # Combinar principal
    collage = np.hstack([debug_res, side_strip])

    # Añadir un pie de foto oscuro con info técnica
    footer_h = 45
    footer = np.zeros((footer_h, collage.shape[1], 3), dtype=np.uint8)
    
    txt_main = f"CHESS REKOGNITION - REPORT | {occ}/{total} OCUPADAS"
    txt_tech = f"AUTO-THRESH: STD > {round(stdt, 1)} | EDGE > {round(edget, 0)} | CROP: {int(INNER_CROP_PCT*100)}%"
    
    cv2.putText(footer, txt_main, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(footer, txt_tech, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    
    # Añadir marca de tiempo
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(footer, ts, (collage.shape[1]-150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    return np.vstack([collage, footer])


class VisionService:

    @staticmethod
    def detect_and_rectify(image_bytes: bytes) -> dict:
        """
        Pipeline completo:
        1. Decodificar imagen
        2. Detectar tablero (findChessboardCornersSB)
        3. Calcular esquinas exteriores
        4. Rectificar (homografía)
        5. Analizar 64 casillas
        6. Generar las 3 imágenes de respuesta
        """
        try:
            # 1. Decodificar
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"success": False,
                        "error": "No se pudo decodificar la imagen"}

            # 2. Detectar tablero
            found, corners = _detectar_tablero(frame)
            if not found:
                return {"success": False,
                        "error": "Tablero no detectado. Asegúrate de que "
                                 "el tablero esté bien encuadrado y con buena luz."}

            # 3. Esquinas exteriores
            exterior = _calcular_esquinas_exteriores(corners)

            # 4. Rectificar
            warped = _rectificar(frame, exterior)

            # 5. Analizar casillas
            squares, std_thresh_auto, edge_thresh_auto = _analizar_casillas(warped)
            occupied_count = sum(1 for s in squares if s["occupied"])

            # 6. Generar imágenes
            vista_real  = _generar_vista_real(warped, squares)
            vista_2d    = _generar_vista_2d(squares)
            debug_image = _generar_debug(frame, corners, exterior)
            
            collage = _generar_collage(
                debug_image, vista_real, vista_2d, 
                occupied_count, len(squares), std_thresh_auto, edge_thresh_auto
            )

            return {
                "success": True,
                "status": "OK",
                "rectified_real": _encode_image(vista_real),
                "rectified_2d":   _encode_image(vista_2d),
                "debug_image":    _encode_image(debug_image),
                "export_image":   _encode_image(collage),
                "squares":        squares,
                "occupied_count": occupied_count,
                "num_squares":    len(squares),
                "config": {
                    "std_thresh":  round(std_thresh_auto, 1),
                    "edge_thresh": round(edge_thresh_auto, 1),
                    "crop_pct":    INNER_CROP_PCT
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detail": traceback.format_exc()
            }
