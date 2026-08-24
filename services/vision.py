"""
Servicio para la rectificación de la homografía con OpenCV para la detección y rectificación del tablero de ajedrez.

Proporciona funciones:
    - Detectar el tablero mediante findChessboardCornersSB.
    - Calcular sus esquinas exteriores.
    - Rectificar la perspectiva a vista cenital (homografía).
    - Analizar las 64 casillas y generar imágenes de diagnóstico. 
    
La clase VisionService orquesta el pipeline completo.
"""
import chess
import cv2
import numpy as np
import base64
import traceback

# Dimensiones y constantes del tablero
BOARD_SIZE = 400
CELL_SIZE = BOARD_SIZE // 8
COLS = "abcdefgh"

# Porcentaje del centro de cada casilla que se utiliza para el análisis (el 65% central)
INNER_CROP_PCT = 0.65

# Método auxiliar para codificar imágenes en formato base64 para el frontend
def _encode_image(img: np.ndarray) -> str:
    """
    Convierte una imagen en formato numpy array a una cadena base64 para su envío al frontend.

    Codifica la imagen en formato JPEG con calidad 85 y la empaqueta como data URI.

    Args:
        img: Imagen como array numpy (BGR).

    Returns:
        Cadena en formato data URI lista para usar en etiquetas img del frontend.
    """
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

# Ordena los cuatro puntos en el orden estándar: top-left, top-right, bottom-right, bottom-left
def _ordenar_puntos(pts: np.ndarray) -> np.ndarray:
    # Ordenamos los puntos por su coordenada X
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    # Los dos puntos de la izquierda y los dos de la derecha
    left_pts = x_sorted[:2, :]
    right_pts = x_sorted[2:, :]
    
    # Ordenamos los de la izquierda por la coordenada Y para obtener tl y bl
    left_sorted = left_pts[np.argsort(left_pts[:, 1]), :]
    tl = left_sorted[0]
    bl = left_sorted[1]
    
    # Ordenamos los de la derecha por la coordenada Y para obtener tr y br
    right_sorted = right_pts[np.argsort(right_pts[:, 1]), :]
    tr = right_sorted[0]
    br = right_sorted[1]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

# Detecta el contorno del tablero de ajedrez mediante detección de bordes
def _detectar_tablero_contorno(frame: np.ndarray) -> np.ndarray | None:
    # Convertir a escala de grises y suavizar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = frame.shape[:2]
    min_area = (h * w) * 0.15

    # Intento 1: Canny + Dilatación para conectar contornos
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) == 4:
            return _ordenar_puntos(approx.reshape(4, 2).astype(np.float32))

    # Intento 2: Umbralizado adaptativo
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) == 4:
            return _ordenar_puntos(approx.reshape(4, 2).astype(np.float32))

    return None

# Calcula un encuadre por defecto en el centro de la imagen
def _obtener_tablero_defecto(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    # Usamos el 82% de la dimensión menor para centrar el recuadro
    size = int(min(h, w) * 0.82)
    cx, cy = w // 2, h // 2
    half = size // 2
    
    tl = np.array([cx - half, cy - half], dtype=np.float32)
    tr = np.array([cx + half, cy - half], dtype=np.float32)
    br = np.array([cx + half, cy + half], dtype=np.float32)
    bl = np.array([cx - half, cy + half], dtype=np.float32)
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

# Método para detectar el tablero de ajedrez en una imagen
def _detectar_tablero(frame: np.ndarray):
    """
    Detecta el tablero de ajedrez en una imagen y obtiene las coordenadas del mismo.

    1. Intenta detectar las 49 esquinas internas mediante findChessboardCornersSB o findChessboardCorners.
    2. Si falla, intenta detectar el contorno cuadrangular externo más grande mediante OpenCV.
    3. Si ambos fallan, recurre a un encuadre por defecto (rectángulo centrado).
    """
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Intento principal con findChessboardCornersSB (más robusto ante perspectiva y desenfoque)
    h, w = gray.shape
    scale = 640 / max(h, w)
    if scale < 1:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        small = gray
        scale = 1

    found, corners = cv2.findChessboardCornersSB(
        small, (7, 7),
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        # Fallback al método clásico para tableros con poco contraste
        found, corners = cv2.findChessboardCorners(
            small, (7, 7),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
        )

    if found:
        corners = corners / scale
        # Refinamiento a nivel subpíxel para mayor precisión en la homografía
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners

    # Si falla la detección geométrica de esquinas, recurrimos a detectar el contorno exterior
    exterior_contour = _detectar_tablero_contorno(frame)
    if exterior_contour is not None:
        return True, exterior_contour

    # Si todo falla, generamos el encuadre por defecto (rectángulo centrado)
    default_exterior = _obtener_tablero_defecto(frame)
    return True, default_exterior

def _calcular_esquinas_exteriores(corners: np.ndarray) -> np.ndarray:
    """
    Calcula las cuatro esquinas exteriores del tablero a partir de las 49 esquinas internas o las devuelve directamente.

    Args:
        corners: Array de coordenadas.

    Returns:
        Array numpy con las cuatro esquinas exteriores en orden: tl, tr, br, bl.
    """
    # Normalizamos el formato de entrada: findChessboardCorners/findChessboardCornersSB
    # devuelven las esquinas como (N,1,2) en OpenCV 4.x pero como (N,2) en OpenCV 5.x.
    # Con reshape(-1,2) trabajamos siempre con N filas de puntos (x,y) float32.
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)

    # Si ya se calcularon las 4 esquinas exteriores directamente, las devolvemos sin alterar
    if len(pts) == 4:
        return pts

    # Extraemos las 4 esquinas internas de las 49 detectadas (7x7)
    tl = pts[0];   tr = pts[6]
    bl = pts[42];  br = pts[48]

    # Vector unitario por lado: 7 esquinas = 6 intervalos
    step_h_top    = (tr - tl) / 6.0
    step_h_bottom = (br - bl) / 6.0
    step_v_left   = (bl - tl) / 6.0
    step_v_right  = (br - tr) / 6.0

    # Extrapolación vectorial con margen 1.12 para cubrir el borde exterior del tablero
    MARGIN = 1.12
    board_tl = tl - step_h_top    * MARGIN - step_v_left   * MARGIN
    board_tr = tr + step_h_top    * MARGIN - step_v_right  * MARGIN
    board_bl = bl - step_h_bottom * MARGIN + step_v_left   * MARGIN
    board_br = br + step_h_bottom * MARGIN + step_v_right  * MARGIN
    return np.array(
        [board_tl, board_tr, board_br, board_bl],
        dtype=np.float32
    )

# Función para rectificar la perspectiva del tablero
def _rectificar(frame: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    """
    Aplica una transformación de perspectiva (homografía) para obtener la vista cenital del tablero.

    La imagen resultante se redimensiona a 400x400 píxeles, con cada casilla ocupando 50x50.

    Args:
        frame: Imagen original BGR.
        exterior: Cuatro esquinas exteriores del tablero (de _calcular_esquinas_exteriores).

    Returns:
        Imagen rectificada de 400x400 con el tablero en vista cenital.
    """
    dst = np.array([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE]
    ], dtype=np.float32)

    # Matriz de homografía que mapea la perspectiva original a cenital
    M = cv2.getPerspectiveTransform(exterior, dst)

    # Aplica la transformación de perspectiva
    warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))
    return warped

# Función para calibrar umbrales de detección
def _calibrar_umbrales(warped: np.ndarray) -> tuple:
    """
    Calcula umbrales de detección de forma dinámica según la iluminación del tablero rectificado.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.

    Returns:
        Tupla (std_thresh, edge_thresh) con los umbrales calculados para desviación estándar y detección de bordes.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    stds = []

    # Recorre cada casilla del tablero
    for row in range(8):
        for col in range(8):
            # Calcula la posición y extrae el 65% central de cada casilla
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cell = gray[y:y+CELL_SIZE, x:x+CELL_SIZE]
            h, w = cell.shape
            ch, cw = int(h * INNER_CROP_PCT), int(w * INNER_CROP_PCT)
            yo, xo = (h - ch) // 2, (w - cw) // 2
            stds.append(float(np.std(cell[yo:yo+ch, xo:xo+cw])))

    # Estadísticos robustos (percentiles) para separar ocupadas de vacías
    stds_arr = np.array(stds)
    p25  = float(np.percentile(stds_arr, 25))
    p75  = float(np.percentile(stds_arr, 75))
    spread = p75 - p25

    if spread < 10:
        # Tablero uniforme: umbral conservador basado en mediana + margen
        std_thresh = float(np.median(stds_arr)) + max(spread * 1.5, 15)
    else:
        # Con variación: punto de corte entre percentil 25 y el grupo alto
        std_thresh = p25 + spread * 0.8

    # Acotación segura para evitar falsos positivos/negativos extremos
    std_thresh = float(np.clip(std_thresh, 45, 120))
    edge_thresh = std_thresh * 18

    return std_thresh, edge_thresh

# Función para analizar las casillas del tablero
def _analizar_casillas(warped: np.ndarray) -> tuple:
    """
    Analiza las 64 casillas del tablero rectificado para determinar qué casillas están ocupadas.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.

    Returns:
        Tupla (squares, std_thresh, edge_thresh) dond: 
        - squares es una lista de diccionarios con los datos de cada casilla 
        (id, fila, columna, ocupada, std, bordes, es_clara) y
        - y los dos umbrales calculados.
    """
    # Calibra los umbrales de detección de forma dinámica
    std_thresh, edge_thresh = _calibrar_umbrales(warped)

    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Detecta los bordes de las piezas
    edges = cv2.Canny(gray, 50, 150)

    # Inicializa la lista de casillas
    squares = []

    # Itera sobre cada casilla del tablero
    for row in range(8):
        for col in range(8):
            x, y = col * CELL_SIZE, row * CELL_SIZE

            # Extrae la porción correspondiente de la imagen en escala de grises
            cell_gray = gray[y:y+CELL_SIZE, x:x+CELL_SIZE]

            # Extrae la porción correspondiente de la imagen con bordes detectados
            cell_edges = edges[y:y+CELL_SIZE, x:x+CELL_SIZE]

            # Calcula el tamaño de la porción interna a analizar (65% del tamaño de la casilla)
            h, w = cell_gray.shape
            ch, cw = int(h * INNER_CROP_PCT), int(w * INNER_CROP_PCT)
            y_off, x_off = (h - ch) // 2, (w - cw) // 2

            # Extrae la porción central de cada imagen
            inner_gray = cell_gray[y_off:y_off+ch, x_off:x_off+cw]
            inner_edges = cell_edges[y_off:y_off+ch, x_off:x_off+cw]

            # Calcula la desviación estándar de la porción central
            std = float(np.std(inner_gray))

            # Cuenta los bordes detectados en la porción central
            edge_count = int(np.sum(inner_edges > 0))
            
            # Determina si la casilla está ocupada
            occupied = std > std_thresh or edge_count > edge_thresh

            # Añade los datos de la casilla a la lista
            squares.append({
                "id": f"{COLS[col]}{8 - row}",
                "row": row,
                "col": col,
                "occupied": occupied,
                "std": round(std, 2),
                "edges": edge_count,
                "is_light": (row + col) % 2 == 0
            })

    # Devuelve los datos de las casillas y los umbrales calculados
    return squares, std_thresh, edge_thresh

# Función para generar la vista cenital del tablero
def _generar_vista_real(warped: np.ndarray, squares: list) -> np.ndarray:
    """
    Genera una vista cenital del tablero con overlay visual de ocupación sobre las casillas.

    - Las casillas ocupadas se marcan en rojo y las vacías en verde, 
    - con bordes de colores y una cuadrícula superpuesta para facilitar la lectura.

    Args:
        warped: Imagen del tablero rectificado.
        squares: Lista de diccionarios con los resultados del análisis de casillas.

    Returns:
        Imagen BGR con el overlay de ocupación aplicado.
    """
    # Copia la imagen warped
    output = warped.copy()

    # Itera sobre cada casilla del tablero
    for sq in squares:
        # Calcula la posición de la casilla
        x, y = sq["col"] * CELL_SIZE, sq["row"] * CELL_SIZE

        # Determina el color y la transparencia del overlay
        color = (0, 80, 0) if not sq["occupied"] else (0, 0, 180)
        alpha = 0.18

        # Crea un overlay con el color y la transparencia
        overlay = output.copy()

        # Dibujar el rectángulo de la casilla
        cv2.rectangle(overlay, (x+2, y+2),
                      (x+CELL_SIZE-2, y+CELL_SIZE-2), color, -1)

        # Combina el overlay con la imagen original
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Dibuja el borde de la casilla
        border_color = (0, 200, 80) if not sq["occupied"] else (60, 60, 220)
        cv2.rectangle(output, (x+2, y+2), (x+CELL_SIZE-2, y+CELL_SIZE-2), border_color, 1)

    # Dibuja una cuadrícula sobre el tablero
    for i in range(9):
        cv2.line(output, (i*CELL_SIZE, 0),
                 (i*CELL_SIZE, BOARD_SIZE), (80, 80, 80), 1)
        cv2.line(output, (0, i*CELL_SIZE),
                 (BOARD_SIZE, i*CELL_SIZE), (80, 80, 80), 1)

    # Devuelve la imagen con el overlay de ocupación aplicado
    return output

# Función para generar la vista 2D del tablero
def _generar_vista_2d(squares: list) -> np.ndarray:
    """
    Genera una vista de diagnóstico 2D con un tablero sintético de colores clásicos.

    Las casillas ocupadas se muestran como rectángulos rojos con la notación algebraica 
    y las vacías como círculos verdes. 

    Args:
        squares: Lista de diccionarios con los resultados del análisis de casillas.

    Returns:
        Imagen BGR sintética de 400x400 representando el tablero de diagnóstico.
    """
    # Crea una imagen sintética de 400x400 representando el tablero
    output = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    # Definición de colores
    COLOR_LIGHT = (210, 200, 180)   # Casillas claras
    COLOR_DARK  = (100, 70,  50)   # Casillas oscuras
    COLOR_OCC   = (60,  60,  220)   # Casillas ocupadas
    COLOR_FREE  = (40,  180, 80)   # Casillas vacías

    # Itera sobre cada casilla del tablero
    for sq in squares:
        # Calcula la posición de la casilla
        x, y = sq["col"] * CELL_SIZE, sq["row"] * CELL_SIZE

        # Determina el color base según si la casilla es clara u oscura
        base = COLOR_LIGHT if sq["is_light"] else COLOR_DARK

        # Dibujar el rectángulo de la casilla
        cv2.rectangle(output, (x, y), (x+CELL_SIZE, y+CELL_SIZE), base, -1)

        # Dibujar el contenido de la casilla
        if sq["occupied"]:
            cv2.rectangle(output, (x+4, y+4),
                          (x+CELL_SIZE-4, y+CELL_SIZE-4), COLOR_OCC, -1)
            cv2.putText(output, sq["id"], (x+6, y+CELL_SIZE-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        else:
            # Dibuja un círculo verde en el centro de la casilla vacía
            cv2.circle(output,
                       (x + CELL_SIZE//2, y + CELL_SIZE//2),
                       4, COLOR_FREE, -1)

    # Dibuja una cuadrícula fina sobre el tablero
    for i in range(9):
        cv2.line(output, (i*CELL_SIZE, 0), (i*CELL_SIZE, BOARD_SIZE), (50, 50, 50), 1)
        cv2.line(output, (0, i*CELL_SIZE), (BOARD_SIZE, i*CELL_SIZE), (50, 50, 50), 1)

    return output

# Función para generar una imagen de depuración
def _generar_debug(frame: np.ndarray, corners: np.ndarray,
                   exterior: np.ndarray) -> np.ndarray:
    """
    Genera una imagen de depuración sobre el frame original con anotaciones visuales.

    Args:
        frame: Imagen original BGR.
        corners: Coordenadas de las 49 esquinas internas (de _detectar_tablero).
        exterior: Cuatro esquinas exteriores del tablero (de _calcular_esquinas_exteriores).

    Returns:
        Imagen BGR con las anotaciones de depuración superpuestas.
    """
    # Copia la imagen frame
    debug = frame.copy()

    # Dibuja el perímetro del tablero en verde
    pts = exterior.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(debug, [pts], True, (0, 220, 80), 3)

    # 49 esquinas internas en rojo (si están disponibles). El reshape(-1,2) unifica el
    # formato (N,1,2) de OpenCV 4.x y (N,2) de OpenCV 5.x.
    if corners is not None and len(corners) == 49:
        for x, y in np.asarray(corners).reshape(-1, 2):
            cv2.circle(debug, (int(x), int(y)), 4, (0, 0, 220), -1)
    elif corners is not None and len(corners) == 4:
        for x, y in np.asarray(corners).reshape(-1, 2):
            cv2.circle(debug, (int(x), int(y)), 6, (0, 0, 220), -1)

    return debug

# Función para generar un collage con las tres vistas principales del tablero
def _generar_collage(
    debug: np.ndarray, 
    real: np.ndarray, 
    diag: np.ndarray, 
    occ: int, 
    total: int, 
    stdt: float, 
    edget: float
) -> np.ndarray:
    """
    Crea un collage con las tres vistas principales del tablero y metadatos técnicos.

    Args:
        debug: Imagen de depuración con anotaciones.
        real: Vista cenital real con overlay de ocupación.
        diag: Vista 2D de diagnóstico sintética.
        occ: Número de casillas ocupadas detectadas.
        total: Número total de casillas (64).
        stdt: Umbral de desviación estándar utilizado.
        edget: Umbral de detección de bordes utilizado.

    Returns:
        Imagen BGR del collage completo con pie informativo.
    """
    h_main = 450
    w_main = int(debug.shape[1] * (h_main / debug.shape[0]))
    debug_res = cv2.resize(debug, (w_main, h_main))

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
    
    # Retorna el collage
    return np.vstack([collage, footer])

# Función auxiliar para convertir etiquetas de piezas
def label_to_piece(label: str) -> chess.Piece | None:
    """
    Convierte una etiqueta del clasificador ('w_P', 'b_N', 'empty') a chess.Piece.
    """
    if label == "empty":
        return None
    
    # Obtiene el color y el símbolo de la pieza
    color = chess.WHITE if label.startswith("w_") else chess.BLACK
    symbol = label.split("_")[1]

    # Devuelve la pieza
    return chess.Piece.from_symbol(symbol if color == chess.WHITE else symbol.lower())

# Función auxiliar para convertir el estado del tablero a FEN
def board_state_to_fen_board(board_state: dict) -> chess.Board:
    """Convierte un dict {sq_id: label} a un objeto chess.Board."""
    board = chess.Board(fen=None)
    for sq_name, label in board_state.items():
        piece = label_to_piece(label)
        if piece:
            board.set_piece_at(chess.parse_square(sq_name), piece)
    return board

# Clase VisionService
class VisionService:

    @staticmethod
    def obtener_exterior_y_corners(frame: np.ndarray, coords: str = None) -> tuple:
        """
        Obtiene las esquinas exteriores y los puntos de depuración (corners),
        ya sea a partir de coordenadas manuales en formato 'x1,y1,x2,y2,x3,y3,x4,y4'
        (escaladas como porcentajes 0.0-1.0 de las dimensiones del frame) o mediante
        el pipeline automático de detección.
        """
        if coords:
            try:
                parts = [float(x) for x in coords.split(",")]
                if len(parts) == 8:
                    h, w = frame.shape[:2]
                    pts = np.array([
                        [parts[0] * w, parts[1] * h],
                        [parts[2] * w, parts[3] * h],
                        [parts[4] * w, parts[5] * h],
                        [parts[6] * w, parts[7] * h]
                    ], dtype=np.float32)
                    exterior = _ordenar_puntos(pts)
                    return True, exterior, exterior
            except Exception:
                pass
        
        found, corners = _detectar_tablero(frame)
        if not found:
            return False, None, None
        exterior = _calcular_esquinas_exteriores(corners)
        return True, exterior, corners

    @staticmethod
    def detect_and_rectify(image_bytes: bytes, coords: str = None, rotation: int = 0) -> dict:
        """
        Ejecuta el pipeline completo de detección y rectificación del tablero de ajedrez.

        El flujo de procesamiento incluye:
        1. Decodificar la imagen desde bytes.
        2. Detectar el tablero mediante findChessboardCornersSB o usando coordenadas manuales.
        3. Calcular las esquinas exteriores del tablero.
        4. Rectificar la perspectiva mediante homografía.
        5. Aplicar rotación a la imagen rectificada si es necesario.
        6. Analizar las 64 casillas para determinar ocupación.
        7. Generar las imágenes de respuesta (vista real, vista 2D, depuración, collage).

        Args:
            image_bytes: Datos binarios de la imagen de entrada.
            coords: Coordenadas manuales en formato 'x1,y1,x2,y2,x3,y3,x4,y4' (porcentajes).
            rotation: Rotación de la cámara (0, 90, 180, 270 grados).

        Returns:
            Diccionario con los resultados del pipeline. Incluye las claves:
            success (bool), status (str), rectified_real, rectified_2d, debug_image,
            export_image (cadenas base64), squares (lista de casillas),
            occupied_count, num_squares y config (umbrales calculados).
            En caso de error, success es False e incluye error y detail.
        """
        try:
            # 1. Decodificar
            arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"success": False,
                        "error": "No se pudo decodificar la imagen"}

            # 2 y 3. Obtener esquinas exteriores del tablero
            found, exterior, corners = VisionService.obtener_exterior_y_corners(frame, coords)
            if not found:
                return {"success": False,
                        "error": "Tablero no detectado. Asegúrate de que "
                                 "el tablero esté bien encuadrado y con buena luz."}

            # 4. Rectificar
            warped = _rectificar(frame, exterior)

            # Aplicar rotación si es necesario
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
                # Tablero rectificado LIMPIO (sin overlays de diagnóstico). Lo usa la herramienta
                # /yolo-dataset para guardar imágenes de entrenamiento sin marcas rojas/verdes encima.
                "rectified_clean": _encode_image(warped),
                "squares":        squares,
                "corners":        [[float(pt[0]) / frame.shape[1], float(pt[1]) / frame.shape[0]] for pt in exterior],
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
