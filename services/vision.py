# services/vision.py
# Servicio de visión por computador para detectar el tablero de ajedrez y las piezas.
"""Módulo de visión por computador para la detección y rectificación del tablero de ajedrez.

Proporciona funciones para detectar el tablero mediante findChessboardCornersSB,
calcular sus esquinas exteriores, rectificar la perspectiva, analizar las 64 casillas
y generar imágenes de diagnóstico. La clase VisionService orquesta el pipeline completo."""
import cv2
import numpy as np
import base64
import traceback

# Constantes de configuración para el procesamiento de imágenes.
BOARD_SIZE = 400
CELL_SIZE = BOARD_SIZE // 8
COLS = "abcdefgh"
# Usar solo el 65% central de la casilla.
INNER_CROP_PCT = 0.65

# Función auxiliar para convertir una imagen numpy a base64.
# Se usa para enviar la imagen al frontend.
def _encode_image(img: np.ndarray) -> str:
    """Convierte una imagen en formato numpy array a una cadena base64 para su envío al frontend.

    Codifica la imagen en formato JPEG con calidad 85 y la empaqueta como data URI.

    Args:
        img: Imagen como array numpy (BGR).

    Returns:
        Cadena en formato data URI lista para usar en etiquetas img del frontend.
    """
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

# Función para detectar el tablero de ajedrez en una imagen.
# Usa findChessboardCornersSB para detectar las 49 esquinas internas.
def _detectar_tablero(frame: np.ndarray):
    """Detecta el tablero de ajedrez en una imagen y obtiene las 49 esquinas internas (7x7).

    Utiliza findChessboardCornersSB como método principal por su robustez ante perspectiva
    y desenfoque. Si falla, recurre a findChessboardCorners clásico. Las esquinas detectadas
    se refinan a nivel subpíxel para mayor precisión.

    Args:
        frame: Imagen BGR de entrada.

    Returns:
        Tupla (found, corners). Si se detecta el tablero, found es True y corners contiene
        las 49 coordenadas de las esquinas internas. En caso contrario, (False, None).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Intento principal (más robusto)
    found, corners = cv2.findChessboardCornersSB(
        gray, (7, 7),
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )

    if not found:
        # intento clásico
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

# Función para calcular las 4 esquinas exteriores del tablero a partir de las 49 esquinas internas.
# Usa vectores locales por esquina para mayor precisión cuando el tablero está en perspectiva.
def _calcular_esquinas_exteriores(corners: np.ndarray) -> np.ndarray:
    """Calcula las cuatro esquinas exteriores del tablero a partir de las 49 esquinas internas.

    Extrapola una casilla adicional hacia afuera usando vectores locales calculados
    para cada lado del tablero. Esto permite obtener el perímetro completo incluso
    cuando el tablero está en perspectiva.

    Args:
        corners: Array de 49 coordenadas de esquinas internas (salida de _detectar_tablero).

    Returns:
        Array numpy con las cuatro esquinas exteriores en orden: tl, tr, br, bl.
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

    # Devuelve las 4 esquinas exteriores del tablero en formato numpy array.
    return np.array(
        [board_tl, board_tr, board_br, board_bl],
        dtype=np.float32
    )

# Función para aplicar la homografía y obtener la vista cenital del tablero.
def _rectificar(frame: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    """Aplica una transformación de perspectiva (homografía) para obtener la vista cenital del tablero.

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

    # Calcula la matriz de transformación afín
    M = cv2.getPerspectiveTransform(exterior, dst)
    # Aplica la transformación afín a la imagen
    warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))
    return warped

# Función para calibrar los umbrales de detección de piezas.
# Calcula umbrales dinámicamente a partir del tablero rectificado.
def _calibrar_umbrales(warped: np.ndarray) -> tuple:
    """Calcula umbrales de detección de forma dinámica según la iluminación del tablero rectificado.

    Analiza la desviación estándar de cada casilla y aplica percentiles para separar
    casillas ocupadas de vacías. Se adapta automáticamente a cualquier condición de luz
    sin necesidad de calibración manual.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.

    Returns:
        Tupla (std_thresh, edge_thresh) con los umbrales calculados para desviación estándar
        y detección de bordes respectivamente.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    stds = []

    # Itera sobre cada casilla del tablero.
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

# Función para analizar las 64 casillas del tablero.
# Autocalibra los umbrales según la iluminación actual.
def _analizar_casillas(warped: np.ndarray) -> tuple:
    """Analiza las 64 casillas del tablero rectificado para determinar qué casillas están ocupadas.

    Autocalibra los umbrales según la iluminación actual mediante _calibrar_umbrales
    y clasifica cada casilla en ocupada o vacía usando la desviación estándar y el
    recuento de bordes de la porción central de cada casilla.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.

    Returns:
        Tupla (squares, std_thresh, edge_thresh) donde squares es una lista de diccionarios
        con los datos de cada casilla (id, fila, columna, ocupada, std, bordes, es_clara),
        y los dos umbrales calculados.
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

# Función para generar la vista cenital real del tablero con overlay verde/rojo sobre las casillas.
def _generar_vista_real(warped: np.ndarray, squares: list) -> np.ndarray:
    """Genera una vista cenital del tablero con overlay visual de ocupación sobre las casillas.

    Las casillas ocupadas se marcan en rojo y las vacías en verde, con bordes de colores
    y una cuadrícula superpuesta para facilitar la lectura.

    Args:
        warped: Imagen del tablero rectificado.
        squares: Lista de diccionarios con los resultados del análisis de casillas.

    Returns:
        Imagen BGR con el overlay de ocupación aplicado.
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

# Función para generar la vista 2D diagnóstica.
# Tablero sintético con colores clásicos de casillas y marcadores de ocupación.
def _generar_vista_2d(squares: list) -> np.ndarray:
    """Genera una vista de diagnóstico 2D con un tablero sintético de colores clásicos.

    Las casillas ocupadas se muestran como rectángulos rojos con la notación algebraica,
    y las vacías como círculos verdes. Proporciona una representación visual clara del
    estado de ocupación del tablero.

    Args:
        squares: Lista de diccionarios con los resultados del análisis de casillas.

    Returns:
        Imagen BGR sintética de 400x400 representando el tablero de diagnóstico.
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

# Función para generar imagen de debug.
# Frame original con los 49 puntos internos en rojo y el perímetro del tablero en verde.
def _generar_debug(frame: np.ndarray, corners: np.ndarray,
                   exterior: np.ndarray) -> np.ndarray:
    """Genera una imagen de depuración sobre el frame original con anotaciones visuales.

    Dibuja el perímetro del tablero en verde y las 49 esquinas internas detectadas
    como puntos rojos, permitiendo verificar visualmente la precisión de la detección.

    Args:
        frame: Imagen original BGR.
        corners: Coordenadas de las 49 esquinas internas (de _detectar_tablero).
        exterior: Cuatro esquinas exteriores del tablero (de _calcular_esquinas_exteriores).

    Returns:
        Imagen BGR con las anotaciones de depuración superpuestas.
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

# Función para generar un collage con las 3 vistas (debug, real, diagnóstico).
# Incluye metadatos técnicos para diagnóstico profundo.
def _generar_collage(debug: np.ndarray, real: np.ndarray, diag: np.ndarray, 
                     occ: int, total: int, stdt: float, edget: float) -> np.ndarray:
    """Compone un collage con las tres vistas principales del tablero y metadatos técnicos.

    Combina la imagen de depuración, la vista real con overlay y la vista 2D de diagnóstico
    en una sola imagen. Incluye un pie con información técnica (umbrales, ocupación, marca
    de tiempo) para facilitar la exportación y el diagnóstico.

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

# Clase principal VisionService con el método público para procesar imágenes.
# Contiene el pipeline completo de detección y rectificación del tablero.
class VisionService:

    @staticmethod
    def detect_and_rectify(image_bytes: bytes) -> dict:
        """Ejecuta el pipeline completo de detección y rectificación del tablero de ajedrez.

        El flujo de procesamiento incluye:
        1. Decodificar la imagen desde bytes.
        2. Detectar el tablero mediante findChessboardCornersSB.
        3. Calcular las esquinas exteriores del tablero.
        4. Rectificar la perspectiva mediante homografía.
        5. Analizar las 64 casillas para determinar ocupación.
        6. Generar las imágenes de respuesta (vista real, vista 2D, depuración, collage).

        Args:
            image_bytes: Datos binarios de la imagen de entrada.

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
