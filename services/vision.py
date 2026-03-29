import cv2
import numpy as np
import base64
from typing import Optional, List

class VisionService:
    """
    SERVICIO DE VISIÓN - DETECCIÓN ROBUSTA DE TABLERO (VISIÓN CLÁSICA)
    
    Implementa un pipeline avanzado para:
    1. Normalización de iluminación (CLAHE)
    2. Detección de bordes adaptativa (Auto-Canny)
    3. Detección de líneas (Transformada de Hough)
    4. Inferencia de rejilla e intersecciones
    5. Rectificación de perspectiva cential
    6. División en 64 casillas
    """

    # =========================
    # UTILIDADES BÁSICAS
    # =========================
    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Convierte bytes de imagen en matriz OpenCV (BGR)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def encode_image(image: np.ndarray, format: str = '.jpg') -> str:
        """Convierte matriz OpenCV en cadena Base64."""
        _, buffer = cv2.imencode(format, image)
        return base64.b64encode(buffer).decode('utf-8')

    # =========================
    # PREPROCESADO
    # =========================
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Aplica escala de grises, CLAHE y desenfoque gausiano."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CLAHE para normalizar contrastes locales
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Reducción de ruido
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    @staticmethod
    def auto_canny(image: np.ndarray, sigma=0.33):
        """Detección de bordes Canny con umbrales calculados automáticamente."""
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)

    # =========================
    # DETECCIÓN DE LÍNEAS
    # =========================
    @staticmethod
    def detect_lines(edges: np.ndarray):
        """Usa HoughLinesP para encontrar segmentos de línea rectos."""
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=20
        )

    @staticmethod
    def split_lines(lines):
        """Separa las líneas detectadas en horizontales y verticales según su ángulo."""
        horizontals, verticals = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            # Cerca de 0 radianes -> Horizontal
            if angle < np.pi / 6:
                horizontals.append(line[0])
            # Cerca de PI/2 radianes -> Vertical
            elif angle > np.pi / 3:
                verticals.append(line[0])
        return horizontals, verticals

    # =========================
    # INTERSECCIONES
    # =========================
    @staticmethod
    def line_intersection(l1, l2):
        """Calcula el punto de intersección entre dos segmentos de línea."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0:
            return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return [int(px), int(py)]

    # =========================
    # DETECCIÓN DEL TABLERO
    # =========================
    @staticmethod
    def find_board_corners(image: np.ndarray) -> Optional[np.ndarray]:
        """Algoritmo principal basado en líneas para encontrar las esquinas del tablero."""
        gray = VisionService.preprocess(image)
        edges = VisionService.auto_canny(gray)
        lines = VisionService.detect_lines(edges)
        
        if lines is None:
            return None

        h_lines, v_lines = VisionService.split_lines(lines)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Encontrar todas las intersecciones posibles
        points = []
        for h in h_lines:
            for v in v_lines:
                pt = VisionService.line_intersection(h, v)
                if pt is not None:
                    points.append(pt)

        if len(points) < 20: # Un tablero 8x8 genera muchas intersecciones
            return None

        pts = np.array(points)
        # Extraer los límites extremos
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # --- FILTROS DE SEGURIDAD ---
        
        # 1. Filtro de Tamaño Mínimo (no detectar motas de polvo)
        image_area = image.shape[0] * image.shape[1]
        board_area = width * height
        if board_area < (image_area * 0.05): # Al menos 5% de la imagen
            return None
            
        # 2. Filtro de Aspecto (debe ser casi un cuadrado)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 1.5: # Si un lado es 1.5 veces más largo que el otro, no es un tablero
            return None
            
        # 3. Filtro de Densidad (los puntos deben estar distribuidos)
        # Si todos los puntos están en una línea fina, no es una rejilla
        if width < 50 or height < 50:
            return None

        # Si pasa los filtros, devolvemos las 4 esquinas calculadas
        return np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype="float32")

    # =========================
    # FALLBACK
    # =========================
    @staticmethod
    def fallback_chessboard(gray):
        """Plan B: Usa la función nativa de OpenCV si el método de líneas falla."""
        found, corners = cv2.findChessboardCorners(gray, (7, 7))
        if not found:
            return None
        corners = corners.reshape(-1, 2)
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)
        return np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype="float32")

    # =========================
    # HOMOGRAFÍA
    # =========================
    @staticmethod
    def order_points(pts):
        """Ordena 4 puntos en: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape(4, 2)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def rectify(image, pts, size=800):
        """Corrige la perspectiva de la imagen para dejar el tablero cuadrado."""
        rect = VisionService.order_points(pts)
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (size, size))

    # =========================
    # DIVISIÓN 8x8
    # =========================
    @staticmethod
    def split_into_squares(image: np.ndarray) -> List[np.ndarray]:
        """Divide el tablero rectificado de 800x800 en 64 imágenes de casillas individuales."""
        h, w = image.shape[:2]
        squares = []
        step_x = w // 8
        step_y = h // 8
        for i in range(8):
            for j in range(8):
                square = image[
                    i * step_y:(i + 1) * step_y,
                    j * step_x:(j + 1) * step_x
                ]
                squares.append(square)
        return squares

    # =========================
    # PIPELINE FINAL
    # =========================
    @staticmethod
    def detect_and_rectify(image_bytes: bytes) -> dict:
        """Punto de entrada principal para la API."""
        img = VisionService.decode_image(image_bytes)
        if img is None:
            return {"success": False, "error": "Imagen inválida."}

        # Redimensionado para optimizar rendimiento si la imagen es enorme
        max_dim = 1000
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Intento 1: Detección por líneas (Hough)
        corners = VisionService.find_board_corners(img)

        # Intento 2: Fallback si el primero falla
        if corners is None:
            gray = VisionService.preprocess(img)
            corners = VisionService.fallback_chessboard(gray)

        if corners is None:
            return {"success": False, "error": "No se detectó el tablero de ajedrez."}

        # Rectificación y corte
        rectified = VisionService.rectify(img, corners)
        squares = VisionService.split_into_squares(rectified)

        # Generar imagen de depuración con el polígono dibujado
        debug = img.copy()
        cv2.polylines(debug, [corners.astype(int)], True, (0, 255, 0), 2)

        return {
            "success": True,
            "rectified_image": f"data:image/jpeg;base64,{VisionService.encode_image(rectified)}",
            "debug_image": f"data:image/jpeg;base64,{VisionService.encode_image(debug)}",
            "num_squares": len(squares)
        }
