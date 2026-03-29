import cv2
import numpy as np
import base64
from typing import Optional, List

class VisionService:
    # Constantes PRO
    BOARD_SIZE = 400
    CELL_SIZE = BOARD_SIZE // 8
    OCCUPIED_THRESHOLD = 35.0
    EDGE_THRESHOLD = 12.0

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def encode_image(image: np.ndarray, format: str = '.jpg') -> str:
        _, buffer = cv2.imencode(format, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape(4, 2)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left
        return rect

    @staticmethod
    def _is_valid_geometry(pts: np.ndarray) -> bool:
        """Verifica que las esquinas formen algo parecido a un cuadrado."""
        rect = VisionService.order_points(pts)
        w = np.linalg.norm(rect[0] - rect[1])
        h = np.linalg.norm(rect[1] - rect[2])
        ratio = w / (h + 1e-6)
        return 0.7 < ratio < 1.3 and w > 100

    @staticmethod
    def _fix_orientation_pro(board_img: np.ndarray) -> np.ndarray:
        """Usa una puntuación de patrón global para asegurar la orientación h1-blanca."""
        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        
        def get_pattern_score(img_gray):
            score = 0
            for r in range(8):
                for c in range(8):
                    crop = img_gray[r*50:(r+1)*50, c*50:(c+1)*50]
                    brightness = cv2.mean(crop)[0]
                    # En ajedrez estándar, (r+c) par es casilla clara
                    if (r + c) % 2 == 0:
                        score += brightness
                    else:
                        score -= brightness
            return score

        best_img = board_img
        max_score = -1e9
        
        # Probar las 4 rotaciones posibles
        for _ in range(4):
            current_score = get_pattern_score(gray)
            if current_score > max_score:
                max_score = current_score
                best_img = board_img.copy()
            
            board_img = cv2.rotate(board_img, cv2.ROTATE_90_CLOCKWISE)
            gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            
        return best_img

    @staticmethod
    def _draw_diagnostic_2d(squares: list) -> np.ndarray:
        """Reutilizamos tu tablero de diagnóstico que es muy útil."""
        board_2d = np.zeros((400, 400, 3), dtype=np.uint8)
        for sq in squares:
            x1, y1 = sq["col"] * 50, sq["row"] * 50
            x2, y2 = x1 + 50, y1 + 50
            is_light = (sq["row"] + sq["col"]) % 2 == 0
            bg_color = (210, 225, 235) if is_light else (110, 140, 160)
            cv2.rectangle(board_2d, (x1, y1), (x2, y2), bg_color, -1)
            
            # Marcador visual de ocupación
            color = (60, 60, 220) if sq["occupied"] else (60, 220, 60)
            cv2.rectangle(board_2d, (x1+8, y1+8), (x2-8, y2-8), color, 2)
            cv2.putText(board_2d, sq["id"], (x1+3, y2-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
        return board_2d

    @staticmethod
    def detect_and_rectify(image_bytes: bytes) -> dict:
        img = VisionService.decode_image(image_bytes)
        if img is None: return {"success": False, "error": "Imagen inválida"}

        # Preprocesado
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        
        # 1. DETECCIÓN
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(clahe_img, (7, 7), flags)

        if not found:
            return {"success": False, "error": "Tablero no detectado"}

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))

        # 2. HOMOGRAFÍA ROBUSTA (Usando los 49 puntos)
        # Generar puntos destino teóricos en una imagen de 400x400
        # Los 49 puntos internos del tablero (7x7) están en las intersecciones (1,1) a (7,7)
        # siendo cada casilla de 50x50.
        dst_pts = []
        for row in range(1, 8):
            for col in range(1, 8):
                dst_pts.append([col * 50, row * 50])
        dst_pts = np.array(dst_pts, dtype="float32")
        src_pts = corners_refined.reshape(-1, 2)

        # Usamos findHomography con RANSAC para filtrar posibles errores en esquinas
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        rectified = cv2.warpPerspective(img, M, (400, 400))
        
        # Orientación automática h1-blanca
        rectified = VisionService._fix_orientation_pro(rectified)

        # 3. ANÁLISIS DE CASILLAS
        squares = []
        col_letters = "abcdefgh"
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 50, row * 50
                x2, y2 = x1 + 50, y1 + 50
                
                # Tomar el centro de la casilla para evitar el borde de la rejilla
                # Con el homography de 49 puntos, este centro es muchisimo más preciso
                margin = 12 
                crop = rectified[y1+margin:y2-margin, x1+margin:x2-margin]
                
                std = float(crop.std())
                edges = cv2.Canny(crop, 30, 100)
                edge_density = float(edges.mean())
                
                # Umbrales algo más altos para evitar ruido en tableros con textura
                occupied = (std > 45.0 or edge_density > 15.0)
                
                square_id = f"{col_letters[col]}{8-row}"
                squares.append({
                    "id": square_id, "row": row, "col": col, 
                    "occupied": bool(occupied), "std": round(std, 2), "edges": round(edge_density, 2)
                })

        # 4. SALIDA VISUAL
        rectified_viz = rectified.copy()
        for sq in squares:
            x1, y1 = sq["col"] * 50, sq["row"] * 50
            x2, y2 = x1 + 50, y1 + 50
            color = (0, 0, 255) if sq["occupied"] else (0, 255, 0)
            cv2.rectangle(rectified_viz, (x1+1, y1+1), (x2-1, y2-1), color, 1)

        board_2d = VisionService._draw_diagnostic_2d(squares)
        combined = np.hstack([rectified_viz, board_2d])

        # Debug limpio en frame original (Dibujamos el contorno estimado)
        debug = img.copy()
        # Transformamos las esquinas exteriores teóricas de vuelta a la imagen original
        outer_corners = np.array([[0,0],[400,0],[400,400],[0,400]], dtype="float32").reshape(-1,1,2)
        M_inv = np.linalg.inv(M)
        outer_pts = cv2.perspectiveTransform(outer_corners, M_inv)
        cv2.polylines(debug, [np.int32(outer_pts)], True, (0, 255, 0), 3, cv2.LINE_AA)
        # Dibujar puntos detectados discretos (no la maraña de colores)
        for pt in src_pts:
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

        return {
            "success": True,
            "rectified_image": f"data:image/jpeg;base64,{VisionService.encode_image(combined)}",
            "debug_image": f"data:image/jpeg;base64,{VisionService.encode_image(debug)}",
            "num_squares": 64,
            "occupied_count": sum(1 for s in squares if s["occupied"]),
            "squares": squares,
            "config": {
                "std_thresh": 45.0,
                "edge_thresh": 15.0
            }
        }
