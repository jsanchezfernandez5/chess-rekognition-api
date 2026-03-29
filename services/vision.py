import cv2
import numpy as np
import base64

class VisionService:
    # --- PERSISTENCIA Y OPTIMIZACIÓN (MEMORIA PRO) ---
    _LAST_VALID_M = None
    _LAST_SRC_PTS = None
    _LAST_ROTATION_CODE = None # Guarda el código de rotación (90, 180, 270)
    _MEMORY_COUNT = 0          # Contador de frames desde la última detección activa
    _MAX_MEMORY_FRAMES = 30    # TTL: 1 segundo a 30fps (Previene deriva si mueves la cámara)
    _SMOOTHING_ALPHA = 0.25    # Factor de suavizado EMA

    # Constantes PRO para análisis de casillas
    BOARD_SIZE = 400
    CELL_SIZE = BOARD_SIZE // 8
    OCCUPIED_THRESHOLD = 45.0
    EDGE_THRESHOLD = 15.0

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def encode_image(image: np.ndarray, format: str = '.jpg') -> str:
        _, buffer = cv2.imencode(format, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def _fix_orientation_pro(board_img: np.ndarray, use_cache: bool = False) -> (np.ndarray, int):
        """
        Cálculo inteligente de la orientación h1-blanca.
        Si use_cache=True y ya conocemos el giro, saltamos el proceso de escaneo (Ahorro CPU).
        """
        if use_cache and VisionService._LAST_ROTATION_CODE is not None:
            # Si el código es 0 (None en CV2), no rotamos, sino aplicamos el giro guardado.
            if VisionService._LAST_ROTATION_CODE == -1: return board_img
            return cv2.rotate(board_img, VisionService._LAST_ROTATION_CODE)

        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        
        def get_pattern_score(img_gray):
            score = 0
            for r in range(8):
                for c in range(8):
                    crop = img_gray[r*50:(r+1)*50, c*50:(c+1)*50]
                    brightness = cv2.mean(crop)[0]
                    if (r + c) % 2 == 0: score += brightness
                    else: score -= brightness
            return score

        best_img = board_img
        best_code = -1 # -1 Representa "sin rotación necesaria"
        max_score = -1e9
        
        # Códigos de rotación de OpenCV: 0=90CW, 1=180, 2=270CW
        rotation_codes = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        
        temp_img = board_img.copy()
        temp_gray = gray.copy()

        for i, code in enumerate(rotation_codes):
            if code is not None:
                temp_img = cv2.rotate(board_img, code)
                temp_gray = cv2.rotate(gray, code)
            
            current_score = get_pattern_score(temp_gray)
            if current_score > max_score:
                max_score = current_score
                best_img = temp_img.copy()
                best_code = code if code is not None else -1
            
        # Guardamos en caché el código ganador para el modo memoria
        VisionService._LAST_ROTATION_CODE = best_code
        return best_img

    @staticmethod
    def detect_and_rectify(image_bytes: bytes) -> dict:
        img = VisionService.decode_image(image_bytes)
        if img is None: return {"success": False, "error": "Imagen inválida"}

        status_msg = "ACTIVO"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        
        # 1. INTENTO DE DETECCIÓN ACTIVA (7x7 internos)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(clahe_img, (7, 7), flags)

        if found:
            VisionService._MEMORY_COUNT = 0 # Reset del TTL
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
            
            src_pts = corners_refined.reshape(-1, 2)
            dst_pts = np.array([[c*50, r*50] for r in range(1,8) for c in range(1,8)], dtype="float32")
            
            M_new, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if VisionService._LAST_VALID_M is not None:
                alpha = VisionService._SMOOTHING_ALPHA
                M = alpha * M_new + (1 - alpha) * VisionService._LAST_VALID_M
            else:
                M = M_new
            
            VisionService._LAST_VALID_M = M
            VisionService._LAST_SRC_PTS = src_pts
            needs_reorientation = True # Detección fresca, chequeamos giro completo
            status_msg = "DETECCIÓN OK"
            
        else:
            # 2. FALLBACK A MEMORIA CON TTL (Time-To-Live)
            VisionService._MEMORY_COUNT += 1
            if VisionService._LAST_VALID_M is not None and VisionService._MEMORY_COUNT < VisionService._MAX_MEMORY_FRAMES:
                M = VisionService._LAST_VALID_M
                src_pts = VisionService._LAST_SRC_PTS
                needs_reorientation = False # REUTILIZAMOS GIRO (Cache) - Gran ahorro CPU
                status_msg = f"MODO MEMORIA ({VisionService._MAX_MEMORY_FRAMES - VisionService._MEMORY_COUNT})"
            else:
                # Si caduca el TTL o no hay memoria, limpiamos todo
                VisionService._LAST_VALID_M = None
                VisionService._LAST_ROTATION_CODE = None
                return {"success": False, "error": "Tablero perdido o movido (Re-calibra)"}

        # 3. RECTIFICACIÓN Y ORIENTACIÓN OPTIMIZADA
        rectified = cv2.warpPerspective(img, M, (400, 400))
        # Si estamos en modo memoria, usamos el ángulo cacheado (use_cache=True)
        rectified = VisionService._fix_orientation_pro(rectified, use_cache=(not needs_reorientation))

        # 4. ANÁLISIS DE CASILLAS
        squares = []
        col_letters = "abcdefgh"
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 50, row * 50
                x2, y2 = x1 + 50, y1 + 50
                margin = 12 
                crop = rectified[y1+margin:y2-margin, x1+margin:x2-margin]
                
                std = float(crop.std())
                edges = cv2.Canny(crop, 30, 100)
                edge_density = float(edges.mean())
                
                occupied = (std > VisionService.OCCUPIED_THRESHOLD or edge_density > VisionService.EDGE_THRESHOLD)
                
                square_id = f"{col_letters[col]}{8-row}"
                squares.append({
                    "id": square_id, "row": row, "col": col, 
                    "occupied": bool(occupied), "std": round(std, 2), "edges": round(edge_density, 2)
                })

        # 5. SALIDA VISUAL SEGMENTADA
        rectified_plain = rectified.copy()
        board_2d = VisionService._draw_diagnostic_2d(squares)
        debug = img.copy()
        
        # Dibujar UI de diagnóstico sobre el frame original
        outer_corners = np.array([[0,0],[400,0],[400,400],[0,400]], dtype="float32").reshape(-1,1,2)
        M_inv = np.linalg.inv(M)
        outer_pts = cv2.perspectiveTransform(outer_corners, M_inv)
        
        line_color = (0, 255, 0) if found else (0, 165, 255)
        cv2.polylines(debug, [np.int32(outer_pts)], True, line_color, 2, cv2.LINE_AA)
        for pt in src_pts:
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

        return {
            "success": True,
            "rectified_real": f"data:image/jpeg;base64,{VisionService.encode_image(rectified_plain)}",
            "rectified_2d": f"data:image/jpeg;base64,{VisionService.encode_image(board_2d)}",
            "debug_image": f"data:image/jpeg;base64,{VisionService.encode_image(debug)}",
            "num_squares": 64,
            "occupied_count": sum(1 for s in squares if s["occupied"]),
            "squares": squares,
            "status": status_msg,
            "config": {
                "std_thresh": VisionService.OCCUPIED_THRESHOLD,
                "edge_thresh": VisionService.EDGE_THRESHOLD
            }
        }
