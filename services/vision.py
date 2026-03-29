import cv2
import numpy as np
import base64
from typing import Tuple, List, Optional

class VisionService:
    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Convierte bytes de imagen en una matriz OpenCV (BGR)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def encode_image(image: np.ndarray, format: str = '.jpg') -> str:
        """Convierte una matriz OpenCV en una cadena Base64"""
        _, buffer = cv2.imencode(format, image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    @staticmethod
    def find_board_corners(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Intenta encontrar las 4 esquinas del tablero de ajedrez.
        Utiliza una combinación de detección de bordes y búsqueda de contornos.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Preprocesamiento: desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Detección de bordes Canny
        edged = cv2.Canny(blurred, 50, 150)
        
        # 3. Dilatación y erosión para cerrar huecos en los bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edged, kernel, iterations=1)
        
        # 4. Encontrar contornos
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar contornos por área de mayor a menor
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            # Aproximar el contorno a un polígono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Si el polígono tiene 4 esquinas, asumimos que es el tablero
            if len(approx) == 4:
                return approx

        # Fallback: cv2.findChessboardCorners (útil si es un tablero de calibración o alto contraste)
        # Esto busca las ESQUINAS INTERNAS, no el borde exterior.
        # Pero podemos usarlo para inferir el borde.
        # Por ahora lo dejamos en None si no encuentra el contorno principal.
        return None

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Ordena las 4 esquinas en formato: TL, TR, BR, BL"""
        pts = pts.reshape((4, 2))
        rect = np.zeros((4, 2), dtype="float32")
        
        # Suma de x+y: TL tiene la mínima suma, BR la máxima suma
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Diferencia y-x: TR tiene la mínima diferencia, BL la máxima
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    @staticmethod
    def rectify_homography(image: np.ndarray, pts: np.ndarray, size: int = 800) -> np.ndarray:
        """Aplica la transformación de perspectiva para obtener una vista cenital del tablero."""
        rect = VisionService.order_points(pts)
        
        # Los puntos de destino serán un cuadrado perfecto
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype="float32")
        
        # Calcular matriz de homografía
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Aplicar transformación
        warped = cv2.warpPerspective(image, M, (size, size))
        
        return warped

    @staticmethod
    def detect_and_rectify(image_bytes: bytes) -> dict:
        """Pipeline completo: recibe bytes y devuelve imagen rectificada en base64"""
        img = VisionService.decode_image(image_bytes)
        if img is None:
            return {"success": False, "error": "No se pudo decodificar la imagen"}
            
        corners = VisionService.find_board_corners(img)
        
        if corners is None:
            return {"success": False, "error": "No se encontró el tablero en la imagen"}
            
        rectified = VisionService.rectify_homography(img, corners)
        img_b64 = VisionService.encode_image(rectified)
        
        # También dibujamos el contorno en la original para depuración
        cv2.drawContours(img, [corners], -1, (0, 255, 0), 2)
        orig_with_contour_b64 = VisionService.encode_image(img)
        
        return {
            "success": True,
            "rectified_image": f"data:image/jpeg;base64,{img_b64}",
            "debug_image": f"data:image/jpeg;base64,{orig_with_contour_b64}"
        }
