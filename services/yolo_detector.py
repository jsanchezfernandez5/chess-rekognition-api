"""
Servicio de detección de piezas y manos mediante un modelo YOLO26 entrenado (Ultralytics).

Motor principal de reconocimiento: localiza objetos reales (piezas blancas/negras y manos)
con sus bounding boxes sobre el tablero ya rectificado a 400x400.

Principio de coordenadas: TODO el trabajo ocurre sobre la vista cenital 400x400 producida por
OpenCV (nunca sobre el frame crudo de la cámara). El mapeo de una caja a su casilla se hace por
el CENTRO de la caja: fila = y_centro // CELL_SIZE, columna = x_centro // CELL_SIZE.

El import de Ultralytics es diferido a propósito: PyTorch es muy pesado y la API debe poder
arrancar aunque el paquete no esté instalado (en ese caso el detector queda simplemente
"no listo" y los endpoints devuelven un mensaje claro en lugar de romper).
"""
import threading

from core.config import settings


class YoloDetector:
    """Wrapper thread-safe alrededor del modelo YOLO entrenado.

    Patrón singleton + RLock para que pueda llamarse desde varios requests
    simultáneos sin condiciones de carrera.
    """

    def __init__(self):
        self.model = None
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------
    def _load(self):
        """Carga (o recarga) el modelo .pt si existe; si no, deja el detector 'no listo'."""
        with self._lock:
            self.model = None
            ruta_modelo = settings.YOLO_MODEL_PATH
            if not self._existe_modelo(ruta_modelo):
                print(f"[YoloDetector] No hay modelo entrenado en {ruta_modelo}. "
                      f"Entrena primero desde /yolo-dataset.")
                return

            YOLO = self._importar_ultralytics()
            if YOLO is None:
                print("[YoloDetector] El paquete 'ultralytics' no está instalado. "
                      "Instálalo (pip install ultralytics) para activar este motor.")
                return

            try:
                self.model = YOLO(ruta_modelo)
                print(f"[YoloDetector] Modelo YOLO cargado correctamente desde {ruta_modelo}")
            except Exception as e:
                self.model = None
                print(f"[YoloDetector] Error al cargar el modelo YOLO: {e}")

    @staticmethod
    def _existe_modelo(ruta_modelo) -> bool:
        import os
        return os.path.exists(ruta_modelo)

    @staticmethod
    def _importar_ultralytics():
        """Import diferido de Ultralytics; devuelve la clase YOLO o None si no está instalada."""
        try:
            from ultralytics import YOLO
            return YOLO
        except ImportError:
            return None

    def reload(self):
        """Recarga el modelo en caliente (tras un nuevo entrenamiento)."""
        self._load()

    def is_ready(self) -> bool:
        """Indica si hay un modelo cargado y utilizable."""
        with self._lock:
            return self.model is not None

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    def detect(self, warped):
        """Detecta piezas y manos sobre el tablero rectificado 400x400 (BGR de OpenCV).

        Devuelve una lista de detecciones:
            { "class_id": int, "class_name": str, "confidence": float,
              "bbox_px": [x, y, w, h] en píxeles, "bbox_norm": [xc, yc, w, h] normalizado 0-1 }

        Las coordenadas normalizadas son las nativas del formato YOLO; las de píxeles son las
        cómodas para dibujar overlays encima del tablero rectificado.
        """
        with self._lock:
            if not self.is_ready():
                raise RuntimeError("El modelo YOLO no está disponible.")

            resultados = self.model.predict(
                source=warped,                       # ndarray BGR: el formato nativo de OpenCV/Ultralytics
                conf=settings.YOLO_CONF_THRESHOLD,
                iou=settings.YOLO_IOU_THRESHOLD,
                imgsz=settings.YOLO_IMG_SIZE,
                device="cpu",                        # Sin GPU en producción (Railway)
                verbose=False,
            )

        detecciones = []
        alto, ancho = warped.shape[:2]
        for resultado in resultados:
            cajas = getattr(resultado, "boxes", None)
            if cajas is None:
                continue
            nombres = getattr(resultado, "names", {}) or {}
            for i in range(len(cajas)):
                class_id = int(cajas.cls[i].item())
                confidence = float(cajas.conf[i].item())
                xc_n, yc_n, w_n, h_n = [float(v) for v in cajas.xywhn[i].tolist()]
                detecciones.append({
                    "class_id": class_id,
                    "class_name": nombres.get(class_id, str(class_id)),
                    "confidence": round(confidence, 4),
                    "bbox_px": [
                        round((xc_n - w_n / 2) * ancho),
                        round((yc_n - h_n / 2) * alto),
                        round(w_n * ancho),
                        round(h_n * alto),
                    ],
                    "bbox_norm": [
                        round(xc_n, 4), round(yc_n, 4),
                        round(w_n, 4), round(h_n, 4),
                    ],
                })
        return detecciones


def boxes_a_board_state(detecciones: list):
    """Convierte detecciones YOLO en un board_state COMPLETO de 64 casillas.

    Cada caja se asigna a su casilla por el centro del bbox (fila = y // CELL_SIZE,
    columna = x // CELL_SIZE). Las casillas sin ninguna caja encima quedan como
    {"label": "empty", "confidence": None}: en detección de objetos la ausencia de caja ES la
    evidencia de casilla vacía, pero no existe una confianza numérica comparable con la de una
    pieza detectada, así que se marca como None.
    El formato de 64 casillas es obligatorio para poder reutilizar detect_move_desde_estado().

    Las detecciones de clase "hand" NO entran en el board_state (no son piezas); se devuelven
    aparte como señal de interferencia humana (mano/dedo sobre el tablero).

    Si varias cajas caen en la misma casilla gana la de mayor confianza; el resto se devuelve
    en una lista de conflictos para poder loguearlos como aviso (nunca como error).

    Returns:
        (board_state, hand_boxes, conflictos) donde:
          - board_state: las 64 casillas {"e4": {"label": "w_P", "confidence": 0.91}, ...}.
          - hand_boxes: lista de detecciones cuya clase es "hand".
          - conflictos: [{"square": "e4", "kept": "w_P@0.91", "discarded": ["b_P@0.30"]}, ...]
    """
    cell = settings.CELL_SIZE
    cols = settings.COLS

    # Estado base: 64 casillas vacías sin confianza numérica.
    board_state = {}
    for fila in range(8):
        for col in range(8):
            square = f"{cols[col]}{8 - fila}"  # La fila superior de la imagen es la fila 8 del tablero.
            board_state[square] = {"label": "empty", "confidence": None}

    hand_boxes = []
    casillas_por_deteccion = []  # (deteccion, nombre_casilla) para detectar duplicados después

    for det in detecciones:
        if det["class_name"] == "hand":
            hand_boxes.append(det)
            continue

        x, y, w, h = det["bbox_px"]
        col = min(max(int((x + w / 2) // cell), 0), 7)
        fila = min(max(int((y + h / 2) // cell), 0), 7)
        rank = 8 - fila
        square = f"{cols[col]}{rank}"
        casillas_por_deteccion.append((det, square))

    for det, square in casillas_por_deteccion:
        existente = board_state[square]
        if existente["label"] == "empty" or det["confidence"] > existente["confidence"]:
            board_state[square] = {
                "label": det["class_name"],
                "confidence": det["confidence"],
            }

    # Conflictos: casillas donde hubo más de una candidata.
    conflictos = []
    vistos = {}
    for det, square in casillas_por_deteccion:
        vistos.setdefault(square, []).append(det)
    for square, grupo in vistos.items():
        if len(grupo) > 1:
            ganadora = max(grupo, key=lambda d: d["confidence"])
            descartadas = [d for d in grupo if d is not ganadora]
            conflictos.append({
                "square": square,
                "kept": f"{ganadora['class_name']}@{ganadora['confidence']}",
                "discarded": [f"{d['class_name']}@{d['confidence']}" for d in descartadas],
            })
    return board_state, hand_boxes, conflictos


# Instancia singleton accesible desde cualquier módulo.
yolo_detector = YoloDetector()
