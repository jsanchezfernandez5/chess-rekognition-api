"""
Evaluación del motor de reconocimiento YOLO26 sobre el dataset de validación.

Evalúa el detector YOLO26 sobre las imágenes de VALIDACIÓN del dataset YOLO
(yolo_dataset/images/val + labels/val, que NUNCA participan en el entrenamiento):

    - square_accuracy: % de casillas bien leídas (las 64 por imagen).
    - fen_exact:       % de tableros completos sin ni un solo error (lo que de verdad importa:
                         un solo fallo invalida la detección del movimiento).
    - piece_precision / piece_recall: calidad centrada SOLO en casillas ocupadas.
    - avg_ms:          tiempo medio de inferencia por imagen.

Este módulo solo informa (nunca cambia configuración).

⚠️ NOTA METODOLÓGICA IMPORTANTE: las etiquetas de validación proceden de anotación
semi-automática, así que la puntuación puede estar sesgada al alza en la medida en que
las predicciones originales se aceptaron sin corrección. Conforme se corrijan cajas a mano,
el sesgo disminuye. Para una evaluación imparcial a largo plazo conviene ir curando a mano
un subconjunto de validación.
"""
import os
import time

from core.config import settings


def _cargar_ground_truth(dataset_dir):
    """Lee imágenes y etiquetas de validación y las convierte en (nombre, board_state_verdad).

    Las etiquetas .txt están en formato YOLO normalizado 0-1; para reutilizar EXACTAMENTE la
    misma lógica de mapeo caja→casilla que usa la inferencia, convertimos cada línea al mismo
    formato de detección que produce yolo_detector.detect() y pasamos por boxes_a_board_state().

    Returns:
        Lista de dicts {"file": str, "board_state": dict_64_casillas}.
    """
    # Import diferido para no cargar ultralytics solo por importar este módulo.
    from services.yolo_detector import boxes_a_board_state

    img_dir = os.path.join(dataset_dir, "images", "val")
    lbl_dir = os.path.join(dataset_dir, "labels", "val")
    if not os.path.isdir(img_dir):
        return []

    muestras = []
    clases = settings.YOLO_CLASSES
    for nombre in sorted(os.listdir(img_dir)):
        if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        ruta_lbl = os.path.join(lbl_dir, os.path.splitext(nombre)[0] + ".txt")
        if not os.path.exists(ruta_lbl):
            continue  # Sin etiqueta no hay verdad terrena: imagen inservible para evaluar.

        detecciones_gt = []
        with open(ruta_lbl, encoding="utf-8") as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) != 5:
                    continue
                class_id = int(partes[0])
                xc_n, yc_n, w_n, h_n = [float(p) for p in partes[1:]]
                if class_id < 0 or class_id >= len(clases):
                    continue
                detecciones_gt.append({
                    "class_name": clases[class_id],
                    "confidence": 1.0,
                    # bbox_px sobre el lienzo virtual 400x400 (las val ya están rectificadas).
                    "bbox_px": [
                        round((xc_n - w_n / 2) * settings.BOARD_SIZE),
                        round((yc_n - h_n / 2) * settings.BOARD_SIZE),
                        round(w_n * settings.BOARD_SIZE),
                        round(h_n * settings.BOARD_SIZE),
                    ],
                })

        board_state, _, _ = boxes_a_board_state(detecciones_gt)
        muestras.append({"file": nombre, "board_state": board_state})
    return muestras


def _leer_imagen(ruta):
    """Carga una imagen de validación como ndarray BGR (OpenCV)."""
    import cv2
    img = cv2.imread(ruta)
    return img


def _metricas(board_pred, board_gt):
    """Compara dos board_states de 64 casillas y devuelve los contadores crudos."""
    aciertos = 0
    tp = fp = fn = 0
    for square, gt in board_gt.items():
        pred_label = board_pred.get(square, {}).get("label", "empty")
        gt_label = gt["label"]
        if pred_label == gt_label:
            aciertos += 1
        if gt_label != "empty":
            if pred_label == gt_label:
                tp += 1
            else:
                fn += 1
        elif pred_label != "empty":
            fp += 1
    return {"aciertos": aciertos, "total": len(board_gt), "tp": tp, "fp": fp, "fn": fn}


def _resumen(acumulado, n_imgs, ms_total):
    """Agrega los contadores crudos en las métricas finales de YOLO."""
    total_casillas = max(acumulado["total"], 1)
    piezas_gt = acumulado["tp"] + acumulado["fn"]
    return {
        "motor": "yolo",
        "square_accuracy": round(acumulado["aciertos"] / total_casillas, 4),
        "fen_exact": round(acumulado["boards_ok"] / max(n_imgs, 1), 4),
        "piece_precision": round(acumulado["tp"] / max(acumulado["tp"] + acumulado["fp"], 1), 4),
        "piece_recall": round(acumulado["tp"] / max(piezas_gt, 1), 4),
        "avg_ms": round(ms_total / max(n_imgs, 1)),
    }


def evaluar_motor(max_images=100):
    """Ejecuta la evaluación del motor YOLO26 sobre la partición de validación.

    ⚠️ RENDIMIENTO EN PRODUCCIÓN: esta evaluación corre DE FORMA SÍNCRONA dentro de la
    petición HTTP, con inferencia YOLO sobre CPU (Railway no tiene GPU). Con max_images
    alto puede tardar varios MINUTOS y el proxy/gateway puede cortar la conexión por
    timeout antes de recibir la respuesta. Se recomienda mantener valores moderados
    (50-100 imágenes) hasta que, si hace falta, se migre a un patrón de ejecución en
    segundo plano con polling de progreso.

    Args:
        max_images: límite de imágenes a evaluar (la validación puede crecer mucho y esto
                    corre sobre CPU en producción).

    Returns:
        Dict con las métricas de YOLO, la nota metodológica sobre el sesgo semi-automático
        y una nota de rendimiento para producción.
    """
    from services.yolo_detector import yolo_detector, boxes_a_board_state

    muestras = _cargar_ground_truth(settings.YOLO_DATASET_DIR)[:max_images]
    if not muestras:
        return {"success": False,
                "error": "No hay imágenes de validación etiquetadas en yolo_dataset/images/val."}

    yolo_listo = yolo_detector.is_ready()
    if not yolo_listo:
        return {"success": False, "error": "Modelo YOLO no disponible."}

    acumulado = {"aciertos": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "boards_ok": 0}
    tiempo_total = 0.0
    n_evaluadas = 0

    for muestra in muestras:
        ruta = os.path.join(settings.YOLO_DATASET_DIR, "images", "val", muestra["file"])
        warped = _leer_imagen(ruta)
        if warped is None:
            continue
        n_evaluadas += 1
        gt = muestra["board_state"]

        # Inferencia YOLO
        t0 = time.perf_counter()
        detecciones = yolo_detector.detect(warped)
        board_yolo, _, _ = boxes_a_board_state(detecciones)
        tiempo_total += (time.perf_counter() - t0) * 1000

        m = _metricas(board_yolo, gt)
        for k in ("aciertos", "total", "tp", "fp", "fn"):
            acumulado[k] += m[k]
        if m["aciertos"] == m["total"]:
            acumulado["boards_ok"] += 1

    return {
        "success": True,
        "n_images": n_evaluadas,
        "engines": {
            "yolo": _resumen(acumulado, n_evaluadas, tiempo_total),
        },
        "nota_metodologica": (
            "Métricas sobre la partición de VALIDACIÓN (nunca entrenamiento). OJO: las etiquetas "
            "proceden de anotación semi-automática, por lo que la puntuación puede estar sesgada "
            "al alza en la medida en que las predicciones originales se aceptaron sin corrección. "
            "Corrige cajas a mano en /yolo-dataset para reducir el sesgo."
        ),
        "nota_rendimiento": (
            "Evaluación SÍNCRONA sobre CPU (inferencia YOLO por imagen). Con "
            "max_images alto la petición puede tardar minutos y el gateway de producción "
            "(Railway) puede cortarla por timeout: usa valores moderados (50-100)."
        ),
    }
