"""
Evaluación comparativa de los motores de reconocimiento sobre el dataset de validación YOLO.

Criterio MEDIBLE de promoción de motor: en lugar de preferencias, este módulo evalúa
TensorFlow, YOLO y su fusión sobre las imágenes de VALIDACIÓN del dataset YOLO
(yolo_dataset/images/val + labels/val, que NUNCA participan en el entrenamiento) y compara:

    - square_accuracy: % de casillas bien leídas (las 64 por imagen).
    - fen_exact:       % de tableros completos sin ni un solo error (lo que de verdad importa:
                         un solo fallo invalida la detección del movimiento).
    - piece_precision / piece_recall: calidad centrada SOLO en casillas ocupadas.
    - avg_ms:          tiempo medio de inferencia por imagen.

La decisión de promover un motor a "por defecto" es MANUAL y se basa en estos números;
este módulo solo informa (nunca cambia configuración).

⚠️ NOTA METODOLÓGICA IMPORTANTE: las etiquetas de validación proceden de anotación
semi-automática ARRANCADA desde las predicciones de TensorFlow, así que la puntuación de TF
está sesgada AL ALZA en este conjunto (parte de sus aciertos están "cocinados" en la verdad
terrena). Conforme se corrijan cajas a mano, el sesgo disminuye. Para una comparativa
imparcial a largo plazo conviene ir curando a mano un subconjunto de validación.
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
    # Import diferido para no cargar ultralytics/TensorFlow solo por importar este módulo.
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


def _resumen(nombre, acumulado, n_imgs, ms_total):
    """Agrega los contadores crudos en las métricas finales de un motor."""
    total_casillas = max(acumulado["total"], 1)
    piezas_gt = acumulado["tp"] + acumulado["fn"]
    return {
        "motor": nombre,
        "square_accuracy": round(acumulado["aciertos"] / total_casillas, 4),
        "fen_exact": round(acumulado["boards_ok"] / max(n_imgs, 1), 4),
        "piece_precision": round(acumulado["tp"] / max(acumulado["tp"] + acumulado["fp"], 1), 4),
        "piece_recall": round(acumulado["tp"] / max(piezas_gt, 1), 4),
        "avg_ms": round(ms_total / max(n_imgs, 1)),
    }


def evaluar_motores(max_images=100):
    """Ejecuta la evaluación completa de los tres modos sobre la validación.

    Args:
        max_images: límite de imágenes a evaluar (la validación puede crecer mucho y esto
                    corre sobre CPU en producción).

    Returns:
        Dict con las métricas por motor, el mejor por fen_exact y la nota metodológica.
        Si un motor no está disponible, su entrada explica el motivo en vez de fallar todo.
    """
    from services.classifier import classifier
    from services.yolo_detector import yolo_detector, boxes_a_board_state
    from services.fusion import fusionar_board_states

    muestras = _cargar_ground_truth(settings.YOLO_DATASET_DIR)[:max_images]
    if not muestras:
        return {"success": False,
                "error": "No hay imágenes de validación etiquetadas en yolo_dataset/images/val."}

    tf_listo = classifier.is_ready()
    yolo_listo = yolo_detector.is_ready()

    acumulados = {
        "tf": {"aciertos": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "boards_ok": 0},
        "yolo": {"aciertos": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "boards_ok": 0},
        "fusion": {"aciertos": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "boards_ok": 0},
    }
    tiempos = {"tf": 0.0, "yolo": 0.0, "fusion": 0.0}
    n_evaluadas = 0

    for muestra in muestras:
        ruta = os.path.join(settings.YOLO_DATASET_DIR, "images", "val", muestra["file"])
        warped = _leer_imagen(ruta)
        if warped is None:
            continue
        n_evaluadas += 1
        gt = muestra["board_state"]

        board_tf = board_yolo = None

        # --- Motor TensorFlow ---
        if tf_listo:
            t0 = time.perf_counter()
            board_tf = classifier.classify_board(warped)
            tiempos["tf"] += (time.perf_counter() - t0) * 1000
            m = _metricas(board_tf, gt)
            for k in ("aciertos", "total", "tp", "fp", "fn"):
                acumulados["tf"][k] += m[k]
            if m["aciertos"] == m["total"]:
                acumulados["tf"]["boards_ok"] += 1

        # --- Motor YOLO ---
        if yolo_listo:
            t0 = time.perf_counter()
            detecciones = yolo_detector.detect(warped)
            board_yolo, _, _ = boxes_a_board_state(detecciones)
            tiempos["yolo"] += (time.perf_counter() - t0) * 1000
            m = _metricas(board_yolo, gt)
            for k in ("aciertos", "total", "tp", "fp", "fn"):
                acumulados["yolo"][k] += m[k]
            if m["aciertos"] == m["total"]:
                acumulados["yolo"]["boards_ok"] += 1

        # --- Fusión (solo tiene sentido con ambos motores operativos) ---
        if tf_listo and yolo_listo:
            t0 = time.perf_counter()
            board_fusion, _ = fusionar_board_states(board_tf, board_yolo, [])
            tiempos["fusion"] += (time.perf_counter() - t0) * 1000
            m = _metricas(board_fusion, gt)
            for k in ("aciertos", "total", "tp", "fp", "fn"):
                acumulados["fusion"][k] += m[k]
            if m["aciertos"] == m["total"]:
                acumulados["fusion"]["boards_ok"] += 1

    resultados = {}
    if tf_listo:
        resultados["tf"] = _resumen("tensorflow", acumulados["tf"], n_evaluadas, tiempos["tf"])
    else:
        resultados["tf"] = {"error": "Modelo TensorFlow no cargado"}
    if yolo_listo:
        resultados["yolo"] = _resumen("yolo", acumulados["yolo"], n_evaluadas, tiempos["yolo"])
    else:
        resultados["yolo"] = {"error": "Modelo YOLO no disponible"}
    if tf_listo and yolo_listo:
        resultados["fusion"] = _resumen("fusion", acumulados["fusion"], n_evaluadas, tiempos["fusion"])
    else:
        resultados["fusion"] = {"error": "Requiere ambos motores operativos"}

    # Mejor motor POR MÉTRICAS (fen_exact primero): solo informativo, no cambia nada solo.
    candidatos = [(v.get("fen_exact", -1), k) for k, v in resultados.items() if "fen_exact" in v]
    mejor = max(candidatos)[1] if candidatos else None

    return {
        "success": True,
        "n_images": n_evaluadas,
        "engines": resultados,
        "mejor_motor_por_metricas": mejor,
        "nota_metodologica": (
            "Métricas sobre la partición de VALIDACIÓN (nunca entrenamiento). OJO: las etiquetas "
            "proceden de anotación semi-automática arrancada desde TensorFlow, por lo que la "
            "puntuación de TF está sesgada al alza aquí; corrige cajas a mano en /yolo-dataset "
            "para reducir el sesgo. La promoción de motor por defecto es una decisión manual."
        ),
    }
