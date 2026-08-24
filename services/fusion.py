"""
Servicio de FUSIÓN por arbitraje casilla a casilla entre MobileNetV2 (TensorFlow) y YOLO26.

Ambos motores leen el MISMO tablero rectificado 400x400 y producen un board_state de 64 casillas.
Esta función decide, casilla por casilla, qué etiqueta gana según la confianza de cada motor,
sin preferencia fija: YOLO solo sobrescribe a TensorFlow cuando sus detecciones son fiables.

Reglas de arbitraje (en orden):
    1. MANO: si una detección "hand" supera HAND_MIN_CONFIDENCE, las casillas que cubre quedan
       marcadas como inciertas (razón "hand"): su lectura visual no es fiable mientras la mano
       tape la posición.
    2. AMBOS DETECTAN PIEZA:
       - Misma pieza → se queda la etiqueta con la confianza máxima (razón "agree").
       - Piezas distintas → gana la de mayor confianza; en empate gana TensorFlow por ser el
         motor por defecto (razones "yolo_over_tf" / "tf_over_yolo").
    3. SOLO UN MOTOR DETECTA PIEZA: esa parte gana SOLO SI supera su umbral de confianza propio;
       si no lo supera, prevalece la lectura del otro motor (razón "low_confidence_rejected").
    4. AUSENCIA DE YOLO (casilla vacía sin caja): solo se acepta como "empty" cuando TensorFlow
       tampoco está seguro de su pieza (conf < FUSION_MIN_TF_CONFIDENCE), porque los falsos
       negativos de YOLO son habituales con datasets pequeños (razón "yolo_absence").
"""
from core.config import settings


def _cuadricula_de_manos(hand_boxes):
    """Calcula qué casillas están tapadas por alguna mano suficientemente confiada.

    Devuelve (conjunto_de_casillas, manos_filtradas).
    """
    cell = settings.CELL_SIZE
    cols = settings.COLS
    casillas = set()
    manos_validas = [h for h in (hand_boxes or []) if h["confidence"] >= settings.HAND_MIN_CONFIDENCE]
    for h in manos_validas:
        x, y, w, hpx = h["bbox_px"]
        col_min = max(int(x // cell), 0)
        col_max = min(int((x + w) // cell), 7)
        fila_min = max(int(y // cell), 0)
        fila_max = min(int((y + hpx) // cell), 7)
        for fila in range(fila_min, fila_max + 1):
            for col in range(col_min, col_max + 1):
                casillas.add(f"{cols[col]}{8 - fila}")
    return casillas, manos_validas


def fusionar_board_states(board_state_tf: dict, board_state_yolo: dict, hand_boxes=None):
    """Arbitra casilla a casilla entre los dos motores y genera el estado final.

    Args:
        board_state_tf: 64 casillas {"e4": {"label": "w_P", "confidence": 0.91}} desde MobileNetV2.
        board_state_yolo: 64 casillas mismo formato desde YOLO (vacías con confidence None).
        hand_boxes: detecciones brutas de clase "hand" del detector YOLO (opcional).

    Returns:
        (board_state_final, resumen) donde resumen incluye contadores por razón de decisión,
        la lista de decisiones NO triviales (desacuerdos, rechazos por confianza baja o mano),
        las manos válidas y las casillas que tapan.
    """
    casillas_mano, manos_validas = _cuadricula_de_manos(hand_boxes)

    board_state_final = {}
    decisiones = []
    contadores = {
        "agree": 0,
        "tf_over_yolo": 0,
        "yolo_over_tf": 0,
        "low_confidence_rejected": 0,
        "yolo_absence": 0,
        "hand": 0,
    }

    for square, tf in board_state_tf.items():
        yolo = board_state_yolo.get(square, {"label": "empty", "confidence": None})
        tf_label, tf_conf = tf["label"], tf.get("confidence")
        yolo_label, yolo_conf = yolo["label"], yolo.get("confidence")

        # Regla 1: la mano invalida la lectura de las casillas que cubre.
        if square in casillas_mano:
            board_state_final[square] = {"label": tf_label, "confidence": tf_conf, "hand": True}
            contadores["hand"] += 1
            decisiones.append({
                "square": square, "reason": "hand",
                "tf": tf_label, "yolo": yolo_label, "final": tf_label
            })
            continue

        final_label, final_conf, razon = tf_label, tf_conf, "agree"

        # Regla 2: ambos ven una pieza.
        if tf_label != "empty" and yolo_label != "empty":
            if tf_label == yolo_label:
                # Misma pieza: nos quedamos con la lectura más segura de las dos.
                if (yolo_conf or 0) > (tf_conf or 0):
                    final_label, final_conf, razon = yolo_label, yolo_conf, "agree"
                else:
                    final_label, final_conf = tf_label, tf_conf
                contadores["agree"] += 1
            elif (yolo_conf or 0) > (tf_conf or 0):
                final_label, final_conf, razon = yolo_label, yolo_conf, "yolo_over_tf"
                contadores["yolo_over_tf"] += 1
            else:
                razon = "tf_over_yolo"
                contadores["tf_over_yolo"] += 1

        # Regla 3: solo uno de los dos ve pieza.
        elif tf_label != "empty" or yolo_label != "empty":
            pieza_es_tf = tf_label != "empty"
            conf_pieza = tf_conf if pieza_es_tf else yolo_conf
            umbral = settings.FUSION_MIN_TF_CONFIDENCE if pieza_es_tf else settings.FUSION_MIN_YOLO_CONFIDENCE

            if conf_pieza is not None and conf_pieza >= umbral:
                # La detección es fiable: se acepta tal cual (razón "agree" simplificada).
                final_label, final_conf = (tf_label, tf_conf) if pieza_es_tf else (yolo_label, yolo_conf)
                contadores["agree"] += 1
            else:
                # Detección poco fiable: prevalece la lectura del otro motor.
                otro_label, otra_conf = (yolo_label, yolo_conf) if pieza_es_tf else (tf_label, tf_conf)
                final_label, final_conf = otro_label, otra_conf
                razon = "low_confidence_rejected"
                contadores["low_confidence_rejected"] += 1
                decisiones.append({
                    "square": square, "reason": razon,
                    "tf": tf_label, "yolo": yolo_label, "final": final_label
                })

        else:
            # Ambos dicen empty: si TensorFlow estaba inseguro de su "empty", lo atribuimos a
            # la ausencia validada por YOLO (sin caja no hay confianza numérica que comparar).
            if tf_conf is not None and tf_conf < settings.FUSION_MIN_TF_CONFIDENCE and yolo_conf is None:
                razon = "yolo_absence"
                contadores["yolo_absence"] += 1
            else:
                contadores["agree"] += 1

        board_state_final[square] = {"label": final_label, "confidence": final_conf}

        # Registramos solo desacuerdos reales entre motores (evita ruido en la respuesta).
        if razon in ("tf_over_yolo", "yolo_over_tf"):
            decisiones.append({
                "square": square, "reason": razon,
                "tf": tf_label, "yolo": yolo_label, "final": final_label
            })

    resumen = {
        "contadores": contadores,
        "decisiones": decisiones,
        "manos_validas": manos_validas,
        "casillas_con_mano": sorted(casillas_mano),
    }
    return board_state_final, resumen
