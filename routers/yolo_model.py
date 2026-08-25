"""
Router de gestión del modelo YOLO26: clases, bounding boxes y borrado del modelo entrenado.

Permite consultar y gestionar las clases del detector YOLO, inspeccionar y borrar
bounding boxes del dataset, y eliminar el modelo entrenado para reentrenar.

Endpoints:
    GET  /yolo-model/classes            | Lista las clases del detector YOLO.
    GET  /yolo-model/boxes              | Lista todas las bounding boxes del dataset, agrupadas por imagen.
    GET  /yolo-model/boxes/{image_id}   | Detalle de las bounding boxes de una imagen concreta.
    DELETE /yolo-model/boxes/{image_id}/{box_index} | Elimina una bounding box concreta de una imagen.
    DELETE /yolo-model/images/{image_id}| Elimina una imagen y todas sus anotaciones del dataset.
    DELETE /yolo-model/model            | Elimina el modelo YOLO entrenado y recarga el detector.
"""
import os
import traceback

from fastapi import APIRouter, Depends, Query

from core.config import settings
from core.dependencies import get_current_user
from services.yolo_detector import yolo_detector

# Router para la gestión del modelo y dataset YOLO.
router = APIRouter(prefix="/yolo-model", tags=["Modelo YOLO"])


def _label_path(image_id: str) -> str:
    """Devuelve la ruta del fichero .txt de etiquetas para una imagen dada."""
    return os.path.join(settings.YOLO_DATASET_DIR, "labels", f"{image_id}.txt")


def _image_path(image_id: str) -> str:
    """Devuelve la ruta de la imagen (JPG) para un id dado."""
    return os.path.join(settings.YOLO_DATASET_DIR, "images", f"{image_id}.jpg")


def _fuentes_path(image_id: str) -> str:
    """Devuelve la ruta de la imagen fuente original para un id dado."""
    return os.path.join(settings.YOLO_DATASET_DIR, "fuentes", f"{image_id}.jpg")


def _list_image_ids() -> list:
    """Devuelve la lista de IDs de imágenes (sin extensión) del dataset YOLO."""
    img_dir = os.path.join(settings.YOLO_DATASET_DIR, "images")
    if not os.path.isdir(img_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _read_boxes(image_id: str) -> list:
    """Lee el fichero .txt de una imagen y devuelve la lista de cajas parseadas.

    Cada caja: {"class_id": int, "class_name": str, "x_center": float,
                 "y_center": float, "width": float, "height": float}.
    """
    path = _label_path(image_id)
    if not os.path.isfile(path):
        return []

    cajas = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            if len(partes) < 5:
                continue
            try:
                class_id = int(partes[0])
                x_c = float(partes[1])
                y_c = float(partes[2])
                w = float(partes[3])
                h = float(partes[4])
                class_name = settings.YOLO_CLASSES[class_id] if 0 <= class_id < len(settings.YOLO_CLASSES) else f"unknown_{class_id}"
                cajas.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "x_center": round(x_c, 6),
                    "y_center": round(y_c, 6),
                    "width": round(w, 6),
                    "height": round(h, 6),
                })
            except (ValueError, IndexError):
                continue
    return cajas


def _write_boxes(image_id: str, cajas: list):
    """Escribe la lista de cajas en el fichero .txt de una imagen (formato YOLO)."""
    path = _label_path(image_id)
    lineas = []
    for c in cajas:
        lineas.append(
            f"{c['class_id']} {c['x_center']:.6f} {c['y_center']:.6f} "
            f"{c['width']:.6f} {c['height']:.6f}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n" if lineas else "")


def _count_boxes_by_class() -> dict:
    """Cuenta el número total de cajas de cada clase en todo el dataset."""
    contadores = {cls: 0 for cls in settings.YOLO_CLASSES}
    lbl_dir = os.path.join(settings.YOLO_DATASET_DIR, "labels")
    if not os.path.isdir(lbl_dir):
        return contadores

    for fname in os.listdir(lbl_dir):
        if not fname.lower().endswith(".txt"):
            continue
        with open(os.path.join(lbl_dir, fname), "r", encoding="utf-8") as f:
            for linea in f:
                partes = linea.strip().split()
                if not partes:
                    continue
                try:
                    cid = int(partes[0])
                    if 0 <= cid < len(settings.YOLO_CLASSES):
                        contadores[settings.YOLO_CLASSES[cid]] += 1
                except ValueError:
                    continue
    return contadores


# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /yolo-model/classes
# Devuelve la lista de clases del detector YOLO con su id y el número de cajas de cada una.
# -------------------------------------------------------------------------------
@router.get(
    "/classes",
    summary="Lista las clases del detector YOLO con el número de cajas de cada una."
)
def get_classes(user=Depends(get_current_user)):
    """Devuelve las 13 clases YOLO (12 piezas + hand) con su id y el recuento de cajas."""
    try:
        contadores = _count_boxes_by_class()
        clases = []
        for idx, nombre in enumerate(settings.YOLO_CLASSES):
            clases.append({
                "id": idx,
                "name": nombre,
                "total_boxes": contadores[nombre],
            })
        return {
            "success": True,
            "total_classes": len(clases),
            "classes": clases,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}


# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /yolo-model/boxes
# Lista todas las imágenes del dataset con sus bounding boxes (paginado).
# -------------------------------------------------------------------------------
@router.get(
    "/boxes",
    summary="Lista todas las imágenes del dataset YOLO con sus bounding boxes."
)
def list_boxes(
    page: int = Query(1, ge=1, description="Página (empieza en 1)"),
    per_page: int = Query(50, ge=1, le=500, description="Imágenes por página"),
    class_filter: str = Query(None, description="Filtrar por nombre de clase (ej: w_P)"),
    user=Depends(get_current_user),
):
    """
    Devuelve la lista de imágenes del dataset con todas sus bounding boxes.
    Opcionalmente se puede filtrar para mostrar solo las imágenes que contienen
    cajas de una clase concreta.
    """
    try:
        ids = _list_image_ids()
        total_images = len(ids)

        # Filtrado por clase si se solicita.
        if class_filter:
            if class_filter not in settings.YOLO_CLASSES:
                return {"success": False, "error": f"Clase desconocida: {class_filter}. Clases válidas: {settings.YOLO_CLASSES}"}
            filtered_ids = []
            target_idx = settings.YOLO_CLASSES.index(class_filter)
            for img_id in ids:
                cajas = _read_boxes(img_id)
                if any(c["class_id"] == target_idx for c in cajas):
                    filtered_ids.append(img_id)
            ids = filtered_ids

        total_filtered = len(ids)

        # Paginación.
        start = (page - 1) * per_page
        end = start + per_page
        page_ids = ids[start:end]

        imagenes = []
        for img_id in page_ids:
            cajas = _read_boxes(img_id)
            imagenes.append({
                "id": img_id,
                "has_source": os.path.isfile(_fuentes_path(img_id)),
                "boxes": cajas,
                "total_boxes": len(cajas),
            })

        return {
            "success": True,
            "total_images": total_filtered,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, -(-total_filtered // per_page)),  # ceil division
            "images": imagenes,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}


# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /yolo-model/boxes/{image_id}
# Devuelve las bounding boxes de una imagen concreta.
# -------------------------------------------------------------------------------
@router.get(
    "/boxes/{image_id}",
    summary="Detalla las bounding boxes de una imagen concreta del dataset YOLO."
)
def get_image_boxes(image_id: str, user=Depends(get_current_user)):
    """Devuelve las cajas de una imagen, o 404 si no existe."""
    try:
        img_file = _image_path(image_id)
        lbl_file = _label_path(image_id)
        fuente_file = _fuentes_path(image_id)

        if not os.path.isfile(img_file) and not os.path.isfile(lbl_file):
            return {"success": False, "error": f"Imagen '{image_id}' no encontrada en el dataset."}

        cajas = _read_boxes(image_id)
        return {
            "success": True,
            "image": {
                "id": image_id,
                "has_image": os.path.isfile(img_file),
                "has_label": os.path.isfile(lbl_file),
                "has_source": os.path.isfile(fuente_file),
            },
            "boxes": cajas,
            "total_boxes": len(cajas),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}


# -------------------------------------------------------------------------------
# [ENDPOINT] - DELETE /yolo-model/boxes/{image_id}/{box_index}
# Elimina una bounding box concreta de una imagen.
# -------------------------------------------------------------------------------
@router.delete(
    "/boxes/{image_id}/{box_index}",
    summary="Elimina una bounding box concreta de una imagen del dataset YOLO."
)
def delete_box(image_id: str, box_index: int, user=Depends(get_current_user)):
    """
    Elimina la bounding box en la posición indicada (0-indexed) del fichero .txt
    de la imagen. Si la imagen queda sin cajas, se eliminan tanto el .txt como la imagen
    y la imagen fuente (si existen).
    """
    try:
        lbl_file = _label_path(image_id)
        if not os.path.isfile(lbl_file):
            return {"success": False, "error": f"No existe anotación para la imagen '{image_id}'."}

        cajas = _read_boxes(image_id)
        if box_index < 0 or box_index >= len(cajas):
            return {"success": False, "error": f"Índice {box_index} fuera de rango (0..{len(cajas) - 1})."}

        caja_eliminada = cajas.pop(box_index)

        if len(cajas) == 0:
            # Sin cajas restantes: elimina imagen, label y fuente.
            for path in [_image_path(image_id), lbl_file, _fuentes_path(image_id)]:
                if os.path.isfile(path):
                    os.remove(path)
            return {
                "success": True,
                "removed_box": caja_eliminada,
                "message": f"Última caja eliminada. Imagen '{image_id}' eliminada del dataset.",
                "deleted_image": True,
            }
        else:
            _write_boxes(image_id, cajas)
            return {
                "success": True,
                "removed_box": caja_eliminada,
                "remaining_boxes": len(cajas),
                "deleted_image": False,
            }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}


# -------------------------------------------------------------------------------
# [ENDPOINT] - DELETE /yolo-model/images/{image_id}
# Elimina una imagen completa del dataset (imagen, label y fuente).
# -------------------------------------------------------------------------------
@router.delete(
    "/images/{image_id}",
    summary="Elimina una imagen completa del dataset YOLO (imagen, anotación y fuente)."
)
def delete_image(image_id: str, user=Depends(get_current_user)):
    """Elimina la imagen, su fichero .txt de etiquetas y la imagen fuente original."""
    try:
        img_file = _image_path(image_id)
        lbl_file = _label_path(image_id)
        fuente_file = _fuentes_path(image_id)

        if not os.path.isfile(img_file) and not os.path.isfile(lbl_file):
            return {"success": False, "error": f"Imagen '{image_id}' no encontrada en el dataset."}

        eliminados = []
        for path, nombre in [(img_file, "imagen"), (lbl_file, "etiqueta"), (fuente_file, "fuente")]:
            if os.path.isfile(path):
                os.remove(path)
                eliminados.append(nombre)

        return {
            "success": True,
            "deleted": image_id,
            "removed_files": eliminados,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}


# -------------------------------------------------------------------------------
# [ENDPOINT] - DELETE /yolo-model/model
# Elimina el modelo YOLO entrenado (.pt) y recarga el detector (quedará "no listo").
# -------------------------------------------------------------------------------
@router.delete(
    "/model",
    summary="Elimina el modelo YOLO entrenado y recarga el detector."
)
def delete_model(user=Depends(get_current_user)):
    """
    Elimina el fichero yolo_chess.pt del directorio de modelos y recarga el detector
    en memoria para que quede en estado "no listo".
    """
    try:
        model_path = settings.YOLO_MODEL_PATH
        if not os.path.isfile(model_path):
            return {"success": False, "error": "No existe ningún modelo YOLO entrenado para eliminar."}

        os.remove(model_path)
        yolo_detector.reload()

        return {
            "success": True,
            "message": f"Modelo YOLO eliminado ({os.path.basename(model_path)}). Detector recargado.",
            "ready": yolo_detector.is_ready(),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "detail": traceback.format_exc()}
