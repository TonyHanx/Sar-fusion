"""
estratos.py (dentro de src/evaluacion)

¿Para qué sirve?
    Es la parte que califica al detector. Compara lo que el modelo creyó ver
    con lo que de verdad había en cada foto y saca tres números: la nota
    general (mAP), qué parte de lo que marcó eran personas de verdad
    (precisión) y qué parte de las personas logró encontrar (recall).

    Lo importante es que sabe calificar por separado según la condición de cada
    foto, que es lo que la tesis necesita comparar.

¿Por qué no se usa simplemente la nota que da el programa de entrenamiento?
    Porque ese programa solo entrega el promedio de todo. Para separar por
    altura de vuelo o por cuánto tapa la vegetación hay que calificar nosotros,
    usando pycocotools, que es la herramienta oficial con la que se miden los
    detectores en la literatura.

Dos técnicas, según de qué tipo sea el grupo que se quiere medir:

- **Estrato de imagen** (la altitud): todas las personas de una imagen comparten
  la altitud, asi que el estrato se evalua restringiendo la evaluacion a esas
  imagenes. Si en vez de eso se dejaran las demas imagenes dentro, las
  detecciones falsas sobre su fondo contarian igual y ensuciarian el resultado.

- **Estrato de persona** (la oclusion): dentro de una misma imagen conviven
  personas de distintos estratos. Ahi se evalua sobre todas las imagenes, pero
  las personas de los otros estratos se marcan con iscrowd=1. COCO trata esas
  cajas como "zona a ignorar": una deteccion que caiga encima no cuenta como
  error, y esa persona tampoco cuenta como no detectada.

El mAP@0.5 lo calcula pycocotools. La precision y el recall se calculan aparte
con un umbral de confianza fijo (0.5 en este proyecto), porque el mAP recorre
todos los umbrales y no responde "de lo que el modelo afirma, cuanto acierta".
"""

from __future__ import annotations

import contextlib
import io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CATEGORIAS = [{"id": 1, "name": "person"}]


def construir_gt(registros_imagenes: list[dict], anotaciones: list[dict],
                 ids_ignorados: set[int] | None = None) -> dict:
    """Arma el diccionario COCO de referencia.

    registros_imagenes: lista de {"id", "file_name", "width", "height"}.
    anotaciones: lista de {"id", "image_id", "bbox"} en pixeles [x, y, ancho, alto].
    ids_ignorados: ids de anotacion que deben marcarse como iscrowd=1.

    El area se recalcula como ancho x alto: POP trae 1.1 en todas, que es un
    relleno sin sentido del exportador y confundiria a pycocotools.
    """
    ignorados = ids_ignorados or set()
    anotaciones_coco = []
    for anotacion in anotaciones:
        x, y, ancho, alto = anotacion["bbox"]
        anotaciones_coco.append(
            {
                "id": anotacion["id"],
                "image_id": anotacion["image_id"],
                "category_id": 1,
                "bbox": [float(x), float(y), float(ancho), float(alto)],
                "area": float(ancho) * float(alto),
                "iscrowd": 1 if anotacion["id"] in ignorados else 0,
            }
        )
    return {"images": registros_imagenes, "annotations": anotaciones_coco, "categories": CATEGORIAS}


def _coco_desde_dict(gt: dict) -> COCO:
    """Crea un objeto COCO a partir del diccionario, sin pasar por disco."""
    coco = COCO()
    coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def map50(gt: dict, detecciones: list[dict], ids_imagenes: list[int] | None = None) -> dict:
    """mAP@0.5 (y mAP@0.5:0.95 de regalo) con pycocotools.

    detecciones: lista COCO de {"image_id", "category_id", "bbox", "score"}.
    ids_imagenes: si se pasa, la evaluacion se limita a esas imagenes.
    """
    ids = ids_imagenes if ids_imagenes is not None else [im["id"] for im in gt["images"]]
    detecciones_usadas = [d for d in detecciones if d["image_id"] in set(ids)]
    if not detecciones_usadas:
        return {"map50": 0.0, "map50_95": 0.0}

    coco_gt = _coco_desde_dict(gt)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(list(detecciones_usadas))
        evaluador = COCOeval(coco_gt, coco_dt, "bbox")
        evaluador.params.imgIds = list(ids)
        evaluador.evaluate()
        evaluador.accumulate()
        evaluador.summarize()
    # stats[0] es el promedio de IoU 0.50 a 0.95; stats[1] es exactamente IoU 0.50.
    return {"map50": float(evaluador.stats[1]), "map50_95": float(evaluador.stats[0])}


def _iou(caja_a, caja_b) -> float:
    """Solape de dos cajas en formato [x, y, ancho, alto]."""
    ax, ay, aw, ah = caja_a
    bx, by, bw, bh = caja_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    interseccion = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if interseccion <= 0:
        return 0.0
    return interseccion / (aw * ah + bw * bh - interseccion)


def _solape_con_zona_ignorada(caja_det, caja_ignorada) -> float:
    """Cuanto de la deteccion cae dentro de una zona ignorada.

    No se usa IoU sino la fraccion de la deteccion cubierta, que es el criterio
    de COCO para las zonas de multitud: una deteccion contenida en una zona a
    ignorar se descarta aunque su IoU sea bajo.
    """
    dx, dy, dw, dh = caja_det
    ix, iy, iw, ih = caja_ignorada
    x1, y1 = max(dx, ix), max(dy, iy)
    x2, y2 = min(dx + dw, ix + iw), min(dy + dh, iy + ih)
    interseccion = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return interseccion / (dw * dh) if dw * dh > 0 else 0.0


def precision_recall(gt: dict, detecciones: list[dict], confianza: float = 0.5,
                     umbral_iou: float = 0.5,
                     ids_imagenes: list[int] | None = None) -> dict:
    """Precision y recall a un umbral de confianza fijo.

    Se empareja cada deteccion (de mayor a menor confianza) con la caja real mas
    solapada que siga libre. Las detecciones que caen sobre una persona marcada
    como iscrowd=1 se descartan: no suman acierto ni error.
    """
    ids = set(ids_imagenes if ids_imagenes is not None else [im["id"] for im in gt["images"]])

    reales_por_imagen: dict[int, list[dict]] = {i: [] for i in ids}
    for anotacion in gt["annotations"]:
        if anotacion["image_id"] in ids:
            reales_por_imagen[anotacion["image_id"]].append(anotacion)

    detecciones_por_imagen: dict[int, list[dict]] = {i: [] for i in ids}
    for deteccion in detecciones:
        if deteccion["image_id"] in ids and deteccion["score"] >= confianza:
            detecciones_por_imagen[deteccion["image_id"]].append(deteccion)

    tp = fp = fn = 0
    for id_imagen in ids:
        reales = [a for a in reales_por_imagen[id_imagen] if a["iscrowd"] == 0]
        ignoradas = [a for a in reales_por_imagen[id_imagen] if a["iscrowd"] == 1]
        usadas = set()

        for deteccion in sorted(detecciones_por_imagen[id_imagen],
                                key=lambda d: d["score"], reverse=True):
            mejor_iou, mejor = 0.0, None
            for indice, real in enumerate(reales):
                if indice in usadas:
                    continue
                valor = _iou(deteccion["bbox"], real["bbox"])
                if valor > mejor_iou:
                    mejor_iou, mejor = valor, indice
            if mejor is not None and mejor_iou >= umbral_iou:
                usadas.add(mejor)
                tp += 1
                continue
            # No acerto a nadie: se descarta si cae sobre una zona ignorada.
            if any(_solape_con_zona_ignorada(deteccion["bbox"], a["bbox"]) >= umbral_iou
                   for a in ignoradas):
                continue
            fp += 1

        fn += len(reales) - len(usadas)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def evaluar(gt: dict, detecciones: list[dict], confianza: float = 0.5,
            umbral_iou: float = 0.5, ids_imagenes: list[int] | None = None) -> dict:
    """mAP@0.5, precision y recall de un mismo conjunto, en un solo llamado."""
    resultado = map50(gt, detecciones, ids_imagenes)
    resultado.update(precision_recall(gt, detecciones, confianza, umbral_iou, ids_imagenes))
    ids = set(ids_imagenes if ids_imagenes is not None else [im["id"] for im in gt["images"]])
    resultado["imagenes"] = len(ids)
    resultado["personas"] = sum(
        1 for a in gt["annotations"] if a["image_id"] in ids and a["iscrowd"] == 0
    )
    return resultado


__all__ = ["construir_gt", "evaluar", "map50", "precision_recall", "CATEGORIAS"]
