"""Lectura del dataset POP (termico) en formato COCO.

Para que sirve: POP trae un .json por particion con las cajas de las personas.
Aqui estan las funciones que varios scripts necesitan: localizar los archivos,
leer las anotaciones, sacar la escena y la altitud del nombre de la imagen y
convertir una mascara de COCO en una imagen binaria (cuando la haya).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Particiones oficiales del dataset.
PARTICIONES = ("train", "val", "test")

# Grupos de altitud, como en Woesle et al. (2026): la altura del nombre casi
# nunca es exacta, asi que se agrupa en tres tramos. (nombre, minimo, maximo)
GRUPOS_ALTITUD = (("30 m", 20.0, 40.0), ("50 m", 40.0, 60.0), ("70 m", 60.0, 80.0))


def ruta_anotaciones(raiz: Path, particion: str) -> Path:
    """Ruta del .json de una particion."""
    return Path(raiz) / particion / "Annotations" / f"{particion}.json"


def ruta_imagenes(raiz: Path, particion: str) -> Path:
    """Carpeta con los .JPG de una particion."""
    return Path(raiz) / particion / "JPEGImages"


def cargar_particion(raiz: Path, particion: str) -> dict:
    """Lee el .json de una particion y lo deja listo para consultar.

    Devuelve un diccionario con:
      - "imagenes": {id de imagen: registro de la imagen}
      - "cajas": {id de imagen: lista de anotaciones de esa imagen}
      - "categorias": las categorias tal como vienen en el archivo
    """
    datos = json.loads(ruta_anotaciones(raiz, particion).read_text(encoding="utf-8"))

    imagenes = {im["id"]: im for im in datos["images"]}
    cajas: dict[int, list[dict]] = {id_imagen: [] for id_imagen in imagenes}
    for anotacion in datos["annotations"]:
        cajas[anotacion["image_id"]].append(anotacion)

    return {"imagenes": imagenes, "cajas": cajas, "categorias": datos.get("categories", [])}


def escena_desde_nombre(nombre_archivo: str) -> str:
    """Escena de vuelo: 'F10_0000_30.145.JPG' -> 'F10'."""
    return Path(nombre_archivo).stem.split("_")[0]


def altitud_desde_nombre(nombre_archivo: str) -> float:
    """Altitud en metros: 'F10_0000_30.145.JPG' -> 30.145.

    El nombre tiene tres campos separados por '_' y el ultimo es la altura.
    """
    campos = Path(nombre_archivo).stem.split("_")
    if len(campos) != 3:
        raise ValueError(f"Nombre inesperado, se esperaban 3 campos: {nombre_archivo}")
    return float(campos[-1])


def grupo_altitud(altitud: float) -> str | None:
    """Agrupa una altitud en '30 m', '50 m' o '70 m'; None si queda fuera."""
    for nombre, minimo, maximo in GRUPOS_ALTITUD:
        if minimo <= altitud < maximo:
            return nombre
    return None


def mascara_de_anotacion(anotacion: dict, alto: int, ancho: int) -> np.ndarray | None:
    """Convierte la 'segmentation' de COCO en una imagen binaria (0 o 1).

    Acepta los dos formatos de COCO: lista de poligonos y RLE comprimido.
    Devuelve None si la anotacion no trae mascara, que es el caso de POP tal
    como se descarga de OSF.
    """
    segmentacion = anotacion.get("segmentation")
    if not segmentacion:
        return None

    if isinstance(segmentacion, list):  # uno o varios poligonos
        import cv2

        mascara = np.zeros((alto, ancho), dtype=np.uint8)
        for poligono in segmentacion:
            puntos = np.array(poligono, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mascara, [puntos.round().astype(np.int32)], 1)
        return mascara

    from pycocotools import mask as mascara_coco  # RLE

    rle = segmentacion
    if isinstance(rle.get("counts"), list):  # RLE sin comprimir
        rle = mascara_coco.frPyObjects(rle, alto, ancho)
    return mascara_coco.decode(rle)
