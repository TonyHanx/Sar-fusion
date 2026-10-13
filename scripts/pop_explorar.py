"""Inspeccion visual de POP (Fase 1).

Para que sirve: comprobar con los ojos que las cajas del dataset caen sobre las
personas, y resumir en numeros lo que contiene cada particion. No modifica nada
dentro de data/raw: solo lee.

Que hace:
  1. Lee las anotaciones COCO de cada particion y cuenta imagenes, cajas,
     escenas y grupos de altitud.
  2. Avisa si las anotaciones traen mascaras de oclusion o no.
  3. Elige N imagenes al azar (con la semilla del archivo de configuracion),
     les dibuja las cajas (y las mascaras, si existen) y las guarda.

Uso:
    python scripts/pop_explorar.py
    python scripts/pop_explorar.py --config configs/pop_explorar.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))

from src import visual  # noqa: E402  (se importan despues de ajustar sys.path)
from src.datos import pop  # noqa: E402


def leer_config(ruta: Path) -> dict:
    """Carga el archivo de configuracion."""
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def resolver(ruta: str | Path) -> Path:
    """Convierte una ruta del archivo de configuracion en ruta absoluta."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_REPO / ruta


def resumir_particion(particion: str, datos: dict) -> dict:
    """Imprime y devuelve el resumen numerico de una particion."""
    imagenes = datos["imagenes"]
    n_cajas = sum(len(c) for c in datos["cajas"].values())

    escenas = Counter(pop.escena_desde_nombre(im["file_name"]) for im in imagenes.values())
    grupos = Counter(
        pop.grupo_altitud(pop.altitud_desde_nombre(im["file_name"])) or "fuera de rango"
        for im in imagenes.values()
    )
    con_mascara = sum(
        1 for lista in datos["cajas"].values() for a in lista if a.get("segmentation")
    )

    print(f"\n--- particion {particion} ---")
    print(f"  imagenes          : {len(imagenes)}")
    print(f"  cajas de persona  : {n_cajas}")
    print(f"  cajas con mascara : {con_mascara}")
    print(f"  escenas           : {dict(sorted(escenas.items()))}")
    print(f"  grupos de altitud : {dict(sorted(grupos.items()))}")
    return {"imagenes": len(imagenes), "cajas": n_cajas, "mascaras": con_mascara}


def dibujar(ruta_imagen: Path, anotaciones: list[dict], particion: str, grosor: int):
    """Dibuja las cajas (y las mascaras si existen) sobre una copia de la imagen."""
    imagen = cv2.imread(str(ruta_imagen))
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")
    alto, ancho = imagen.shape[:2]

    n_mascaras = 0
    for anotacion in anotaciones:
        mascara = pop.mascara_de_anotacion(anotacion, alto, ancho)
        if mascara is not None:
            n_mascaras += 1
            # La mascara se pinta como una capa roja semitransparente.
            capa = imagen.copy()
            capa[mascara > 0] = visual.ROJO
            imagen = cv2.addWeighted(capa, 0.45, imagen, 0.55, 0)

        x, y, w, h = (int(v) for v in anotacion["bbox"])
        visual.dibujar_caja(imagen, x, y, x + w, y + h, visual.VERDE, grosor)

    altitud = pop.altitud_desde_nombre(ruta_imagen.name)
    encabezado = (
        f"{particion} | {ruta_imagen.name} | {len(anotaciones)} personas | "
        f"{altitud:.1f} m ({pop.grupo_altitud(altitud)})"
    )
    visual.texto_con_fondo(imagen, encabezado, (12, 34))
    return imagen, n_mascaras


def main() -> int:
    analizador = argparse.ArgumentParser(description="Inspeccion visual de POP.")
    analizador.add_argument(
        "--config",
        type=Path,
        default=RAIZ_REPO / "configs" / "pop_explorar.yaml",
        help="Archivo de configuracion.",
    )
    argumentos = analizador.parse_args()

    config = leer_config(argumentos.config)
    raiz_pop = resolver(config["raiz_pop"])
    salida = resolver(config["salida"])
    salida.mkdir(parents=True, exist_ok=True)

    print(f"Dataset   : {raiz_pop}")
    print(f"Salida    : {salida}")
    print(f"Semilla   : {config['semilla']}")

    # 1) Resumen por particion y lista de candidatas para el muestreo.
    totales = Counter()
    candidatas: list[tuple[str, dict, list[dict]]] = []
    for particion in config["particiones"]:
        datos = pop.cargar_particion(raiz_pop, particion)
        resumen = resumir_particion(particion, datos)
        totales.update(resumen)
        for id_imagen, registro in datos["imagenes"].items():
            candidatas.append((particion, registro, datos["cajas"][id_imagen]))

    print("\n=== totales de POP ===")
    print(f"  imagenes          : {totales['imagenes']}")
    print(f"  cajas de persona  : {totales['cajas']}")
    print(f"  cajas con mascara : {totales['mascaras']}")
    if totales["mascaras"] == 0:
        print(
            "\n  AVISO: ninguna anotacion trae mascara de oclusion ('segmentation' vacio).\n"
            "  Sin mascaras no se puede calcular la oclusion por persona ni estratificar POP."
        )

    # 2) Muestreo reproducible y dibujo.
    sorteo = random.Random(config["semilla"])
    muestras = sorteo.sample(candidatas, min(int(config["n_muestras"]), len(candidatas)))

    print(f"\n=== {len(muestras)} imagenes dibujadas en {salida.name}/ ===")
    for particion, registro, anotaciones in muestras:
        ruta_imagen = pop.ruta_imagenes(raiz_pop, particion) / registro["file_name"]
        imagen, n_mascaras = dibujar(ruta_imagen, anotaciones, particion, int(config["grosor_caja"]))
        destino = salida / f"{particion}_{ruta_imagen.stem}.jpg"
        cv2.imwrite(str(destino), imagen)
        print(
            f"  {destino.name:<34} {len(anotaciones)} cajas, {n_mascaras} mascaras, "
            f"escena {pop.escena_desde_nombre(registro['file_name'])}, "
            f"{pop.altitud_desde_nombre(registro['file_name']):.1f} m"
        )

    return 0


if __name__ == "__main__":
    try:  # la consola de Windows a veces no acepta acentos
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
