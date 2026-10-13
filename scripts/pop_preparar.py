"""Prepara POP para Ultralytics y describe el dataset (Fase 2).

Para que sirve:
  1. Dejar POP en el formato que entiende Ultralytics (una carpeta de imagenes
     y una de etiquetas .txt), respetando las particiones oficiales.
  2. Construir una tabla con una fila por persona, para poder analizar despues
     el desempeno por altitud (y por oclusion, cuando haya con que calcularla).
  3. Generar la tabla resumen y la figura de distribuciones del articulo.

Que tecnica usa: la conversion de COCO a YOLO es la misma de los autores del
dataset (src/coco2yolo.py publicado en OSF): la caja [x, y, ancho, alto] en
pixeles pasa a centro y tamano relativos al tamano de la imagen, redondeados a
6 decimales.

Nota sobre la oclusion: POP no publica mascaras, asi que las columnas de
oclusion y estrato quedan vacias. Ver el resumen de la Fase 1.

Uso:
    python scripts/pop_preparar.py
    python scripts/pop_preparar.py --config configs/pop_preparar.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")  # sin ventana grafica: solo guarda archivos
import matplotlib.pyplot as plt  # noqa: E402

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))

from src.datos import pop  # noqa: E402

# Paleta Okabe-Ito: disenada para distinguirse tambien con daltonismo.
COLORES = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}


def leer_config(ruta: Path) -> dict:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def resolver(ruta: str | Path) -> Path:
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_REPO / ruta


def enlazar_o_copiar(origen: Path, destino: Path, modo: str) -> None:
    """Lleva una imagen a data/processed sin tocar el original.

    Un enlace duro es un segundo nombre para el mismo archivo en el disco: no
    gasta espacio y no modifica el original. Si no se puede (por ejemplo,
    porque el destino esta en otro disco), se copia.
    """
    if destino.exists():
        destino.unlink()
    if modo == "enlace":
        try:
            destino.hardlink_to(origen)
            return
        except OSError:
            pass  # cae a la copia
    shutil.copy2(origen, destino)


def caja_a_yolo(bbox, ancho_img: int, alto_img: int):
    """[x, y, ancho, alto] en pixeles -> centro y tamano relativos (formato YOLO)."""
    x, y, w, h = bbox
    return (
        round((x + w / 2.0) / ancho_img, 6),
        round((y + h / 2.0) / alto_img, 6),
        round(w / ancho_img, 6),
        round(h / alto_img, 6),
    )


def preparar_particion(raiz_pop: Path, particion: str, salida: Path, modo: str, lado_minimo: int):
    """Convierte una particion y devuelve (filas de personas, cajas descartadas)."""
    datos = pop.cargar_particion(raiz_pop, particion)
    carpeta_imagenes = salida / "images" / particion
    carpeta_etiquetas = salida / "labels" / particion
    carpeta_imagenes.mkdir(parents=True, exist_ok=True)
    carpeta_etiquetas.mkdir(parents=True, exist_ok=True)

    filas = []
    descartadas = 0

    for id_imagen, registro in datos["imagenes"].items():
        nombre = registro["file_name"]
        ancho_img, alto_img = registro["width"], registro["height"]
        altitud = pop.altitud_desde_nombre(nombre)
        grupo = pop.grupo_altitud(altitud)
        escena = pop.escena_desde_nombre(nombre)

        enlazar_o_copiar(
            pop.ruta_imagenes(raiz_pop, particion) / nombre,
            carpeta_imagenes / nombre,
            modo,
        )

        lineas = []
        for anotacion in datos["cajas"][id_imagen]:
            x, y, w, h = (int(v) for v in anotacion["bbox"])
            if w < lado_minimo or h < lado_minimo:
                descartadas += 1
                continue
            cx, cy, aw, ah = caja_a_yolo([x, y, w, h], ancho_img, alto_img)
            lineas.append(f"0 {cx} {cy} {aw} {ah}")
            filas.append(
                {
                    "imagen": nombre,
                    "particion": particion,
                    "escena": escena,
                    "altitud_m": altitud,
                    "grupo_altitud": grupo,
                    "id_anotacion": anotacion["id"],
                    "x": x,
                    "y": y,
                    "ancho": w,
                    "alto": h,
                    "area_px": w * h,
                    "oclusion": pd.NA,  # POP no publica mascaras: no se puede calcular
                    "estrato": pd.NA,
                }
            )

        texto = "\n".join(lineas)
        if lineas:
            texto += "\n"
        (carpeta_etiquetas / f"{Path(nombre).stem}.txt").write_text(texto, encoding="utf-8")

    return filas, descartadas


def escribir_config_ultralytics(ruta: Path, salida: Path) -> None:
    """Genera el yaml del dataset que lee Ultralytics."""
    contenido = (
        "# Dataset POP en formato YOLO.\n"
        "# ARCHIVO GENERADO por scripts/pop_preparar.py: no editar a mano,\n"
        "# se vuelve a escribir cada vez que se preparan los datos.\n"
        "# 'path' es absoluto porque Ultralytics resuelve las rutas relativas\n"
        "# contra su propia carpeta de datasets, no contra este repositorio.\n"
        "# Al preparar los datos en el laboratorio, el archivo se regenera con\n"
        "# la ruta de esa maquina.\n"
        f"path: {salida.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: person\n"
    )
    ruta.write_text(contenido, encoding="utf-8")


def figura_distribuciones(personas: pd.DataFrame, imagenes: pd.DataFrame, destino: Path) -> None:
    """Dos histogramas: altitud de vuelo y tamano de las personas en pixeles.

    Cada particion se dibuja como porcentaje de si misma, porque tienen tamanos
    muy distintos y lo que interesa comparar es la forma, no la altura.
    """
    figura, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.2))

    for particion, color in COLORES.items():
        alturas = imagenes.loc[imagenes["particion"] == particion, "altitud_m"]
        izq.hist(
            alturas,
            bins=range(20, 82, 1),
            weights=[100.0 / len(alturas)] * len(alturas),
            histtype="step",
            linewidth=2,
            color=color,
            label=particion,
        )

        lados = personas.loc[personas["particion"] == particion, "alto"]
        der.hist(
            lados,
            bins=range(0, 135, 5),
            weights=[100.0 / len(lados)] * len(lados),
            histtype="step",
            linewidth=2,
            color=color,
            label=particion,
        )

    izq.set_xlabel("altitud de vuelo (m)")
    izq.set_ylabel("% de imagenes de la particion")
    izq.set_title("Altitud de vuelo", loc="left")

    der.set_xlabel("alto de la caja de la persona (px)")
    der.set_ylabel("% de personas de la particion")
    der.set_title("Tamano aparente de las personas", loc="left")
    der.set_xlim(0, 130)  # mas alla no hay datos: 161 px es el maximo absoluto

    for eje in (izq, der):
        eje.legend(frameon=False)
        eje.grid(axis="y", alpha=0.25, linewidth=0.8)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)

    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=200)
    plt.close(figura)


def main() -> int:
    analizador = argparse.ArgumentParser(description="Prepara POP en formato YOLO.")
    analizador.add_argument(
        "--config",
        type=Path,
        default=RAIZ_REPO / "configs" / "pop_preparar.yaml",
        help="Archivo de configuracion.",
    )
    argumentos = analizador.parse_args()

    config = leer_config(argumentos.config)
    raiz_pop = resolver(config["raiz_pop"])
    salida = resolver(config["salida"])
    modo = config["modo_imagenes"]
    lado_minimo = int(config["lado_minimo_caja"])

    print(f"Origen (no se modifica): {raiz_pop}")
    print(f"Destino                : {salida}")
    print(f"Imagenes por           : {modo}\n")

    todas_las_filas = []
    total_descartadas = 0
    for particion in config["particiones"]:
        filas, descartadas = preparar_particion(raiz_pop, particion, salida, modo, lado_minimo)
        todas_las_filas += filas
        total_descartadas += descartadas
        print(f"  {particion:<6} {len(filas):>6} personas  ({descartadas} cajas degeneradas descartadas)")

    personas = pd.DataFrame(todas_las_filas)
    ruta_personas = resolver(config["tabla_personas"])
    ruta_personas.parent.mkdir(parents=True, exist_ok=True)
    personas.to_csv(ruta_personas, index=False, encoding="utf-8")
    print(f"\nTabla por persona  -> {ruta_personas}  ({len(personas)} filas)")

    # Una fila por imagen, para contar imagenes sin repetirlas.
    imagenes = personas.drop_duplicates(subset=["particion", "imagen"])[
        ["particion", "imagen", "altitud_m", "grupo_altitud"]
    ]

    conteo_personas = (
        personas.groupby(["particion", "grupo_altitud"], dropna=False)["id_anotacion"]
        .count()
        .rename("personas")
    )
    conteo_imagenes = (
        imagenes.groupby(["particion", "grupo_altitud"], dropna=False)["imagen"]
        .count()
        .rename("imagenes")
    )
    resumen = pd.concat([conteo_imagenes, conteo_personas], axis=1).reset_index()
    # Columna explicita para que la tabla del articulo muestre por que todavia
    # no hay estratos de oclusion.
    resumen["estrato_oclusion"] = "sin dato (POP no publica mascaras)"

    ruta_resumen = resolver(config["tabla_resumen"])
    ruta_resumen.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8")
    print(f"Tabla resumen      -> {ruta_resumen}\n")
    print(resumen.to_string(index=False))

    figura_distribuciones(personas, imagenes, resolver(config["figura"]))
    print(f"\nFigura             -> {resolver(config['figura'])}")

    escribir_config_ultralytics(resolver(config["config_ultralytics"]), salida)
    print(f"Config Ultralytics -> {resolver(config['config_ultralytics'])}")

    # Verificacion: una etiqueta por imagen y una linea por persona.
    print("\n=== verificacion ===")
    todo_ok = True
    for particion in config["particiones"]:
        n_imagenes = len(list((salida / "images" / particion).glob("*.JPG")))
        archivos_txt = list((salida / "labels" / particion).glob("*.txt"))
        n_lineas = sum(
            len([linea for linea in t.read_text(encoding="utf-8").splitlines() if linea.strip()])
            for t in archivos_txt
        )
        esperadas = int((personas["particion"] == particion).sum())
        bien = n_imagenes == len(archivos_txt) and n_lineas == esperadas
        todo_ok = todo_ok and bien
        print(
            f"  {particion:<6} imagenes {n_imagenes:>5} | etiquetas {len(archivos_txt):>5} | "
            f"lineas {n_lineas:>6} | personas en la tabla {esperadas:>6}  "
            f"[{'OK' if bien else 'REVISAR'}]"
        )

    print(f"\nCajas descartadas por degeneradas: {total_descartadas}")
    print("AVISO: las columnas 'oclusion' y 'estrato' quedan vacias porque POP no publica mascaras.")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    try:  # la consola de Windows a veces no acepta acentos
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
