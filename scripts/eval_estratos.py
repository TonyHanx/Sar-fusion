"""Evalua un modelo entrenado sobre una particion, global y por estrato.

Para que sirve: el numero global esconde lo que de verdad interesa. Este script
separa el desempeno por altitud de vuelo y por nivel de oclusion, que es la
comparacion que sostiene la tesis.

Que hace:
  1. Corre el modelo sobre la particion elegida y guarda las predicciones en
     formato COCO (quedan en runs/, para poder re-evaluar sin volver a inferir).
  2. Calcula con pycocotools el mAP@0.5, y aparte la precision y el recall al
     umbral de confianza fijado (0.5).
  3. Repite el calculo por cada estrato y lo guarda todo en un CSV.
  4. Anota la evaluacion en la bitacora.

Nota: la oclusion depende de personas.csv. POP no publica mascaras, asi que esa
columna esta vacia y el script lo informa en vez de inventar estratos.

Uso:
    python scripts/eval_estratos.py
    python scripts/eval_estratos.py --config configs/eval_estratos.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))

from src.datos import pop  # noqa: E402
from src.evaluacion import estratos  # noqa: E402

COLUMNAS = [
    "fecha", "experimento", "modelo", "particion", "tipo_estrato", "estrato",
    "imagenes", "personas", "map50", "map50_95", "precision", "recall",
    "tp", "fp", "fn", "confianza", "iou",
]

COLUMNAS_BITACORA = [
    "fecha", "maquina", "experimento", "modelo", "dataset", "archivo_config",
    "carpeta_resultados", "map50", "recall", "precision", "notas",
]


def leer_config(ruta: Path) -> dict:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def resolver(ruta) -> Path:
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_REPO / ruta


def ruta_relativa(ruta: Path) -> str:
    return ruta.relative_to(RAIZ_REPO).as_posix() if ruta.is_relative_to(RAIZ_REPO) else str(ruta)


def predecir(modelo, datos: dict, carpeta_imagenes: Path, config: dict) -> list[dict]:
    """Corre el modelo y devuelve las detecciones en formato COCO.

    Se usa una confianza minima muy baja a proposito: el mAP necesita la curva
    completa de precision y recall. El umbral de 0.5 se aplica despues, solo
    para la precision y el recall que se reportan.
    """
    por_nombre = {registro["file_name"]: id_imagen for id_imagen, registro in datos["imagenes"].items()}
    rutas = [str(carpeta_imagenes / nombre) for nombre in por_nombre]

    # Las rutas se procesan por grupos pequenos. Si se le pasa la lista completa,
    # Ultralytics la toma como un unico lote y a 1280 px eso son mas de 13 GB:
    # no entra en una GPU de 6 GB.
    lote = int(config["lote"])
    detecciones = []
    for inicio in range(0, len(rutas), lote):
        grupo = rutas[inicio:inicio + lote]
        for resultado in modelo.predict(
            source=grupo,
            conf=float(config["confianza_prediccion"]),
            iou=0.7,                   # NMS, el valor por omision de Ultralytics
            imgsz=int(config["imgsz"]),
            max_det=int(config["max_det"]),
            stream=True,
            verbose=False,
        ):
            id_imagen = por_nombre[Path(resultado.path).name]
            for (x1, y1, x2, y2), score in zip(
                resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist()
            ):
                detecciones.append(
                    {
                        "image_id": id_imagen,
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO usa [x, y, ancho, alto]
                        "score": float(score),
                    }
                )
        if (inicio // lote) % 20 == 0:
            print(f"    {min(inicio + lote, len(rutas))}/{len(rutas)} imagenes")
    return detecciones


def estratos_de_oclusion(ruta_personas: Path, particion: str) -> dict[str, set[int]]:
    """Lee personas.csv y agrupa los ids de anotacion por estrato de oclusion.

    Devuelve un diccionario vacio si la columna no tiene datos, que es el caso
    de POP hoy.
    """
    if not ruta_personas.exists():
        return {}
    tabla = pd.read_csv(ruta_personas)
    tabla = tabla[tabla["particion"] == particion]
    if "estrato" not in tabla.columns or tabla["estrato"].isna().all():
        return {}
    return {
        str(nombre): set(grupo["id_anotacion"].astype(int))
        for nombre, grupo in tabla.groupby("estrato")
    }


def anotar_en_bitacora(ruta: Path, fila: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.exists() and ruta.stat().st_size > 0
    with ruta.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_BITACORA)
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def main() -> int:
    analizador = argparse.ArgumentParser(description="Evaluacion global y por estrato.")
    analizador.add_argument(
        "--config", type=Path, default=RAIZ_REPO / "configs" / "eval_estratos.yaml"
    )
    argumentos = analizador.parse_args()
    ruta_config = resolver(argumentos.config)
    config = leer_config(ruta_config)

    pesos = resolver(config["pesos"])
    if not pesos.exists():
        print(f"ERROR: no existe el archivo de pesos {pesos}")
        return 1

    raiz_pop = resolver(config["raiz_pop"])
    particion = config["particion"]
    confianza = float(config["confianza_metricas"])
    umbral_iou = float(config["iou"])

    print(f"Modelo    : {pesos}")
    print(f"Particion : {particion}")
    print(f"Metricas  : mAP@0.5 con pycocotools; precision y recall con confianza >= {confianza}\n")

    datos = pop.cargar_particion(raiz_pop, particion)
    registros = [
        {"id": id_imagen, "file_name": r["file_name"], "width": r["width"], "height": r["height"]}
        for id_imagen, r in datos["imagenes"].items()
    ]
    anotaciones = [a for lista in datos["cajas"].values() for a in lista]

    from ultralytics import YOLO

    modelo = YOLO(str(pesos))
    print(f"Infiriendo sobre {len(registros)} imagenes...")
    detecciones = predecir(modelo, datos, pop.ruta_imagenes(raiz_pop, particion), config)
    print(f"  detecciones obtenidas: {len(detecciones)}")

    ruta_predicciones = resolver(config["salida_predicciones"])
    ruta_predicciones.parent.mkdir(parents=True, exist_ok=True)
    ruta_predicciones.write_text(json.dumps(detecciones), encoding="utf-8")
    print(f"  predicciones guardadas en {ruta_predicciones} "
          f"({ruta_predicciones.stat().st_size / 1048576:.1f} MB)\n")

    filas = []
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")

    def agregar(tipo: str, nombre: str, resultado: dict) -> None:
        filas.append(
            {
                "fecha": fecha,
                "experimento": config["experimento"],
                "modelo": ruta_relativa(pesos),
                "particion": particion,
                "tipo_estrato": tipo,
                "estrato": nombre,
                "imagenes": resultado["imagenes"],
                "personas": resultado["personas"],
                "map50": round(resultado["map50"], 4),
                "map50_95": round(resultado["map50_95"], 4),
                "precision": round(resultado["precision"], 4),
                "recall": round(resultado["recall"], 4),
                "tp": resultado["tp"], "fp": resultado["fp"], "fn": resultado["fn"],
                "confianza": confianza, "iou": umbral_iou,
            }
        )
        print(f"  {tipo:<9} {nombre:<12} imagenes {resultado['imagenes']:>5} | "
              f"personas {resultado['personas']:>5} | mAP@0.5 {resultado['map50']:.4f} | "
              f"P {resultado['precision']:.4f} | R {resultado['recall']:.4f}")

    # 1) Global.
    gt_completo = estratos.construir_gt(registros, anotaciones)
    print("=== resultados ===")
    agregar("global", "todas", estratos.evaluar(gt_completo, detecciones, confianza, umbral_iou))

    # 2) Por altitud: la altitud es propia de la imagen, asi que el estrato se
    #    arma restringiendo la evaluacion a las imagenes de ese grupo.
    grupos: dict[str, list[int]] = {}
    for registro in registros:
        grupo = pop.grupo_altitud(pop.altitud_desde_nombre(registro["file_name"]))
        grupos.setdefault(grupo or "fuera de rango", []).append(registro["id"])
    for nombre in sorted(grupos):
        agregar(
            "altitud", nombre,
            estratos.evaluar(gt_completo, detecciones, confianza, umbral_iou, grupos[nombre]),
        )

    # 3) Por oclusion: es propia de cada persona, asi que se evalua sobre todas
    #    las imagenes marcando como iscrowd=1 a las personas de otros estratos.
    por_oclusion = estratos_de_oclusion(resolver(config["tabla_personas"]), particion)
    if not por_oclusion:
        print("\n  AVISO: no hay estratos de oclusion en personas.csv.")
        print("  POP no publica mascaras, asi que esa parte de la tabla queda pendiente.")
    else:
        todos_los_ids = {a["id"] for a in anotaciones}
        for nombre in sorted(por_oclusion):
            ids_del_estrato = por_oclusion[nombre]
            gt = estratos.construir_gt(registros, anotaciones, todos_los_ids - ids_del_estrato)
            agregar("oclusion", nombre, estratos.evaluar(gt, detecciones, confianza, umbral_iou))

    # 4) Control: el mAP@0.5 que calcula Ultralytics por su cuenta. Sirve para
    #    dos cosas: compararlo con el 0.87 publicado y comprobar que nuestro
    #    calculo con pycocotools no se desvia (son implementaciones distintas,
    #    asi que una diferencia de pocas centesimas es normal).
    if config.get("control_ultralytics", True):
        print("\n=== control: evaluacion propia de Ultralytics ===")
        metricas = modelo.val(
            data=str(resolver(config["datos_ultralytics"])),
            split=particion,
            imgsz=int(config["imgsz"]),
            batch=int(config["lote"]),
            conf=0.001,
            iou=umbral_iou,
            plots=False,
            verbose=False,
        )
        filas.append(
            {
                "fecha": fecha,
                "experimento": config["experimento"],
                "modelo": ruta_relativa(pesos),
                "particion": particion,
                "tipo_estrato": "control",
                "estrato": "ultralytics",
                "imagenes": len(registros),
                "personas": len(anotaciones),
                "map50": round(float(metricas.box.map50), 4),
                "map50_95": round(float(metricas.box.map), 4),
                "precision": round(float(metricas.box.mp), 4),
                "recall": round(float(metricas.box.mr), 4),
                "tp": "", "fp": "", "fn": "",
                "confianza": 0.001, "iou": umbral_iou,
            }
        )
        print(f"  Ultralytics  mAP@0.5 {metricas.box.map50:.4f} | "
              f"nuestro calculo {filas[0]['map50']:.4f} | "
              f"diferencia {abs(metricas.box.map50 - filas[0]['map50']):.4f}")

    ruta_csv = resolver(config["salida_resultados"])
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    with ruta_csv.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nResultados -> {ruta_csv}")

    global_ = filas[0]
    anotar_en_bitacora(
        resolver(config["bitacora"]),
        {
            "fecha": fecha,
            "maquina": socket.gethostname(),
            "experimento": f"{config['experimento']} (evaluacion {particion})",
            "modelo": ruta_relativa(pesos),
            "dataset": f"POP {particion}",
            "archivo_config": ruta_relativa(ruta_config),
            "carpeta_resultados": ruta_relativa(ruta_csv),
            "map50": global_["map50"],
            "recall": global_["recall"],
            "precision": global_["precision"],
            "notas": config.get("notas", ""),
        },
    )
    print(f"Bitacora   -> {resolver(config['bitacora'])}")
    return 0


if __name__ == "__main__":
    try:  # la consola de Windows a veces no acepta acentos
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
