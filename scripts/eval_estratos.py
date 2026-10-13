"""
eval_estratos.py

¿Para qué sirve?
    Tomar la nota del detector, pero separada por condiciones. Un solo número
    para todo el dataset esconde lo que de verdad importa: no es lo mismo
    encontrar a una persona con el dron a 30 metros que a 70, ni con el cuerpo
    despejado que medio tapado por un árbol. Esa comparación es justamente la
    pregunta de la tesis, así que aquí se mide por separado.

¿Qué recibe?
    Un modelo ya entrenado y el grupo de fotos del examen final (las que el
    modelo nunca vio durante el entrenamiento).

¿Qué produce?
    - runs/eval/: todo lo que el modelo detectó, guardado en un archivo. Así se
      puede volver a calcular cualquier medida sin repetir el trabajo pesado.
    - reports/exp1_resultados.csv: una fila con los resultados generales y una
      fila por cada altura de vuelo.
    - Una fila nueva en la bitácora del proyecto.

¿Cómo se usa?
    python scripts/eval_estratos.py
    python scripts/eval_estratos.py --config configs/eval_estratos.yaml

Sobre la oclusión: el script también sabe medir por cuánto tapa la vegetación a
cada persona, pero ese dato no viene en POP. Cuando falta, avisa en pantalla en
lugar de inventar los grupos.
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
    """Le muestra al modelo cada foto y anota todo lo que cree ver.

    Se le pide que reporte incluso las cajas en las que apenas confía. Parece
    raro, pero la nota general del detector se calcula justamente recorriendo
    todos los niveles de confianza, de los más dudosos a los más seguros. El
    corte en 0.5 se aplica más adelante, solo para la precisión y el recall.
    """
    por_nombre = {registro["file_name"]: id_imagen for id_imagen, registro in datos["imagenes"].items()}
    rutas = [str(carpeta_imagenes / nombre) for nombre in por_nombre]

    # Las fotos se procesan de a pocas. Si se le entrega la lista completa, el
    # programa intenta cargar las 900 de golpe en la tarjeta grafica: son mas
    # de 13 GB y la tarjeta de esta PC tiene 6.
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

    # 2) Por altura de vuelo. Todas las personas de una foto fueron vistas
    #    desde la misma altura, asi que basta con medir usando solo las fotos
    #    de ese grupo. Si se dejaran las demas, sus falsas alarmas contarian
    #    igual y el resultado no diria nada sobre esa altura.
    grupos: dict[str, list[int]] = {}
    for registro in registros:
        grupo = pop.grupo_altitud(pop.altitud_desde_nombre(registro["file_name"]))
        grupos.setdefault(grupo or "fuera de rango", []).append(registro["id"])
    for nombre in sorted(grupos):
        agregar(
            "altitud", nombre,
            estratos.evaluar(gt_completo, detecciones, confianza, umbral_iou, grupos[nombre]),
        )

    # 3) Por cuanto tapa la vegetacion. Aqui el caso es distinto: en una misma
    #    foto puede haber una persona despejada y otra casi escondida. Se usan
    #    todas las fotos, pero las personas de los otros grupos se marcan como
    #    "zona a ignorar": no cuentan ni como acierto ni como error.
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

    # 4) Control de calidad. Se le pide al programa de entrenamiento que mida
    #    el mismo modelo por su cuenta, con su propio codigo. Si su nota y la
    #    nuestra coinciden, es senal de que nuestra forma de medir es correcta.
    #    Son dos programas distintos, asi que una diferencia minima es normal.
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
