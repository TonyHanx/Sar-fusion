"""Entrena un detector con Ultralytics a partir de un archivo de configuracion.

Para que sirve: tener una sola puerta de entrada al entrenamiento. Todo lo que
cambia entre experimentos (modelo, datos, epocas, lote, resolucion, semilla)
vive en un archivo de configs/, y cada corrida queda anotada en la bitacora.

Que tecnica usa: Ultralytics recibe los parametros del bloque "ultralytics" del
archivo tal cual, sin traducirlos, para que el archivo sea la descripcion exacta
de lo que se ejecuto. Las metricas que van a la bitacora NO se inventan ni se
leen de la pantalla: salen del results.csv que Ultralytics escribe en la carpeta
de la corrida.

Uso:
    python scripts/train.py --config configs/exp1_pop_termico_prueba.yaml
    python scripts/train.py --config configs/exp1_pop_termico.yaml
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
from datetime import datetime
from pathlib import Path

import yaml

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))

# Claves del bloque "ultralytics" que son rutas y hay que volver absolutas,
# porque Ultralytics las resuelve contra su propia carpeta de trabajo.
CLAVES_DE_RUTA = ("data", "project")

# Columnas de la bitacora, en el orden acordado en CLAUDE.md.
COLUMNAS_BITACORA = [
    "fecha",
    "maquina",
    "experimento",
    "modelo",
    "dataset",
    "archivo_config",
    "carpeta_resultados",
    "map50",
    "recall",
    "precision",
    "notas",
]


def resolver(ruta) -> Path:
    """Convierte una ruta del archivo de configuracion en ruta absoluta."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_REPO / ruta


def ruta_relativa(ruta: Path) -> str:
    """Ruta relativa al repositorio para la bitacora; absoluta si queda fuera.

    Las corridas de diagnostico pueden usar configuraciones de fuera del
    repositorio, asi que no se puede dar por hecho que esten dentro.
    """
    return ruta.relative_to(RAIZ_REPO).as_posix() if ruta.is_relative_to(RAIZ_REPO) else str(ruta)


def leer_metricas(carpeta_corrida: Path) -> dict:
    """Lee la ultima fila de results.csv y saca mAP@0.5, precision y recall.

    Ultralytics nombra las columnas con sufijos que cambian entre versiones
    (por ejemplo "metrics/mAP50(B)"), asi que se buscan por el trozo de nombre
    en lugar de por el nombre exacto.
    """
    archivo = carpeta_corrida / "results.csv"
    if not archivo.exists():
        return {"map50": "", "recall": "", "precision": ""}

    with archivo.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return {"map50": "", "recall": "", "precision": ""}

    ultima = {k.strip(): v for k, v in filas[-1].items() if k}

    def buscar(trozo: str) -> str:
        for clave, valor in ultima.items():
            if trozo in clave:
                try:
                    return f"{float(valor):.4f}"
                except (TypeError, ValueError):
                    return ""
        return ""

    return {
        "map50": buscar("mAP50(B)") or buscar("mAP50"),
        "recall": buscar("recall(B)") or buscar("recall"),
        "precision": buscar("precision(B)") or buscar("precision"),
    }


def anotar_en_bitacora(ruta_bitacora: Path, fila: dict) -> None:
    """Agrega una fila a experiments/bitacora.csv, creando el encabezado si falta."""
    ruta_bitacora.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta_bitacora.exists() and ruta_bitacora.stat().st_size > 0
    with ruta_bitacora.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_BITACORA)
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def main() -> int:
    analizador = argparse.ArgumentParser(description="Entrena un detector con Ultralytics.")
    analizador.add_argument("--config", type=Path, required=True, help="Archivo de configs/.")
    argumentos = analizador.parse_args()

    ruta_config = resolver(argumentos.config)
    config = yaml.safe_load(ruta_config.read_text(encoding="utf-8"))
    parametros = dict(config["ultralytics"])
    for clave in CLAVES_DE_RUTA:
        if clave in parametros:
            parametros[clave] = str(resolver(parametros[clave]))

    print(f"Experimento : {config['experimento']}")
    print(f"Config      : {ruta_config}")
    print(f"Modelo      : {config['modelo']}")
    print(f"Datos       : {parametros.get('data')}")
    print(f"Epocas      : {parametros.get('epochs')} | lote: {parametros.get('batch')} | "
          f"imgsz: {parametros.get('imgsz')} | fraccion: {parametros.get('fraction', 1.0)}")
    print()

    # Se importa aqui para que --help no tenga que cargar PyTorch.
    from ultralytics import YOLO

    modelo = YOLO(config["modelo"])
    modelo.train(**parametros)

    # La carpeta real de la corrida la decide Ultralytics (puede anadir sufijos
    # si el nombre ya existia), asi que se pregunta al entrenador.
    carpeta = Path(modelo.trainer.save_dir)
    metricas = leer_metricas(carpeta)

    fila = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "maquina": socket.gethostname(),
        "experimento": config["experimento"],
        "modelo": config["modelo"],
        "dataset": Path(parametros.get("data", "")).name,
        "archivo_config": ruta_relativa(ruta_config),
        "carpeta_resultados": ruta_relativa(carpeta),
        "map50": metricas["map50"],
        "recall": metricas["recall"],
        "precision": metricas["precision"],
        "notas": config.get("notas", ""),
    }
    anotar_en_bitacora(resolver(config["bitacora"]), fila)

    print("\n=== corrida terminada ===")
    print(f"  resultados : {carpeta}")
    print(f"  pesos      : {carpeta / 'weights' / 'best.pt'}")
    print(f"  mAP@0.5    : {fila['map50'] or 'no disponible'}")
    print(f"  precision  : {fila['precision'] or 'no disponible'}")
    print(f"  recall     : {fila['recall'] or 'no disponible'}")
    print(f"  bitacora   : {resolver(config['bitacora'])}")
    return 0


if __name__ == "__main__":
    try:  # la consola de Windows a veces no acepta acentos
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
