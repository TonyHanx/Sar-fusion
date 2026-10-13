"""
train.py

¿Para qué sirve?
    Entrenar al detector, que es el paso donde el programa aprende a encontrar
    personas mirando miles de fotos ya marcadas. Este script existe para que
    haya una sola puerta de entrada: nada se escribe a mano en el momento de
    correrlo. Todo lo que cambia entre un experimento y otro está escrito en un
    archivo de la carpeta configs/, así que ese archivo queda como la
    descripción exacta de lo que se hizo.

¿Qué recibe?
    Un archivo de configuración de configs/, que dice qué detector usar, con
    qué fotos, cuántas vueltas dar y con qué ajustes.

¿Qué produce?
    - runs/: el modelo entrenado, las curvas de aprendizaje y las estadísticas
      de la corrida.
    - Una fila nueva en experiments/bitacora.csv con la fecha, la máquina, la
      configuración usada y los resultados. Es el cuaderno de laboratorio del
      proyecto.

¿Cómo se usa?
    python scripts/train.py --config configs/exp1_pop_termico_prueba.yaml
    python scripts/train.py --config configs/exp1_pop_termico.yaml

Una regla del proyecto: los números de la bitácora no se copian de la pantalla.
Se leen del archivo de resultados que escribe el propio programa de
entrenamiento, para que nadie pueda transcribirlos mal.
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

# Estos dos ajustes del archivo de configuracion son rutas de carpetas. Hay que
# escribirlas completas antes de pasarlas, porque el programa de entrenamiento
# las busca a partir de su propia carpeta y no de la del proyecto.
CLAVES_DE_RUTA = ("data", "project")

# Columnas de la bitacora, en el orden acordado para el proyecto.
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
    """Saca los resultados finales del archivo que escribe el entrenamiento.

    El programa de entrenamiento va anotando en un archivo cómo le fue en cada
    vuelta. Aquí se lee la última línea, que es el resultado final, y se toman
    tres números: la nota general del detector (mAP), qué parte de lo que marcó
    eran personas de verdad (precisión) y qué parte de las personas encontró
    (recall).

    Los nombres de las columnas cambian de una versión a otra del programa, así
    que se buscan por un trozo del nombre en lugar de por el nombre exacto.
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

    # El programa de entrenamiento se carga aqui y no al inicio del archivo
    # porque tarda varios segundos en arrancar. Asi, pedir la ayuda del script
    # con --help responde al instante.
    from ultralytics import YOLO

    modelo = YOLO(config["modelo"])
    modelo.train(**parametros)

    # La carpeta donde quedaron los resultados la elige el programa de
    # entrenamiento, y le agrega un numero si ya existia una con ese nombre.
    # Por eso se le pregunta a el en vez de suponerla.
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
