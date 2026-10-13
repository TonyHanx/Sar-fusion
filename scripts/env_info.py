"""
env_info.py

¿Para qué sirve?
    Deja constancia de en qué computadora se hizo cada experimento. Un mismo
    programa puede dar resultados distintos en máquinas distintas, así que el
    artículo tiene que decir con qué equipo y con qué versiones se trabajó.
    Este script reúne esos datos en una sola hoja.

¿Qué recibe?
    Nada. Le pregunta a la propia computadora.

¿Qué produce?
    - Un informe en pantalla: sistema operativo, versión de Python, de los
      programas de inteligencia artificial que usamos, y el modelo de tarjeta
      gráfica.
    - El mismo informe en un archivo de texto, si se le indica con --salida.
    - Además avisa si la tarjeta gráfica NO está disponible, que es el error
      más común al instalar todo en una máquina nueva.

¿Cómo se usa?
    python scripts/env_info.py
    python scripts/env_info.py --salida reports/entorno.txt
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

# Programas de los que reportamos la versión en la tabla del artículo.
# Cada par es (nombre con el que se importa, nombre con el que se instala),
# porque a veces no coinciden: "cv2" se instala como "opencv-python".
LIBRERIAS_EXTRA = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("thop", "thop"),
    ("pycocotools", "pycocotools"),
]


def version_de(nombre_modulo: str, nombre_paquete: str | None = None) -> str:
    """Averigua qué versión de un programa está instalada.

    Primero se le pregunta al instalador, que siempre conoce la versión exacta.
    Algunos programas, como pycocotools, no la dicen por sí mismos, y entonces
    se recurre al propio módulo como segunda opción.
    """
    try:
        modulo = importlib.import_module(nombre_modulo)
    except Exception:  # noqa: BLE001 - si no se puede importar, solo se informa
        return "no instalado"
    try:
        return metadata.version(nombre_paquete or nombre_modulo)
    except metadata.PackageNotFoundError:
        return str(getattr(modulo, "__version__", "version desconocida"))


def bloque_sistema() -> list[str]:
    """Datos de la computadora: qué sistema tiene y qué versión de Python usa."""
    return [
        "== Sistema ==",
        f"Fecha              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Maquina (hostname) : {socket.gethostname()}",
        f"Sistema operativo  : {platform.platform()}",
        f"Procesador         : {platform.processor() or 'desconocido'}",
        f"Arquitectura       : {platform.machine()}",
        "",
        "== Python ==",
        f"Version            : {platform.python_version()}",
        f"Interprete         : {sys.executable}",
    ]


def bloque_pytorch() -> list[str]:
    """Datos de la tarjeta gráfica y del programa que la maneja.

    PyTorch es el programa que hace los cálculos del entrenamiento, y CUDA es
    la pieza que le permite usar la tarjeta gráfica en lugar del procesador.
    Sin esa pieza todo funciona igual, pero cien veces más lento, así que
    conviene comprobarlo antes de empezar.
    """
    lineas = ["", "== PyTorch y CUDA =="]
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        lineas.append(f"PyTorch            : no instalado ({error})")
        return lineas

    cuda_disponible = torch.cuda.is_available()
    lineas += [
        f"PyTorch            : {torch.__version__}",
        f"CUDA de PyTorch    : {torch.version.cuda or 'compilado sin CUDA (solo CPU)'}",
        f"cuDNN              : {torch.backends.cudnn.version() or 'no disponible'}",
        f"CUDA disponible    : {'SI' if cuda_disponible else 'NO'}",
    ]

    if not cuda_disponible:
        lineas.append(
            "AVISO              : PyTorch no ve la GPU. Revisa el driver de NVIDIA "
            "y que se haya instalado la version de PyTorch con CUDA."
        )
        return lineas

    # Datos de cada tarjeta encontrada. La memoria es el limite real del
    # proyecto: decide cuantas imagenes se pueden procesar a la vez.
    lineas += ["", "== GPU =="]
    lineas.append(f"GPUs detectadas    : {torch.cuda.device_count()}")
    for indice in range(torch.cuda.device_count()):
        propiedades = torch.cuda.get_device_properties(indice)
        memoria_gb = propiedades.total_memory / (1024 ** 3)
        lineas += [
            f"  [{indice}] Nombre        : {propiedades.name}",
            f"  [{indice}] Memoria       : {memoria_gb:.1f} GB",
            f"  [{indice}] Capacidad     : sm_{propiedades.major}{propiedades.minor}",
        ]
    return lineas


def bloque_librerias() -> list[str]:
    """Versiones de los programas que usa el proyecto."""
    lineas = ["", "== Librerias =="]
    lineas.append(f"{'ultralytics':<18} : {version_de('ultralytics')}")
    for nombre_modulo, nombre_paquete in LIBRERIAS_EXTRA:
        lineas.append(f"{nombre_modulo:<18} : {version_de(nombre_modulo, nombre_paquete)}")
    return lineas


def main() -> int:
    analizador = argparse.ArgumentParser(description="Informe del entorno de computo.")
    analizador.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Archivo donde guardar el informe (por ejemplo reports/entorno.txt).",
    )
    argumentos = analizador.parse_args()

    informe = "\n".join(bloque_sistema() + bloque_pytorch() + bloque_librerias())
    print(informe)

    # Si se pidio un archivo, se guarda ahi el mismo texto que se mostro.
    if argumentos.salida is not None:
        ruta = argumentos.salida
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(informe + "\n", encoding="utf-8")
        print(f"\nInforme guardado en: {ruta.resolve()}")

    # El script termina con un aviso de error cuando no hay tarjeta grafica.
    # Asi, en el laboratorio, otro programa puede detenerse en vez de empezar
    # un entrenamiento de horas que iria por el camino lento.
    try:
        import torch

        return 0 if torch.cuda.is_available() else 1
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    try:  # la consola de Windows a veces no acepta acentos
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
