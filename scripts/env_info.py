"""Informe del entorno de cómputo.

Para qué sirve: la tesis debe reportar en qué máquina se corrió cada
experimento (sistema operativo, versiones de Python / PyTorch / CUDA /
Ultralytics y modelo de GPU). Este script imprime esos datos y, con la
opción --salida, los guarda en un archivo de texto.

Uso:
    python scripts/env_info.py
    python scripts/env_info.py --salida reports/entorno.txt
"""

from __future__ import annotations

import argparse
import importlib
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

# Librerías cuya versión se reporta en la tabla del entorno.
LIBRERIAS_EXTRA = ["numpy", "cv2", "pandas", "scipy", "matplotlib", "yaml", "thop", "pycocotools"]


def version_de(nombre_modulo: str) -> str:
    """Devuelve la versión de una librería instalada, o 'no instalado'."""
    try:
        modulo = importlib.import_module(nombre_modulo)
    except Exception:  # noqa: BLE001 - si falla la importación, solo se informa
        return "no instalado"
    return str(getattr(modulo, "__version__", "version desconocida"))


def bloque_sistema() -> list[str]:
    """Sistema operativo, máquina y Python."""
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
    """PyTorch, CUDA y GPU detectadas."""
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
    """Ultralytics y el resto de librerías del proyecto."""
    lineas = ["", "== Librerias =="]
    lineas.append(f"{'ultralytics':<18} : {version_de('ultralytics')}")
    for nombre in LIBRERIAS_EXTRA:
        lineas.append(f"{nombre:<18} : {version_de(nombre)}")
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

    if argumentos.salida is not None:
        ruta = argumentos.salida
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(informe + "\n", encoding="utf-8")
        print(f"\nInforme guardado en: {ruta.resolve()}")

    # Codigo de salida 1 si no hay GPU, para notarlo en los scripts del laboratorio.
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
