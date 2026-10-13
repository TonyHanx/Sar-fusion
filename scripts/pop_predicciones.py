"""
pop_predicciones.py

¿Para qué sirve?
    Mostrar con imágenes qué está haciendo el detector, sin que nadie tenga que
    ejecutar nada. Una tabla de números dice cuánto acierta; estas figuras
    dejan ver en qué acierta y en qué se equivoca, que es lo que se entiende
    de un vistazo en una exposición.

¿Qué recibe?
    Un modelo ya entrenado y las fotos del examen final.

¿Qué produce?
    - reports/predicciones: seis fotos marcadas. En verde las personas reales y
      en rojo lo que el modelo creyó ver, con su nivel de confianza. Una "X"
      señala una falsa alarma. Tres son casos donde acertó del todo y tres son
      casos donde falló, a propósito: un informe honesto muestra las dos caras.
    - Una copia de las curvas de entrenamiento en reports/, porque la carpeta
      donde las guarda el entrenamiento no se sube al repositorio.

¿Cómo se usa?
    python scripts/pop_predicciones.py
    python scripts/pop_predicciones.py --config configs/pop_predicciones.yaml

Las imágenes se eligen solas: el script revisa 200 fotos, cuenta en cada una
cuántas personas encontró y cuántas se le pasaron, y se queda con ejemplos de
distintas alturas de vuelo.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import yaml

RAIZ_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_REPO))

from src import visual  # noqa: E402
from src.datos import pop  # noqa: E402


def leer_config(ruta: Path) -> dict:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def resolver(ruta) -> Path:
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ_REPO / ruta


def iou(caja_a, caja_b) -> float:
    """Mide cuánto se superponen dos rectángulos, de 0 a 1.

    Es la forma estándar de decidir si una detección "le atinó" a una persona.
    Se divide la parte común entre los dos rectángulos por la superficie total
    que ocupan juntos: 1 significa que coinciden exactamente y 0 que no se
    tocan. En este proyecto se considera acierto a partir de 0.5, es decir,
    cuando coinciden al menos a la mitad.
    """
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    ancho, alto = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    interseccion = ancho * alto
    if interseccion <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return interseccion / (area_a + area_b - interseccion)


def emparejar(reales, detecciones, umbral_iou: float):
    """Decide qué detecciones acertaron y cuáles son falsas alarmas.

    Se revisan las detecciones empezando por aquellas en las que el modelo
    confía más, y cada una se asigna a la persona con la que más coincide. Una
    persona ya emparejada no puede volver a usarse: si el modelo dibuja tres
    cajas sobre la misma persona, una cuenta como acierto y las otras dos como
    falsas alarmas. Es el mismo criterio con el que se califica oficialmente a
    los detectores.
    """
    reales_encontradas, detecciones_correctas = set(), set()
    orden = sorted(range(len(detecciones)), key=lambda i: detecciones[i][4], reverse=True)
    for indice_det in orden:
        mejor_iou, mejor_real = 0.0, None
        for indice_real, caja_real in enumerate(reales):
            if indice_real in reales_encontradas:
                continue
            valor = iou(detecciones[indice_det][:4], caja_real)
            if valor > mejor_iou:
                mejor_iou, mejor_real = valor, indice_real
        if mejor_real is not None and mejor_iou >= umbral_iou:
            reales_encontradas.add(mejor_real)
            detecciones_correctas.add(indice_det)
    return reales_encontradas, detecciones_correctas


def dibujar(ruta_imagen: Path, reales, detecciones, encontradas, correctas, etiqueta_modelo: str):
    """Pinta las cajas reales en verde y las detecciones en rojo."""
    imagen = cv2.imread(str(ruta_imagen))

    # Primero las detecciones y encima las cajas reales: cuando una deteccion
    # acierta, las dos cajas casi coinciden y la de abajo quedaria tapada. Lo
    # real tiene que verse siempre.
    for indice, (x1, y1, x2, y2, confianza) in enumerate(detecciones):
        marca = "" if indice in correctas else " X"  # X: deteccion que no corresponde a nadie
        visual.dibujar_caja(
            imagen, int(x1), int(y1), int(x2), int(y2), visual.ROJO, grosor=2,
            etiqueta=f"{confianza:.2f}{marca}",
        )

    for caja in reales:
        x1, y1, x2, y2 = (int(v) for v in caja)
        visual.dibujar_caja(imagen, x1, y1, x2, y2, visual.VERDE, grosor=2)

    perdidas = len(reales) - len(encontradas)
    falsas = len(detecciones) - len(correctas)
    altitud = pop.altitud_desde_nombre(ruta_imagen.name)

    visual.texto_con_fondo(
        imagen,
        f"{ruta_imagen.name} | {altitud:.1f} m ({pop.grupo_altitud(altitud)})",
        (12, 34),
    )
    visual.texto_con_fondo(
        imagen,
        f"reales {len(reales)} | detectadas {len(encontradas)} | "
        f"no detectadas {perdidas} | falsas {falsas}",
        (12, 70),
    )
    visual.texto_con_fondo(imagen, f"modelo: {etiqueta_modelo}", (12, 106), escala=0.55, grosor=1)
    return imagen, perdidas, falsas


def elegir(casos: list[dict], cuantos: int, con_fallos: bool) -> list[dict]:
    """Escoge los ejemplos que se van a guardar como figura.

    Busca casos de distintas alturas de vuelo, para que las figuras no sean
    todas del mismo tipo de foto. Pidiendo con_fallos=False devuelve ejemplos
    donde el modelo acertó del todo; con con_fallos=True devuelve los que más
    se equivocó, que son los que hay que mostrar para no dar una impresión
    falsa de lo bien que funciona.
    """
    if con_fallos:
        candidatos = [c for c in casos if c["perdidas"] > 0 or c["falsas"] > 0]
        candidatos.sort(key=lambda c: (c["perdidas"] + c["falsas"]), reverse=True)
    else:
        candidatos = [c for c in casos if c["perdidas"] == 0 and c["falsas"] == 0 and c["reales"] > 0]
        candidatos.sort(key=lambda c: c["reales"], reverse=True)

    elegidos, grupos_usados = [], set()
    for caso in candidatos:  # primera pasada: un ejemplo por grupo de altitud
        if len(elegidos) >= cuantos:
            break
        if caso["grupo"] not in grupos_usados:
            elegidos.append(caso)
            grupos_usados.add(caso["grupo"])
    for caso in candidatos:  # segunda pasada: completar si faltan
        if len(elegidos) >= cuantos:
            break
        if caso not in elegidos:
            elegidos.append(caso)
    return elegidos


def main() -> int:
    analizador = argparse.ArgumentParser(description="Figuras de prediccion sobre POP.")
    analizador.add_argument(
        "--config", type=Path, default=RAIZ_REPO / "configs" / "pop_predicciones.yaml"
    )
    argumentos = analizador.parse_args()
    config = leer_config(argumentos.config)

    pesos = resolver(config["pesos"])
    if not pesos.exists():
        print(f"ERROR: no existe el archivo de pesos {pesos}")
        return 1

    salida = resolver(config["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    raiz_pop = resolver(config["raiz_pop"])
    particion = config["particion"]

    print(f"Modelo    : {pesos}")
    print(f"Particion : {particion}")
    print(f"Salida    : {salida}\n")

    datos = pop.cargar_particion(raiz_pop, particion)
    registros = list(datos["imagenes"].items())
    sorteo = random.Random(config["semilla"])
    muestra = sorteo.sample(registros, min(int(config["candidatas"]), len(registros)))

    from ultralytics import YOLO

    modelo = YOLO(str(pesos))

    casos = []
    for id_imagen, registro in muestra:
        ruta_imagen = pop.ruta_imagenes(raiz_pop, particion) / registro["file_name"]

        reales = [
            (a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
            for a in datos["cajas"][id_imagen]
        ]
        resultado = modelo.predict(
            source=str(ruta_imagen),
            conf=float(config["confianza"]),
            imgsz=int(config["imgsz"]),
            verbose=False,
        )[0]
        detecciones = [
            (float(x1), float(y1), float(x2), float(y2), float(c))
            for (x1, y1, x2, y2), c in zip(
                resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist()
            )
        ]

        encontradas, correctas = emparejar(reales, detecciones, float(config["iou"]))
        altitud = pop.altitud_desde_nombre(registro["file_name"])
        casos.append(
            {
                "ruta": ruta_imagen,
                "reales_cajas": reales,
                "detecciones": detecciones,
                "encontradas": encontradas,
                "correctas": correctas,
                "reales": len(reales),
                "perdidas": len(reales) - len(encontradas),
                "falsas": len(detecciones) - len(correctas),
                "grupo": pop.grupo_altitud(altitud),
            }
        )

    print(f"Imagenes analizadas: {len(casos)}")
    print(f"  con todo correcto : {sum(1 for c in casos if c['perdidas'] == 0 and c['falsas'] == 0)}")
    print(f"  con algun error   : {sum(1 for c in casos if c['perdidas'] > 0 or c['falsas'] > 0)}\n")

    seleccion = [(c, "acierto") for c in elegir(casos, int(config["n_aciertos"]), con_fallos=False)]
    seleccion += [(c, "fallo") for c in elegir(casos, int(config["n_fallos"]), con_fallos=True)]

    for caso, tipo in seleccion:
        imagen, perdidas, falsas = dibujar(
            caso["ruta"], caso["reales_cajas"], caso["detecciones"],
            caso["encontradas"], caso["correctas"], config["etiqueta_modelo"],
        )
        destino = salida / f"{tipo}_{caso['ruta'].stem}.jpg"
        cv2.imwrite(str(destino), imagen, [cv2.IMWRITE_JPEG_QUALITY, int(config["calidad_jpg"])])
        print(
            f"  {destino.name:<34} {caso['reales']} reales | "
            f"{len(caso['encontradas'])} detectadas | {perdidas} perdidas | {falsas} falsas | "
            f"{caso['grupo']} | {destino.stat().st_size / 1024:.0f} KB"
        )

    origen_curvas = resolver(config["curvas_origen"])
    destino_curvas = resolver(config["curvas_destino"])
    if origen_curvas.exists():
        shutil.copy2(origen_curvas, destino_curvas)
        print(f"\nCurvas de entrenamiento copiadas a {destino_curvas}")
    else:
        print(f"\nAVISO: no se encontro {origen_curvas}; no se copiaron las curvas.")

    return 0


if __name__ == "__main__":
    try:  # la consola de Windows a veces no acepta acentos
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
