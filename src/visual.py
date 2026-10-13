"""Dibujo de cajas y texto sobre las imagenes, para las figuras del articulo.

Para que sirve: las imagenes de POP son termicas en escala de grises, asi que
el texto blanco o negro se pierde segun la zona. Aqui esta el dibujo comun que
usan los scripts de inspeccion y de predicciones, para que todas las figuras se
vean igual.
"""

from __future__ import annotations

import cv2

# Colores en el orden que usa OpenCV (azul, verde, rojo).
VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)


def texto_con_fondo(imagen, texto: str, origen, color_texto=BLANCO, escala: float = 0.7,
                    grosor: int = 2) -> None:
    """Escribe texto sobre una banda oscura para que se lea en cualquier fondo."""
    (ancho, alto), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x, y = origen
    cv2.rectangle(imagen, (x - 4, y - alto - 6), (x + ancho + 4, y + base), NEGRO, -1)
    cv2.putText(imagen, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color_texto, grosor,
                cv2.LINE_AA)


def dibujar_caja(imagen, x1: int, y1: int, x2: int, y2: int, color, grosor: int = 2,
                 etiqueta: str | None = None) -> None:
    """Dibuja un rectangulo y, si se pide, una etiqueta pegada a su borde superior."""
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, grosor)
    if etiqueta is None:
        return
    escala, grosor_texto = 0.5, 1
    (ancho, alto), base = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor_texto)
    # Si la caja esta muy arriba, la etiqueta va por dentro para no salirse.
    y_texto = y1 - 4 if y1 - alto - 6 > 0 else y2 + alto + 6
    cv2.rectangle(imagen, (x1, y_texto - alto - base), (x1 + ancho + 4, y_texto + base), color, -1)
    cv2.putText(imagen, etiqueta, (x1 + 2, y_texto), cv2.FONT_HERSHEY_SIMPLEX, escala, NEGRO,
                grosor_texto, cv2.LINE_AA)
