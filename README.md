# Detección de personas ocultas por vegetación con fusión RGB-térmica

Tesis de pregrado en Ingeniería de Sistemas, Universidad de Lima.

La pregunta es **qué forma de combinar una imagen RGB con una térmica detecta mejor a las personas tapadas por vegetación desde un dron**. Se comparan tres formas de combinarlas: fusión temprana (se juntan las imágenes antes de entrar a la red), fusión tardía (cada imagen tiene su detector y se combinan las detecciones) y fusión adaptativa (la red aprende cuánto confiar en cada imagen según la zona).

Esta etapa es una prueba de concepto: comprobar que el flujo completo funciona y obtener los primeros resultados. El plan por fases está en [FASES.md](FASES.md) y las reglas de trabajo en [CLAUDE.md](CLAUDE.md).

## Estado

| Fase | Qué cubre | Estado |
|---|---|---|
| 0 | Entorno y repositorio | Terminada |
| 1 | Conocer POP | Terminada |
| 2 | POP en formato YOLO y conteos | Terminada |
| 3a | Entrenador, bitácora y prueba en PC | Terminada |
| 3b | Evaluación por estrato | Pendiente |
| 4–10 | WiSARD, registro y las tres fusiones | Pendientes |

---

## 1. Cómo son los datos

POP son imágenes térmicas tomadas desde un dron. Las personas aparecen como manchas claras, porque el calor del cuerpo contrasta con el fondo.

![Personas a 31 metros](reports/pop_muestras/train_F7_0184_31.135.jpg)

*A 31 metros de altura se distingue la silueta completa: se ven brazos y piernas. Las cajas verdes son las anotaciones que trae el dataset.*

![Personas a 71 metros](reports/pop_muestras/train_F8_0593_71.069.jpg)

*A 71 metros la misma persona ocupa unos 20 píxeles y queda reducida a una mancha. Esta es la dificultad central del problema.*

Las otras ocho muestras están en [`reports/pop_muestras/`](reports/pop_muestras).

## 2. Qué contiene el dataset

![Distribuciones de POP](reports/pop_distribuciones.png)

*Izquierda: los vuelos se hicieron a tres altitudes (30, 50 y 70 m), salvo en la partición de prueba, que es un barrido continuo entre 45 y 55 m. Derecha: las personas de prueba son más pequeñas que las de entrenamiento, así que la prueba es más exigente.*

| Partición | Escenas | Imágenes | Personas |
|---|---|---|---|
| Entrenamiento | F1–F12 | 5548 | 18 425 |
| Validación | F13–F16 | 2320 | 6 545 |
| Prueba | T1–T3 | 900 | 1 840 |
| **Total** | | **8768** | **26 810** |

Los conteos salen de los archivos de anotación, no del artículo: ver [`reports/tabla9_pop.csv`](reports/tabla9_pop.csv) para el detalle por altitud. Se descartó una caja degenerada de 1×1 px en validación.

## 3. Entrenamiento

![Curvas de entrenamiento](reports/exp1_prueba_5epocas_curvas.png)

*Las tres pérdidas de arriba bajan y las métricas de la derecha suben: el modelo aprende. La pérdida de clasificación cayó de 3.39 a 0.61 y el mAP@0.5 llegó a 0.828.*

> **Importante:** estas curvas son de un **diagnóstico de 5 épocas con el 10 % de los datos**, hecho en una PC para comprobar que el flujo funciona. No son el resultado del artículo. La corrida que se reporta son 100 épocas con los datos completos, en el laboratorio, y busca reproducir el **mAP@0.5 = 0.87** publicado por Song et al. (2025).

## 4. Qué detecta el modelo

En las siguientes figuras, **verde = persona real** y **rojo = detección del modelo**, con su nivel de confianza. Una `X` marca una detección que no corresponde a ninguna persona.

![Detección correcta a 49 metros](reports/predicciones/acierto_T3_0030_48.818.jpg)

*Caso bueno: las tres personas detectadas, con confianzas de 0.65 a 0.89, y ninguna detección falsa.*

![Fallo a 30 metros](reports/predicciones/fallo_T1_0206_29.603.jpg)

*Caso malo: encuentra a la persona, pero además marca diez manchas de vegetación caliente como si fueran personas. Es el error típico de un modelo poco entrenado.*

![Fallo a 70 metros](reports/predicciones/fallo_T1_0247_69.602.jpg)

*A 70 metros se le pasa una de las dos personas y produce cinco detecciones falsas: cuanto más alto vuela el dron, más difícil se vuelve todo.*

Los seis ejemplos están en [`reports/predicciones/`](reports/predicciones).

---

## Una limitación que condiciona el proyecto

**POP no publica máscaras de oclusión.** Las 26 811 anotaciones traen `segmentation: null`, y en el repositorio de OSF solo hay imágenes, videos y un script de conversión. El artículo define la oclusión como λ = 1 − píxeles visibles / píxeles totales, pero esa medición se hizo en un experimento aparte y no se anotó persona por persona.

Como la tesis compara el desempeño **por nivel de oclusión**, esto está pendiente de resolver: se consultó a los autores del dataset. Mientras tanto, POP se estratifica por altitud, que sí se puede obtener del nombre de cada archivo.

## Cómo se reproduce

```bash
conda activate sar
python scripts/env_info.py                 # entorno: SO, Python, PyTorch, CUDA, GPU
python scripts/pop_explorar.py             # muestras de POP con sus cajas
python scripts/pop_preparar.py             # POP en formato YOLO + tablas + figura
python scripts/train.py --config configs/exp1_pop_termico.yaml
python scripts/pop_predicciones.py         # figuras de predicción
```

Todo lo que cambia entre corridas (rutas, épocas, lote, semilla) está en [`configs/`](configs), y cada entrenamiento agrega una fila a [`experiments/bitacora.csv`](experiments/bitacora.csv). La semilla es 0 en todo. Los datos (`data/`) y las salidas de entrenamiento (`runs/`) no se versionan.

## Créditos

Las imágenes térmicas provienen del dataset **POP**, publicado con licencia [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) en [osf.io/kmcva](https://osf.io/kmcva):

> Song, Z., Yan, Y., Cao, Y., Jin, S., Qi, F., Li, Z., Lei, T., Chen, L., Jing, Y., Xia, J., Liang, X., & Lu, G. (2025). An infrared dataset for partially occluded person detection in complex environment for search and rescue. *Scientific Data*. https://doi.org/10.1038/s41597-025-04600-0
