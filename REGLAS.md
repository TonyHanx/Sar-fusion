# Proyecto: detección de personas ocultas por vegetación con fusión RGB-térmica

## Contexto

Tesis de pregrado de Ingeniería de Sistemas (Universidad de Lima), curso Seminario de Investigación I.

La pregunta de la investigación es qué forma de combinar una imagen RGB (cámara normal) con una térmica detecta mejor a personas tapadas por vegetación desde un dron. Se comparan tres formas de combinarlas:

- **Fusión temprana:** se juntan las dos imágenes antes de entrar a la red.
- **Fusión tardía:** cada imagen tiene su propio detector y al final se combinan las detecciones.
- **Fusión adaptativa:** la red aprende cuánto confiar en cada imagen según la zona.

Los resultados se miden por nivel de oclusión (qué porcentaje del cuerpo está tapado) y por altitud de vuelo. En esta etapa solo se hace una prueba de concepto: comprobar que todo el flujo funciona y obtener los primeros resultados.

## Cómo explicarme las cosas

Soy estudiante y no domino PyTorch ni Ultralytics. En cada paso, primero dime para qué sirve y después qué técnica usas. Explica los términos técnicos con palabras simples y no me llenes de detalles que no necesito para decidir.

## Reglas de trabajo

- **Trabaja por fases.** Las fases están en FASES.md. Al terminar una, detente y resume tres cosas: qué hiciste, qué salió y qué debo revisar yo. No pases a la siguiente fase hasta que te lo pida.
- **No inventes números.** Todo valor que reportes debe salir de un archivo o de una ejecución.
- **Revisa los datos antes de suponer.** No asumas el formato de un dataset: inspecciónalo y muéstrame lo que encontraste.
- **No toques los originales.** Nunca modifiques `data/raw`. Todo lo que se derive de los datos va a `data/processed`.
- **Nada fijo dentro del código.** Rutas, épocas, tamaño de lote y semilla van en archivos de `configs/`.
- **Bitácora.** Cada entrenamiento o evaluación agrega una fila a `experiments/bitacora.csv` con estos campos: fecha, máquina, experimento, modelo, dataset, archivo de configuración, carpeta de resultados y métricas principales.
- **Semilla 0 en todo.** Así los resultados se pueden repetir.
- **Windows y Linux.** El código debe correr en mi PC con Windows y en el laboratorio con Linux. Usa `pathlib` para las rutas.
- **Español.** Comentarios y mensajes en español.
- **Commits limpios.** Los mensajes de commit no llevan líneas de atribución
  del asistente ni coautores automáticos. Solo describen el cambio.

## Máquinas

- **PC personal (solo para depurar):** Windows, Intel i3-10105F, 16 GB de RAM, NVIDIA RTX 3050 de 6 GB. Aquí solo van pruebas cortas: una fracción pequeña de los datos, de 1 a 3 épocas y lote de 2 a 4.
- **Laboratorio (experimentos que se reportan):** Lenovo ThinkStation P5, Intel Xeon w7-2495X, 256 GB de RAM, NVIDIA RTX 6000 Ada de 48 GB. Todos los resultados del artículo se generan aquí.
- **Versiones:** `scripts/env_info.py` debe imprimir sistema operativo, Python, PyTorch, CUDA, Ultralytics y GPU. Se usa para la tabla del entorno de cómputo.

## Estructura del repositorio

```
sar-fusion/
  REGLAS.md
  FASES.md
  README.md
  requirements.txt
  configs/                yaml de datasets y de entrenamiento
  data/raw/pop/           original, no va a git
  data/raw/wisard/        original, no va a git
  data/processed/         datos preparados, no va a git
  src/                    código reutilizable (datos, modelos, evaluación)
  scripts/                lo que se ejecuta (preparar, entrenar, evaluar)
  runs/                   salidas de entrenamiento, no va a git
  experiments/bitacora.csv
  reports/                tablas y figuras para el artículo
```

## Datos

### POP (solo térmico)

- **Volumen:** 8768 imágenes JPG de 1280 × 1024 y 25 811 cajas de persona.
- **Anotaciones:** formato COCO.
- **Nombre de archivo:** `<escena>_<número>_<altura>.JPG`. La altitud sale del nombre.
- **Particiones oficiales:** entrenamiento F1 a F12, validación F13 a F16 y prueba T1 a T3. Ya están separadas por escena, así que se usan tal cual.
- **Altitudes:** 30, 50 y 70 m. Si la altura del nombre no es exacta, se agrupa en 20 a 40, 40 a 60 y 60 a 80 m, como hicieron Woesle et al. (2026).
- **Máscaras de oclusión:** según la literatura, POP tiene máscaras a nivel de píxel. Hay que verificar si existen para todas las personas y en qué formato vienen.
- **Oclusión alta:** según Woesle et al. (2026), POP casi no tiene personas por encima del 65 % de oclusión.

### WiSARD (RGB y térmico)

- **Pares:** el subconjunto multimodal tiene 15 453 pares tomados al mismo tiempo.
- **Resolución:** RGB de 3840 × 2160 (algunas de 1920 × 1080) y térmica de 640 × 512.
- **Etiquetas:** archivos .txt en formato YOLO (clase 0 = persona, y centro, ancho y alto relativos al tamaño de la imagen). Cada modalidad tiene sus propias etiquetas.
- **Metadatos:** un .yaml por carpeta con las condiciones de vuelo. De ahí sale a qué sitio pertenece cada carpeta.
- **Sitios y fotogramas:** hay 12 sitios. Se anotó 1 de cada 6 fotogramas de video, así que imágenes vecinas son casi iguales. Por eso la partición es por sitio y no por imagen.
- **Lo que no trae:** oclusión ni altitud por imagen.

## Decisiones fijas del diseño (no cambiar sin consultarme)

### Entrenamiento

- **Detector base:** YOLOv8s con pesos preentrenados en COCO (`yolov8s.pt`).
- **Configuración:** 100 épocas, lote 16, `lr0=0.001`, `seed=0` y `deterministic=True`.
- **Optimizador:** `optimizer=SGD`, escrito explícitamente. Con `optimizer=auto`, Ultralytics ignora `lr0`.
- **Resolución:** 1280 en POP y 640 en WiSARD.
- **Aumento de datos:** volteo, mosaico y cambio de escala. En los pares, la misma transformación se aplica a las dos imágenes.

### Oclusión

- **Cálculo:** la oclusión de una persona es λ = píxeles tapados / píxeles totales de su silueta.
- **Estratos:**

| Estrato | Rango |
|---|---|
| Baja | 0 a 35 % |
| Moderada | 35 a 70 % |
| Severa | 70 a 90 % |
| Extrema | más de 90 % |

### WiSARD

- **Registro:** cada imagen RGB se lleva a la geometría de la térmica (640 × 512) con una homografía calculada a partir de puntos de control que marco a mano. Una homografía es una transformación que "encaja" una imagen sobre la otra.
- **Etiquetas de referencia:** las térmicas, para todos los modelos.
- **Partición:** por sitio. Cada sitio queda en una sola partición.

### Modelos

| Modelo | Descripción |
|---|---|
| YOLOv8s RGB | Detector con una sola modalidad, RGB registrada |
| YOLOv8s térmico | Detector con una sola modalidad, térmica |
| Fusión temprana | Una imagen de 4 canales (3 RGB + 1 térmico) |
| Fusión tardía | Los dos YOLOv8s anteriores. Sus detecciones se combinan con un peso por modalidad, ajustado en validación, y luego se eliminan las cajas repetidas con supresión de no máximos (NMS) |
| Fusión adaptativa | Dos redes troncales YOLOv8s, una por modalidad. Se unen en los niveles P3, P4 y P5 con atención espacial y por canal, como en Zou et al. (2024), y siguen con el cuello y la cabeza de YOLOv8s |

### Métricas

- **Desempeño:** mAP@0.5, recall y precisión. El recall y la precisión se calculan con umbral de confianza 0.5.
- **Costo:** número de parámetros y GFLOPs, con entrada de 640 × 640 y la misma herramienta para todos los modelos.
- **Evaluación por estrato:** script propio con `pycocotools`. Al evaluar un estrato, las personas de los otros estratos se marcan como `iscrowd=1` para que no cuenten ni como acierto ni como error.
- **Control de POP:** además, se reporta el mAP@0.5 global que da Ultralytics, para compararlo con el 0.87 publicado por Z. Song et al. (2025).
