# Fases de la experimentación

Cada fase tiene un mensaje para pegar en Claude Code y una lista de lo que debes revisar antes de pasar a la siguiente. No avances si la revisión no sale bien.

---

## Fase 0. Entorno y repositorio (PC)

**Para qué:** tener un lugar ordenado donde vive el código y un entorno de Python con todo lo necesario, igual en la PC y en el laboratorio.

**Mensaje:**

> Lee CLAUDE.md. Vamos con la Fase 0:
>
> 1. Crea la estructura de carpetas del repositorio.
> 2. Crea un `.gitignore` que excluya `data/`, `runs/` y los archivos `*.pt`.
> 3. Crea un `requirements.txt` con ultralytics, pycocotools, opencv-python, pandas, scipy, matplotlib, pyyaml y thop.
> 4. Escribe `scripts/env_info.py`.
> 5. Dame los comandos exactos para crear un entorno con conda e instalar PyTorch con CUDA para mi RTX 3050 en Windows.
>
> Explícame cada comando antes de que lo ejecute.

**Revisa:**

- `python scripts/env_info.py` muestra tu RTX 3050 y dice que CUDA está disponible.
- El repositorio está subido a GitHub (privado), sin la carpeta `data`.

---

## Fase 1. Conocer POP

**Para qué:** entender cómo vienen los datos antes de programar nada encima. Es lo que más errores evita.

**Antes:** descarga POP desde osf.io/kmcva y ponlo en `data/raw/pop`.

**Mensaje:**

> Fase 1. Descargué POP en `data/raw/pop`. Inspecciona, sin modificar nada:
>
> - las carpetas;
> - los archivos de anotación y su formato;
> - si existen máscaras de oclusión, en qué formato están y cómo se relacionan con cada persona;
> - cómo sacar la altitud del nombre de archivo.
>
> Después escribe `scripts/pop_explorar.py`, que dibuje las cajas (y las máscaras, si existen) sobre 10 imágenes al azar y las guarde en `reports/pop_muestras/`. Resúmeme en palabras simples lo que encontraste.

**Revisa:**

- Abre las 10 imágenes: las cajas deben estar sobre las personas.
- Confirma que hay 8768 imágenes y 25 811 cajas, y en qué carpeta está cada partición.
- Averigua si hay máscaras de oclusión. Esto es clave: sin máscaras no se puede estratificar POP. Si no las hay, avísame antes de seguir.

---

## Fase 2. Preparar POP y contar por estrato

**Para qué:**

- Dejar POP en el formato que entiende Ultralytics.
- Saber cuántas personas hay en cada nivel de oclusión. Esos números van a la Tabla 9 del artículo y dicen si el tramo de más del 70 % tiene casos suficientes.

**Mensaje:**

> Fase 2.
>
> 1. Convierte las anotaciones de POP al formato YOLO en `data/processed/pop`, manteniendo las particiones oficiales.
> 2. Para cada persona, calcula la oclusión λ con su máscara y saca la altitud del nombre del archivo.
> 3. Guarda una tabla con una fila por persona (imagen, partición, altitud, oclusión y estrato) en `data/processed/pop/personas.csv`.
> 4. Genera `reports/tabla9_pop.csv` con el número de imágenes y de personas por partición y por estrato, y un histograma de la oclusión.
> 5. Crea `configs/pop.yaml` para Ultralytics.

**Revisa:**

- Los totales coinciden con los de la Fase 1.
- Cuántas personas quedan por encima del 70 % en la partición de prueba. Pásame la tabla.

---

## Fase 3. Experimento 1: YOLOv8s térmico en POP

**Para qué:** comprobar que tu flujo completo (datos, entrenamiento y evaluación) funciona. Para eso intenta reproducir un resultado ya publicado: si sale un mAP@0.5 cerca de 0.87, el flujo está bien y puedes confiar en él para lo demás.

**Mensaje 3a:**

> Fase 3.
>
> 1. Escribe `scripts/train.py`, que lea una configuración de `configs/`, entrene YOLOv8s con Ultralytics y registre la corrida en la bitácora.
> 2. Haz una prueba corta en mi PC: 10 % de los datos, 1 época y lote 4.
> 3. Cuando funcione, dame el comando exacto para la corrida completa en el laboratorio.

**Mensaje 3b**, mientras corre el entrenamiento en el laboratorio:

> Escribe `scripts/eval_estratos.py`. Debe:
>
> 1. tomar un modelo entrenado y la partición de prueba;
> 2. guardar las predicciones;
> 3. calcular con pycocotools el mAP@0.5, el recall y la precisión (con confianza 0.5): global, por estrato de oclusión y por altitud;
> 4. guardar el resultado en `reports/exp1_resultados.csv`.
>
> Pruébalo con el modelo de la prueba corta.

**Revisa:**

- El mAP@0.5 global de Ultralytics está cerca de 0.87.
- El mAP@0.5 baja a medida que sube la oclusión. Es lo que se espera.

---

## Fase 4. Conocer WiSARD y partirlo por sitio

**Para qué:** encontrar los pares RGB-térmicos y repartir los sitios entre entrenamiento, validación y prueba. Así el modelo se evalúa en lugares que nunca vio.

**Antes:** descarga WiSARD y ponlo en `data/raw/wisard`.

**Mensaje:**

> Fase 4. Descargué WiSARD en `data/raw/wisard`. Inspecciona:
>
> - cómo están organizados los pares multimodales;
> - cómo se empareja cada RGB con su térmica;
> - qué dicen los yaml de metadatos;
> - cómo saber a qué sitio pertenece cada carpeta.
>
> Después propón una partición por sitio que se acerque a 70 % para entrenamiento, 15 % para validación y 15 % para prueba, contando pares y personas. Muéstramela antes de guardarla.

**Revisa:**

- Ningún sitio aparece en dos particiones.
- La prueba no quedó con un solo tipo de terreno.

---

## Fase 5. Registrar RGB sobre la térmica

**Para qué:** las dos cámaras no ven exactamente lo mismo, porque tienen distinto tamaño y ángulo. Para combinarlas, cada píxel RGB debe caer sobre su píxel térmico.

**Mensaje:**

> Fase 5.
>
> 1. Haz una herramienta con OpenCV, `scripts/marcar_puntos.py`, para que yo marque al menos 4 puntos correspondientes entre una imagen RGB y su térmica. Lo haré por carpeta de vuelo, o por cámara si comprobamos que la relación entre ambas es fija.
> 2. Con `cv2.findHomography`, calcula la transformación y guárdala.
> 3. Escribe `scripts/registrar.py`, que lleve cada RGB a 640 × 512 en la geometría de la térmica y guarde los pares en `data/processed/wisard`.
> 4. Para verificar el registro:
>    - superpón ambas imágenes en 20 pares y guarda esas imágenes en `reports/registro/`;
>    - mide la distancia en píxeles entre los centros de las cajas RGB transformadas y los de las cajas térmicas.

**Revisa:**

- En las superposiciones, las personas coinciden en las dos imágenes.
- La distancia promedio entre centros es de pocos píxeles. Anota el valor: va al artículo.

---

## Fase 6. Referencias de una sola modalidad en WiSARD

**Para qué:** sin estas referencias no sabrías si la fusión ayuda. Son el punto de comparación: lo que logra cada cámara por separado.

**Mensaje:**

> Fase 6.
>
> 1. Entrena un YOLOv8s solo con las térmicas y otro solo con las RGB registradas, usando la partición de WiSARD y las etiquetas térmicas.
> 2. Haz una prueba corta en mi PC y dame los comandos para el laboratorio.
> 3. Evalúa ambos con `eval_estratos.py`, solo de forma global, porque WiSARD no tiene oclusión.

**Revisa:**

- Los dos modelos tienen resultados razonables y la bitácora tiene sus filas.

---

## Fase 7. Fusión tardía

**Para qué:** es la fusión más fácil de armar, porque reutiliza los dos modelos de la Fase 6 y solo combina sus respuestas.

**Mensaje:**

> Fase 7.
>
> 1. Implementa la fusión tardía con los dos modelos de la Fase 6. En cada imagen:
>    - multiplica las confianzas de la térmica por un peso w y las de la RGB por (1 − w);
>    - junta todas las cajas y aplica NMS.
> 2. Elige w en validación, probando de 0.0 a 1.0 de 0.1 en 0.1.
> 3. Evalúa en prueba con ese w.

**Revisa:**

- El w elegido y si la fusión supera a la mejor modalidad sola.

---

## Fase 8. Fusión temprana

**Para qué:** probar la forma más simple de unir las imágenes: apilarlas como si fueran una sola imagen de 4 canales.

**Mensaje:**

> Fase 8. Implementa la fusión temprana con una entrada de 4 canales (RGB + térmica).
>
> 1. Antes de programar, revisa si la versión instalada de Ultralytics acepta imágenes de más de 3 canales. Si no, explícame cómo adaptarlo.
> 2. Inicializa la primera capa así: los 3 canales RGB con los pesos de COCO, y el canal térmico con el promedio de esos pesos.
> 3. Aplica el aumento de datos igual a los 4 canales.
> 4. Haz una prueba corta en la PC.

**Revisa:**

- La prueba corta entrena sin errores y la pérdida baja.

---

## Fase 9. Fusión adaptativa

**Para qué:** es el modelo que, según la hipótesis, mejor resiste la oclusión, porque decide en cada zona en qué cámara confiar. Es el más difícil de programar.

**Mensaje:**

> Fase 9. Antes de programar, explícame en palabras simples el diseño:
>
> - dos redes troncales YOLOv8s, una por modalidad;
> - un módulo de atención espacial y por canal que une sus características en P3, P4 y P5;
> - el cuello y la cabeza de YOLOv8s.
>
> Dime cómo se entrenará (con el entrenador de Ultralytics o con un bucle propio que use su función de pérdida) y qué riesgos ves. Cuando lo apruebe, impleméntalo y haz una prueba corta.

**Revisa:**

- Entiendes el diseño antes de aprobarlo.
- La prueba corta entrena sin errores y la memoria de la GPU alcanza.

---

## Fase 10. Costo y tablas finales

**Para qué:** juntar todo lo que va al artículo.

**Mensaje:**

> Fase 10.
>
> 1. Cuenta parámetros y GFLOPs de los cinco modelos a 640 × 640 con la misma herramienta. En la fusión tardía, suma los dos detectores.
> 2. Junta todos los resultados en `reports/resultados_poc.csv`.
> 3. Ejecuta `env_info.py` en el laboratorio y guarda la salida en `reports/entorno.txt`.

**Lo que me traes para redactar Resultados:**

- `reports/tabla9_pop.csv`
- `reports/exp1_resultados.csv`
- `reports/resultados_poc.csv`
- `reports/entorno.txt`
- `experiments/bitacora.csv`
- la distancia promedio del registro
