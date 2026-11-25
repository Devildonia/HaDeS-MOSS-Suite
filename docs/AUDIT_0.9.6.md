# Auditoría técnica de HaDeS MOSS Suite

**Nota global: 8,2 / 10**  
**Fecha:** 25 de noviembre de 2026 · **Suite:** 0.9.6 · **Revisión local:** `62f9e1bc012f4178a6bf644fa8a9a96353d42e01`

## 1. Dictamen

HaDeS MOSS Suite ya funciona como una herramienta completa para trabajar con MOSS-Image en ComfyUI. Sus puntos fuertes son la integración entre nodos, la posibilidad de seguir la receta de un modelo, la exportación a formatos nativos y la disciplina de medir los presets y las afirmaciones públicas. La batería local suma **146 pruebas superadas**, con las dos comprobaciones CUDA de exportación incluidas. La inspección inventarió **133 archivos versionados**, **23 nodos de distribución**, **2 de desarrollo** y **16 workflows de ejemplo**.

El riesgo principal está en la fiabilidad de la procedencia y de los artefactos. Un entrenamiento reanudado puede reutilizar datos de otra configuración en la misma carpeta. Una comparación puede mostrar correctamente los nombres de los modelos sobre la imagen y perderlos en la telemetría de cada rama. El lector puede presentar una descripción de precisión incorrecta porque deduce el formato con heurísticas. A esto se añaden condiciones de carrera en la escritura de PNG y un fallo reproducido en el modo ciego con imágenes de distinto tamaño.

**Prioridad inmediata:** vincular las cachés de entrenamiento a sus entradas, conservar la identidad de cada rama y proteger las escrituras PNG. Son cambios acotados con impacto directo en el trabajo que ya se publica.

| Área | Peso | Nota | Fundamento |
|---|---:|---:|---|
| Arquitectura e integración | 20 % | 8,7 | Módulos con responsabilidades identificables, ModelPatcher y formatos del core, Dev Tools separadas |
| Corrección funcional | 18 % | 8,2 | Buenas regresiones numéricas y de exportación, con casos límite en comparación, layout y LoRA |
| Interfaz y workflows | 15 % | 7,9 | Tres vistas contempladas, widgets con pruebas y ejemplos coherentes, pendiente repetir la aceptación visual |
| Robustez, procedencia y privacidad | 15 % | 7,3 | Buen patrón de escritura para pesos, huecos en PNG, cachés e identidad de recursos |
| Pruebas y distribución | 12 % | 8,2 | Batería local fuerte, CI con dos líneas de ComfyUI, cobertura limitada del navegador y del paquete real en esta auditoría |
| Herramientas y reproducibilidad | 10 % | 8,0 | Checkpoints, RNG, informes y herramientas de evaluación, con identidad insuficiente en cachés |
| Documentación y evidencia | 10 % | 8,7 | Resultados precisos y extensos, algunos enlaces rotos y afirmaciones que necesitan delimitar la muestra |

Media ponderada de 8,15 antes del redondeo. Nota de ingeniería de la revisión local, con valoración conservadora de las áreas que solo tienen evidencia histórica o estática.

## 2. Alcance y método

| Superficie | Inventario | Revisión |
|---|---:|---|
| Backend Python en raíz | 17 archivos | Registro, contratos de nodos, errores, tensores, persistencia, interoperabilidad con ComfyUI |
| Frontend | 5 módulos JS y 1 icono | Sintaxis, comportamiento automatizado, fuentes de datos, enlaces de sockets, caché y serialización desde código |
| Workflows | 14 públicos y 2 de desarrollo | JSON, referencias y tipos de enlaces, contratos de widgets contrastados con las clases reales de la Suite y del core |
| Pruebas | 5 archivos Python y 1 JS | Ejecución de las cuatro suites Python y de widgets, revisión del verificador de distribución |
| Herramientas | 45 scripts Python | Inventario y sintaxis completos, revisión focalizada de entrenamiento, cuantización, evaluación, datos, ejemplos y grabación |
| Documentación y activos | README, CHANGELOG, 6 auditorías, RESULTS, 4 tarjetas y sus activos | Versión, enlaces locales, trazabilidad de cifras, comparación con archivos de evaluación disponibles |
| Distribución | pyproject, .comfyignore, GitHub Actions, licencia | Presencia en paquete, dependencias, matriz y cobertura prevista |

Se revisó el árbol local. Los scripts de investigación se analizaron según su criticidad y puntos de entrada, con análisis sintáctico de todos los Python. La ejecución experimental se centró en flujos sintéticos pequeños. No se reentrenaron modelos, no se recalibró un MOSS completo y no se volvió a votar la calidad visual del benchmark. El repositorio, el workflow activo y la sesión principal de ComfyUI quedaron intactos.

**Estados de evidencia.** Reproducido significa ejecución controlada en esta auditoría. Confirmado por código significa que el contrato o la ruta está identificada en el código y se especifica su condición de aparición. Evidencia histórica corresponde a RESULTS o a archivos del experimento existente.

## 3. Hallazgos

### F01 · Alta · Las cachés de entrenamiento pueden mezclar dos experimentos

**Confirmado por código.** DMD y Slider identifican los embeddings por texto. Edición identifica los latentes por resolución y ruta del archivo, y los embeddings por texto. En DMD, los pares de regresión se recuperan por índice, pasos del profesor y tamaño. El identificador omite el prompt que ocupa ese índice, las identidades de profesor y encoder, y otros ajustes que cambian el objetivo. Los checkpoints conservan los argumentos, pero al restaurarlos los tres entrenadores cargan sus estados sin contrastar esa configuración con la actual.

**Impacto.** Al reusar `--out` después de cambiar de encoder, dataset o Base/SFT se pueden combinar latentes, condicionamiento y targets de experimentos distintos. El entrenamiento puede terminar y producir pesos válidos sin que las métricas representen la receta declarada. La auditoría no atribuye este problema a ninguna ejecución publicada.

**Corrección y aceptación.** Sellar cada ejecución con las identidades de modelos, VAE, prompts, dataset y parámetros estructurales. Rechazar los cambios incompatibles antes de cargar el checkpoint. Validar por separado cada caché y permitir cambios operativos explícitos como ampliar el total de pasos. Probar que dos lanzamientos idénticos reanudan y que cambiar solo el prompt de un índice invalida su target.

**Referencias:** `tools/moss_dmd_train.py:208`, `:267`, `:379`, `tools/moss_edit_train.py:149`, `:249`, `tools/moss_slider_train.py:157`, `:206`.

### F02 · Media · Compare Sampler pierde la identidad del modelo en la telemetría de rama

**Reproducido en composición de salidas con un sampler simulado.** Dos ramas procedentes de Load Diffusion Model reciben etiquetas correctas en la lámina, pero sus registros de rama solo conservan KSampler sin `model_files`. Compare Sampler resuelve el origen para la etiqueta después de invocar internamente a KSampler sin su grafo. Para modelos cargados por nodos del core, la rama carece de attachment de la Suite que supla esa pérdida.

**Impacto.** La imagen dice qué modelo se comparó, pero un informe o script que consuma el JSON estructurado puede perder la identificación. Esto afecta directamente al uso de Compare Sampler como fuente de procedencia. En un merge también conviene comprobar que los nombres del modelo base y de los contribuyentes reflejen la receta según el método.

**Corrección y aceptación.** Resolver las fuentes de cada rama una vez y conservar el dato en la misma estructura que lee la telemetría. Verificar Loader de la Suite, loader core, LoRA activa e inactiva y merges. Comprobar que dos KSamplers con ajustes idénticos y modelos distintos mantienen identidades diferenciadas.

**Referencia:** `moss_compare_nodes.py:385`.

### F03 · Media · Dos instancias que comparten salida pueden sobrescribir PNG

**Confirmado por código.** Image Saver y Compare Sheet calculan un contador de archivo y escriben al destino final. Entre los dos pasos otro proceso puede elegir el mismo nombre. Una escritura interrumpida también puede dejar un PNG incompleto en el destino definitivo. Model Saver y las herramientas LoRA ya tienen un patrón de reserva exclusiva y reemplazo atómico.

**Impacto.** Riesgo de pérdida o sustitución de imágenes cuando coinciden carpeta y prefijo, por ejemplo entre ComfyUI principal y una instancia de pruebas que se configure con la misma salida. El uso secuencial típico reduce la exposición.

**Corrección y aceptación.** Aplicar reserva y escritura atómica por imagen persistente. Una prueba con dos escritores debe dejar dos PNG íntegros y distintos. Un fallo de codificación debe preservar la salida previa y limpiar sus temporales.

**Referencias:** `moss_image_saver_nodes.py:228`, `:243`, `moss_compare_nodes.py:273`, `:288`, `moss_common.py:47`, `:72`.

### F04 · Media · Loader y Metadata Reader pueden clasificar mal la cuantización

**Confirmado por código, con prueba de la heurística sobre entradas sintéticas.** Loader usa la categoría genérica `int8` para cualquier QuantizedTensor cuyo nombre no parezca FP8. Metadata Reader usa la coexistencia de una clave `comfy_quant` y un tensor I8 del archivo como señal de int8 ConvRot. No lee el identificador concreto de cada proyección para esta decisión. Con una proyección empaquetada U8 y una configuración de cuantización presente, la heurística del lector tampoco puede distinguirla de un archivo que el resumen llame «not quantized».

**Impacto.** HMS v1 es una receta mixta W4A8/W6A8. El usuario puede recibir un diagnóstico genérico o incorrecto en un nodo cuya función es explicar el archivo. El peso puede cargar y generar correctamente.

**Corrección y aceptación.** Reportar los layouts reales de QuantizedTensor y leer los descriptores de `comfy_quant` al inspeccionar archivos. Probar BF16, FP8, int8 con y sin ConvRot y el mix HMS. Mostrar formato desconocido si el descriptor no se reconoce.

**Referencias:** `moss_loader_nodes.py:16`, `moss_metadata_nodes.py:125`, `:359`, `moss_hms.py:259`.

### F05 · Media · El azar del modo ciego altera la escala y la resolución de la lámina

**Reproducido.** El contrato de Compare Sheet dice que A fija la altura. Con A de 32 × 32 y B de 64 × 64, la salida sin ciego y con ciego de semilla 0 fue de 80 × 74. Con ciego de semilla 1 pasó a 144 × 106 al colocar B primero. El ancho y alto indicados incluyen cabecera y separación.

**Impacto.** Al comparar entradas de distinto tamaño, cambiar la semilla modifica el remuestreo y la superficie de lectura, además del orden. El informe dice que se tomó la altura de A aunque se haya tomado la de B. Las 133 láminas de Turbo contienen ramas del mismo tamaño, así que este defecto no explica sus resultados.

**Corrección y aceptación.** Fijar la altura desde A antes de barajar. Probar las dos permutaciones con imágenes desiguales, escalar miniaturas extremas a un ancho mínimo de un píxel y mantener documentado el truncado de lotes desiguales.

**Referencias:** `moss_compare_nodes.py:153`, `:200`, `:255`, `:266`, `:306`.

### F06 · Media · Desactivar el workflow incrustado conserva metadatos de la Suite

**Confirmado por código.** `embed_workflow=False` desactiva los bloques ComfyUI, pero Image Saver mantiene `parameters` y `hades_moss`. Compare Sheet mantiene nombres, metadatos de rama y clave del ciego. Con `--disable-metadata` se omiten esos bloques, pero una etiqueta o pie dibujado sobre la imagen forma parte de sus píxeles. El modo ciego actual anuncia que su clave se guarda en PNG e info.

**Impacto.** Al enviar imágenes de muestra o compartir una lámina a ciegas, un PNG puede llevar nombres de modelos, LoRAs o prompts aunque se haya desmarcado el workflow. Ningún nombre privado se incluyó en este informe. El riesgo es de publicación y de protocolo experimental.

**Corrección y aceptación.** Dar controles separados para workflow, procedencia y clave ciega, preservando los valores antiguos en workflows previos. Probar cada combinación con inspección de chunks y de las etiquetas visibles. Para evaluaciones ciegas, conservar la clave en un artefacto separado del que lee el evaluador.

**Referencias:** `moss_image_saver_nodes.py:232`, `moss_compare_nodes.py:192`, `:276`.

### F07 · Media · El Prompt Enhancer puede devolver un layout que su propio compilador rechaza

**Reproducido con respuestas sintéticas del generador.** Un `summary.caption` numérico sobrevive a la normalización de Enhancer y falla en el compilador. Una coordenada infinita supera el test de tipo y falla en `round`. `summary.canvas` y algunas listas se conservan con una validación más débil que la del consumidor.

**Impacto.** Un layout generado en una ejecución larga puede abortar por un detalle del JSON o perder la oportunidad de que el usuario recupere las cajas. La reparación de elementos sin cajas ya existe y puede reutilizarse.

**Corrección y aceptación.** Normalizar un único contrato de salida antes de compilar. Descartar o marcar cajas no finitas y manejar tipos inesperados con diagnóstico claro. Añadir casos de resumen mal tipado, elementos repetidos y coordenadas no finitas.

**Referencias:** `moss_enhancer_nodes.py:188`, `:214`, `moss_pipeline_nodes.py:123`.

### F08 · Media · Leer un hash al guardar no acredita qué archivo se usó al muestrear

**Confirmado por código.** El Saver resuelve el nombre de modelo o LoRA en el momento del guardado y calcula su SHA-256. El nombre del archivo que se obtuvo del grafo no prueba que los bytes presentes en disco sigan siendo los mismos que cargó el ModelPatcher. Esto importa al sobrescribir o reemplazar pesos con el mismo nombre mientras sigue vivo un modelo en caché. El lector de recetas sí contempla cambios en dependencias para invalidar su propio resultado.

**Impacto.** En ese caso se puede atribuir una imagen a la revisión equivocada de un recurso. Los hashes siguen siendo correctos como hashes del archivo que el Saver encontró en disco.

**Corrección y aceptación.** Capturar la identidad al cargar, o propagar una firma del archivo y marcar el hash como no verificado si la firma cambió antes de guardar. Probar un reemplazo controlado entre la carga y el guardado con archivos sintéticos.

**Referencias:** `moss_sampler_nodes.py:29`, `moss_image_saver_nodes.py:21`, `:113`, `moss_metadata_nodes.py:412`.

### F09 · Media · El bloqueo de los entrenadores tiene una ventana de carrera

**Confirmado por código.** Los tres entrenadores comprueban si existe un archivo de bloqueo y luego lo crean con modo `w`. Dos arranques cercanos pueden pasar la comprobación y usar el mismo destino. Los checkpoints se guardan mediante temporales y reemplazo, pero su archivo temporal se comparte si concurren dos ejecuciones. DMD y Slider escriben directamente la caché de prompts.

**Impacto.** Dos procesos pueden competir por checkpoints, informes y cachés del mismo experimento. Una interrupción durante una escritura directa de caché también puede impedir reanudar sin recuperar esa caché.

**Corrección y aceptación.** Crear el bloqueo de forma exclusiva y unificar escritura atómica de cachés, informes y checkpoints. Probar arranque simultáneo, interrupción y reanudación sobre un destino descartable.

**Referencias:** `tools/moss_dmd_train.py:190`, `:219`, `tools/moss_edit_train.py:141`, `tools/moss_slider_train.py:128`, `:167`.

### F10 · Media · Algunos diagnósticos de telemetría producen una excepción antes de validar

**Reproducido.** Una entrada `compare_sampler` con `branches` como lista hace que la telemetría lance AttributeError antes de llegar al manejo de entradas incompletas. Adicionalmente, con cero entradas válidas la redacción «No findings» puede interpretarse como una verificación positiva aunque no se haya comprobado ningún nodo.

**Impacto.** Un informe desactualizado o mal formado puede detener un workflow que debía diagnosticarlo. Una telemetría sin datos puede dar más confianza de la que corresponde.

**Corrección y aceptación.** Validar el tipo del contenedor y de cada rama antes de extraer sus entradas. Distinguir explícitamente «sin datos» de «comprobado sin hallazgos».

**Referencia:** `moss_telemetry_nodes.py:153`.

### F11 · Media · Compatibilidad HMS más acotada que la presentación general

**Confirmado por código y documentación.** HMS requiere proyecciones fusionadas y usa funciones privadas de comfy-kitchen verificadas con 0.2.40. La comprobación actual de layout se apoya en el número total de nombres, y el nodo comprueba que exista un CLIP conectado. La documentación corta anuncia soporte para cualquier modelo o merge MOSS y empaqueta 3,96 GB, 11,6 GB y ~2× como parte de la descripción del ejemplo.

**Impacto.** Formatos alternativos que ya carguen en ComfyUI pueden quedar fuera del converter. Las cifras dependen de la arquitectura y del equipo. RESULTS documenta las condiciones y guarda evidencia de modelos concretos, incluidas desviaciones de la puerta visual en un merge. El registro experimental tiene más matiz que el resumen.

**Corrección y aceptación.** Comprobar familia, proyecciones de cada capa, encoder y disponibilidad de kernels antes de calibrar. Mostrar en README el alcance exacto por variante y el equipo donde se midió el rendimiento. Separar evidencia de lectura del archivo, equivalencia de pesos, imagen y velocidad.

**Referencias:** `moss_saver_nodes.py:198`, `moss_hms.py:51`, `README.md:110`, `:282`, `tools/RESULTS.md:2881`, `:2918`, `:2987`.

### F12 · Media · Un mismo flujo puede contar una LoRA sin aplicar cambios

**Confirmado por código.** La ruta por grupos cuenta cada peso mapeado antes de comprobar el valor efectivo `strength × group`. Con pesos de todos los grupos a cero, no añade parches, pero puede conservar el nombre de la LoRA, los pesos contados y la detección few-step en telemetría. Image Saver consume esa lista de recursos.

**Impacto.** Atribución incorrecta y advertencias turbo cuando la LoRA quedó neutralizada por grupos. El cero global tiene una ruta diferenciada, por lo que el fallo depende de cómo se haya expresado la desactivación.

**Corrección y aceptación.** Separar en el modelo de datos los pesos mapeados de los efectivamente aplicados. Probar cero global, todos los grupos a cero y un único grupo no nulo. El JSON y los hashes embebidos deben reflejar los recursos efectivos.

**Referencias:** `moss_lora_nodes.py:169`, `:270`, `moss_image_saver_nodes.py:71`.

### F13 · Baja · Enlaces rotos según el canal de instalación

**Verificado en los archivos locales.** La tarjeta de Turbo referencia dos imágenes ausentes del checkout mediante `images/showcase.jpg` e `images/variety_cafe.jpg`. Desde README, la tarjeta Turbo y `docs/AUDIT_0.6.1.md` se enlazan de forma relativa y la política de Registry excluye ambos destinos. Esto corresponde al checkout y al contenido previsto de Registry. No se inspeccionaron las copias desplegadas de la tarjeta en plataformas externas.

**Corrección y aceptación.** Enlazar los destinos excluidos con URL canónica del repositorio y sincronizar o reemplazar los activos ausentes. Incluir enlaces relativos en la validación de empaquetado.

**Referencias:** `.comfyignore`, `README.md`, `tools/model_cards/HaDeS_MOSS_Turbo/README.md`.

### F14 · Baja · Desfase entre unidades, mensajes y referencias internas

**Confirmado por código y documentación.** Model Saver divide por 1024³ y muestra GB. Ejemplo documentado: 3.955.670.084 bytes corresponden a 3,96 GB decimales y aproximadamente 3,68 GiB. CLAUDE.md mantiene una foto del proyecto en 0.9.2 mientras el código, pyproject y changelog están en 0.9.6. El aviso del guardado por etapa menciona que falta MOSS Loader cuando un modelo core correctamente trazado también puede cubrir esa información. Este último caso afecta solo a la redacción del caso sin procedencia.

**Corrección y aceptación.** Usar bytes, GB y GiB de forma consistente y mantener la guía de trabajo sincronizada con un estado de versión fechado.

**Referencias:** `moss_saver_nodes.py:195`, `moss_image_saver_nodes.py:214`, `CLAUDE.md`, `tools/RESULTS.md:2987`.

### F15 · Baja · La suite de pruebas usa parte del espacio temporal de ComfyUI

**Reproducido.** En la invocación directa, cuatro tests de previsualización de Compare Sheet fallaron por permisos al escribir en el directorio temporal por defecto de ComfyUI. Al apuntar ese directorio a un temporal del test, las 114 pruebas de la suite principal pasaron con CUDA habilitada. Los tests ya aíslan la salida general y la carpeta LoRA, pero dejan fuera los temporales de previews.

**Impacto.** Resultados engañosos en sandboxes y residuos en instalaciones que ejecutan los tests sobre el ComfyUI habitual.

**Corrección y aceptación.** Añadir el temporal del frontend al fixture de la suite. Ejecutar la suite con la instalación de ComfyUI en solo lectura y verificar que pasa sin tocar su carpeta temporal.

**Referencias:** `tests/test_moss_suite.py:45`, `moss_compare_nodes.py:292`.

### F16 · Baja · El verificador de pares emite un veredicto más fuerte que su cobertura

**Confirmado por código.** El script clasifica como «fair pair» todo caso sin discrepancias observadas, incluso si hay campos sin verificar. Su salida sí enumera esos campos. El verificador tampoco espera metadatos de Compare Sheet, que tienen su propio chunk. Es una diferencia de contrato entre las dos herramientas, con impacto en automatizaciones que se apoyen solo en la primera línea o el código de salida.

**Corrección y aceptación.** Introducir un modo estricto que falle ante la falta de campos obligatorios. Para uso histórico, distinguir igualdad de los campos disponibles y ausencia de evidencia suficiente. Dar soporte a ambas ramas de Compare Sheet mediante entrada explícita o documentar el paso de extracción.

**Referencia:** `tools/moss_pair_check.py:43`, `:56`, `:109`.

## 4. Revisión por familias

| Familia | Qué sostiene la nota | Qué queda por cubrir |
|---|---|---|
| Loader, Latent y KSampler | Comprobación de arquitectura, encoder y canales del VAE. Separación de schedule core/exact, tratamiento de denoise cero y pruebas de schedules reales | Consistencia del formato reportado, identidad de recurso en caché, inferencia completa con distintos backends |
| Model Block Merger y Multi-Model Merger | Parches lazy, validación de compatibilidad y pruebas de ratios, TIES, SLERP, precisión y rescalado | Generalizar validación end-to-end a más métodos y combinaciones de pesos. Medir coste cuando conviven varias ramas activas |
| Extractor, Block Weight, Multi-LoRA, Merger y Resize | Comparación contra el core, carga de exportaciones y guardas frente a sobrescribir entradas, incluidas variantes de ruta y mayúsculas | Distinguir LoRA reconocida de LoRA aplicada. Aclarar familias válidas para cargar frente a las que se pueden recomprimir |
| Model Saver y HMS | Exportaciones con reserva y reemplazo, preservación de layout int8, redondeo medido y pruebas CUDA de layout nativo | Compatibilidad fuera del layout medido, captura integral con interrupción durante hooks y matrices de VRAM en otras GPUs |
| Prompt Enhancer, Layout Editor, Layout Compiler y Control | Contrato estricto del compilador, referencia histórica con OpenMOSS, edición parcial del grafo y adaptación de mapas | Normalización del JSON generativo, tamaños máximos para drafts externos y regresión visual de los tres modos |
| Compare Sheet, Compare Sampler, Text Check y Telemetry | Comparativas reproducibles, reportes legibles, comparación por regiones y vía local sin otro modelo OCR | Procedencia de ramas, integridad del protocolo ciego y validación de entradas. Text Check comprueba la primera imagen del lote y tolera caso y puntuación, tal como declara su contrato |
| Image Saver, Metadata Reader y Sampler Metadata | Selección explícita de etapa, parámetros A1111, hashing local, seguimiento de recetas y caché invalidada por dependencias del lector | Hash de procedencia, controles de publicación, escritura PNG y precisión de etiquetas |
| Dev Tools | Cálculo de regiones y diferencias, informes de solo lectura y evidencia histórica de uso en tres vistas | Volver a comprobar UI al cambiar de frontend. Mantener ejemplos de desarrollo fuera del paquete público |
| Entrenamiento y datasets | Checkpoints con optimizador, EMA y RNG, pipeline documentado de selección, preparación y deduplicación | Identidad de datasets y modelos por ejecución, protección del destino y pruebas de reanudación. La auditoría no certifica licencias entrada por entrada |
| Cuantización de laboratorio | Comparadores, pilotos, barridos y recetas documentadas, con separación entre formatos experimentales y el elegido | Probar paridad entre script y nodo en una matriz reducida y versionar la receta junto a cada medición |
| Grabación y ejemplos | Flujos reproducibles y separación entre servidor personal y perfiles de grabación según el método | No se regrabaron los vídeos ni se inspeccionaron exportaciones privadas. Conviene regenerar ejemplos solo cuando cambien sus contratos |

## 5. Pruebas y compatibilidad

| Comprobación | Resultado |
|---|---|
| `test_moss_suite.py` con `HADES_MOSS_TEST_CUDA=1` y temporales aislados | 114/114, incluidas las dos pruebas CUDA |
| `test_moss_dev_nodes.py` | 15/15 |
| `test_moss_edit_eval.py` | 4/4 |
| `test_moss_pair_check.py` | 5/5 |
| `node --test tests/test_moss_widgets.mjs` | 8/8 |
| Sintaxis | 67 archivos Python parseados, 5 JS comprobados con Node, JSON leído correctamente |
| Integridad estructural de los ejemplos | 16/16 JSON con enlaces internos coherentes. 179 instancias de nodo coinciden con el número de widgets declarado por las clases reales |
| Hardware y software local | RTX 5090, Windows, Python 3.11.9, PyTorch 2.9.1+cu130, comfy-kitchen 0.2.40, ComfyUI 0.43.0 y frontend instalado 1.57.4 |
| Registry y CI remotos | Revisión de configuración local. Sin reconstrucción del zip con comfy-cli ni consulta del estado remoto |
| Aceptación visual actual | Sin ejecución de navegador. El acceso a la instancia aislada por loopback devolvió un error de permisos de socket en el entorno de auditoría |
| Cobertura multi-OS | La matriz declarada incluye Windows y Linux con el baseline y Linux con master. Esta auditoría ejecutó Windows |

Las pruebas JS cubren el cambio de estilos, la persistencia visual de informes y previews y la asignación del slot al soltar un enlace sobre un widget de Nodes 2.0. La validación visual anterior en RESULTS de frontend 1.57.4 aporta evidencia por nodo y escenario. Para aceptar de nuevo las tres vistas quedarían pendientes interacción real, zoom, redimensionado, ejecución y caché, informes y previews, y guardado y recarga en el editor. No se ha extendido esa aceptación a conclusiones sobre frontend posteriores.

CI tiene buena estructura. Usa pesos sintéticos para probar contratos y una segunda tarea para comprobar el contenido del paquete y el registro del servidor. En el pipeline publicado los datos no dependen de librerías nuevas de la Suite. Dos dependencias de integración merecen seguimiento: funciones privadas de comfy-kitchen en HMS y contratos internos del frontend en los widgets personalizados. Confirmar compatibilidad real requiere pruebas con sus versiones concretas.

## 6. Auditoría de la evidencia de Turbo 2.0

- Inspeccioné los metadatos de las **133 láminas**, de la 00245 a la 00377. En las 133 parejas coinciden entre ramas los diez campos comprobados: semilla, pasos, CFG, sampler, scheduler, anchura, altura, denoise, shift y SLG. El tamaño declarado es **1344 × 1792**. La distribución es **80 con CFG 1,0, 51 con 1,2 y 2 con 1,5**. Los originales incluyen workflow y carecen de clave ciega nativa. Eso es compatible con una revisión externa que recortara etiquetas y barajara las dos imágenes.
- El archivo de votos de Claude contiene **133 identificadores únicos y completos, 76 a favor de v2.0 y 57 a favor de v1.0**, que son **57,1 % y 42,9 %**. Esta comprobación acredita el recuento de ese fichero, sin reevaluar visualmente cada pick ni confundirlo con otros evaluadores. Según el recuento de HaDeS aportado, su voto fue **87–46 para v2.0**, equivalente a **65,4 %**. El segundo revisor figura en RESULTS con dos denominadores según se admitan empates. Su ficha individual no se verificó.
- El SHA-256 local del adaptador v2.0 FP16 coincide con el publicado en RESULTS. Esto verifica la identidad del archivo local contrastado. Los metadatos de muestreo por sí solos no prueban que cada imagen se generara con exactamente esos bytes.
- El **13/15 de Claude en texto y carteles** es una señal valiosa para esa muestra. La categoría es pequeña, y el informe solo lista el desglose de parte de las láminas. La expresión «the text result is» conclusiva debe moderarse. Conviene guardar categorías, votos, empates, protocolo y mapa ciego de los evaluadores en un manifiesto común antes de publicar una comparación global.
- Para la cuantización FP16, los dos controles entre sesiones muestran que el cambio de sesión es un factor de confusión. Son compatibles con una diferencia pequeña de precisión, pero la atribución causal pide un diseño cruzado con FP32 y FP16 replicados en cada sesión. Los resultados de sesgo de apariencia están acotados a los prompts con caras visibles y a la observación del evaluador. La tarjeta debe mantener ese alcance.

**Referencia:** `tools/RESULTS.md:2995`, `:3052`, `tools/model_cards/HaDeS_MOSS_Turbo/README.md` y metadatos de las 133 láminas locales.

## 7. Orden de corrección propuesto

| Orden | Acciones | Criterio de cierre |
|---:|---|---|
| 1 | F01 y F09, identidad de cachés, reanudación y bloqueo | Misma receta reanuda. Cambios de modelo, dataset o prompt invalidan el artefacto correspondiente. Dos procesos no comparten una ejecución |
| 2 | F02, F04, F08 y F12, procedencia, formato e identidad efectiva | Imagen, telemetría y archivo de origen cuentan la misma historia en modelo core, merge, LoRA en cero y HMS |
| 3 | F03, F05 y F06, guardado y modo ciego | Dos escritores producen dos archivos válidos. A fija la altura. Los archivos destinados a evaluación no contienen ni muestran el mapa ciego |
| 4 | F07, F10 y F11, contratos y diagnóstico | Una salida generativa irregular se recupera o falla con explicación. Telemetry sigue funcionando con una rama inválida. HMS da un error temprano para formatos no admitidos |
| 5 | F13 a F16, documentación y cobertura | Registry sin enlaces relativos rotos. Unidades consistentes. Tests sin escribir en el ComfyUI anfitrión. Verificador de pares distingue coincidencia de cobertura |
| 6 | Aceptación de release | Repetir las tres vistas con workflows versionados, ejecutar zip real de Registry y registrar la matriz de versiones y límites en RESULTS cuando se autoricen cambios |

La tesis de la auditoría es práctica. Las piezas de mayor valor ya están presentes y sus caminos principales están bien probados. La mejora más importante está en hacer que la Suite preserve una relación consistente entre **lo que se ejecutó, lo que se guardó y lo que se afirma sobre el resultado**.

