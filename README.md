INTEGRANTES:
-JOEL MATEO MANRIQUE VELASQUEZ
-PEREZ RAMOS LUIS ENRRIQUES
-JACINTO CASTRO GINNO

Resumen de datos y preprocesamiento
En la Parte A utilicé el dataset Heart Disease (UCI). La variable objetivo original (num ∈ {0,1,2,3,4}) se recodificó a binaria: 0 si num=0, 1 en caso contrario. El análisis de valores faltantes mostró 4 nulos en ca (≈1.32%) y 2 en thal (≈0.66%); el resto sin nulos. Apliqué estandarización a las variables numéricas y one-hot a categóricas (sex, cp, fbs, restecg, exang, slope, ca, thal). Tras el pipeline se obtuvieron 29 columnas (incluido el intercepto). La partición fue 70/30 con estratificación. Para las partes B y C usé Wine (sklearn), con 13 variables químicas y 3 clases; ahí solo fue necesario escalar y estratificar (70/30).

Diferencias del gradiente: binario, OvA y multinomial
El gradiente tiene la misma estructura conceptual (“predicho − verdadero”) multiplicada por las características, pero cambia el cálculo de probabilidades y cuántos parámetros se actualizan a la vez.
• Binario: un único vector de pesos w y la función sigmoide p=σ(Xw). El gradiente de la pérdida logarítmica (NLL) es (p−y)ᵀX / n.
• OvA: se resuelven K problemas binarios independientes (clase k vs resto) con el mismo gradiente del caso binario; lo único que cambia es la etiqueta objetivo (1 si y=k, 0 en otro caso). Se entrenan K vectores w_k por separado y luego se predice por argmax de las probabilidades sigmoides de cada clasificador.
• Multinomial (softmax): un único modelo con matriz W∈ℝ^{K×(d+1)}. Se calculan puntajes Z=XWᵀ, probabilidades P=softmax(Z) y la NLL. El gradiente conjunto es ∇_W NLL = (1/n) (P−Y)ᵀ X, lo que acopla a todas las clases (cada actualización de w_k depende de las demás).

Estabilidad numérica en softmax
Exponentiar puntajes puede causar overflow si Z tiene magnitudes grandes. Para evitarlo usé la versión estable: por fila, restar el máximo antes de la exponencial (Z ← Z − max(Z, fila)). Además, en la NLL agregué un pequeño eps dentro del log. Con esto, el entrenamiento fue estable y sin NaN. En Wine, las curvas de NLL cayeron rápido y de forma monótona para lr=0.01 y lr=0.005; la tasa 0.01 convergió más rápido y a menor pérdida final.

Resultados experimentales

Parte A — Binaria (Heart Disease)
El modelo desde cero con descenso de gradiente alcanzó en test: Accuracy 86.81%, Precision 92.68%, Recall 80.85% y F1 86.36%. La matriz de confusión fue [[41, 3], [9, 38]]: tres falsos positivos y nueve falsos negativos. Con LogisticRegression de sklearn obtuve Accuracy 85.71%, Precision 92.50%, Recall 78.72% y F1 85.06%. Es decir, mi implementación superó a sklearn en +1.10 puntos de accuracy y +2.13 puntos de recall, manteniendo una precisión similar; la curva de log-verosimilitud mostró convergencia estable con dos learning rates (0.1 y 0.01).

Parte B — Multiclase OvA (Wine)
Mi OvA (tres binarios k vs resto con el optimizador de la Parte A) logró Accuracy 0.9630. La matriz de confusión fue [[18, 0, 0], [0, 19, 2], [0, 0, 15]]: los dos errores se dieron en la clase 1 confundida como clase 2. El reporte por clase fue:
– Clase 0: P=1.0000, R=1.0000, F1=1.0000 (18/18 correctos).
– Clase 1: P=1.0000, R=0.9048, F1=0.9500 (20/21 correctos).
– Clase 2: P=0.8824, R=1.0000, F1=0.9375 (15/15 correctos).
Con sklearn (One-vs-Rest) obtuve Accuracy 0.9815 y matriz [[18, 0, 0], [1, 20, 0], [0, 0, 15]] (un error, clase 1→0). En la comparación de coeficientes (top-5 |coef| por clase), los signos y las variables más influyentes coincidieron en gran medida entre mi modelo y sklearn. Por ejemplo, para la clase 0 destacan proline (+), alcohol (+) y flavanoids (+), mientras que alcalinity_of_ash (−) y el sesgo/bias (−) actúan en sentido contrario; las magnitudes difieren moderadamente, algo esperable por el solver y la regularización de sklearn frente a un GD “puro”.

Parte C — Multinomial (Softmax, Wine)
El modelo softmax desde cero alcanzó Accuracy 0.9815 con matriz [[18, 0, 0], [0, 20, 1], [0, 0, 15]]: un único error (clase 1→2). Con LogisticRegression multinomial (lbfgs) también obtuve Accuracy 0.9815, matriz [[18, 0, 0], [1, 20, 0], [0, 0, 15]]. Es decir, la precisión global fue idéntica, aunque los dos modelos discreparon en qué muestra de la clase 1 confundir: mi softmax erró hacia la clase 2; sklearn, hacia la clase 0. Esta diferencia ilustra que, aun con la misma exactitud, OvA y multinomial pueden separar el espacio de decisión de manera distinta.

¿Cuándo divergen OvA y multinomial?
Aunque en Wine ambos enfoques rindieron casi igual (mismo accuracy 0.9815 para softmax y 0.9815 para sklearn multinomial; y 0.9630 para mi OvA), no son equivalentes. OvA entrena modelos independientes y decide por argmax; no garantiza que las probabilidades sumen a uno y puede producir inconsistencias si los clasificadores quedan desbalanceados. El softmax, en cambio, acopla las clases y normaliza las probabilidades, lo que suele mejorar la calibración y la coherencia cuando hay solapamiento, desbalance o fronteras complejas. En mis resultados, las discrepancias se concentraron en el límite de decisión de la clase 1 (una muestra), mostrando cómo cada esquema “elige” lados diferentes de la frontera.

Conclusión
Las tres variantes funcionaron correctamente, con curvas de convergencia limpias y métricas altas. En Heart Disease, mi modelo binario superó levemente a sklearn, especialmente en recall. En Wine, mi OvA entregó 0.9630 de accuracy y sklearn OvR 0.9815, con coeficientes alineados en variables clave. El softmax desde cero igualó a sklearn multinomial en 0.9815, con estabilidad numérica garantizada por la versión estable del softmax. En conjunto, los experimentos validan la implementación de los gradientes y muestran las diferencias prácticas entre descomponer en K binarios (OvA) y optimizar todas las clases de forma conjunta (softmax).
