# Sistema de Diagnóstico Médico Interactivo con Visualización de Métricas

Este sistema está compuesto por dos bloques principales:

1. **Diagnóstico inteligente basado en preguntas al usuario (Chatbot)**
2. **Visualización de resultados acumulados mediante gráficas**

---

## 1. Código de Diagnóstico: `CHATBOTMED_KNN_METRICAS.py`

### Descripción General
Este archivo implementa un sistema médico interactivo que realiza preguntas al usuario sobre factores clínicos y síntomas, y realiza un diagnóstico de enfermedades utilizando dos métodos combinados:
- Coincidencia clásica de síntomas (score directo)
- Similitud por K-Nearest Neighbors (distancia de Hamming)

### Fases del Diagnóstico

- **Ingreso de datos personales:** edad, género, peso y altura (se calcula el IMC).
- **Identificación de factores de riesgo:** desnutrición, sobrepeso, obesidad, grupo etario, etc.
- **Validación de síntomas iniciales:** el usuario introduce síntomas por texto y el sistema los interpreta mediante sinónimos.
- **Fase 1 (preguntas guiadas):** se preguntan los síntomas más frecuentes en las enfermedades posibles.
- **Fase 2 (preguntas adaptativas):** se priorizan los síntomas faltantes de las enfermedades con mayor score.
- **Cálculo final de score:** combinación ponderada entre score clásico (60%) y score KNN (40%).
- **Presentación del Top 5:** se muestran hasta cinco enfermedades ordenadas por mayor score.

### Registro de Resultados
Al final de cada sesión, el sistema genera automáticamente un registro en un archivo CSV (`registro_sesiones.csv`), incluyendo:

- ID de sesión (formato: `SYYYYMMDDHHMMSSxxx`)
- Fecha y hora
- Score clásico, KNN y final
- Preguntas realizadas y síntomas confirmados
- Enfermedad Top 1 y su score

---

## 2. Visualizador de Métricas: `METRICAS_CHATBOTMED_KNN.py`

### Descripción General
Este script analiza el archivo `registro_sesiones.csv` y genera **cinco gráficas clínicas** para evaluar el rendimiento del sistema.

### Requisitos
- Python 3.x
- Librerías: `pandas`, `matplotlib`, `seaborn`

Instalación de dependencias:
```bash
pip install pandas matplotlib seaborn
```

### Ejecución
```bash
python METRICAS_CHATBOTMED_KNN.py
```

### Gráficas Generadas

1. **Distribución del Score Final**
   - Histograma con curva de densidad. Permite visualizar si los scores se concentran en zonas altas o bajas.

2. **Relación entre número de preguntas y score final**
   - Gráfica de dispersión. Mide si hacer más preguntas lleva a un mejor resultado.

3. **Comparación Score Clásico vs KNN**
   - Scatter plot con tamaño y color basados en el score final. Evalúa la coherencia entre ambos métodos.

4. **Boxplot de preguntas por sesión**
   - Resume cuántas preguntas hace el sistema en promedio.

5. **Distribución de síntomas confirmados**
   - Indica qué tanta evidencia clínica se logra en cada sesión.

---

## Recomendaciones Finales

- Ejecuta `CHATBOTMED_KNN_METRICAS.py` múltiples veces para generar sesiones reales.
- El archivo `registro_sesiones.csv` se va llenando automáticamente.
- Luego corre `METRICAS_CHATBOTMED_KNN.py` para analizar el rendimiento global.
- Asegúrate de mantener los nombres de columnas originales.

---

Este sistema no reemplaza a un profesional médico. Su objetivo es demostrar un flujo de inferencia clínica guiada con respaldo estadístico.

