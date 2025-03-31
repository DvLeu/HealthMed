# Importa la biblioteca pandas para manipulación y análisis de datos tabulares (como DataFrames)
import pandas as pd

# Importa matplotlib para generación de gráficos estáticos y personalizados
import matplotlib.pyplot as plt

# Importa seaborn, una biblioteca basada en matplotlib para visualizaciones estadísticas más estilizadas
import seaborn as sns


# ================================
# CARGA DEL ARCHIVO DE REGISTRO
# ================================

# Ruta del archivo CSV que contiene los resultados registrados después de cada sesión
archivo = 'CODING_SAMSUNG/chatbot_medico/Dataset/registro_sesiones.csv'

# Se intenta cargar el archivo con los datos de las sesiones
try:
    df = pd.read_csv(archivo)
except FileNotFoundError:
    # Si no se encuentra el archivo, se muestra un mensaje y se detiene la ejecución
    print(f"El archivo '{archivo}' no fue encontrado.")
    exit()

# ================================
# VALIDACIÓN DE COLUMNAS NECESARIAS
# ================================

# Lista de columnas mínimas que se requieren para generar las gráficas
columnas_requeridas = [
    'score_clasico', 'score_knn', 'score_final',
    'preguntas_realizadas', 'sintomas_confirmados'
]

# Si falta alguna de las columnas necesarias, se notifica y se detiene el script
if not all(col in df.columns for col in columnas_requeridas):
    print("Faltan columnas requeridas en el archivo de registros.")
    exit()

# ================================
# VISUALIZACIONES DE MÉTRICAS
# ================================

# === GRÁFICA 1: Score Final (Histograma) ===
plt.figure(figsize=(8, 5))
sns.histplot(df['score_final'], bins=10, kde=True)  # kde=True añade la curva de densidad
plt.title('Distribución del Score Final')
plt.xlabel('Score Final (%)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.tight_layout()
plt.show()

# === GRÁFICA 2: Preguntas vs Score Final ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='preguntas_realizadas', y='score_final')
plt.title('Relación entre número de preguntas y score final')
plt.xlabel('Preguntas realizadas')
plt.ylabel('Score Final (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === GRÁFICA 3: Score Clásico vs KNN ===
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='score_clasico',         # Eje X: score del método clásico
    y='score_knn',             # Eje Y: score del método KNN
    size='score_final',        # Tamaño de cada punto según el score final combinado
    hue='score_final',         # Color de cada punto según el score final
    palette='coolwarm',        # Paleta de colores que indica intensidad
    sizes=(50, 200),           # Rango de tamaños de los puntos
    legend=False               # No mostrar la leyenda para evitar saturación
)
plt.title('Score Clásico vs Score KNN')
plt.xlabel('Score Clásico (%)')
plt.ylabel('Score KNN (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === GRÁFICA 4: Distribución de preguntas (Boxplot) ===
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['preguntas_realizadas'])
plt.title('Distribución de preguntas por sesión')
plt.xlabel('Número de preguntas')
plt.tight_layout()
plt.show()

# === GRÁFICA 5: Síntomas confirmados (Histograma) ===
plt.figure(figsize=(8, 5))
sns.histplot(df['sintomas_confirmados'], bins=10, kde=False)  # kde=False para un histograma clásico
plt.title('Síntomas confirmados por sesión')
plt.xlabel('Síntomas confirmados')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.tight_layout()
plt.show()
