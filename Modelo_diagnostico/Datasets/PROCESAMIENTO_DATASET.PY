import pandas as pd

# Cargar datos originales desde CSV
df = pd.read_csv('Enfermedades.csv')

# Limpiar la columna de síntomas: eliminar nulos, espacios extra, puntos finales y estandarizar a minúsculas
df['Sintomas'] = df['Sintomas'].fillna('').str.strip().str.rstrip('.').str.lower()

# Separar la cadena de síntomas en una lista Python (usando la coma como separador)
df['Lista_Sintomas'] = df['Sintomas'].str.split(',')

# Quitar espacios en blanco sobrantes de cada síntoma en las listas
df['Lista_Sintomas'] = df['Lista_Sintomas'].apply(lambda lista: [s.strip() for s in lista if s.strip() != ''])

# Identificar todos los síntomas únicos presentes en el dataset
todos_sintomas = sorted({s for lista in df['Lista_Sintomas'] for s in lista})

# Para cada síntoma único, crear una columna nueva e indicar con 1/0 su presencia en cada enfermedad
for sintoma in todos_sintomas:
    df[sintoma] = df['Lista_Sintomas'].apply(lambda lista: 1 if sintoma in lista else 0)

# Eliminar las columnas temporales de texto si ya no son necesarias
df = df.drop(['Sintomas', 'Lista_Sintomas'], axis=1)

# Guardar el DataFrame resultante a un nuevo CSV (opcional)
df.to_csv('Dataset_Enfermedades_FV.csv', index=False)
