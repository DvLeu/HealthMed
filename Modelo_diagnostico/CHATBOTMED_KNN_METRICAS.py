# ========================
# BLOQUE DE IMPORTACIONES
# ========================

# Importa pandas para manipulación eficiente de datos tabulares (CSV, DataFrames)
import pandas as pd

# Importa numpy para cálculos numéricos, operaciones con vectores y matrices
import numpy as np

# Importa datetime para trabajar con fechas y horas (registro de sesiones, IDs únicos)
from datetime import datetime

# Importa csv para escribir datos en formato CSV manualmente (registro de resultados)
import csv

# Importa el algoritmo K-Nearest Neighbors del módulo de machine learning scikit-learn
from sklearn.neighbors import KNeighborsClassifier

# Importa difflib para encontrar coincidencias aproximadas entre cadenas de texto (útil para sinónimos y correcciones)
import difflib

# Importa os para verificar existencia de archivos o rutas (útil al registrar sesiones)
import os

# Importa re (expresiones regulares) para procesar entradas de texto del usuario
import re

# Importa random para generar números aleatorios (IDs de sesión, simulaciones de datos)
import random


# =========================================
# BLOQUE 1: DEFINICIÓN DE GRUPOS EXCLUSIVOS
# =========================================
# Algunos síntomas son variantes del mismo (por ejemplo: fiebre leve y fiebre alta).
# Agrupamos estos síntomas para evitar preguntas repetidas y facilitar el análisis.

grupos_exclusivos = {
    'fiebre': [
        'fiebre','fiebre_baja', 'fiebre_leve', 'fiebre_alta', 'fiebre_alta_o_hipotermia',
        'fiebre_intermitente', 'fiebre_nocturna', 'fiebre_en_casos_graves',
        'fiebre_persistente', 'fiebre_prolongada', 'fiebre_alta_y_prolongada',
        'fiebre_alta_repentina'
    ],
    'tos': [
        'tos','tos_seca', 'tos_con_flema', 'tos_con_expectoración',
        'tos_crónica', 'tos_persistente', 'tos_crónica_con_flemas', 'tos_leve'
    ],
    'presion_arterial': [
        'presión_arterial_alta', 'presión_arterial_baja', 'hipertensión',
        'hipotensión', 'hipertensión_arterial'
    ],
    'fatiga': [
        'fatiga','fatiga_extrema', 'fatiga_diurna', 'fatiga_persistente', 'fatiga_crónica'
    ]
}

# Creamos un diccionario inverso para saber a qué grupo pertenece cada síntoma.
# Por ejemplo, si un síntoma es 'fiebre_alta', podremos saber que pertenece al grupo 'fiebre'.
grupo_por_sintoma = {}
for grupo, sintomas in grupos_exclusivos.items():
    for s in sintomas:
        grupo_por_sintoma[s] = grupo

# =========================================
# BLOQUE 2: FUNCIONES BÁSICAS DE INTERACCIÓN
# =========================================
# Estas funciones sirven para hacer preguntas al usuario y asegurarse que las respuestas sean válidas.

# Esta función hace una pregunta tipo sí/no y valida la respuesta del usuario.
# Devuelve 1 si el usuario responde 'sí', y 0 si responde 'no'.
def preguntar_binario(pregunta):
    while True:
        respuesta = input(pregunta).strip().lower()
        if respuesta in ['s', 'si']:
            return 1  # 1 significa "sí tengo ese síntoma"
        elif respuesta in ['n', 'no']:
            return 0  # 0 significa "no tengo ese síntoma"
        else:
            print("Entrada inválida. Responda con 's' o 'n'.")

# Esta función convierte un nombre de síntoma del tipo 'dolor_de_cabeza' a un formato legible: 'Dolor de cabeza'
def formato_legible(sintoma):
    return sintoma.replace('_', ' ').capitalize()

# =========================================
# BLOQUE 3: CARGA DEL DATASET DE ENFERMEDADES
# =========================================
# Se lee el archivo CSV que contiene las enfermedades y sus síntomas relacionados.

# Cargamos el archivo CSV con los datos. Cada fila representa una enfermedad.
# Cada columna representa un síntoma o factor de riesgo. 1 = presente, 0 = ausente.
df = pd.read_csv('CODING_SAMSUNG/chatbot_medico/Dataset/Dataset_Enfermedades_FV.csv', encoding='utf-8', encoding_errors='replace')

# Normalizamos los nombres de las columnas para que sean consistentes:
# - Quitamos espacios
# - Convertimos a minúsculas
# - Reemplazamos espacios por guiones bajos
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convertimos las columnas de síntomas a números enteros (0 o 1), excepto las columnas descriptivas
for col in df.columns:
    if col not in ['nombre_de_la_enfermedad', 'breve_descripción', 'tratamiento']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Definimos qué columnas corresponden a factores de riesgo (edad, género, IMC, etc.)
risk_cols = ['hombre', 'mujer', 'obesidad', 'sobrepeso', 'desnutricion', 'niño', 'adolescente', 'adulto', 'adulto_mayor']

# El resto de las columnas (que no son de texto ni factores) se consideran síntomas
symptom_cols = [col for col in df.columns if col not in ['nombre_de_la_enfermedad', 'breve_descripción', 'tratamiento'] + risk_cols]

# =========================================
# BLOQUE 4: DATOS DEL USUARIO Y CLASIFICACIÓN DE RIESGO
# =========================================
# Solicitamos datos personales del usuario como edad, género, peso y altura.
# Luego calculamos el IMC (Índice de Masa Corporal) y lo clasificamos en categorías de riesgo.
# También determinamos en qué grupo etario se encuentra (niño, adulto, etc.).

# Función para pedir un número decimal (ej. peso o altura)
def pedir_float(msg):
    while True:
        try:
            valor = float(input(msg))
            if valor > 0:
                return valor
            print("Debe ser mayor que 0.")
        except ValueError:
            print("Entrada inválida.")

# Función para pedir un número entero (ej. edad)
def pedir_int(msg):
    while True:
        try:
            valor = int(input(msg))
            if valor > 0:
                return valor
            print("Debe ser mayor que 0.")
        except ValueError:
            print("Entrada inválida.")

# Función para pedir el género (M para masculino, F para femenino)
def pedir_genero():
    while True:
        g = input("Ingrese su género (M/F): ").strip().upper()
        if g in ['M', 'F']:
            return g
        print("Debe ingresar 'M' o 'F'.")

# Recolección de datos del usuario
edad = pedir_int("Ingrese su edad: ")
genero = pedir_genero()
peso = pedir_float("Ingrese su peso en kg: ")
altura = pedir_float("Ingrese su altura en metros: ")

# Calculamos el IMC usando la fórmula peso / altura^2
imc = peso / (altura ** 2)
print(f"Su IMC es de {imc:.1f}", end=", ")

# Creamos el vector del usuario con todos los síntomas y factores en 0 inicialmente
user_vector = {s: 0 for s in symptom_cols + risk_cols}
grupos_ya_confirmados = set()  # Aquí guardamos grupos clínicos ya confirmados (como 'fiebre')

# Clasificamos el IMC y marcamos el factor correspondiente
if imc < 18.5:
    print("lo que indica desnutrición.")
    user_vector['desnutricion'] = 1
elif imc < 25:
    print("lo que indica un peso normal.")
elif imc < 30:
    print("lo que indica sobrepeso.")
    user_vector['sobrepeso'] = 1
else:
    print("lo que indica obesidad.")
    user_vector['obesidad'] = 1

# Marcamos el género en el vector del usuario
user_vector['hombre' if genero == 'M' else 'mujer'] = 1

# Clasificamos al usuario en grupo etario
if edad <= 12:
    user_vector['niño'] = 1
elif edad <= 18:
    user_vector['adolescente'] = 1
elif edad <= 59:
    user_vector['adulto'] = 1
else:
    user_vector['adulto_mayor'] = 1


# =========================================
# BLOQUE 5: EXCLUSIÓN POR GÉNERO Y ENFERMEDADES IRRELEVANTES
# =========================================
# Se eliminan síntomas y enfermedades que son exclusivos del género opuesto al usuario.
# Esto evita mostrar preguntas o diagnósticos que no corresponden biológicamente.

# Síntomas exclusivos de hombres (no se deben mostrar a mujeres)
sintomas_exclusivos_hombre = {
    'dolor_testicular', 'masa_testicular', 'disfuncion_erectil', 'problemas_prostaticos',
    'crecimiento_prostata', 'dificultad_eyaculacion', 'sangre_en_eyaculacion'
}

# Enfermedades exclusivas de hombres
enfermedades_exclusivas_hombre = {
    'cáncer de próstata', 'cáncer de mama masculino', 'disfunción eréctil',
    'hiperplasia prostática benigna'
}

# Síntomas exclusivos de mujeres (no se deben mostrar a hombres)
sintomas_exclusivos_mujer = {
    'sangrado_vaginal', 'flujo_vaginal', 'dolor_pelvico_ciclo_menstrual',
    'dolor_durante_relaciones_sexuales', 'ausencia_menstruacion', 'menstruacion_irregular',
    'menopausia', 'sindrome_ovario_poliquistico', 'amenorrea', 'endometriosis',
    'vaginismo', 'cancer_de_cuello_uterino'
}

# Enfermedades exclusivas de mujeres
enfermedades_exclusivas_mujer = {
    'cáncer de mama femenino', 'síndrome de ovario poliquístico', 'menopausia',
    'amenorrea', 'endometriosis', 'vaginismo', 'cáncer de cuello uterino'
}

# Según el género, determinamos qué síntomas y enfermedades se deben excluir
if genero == 'M':
    sintomas_excluir = sintomas_exclusivos_mujer
    enfermedades_excluir = enfermedades_exclusivas_mujer
else:
    sintomas_excluir = sintomas_exclusivos_hombre
    enfermedades_excluir = enfermedades_exclusivas_hombre

# =========================================
# BLOQUE 6: ENTRADA DE SÍNTOMAS Y EXPANSIÓN INTELIGENTE
# =========================================
# El usuario escribe sus síntomas en texto libre (ej. "dolor de cabeza, fiebre").
# Usamos sinónimos personalizados para traducirlos al lenguaje del dataset.
# Luego buscamos coincidencias aproximadas (con difflib) y expandimos síntomas por grupo clínico.

# Diccionario de sinónimos personalizados para interpretar mejor la entrada del usuario
sinonimos_personalizados = {
    # Dolor general
    'busto': 'dolor_en_el_seno',
    'seno': 'dolor_en_el_seno',
    'pecho': 'dolor_en_el_seno',
    'vientre': 'dolor_abdominal',
    'abdomen': 'dolor_abdominal',
    'panza': 'dolor_abdominal',
    'tripa': 'dolor_abdominal',
    'barriga': 'dolor_abdominal',
    'estómago': 'dolor_abdominal',
    'nuca': 'dolor_de_cabeza',
    'cefalea': 'dolor_de_cabeza',
    'cabeza': 'dolor_de_cabeza',
    'mandíbula': 'dolor_dental',
    'dientes': 'dolor_dental',
    'muelas': 'dolor_dental',

    # Síntomas digestivos
    'náuseas': 'nauseas',
    'nausea': 'nauseas',
    'vomito': 'vomitos',
    'vómito': 'vomitos',
    'vomitar': 'vomitos',
    'diarrea': 'diarrea',
    'heces_sueltas': 'diarrea',
    'estreñimiento': 'constipacion',
    'constipado': 'constipacion',
    'constipación': 'constipacion',
    'acidez': 'reflujo_gastroesofagico',
    'agruras': 'reflujo_gastroesofagico',

    # Fatiga y sueño
    'cansancio': 'fatiga',
    'agotamiento': 'fatiga',
    'falta_de_energía': 'fatiga',
    'somnolencia': 'somnolencia',
    'sueño_excesivo': 'somnolencia',

    # Respiratorios
    'mucosidad': 'congestion_nasal',
    'nariz_tapada': 'congestion_nasal',
    'dificultad_para_respirar': 'dificultad_respiratoria',
    'falta_de_aire': 'dificultad_respiratoria',
    'respiración_agitada': 'dificultad_respiratoria',
    'dolor_pecho_al_respirar': 'dolor_toracico',
    'dolor_pecho': 'dolor_toracico',

    # Cardiovasculares
    'palpitaciones': 'latidos_rapidos',
    'taquicardia': 'latidos_rapidos',
    'presion_alta': 'presión_arterial_alta',
    'hipertensión': 'presión_arterial_alta',
    'presion_baja': 'presión_arterial_baja',
    'hipotensión': 'presión_arterial_baja',

    # Piel y fiebre
    'piel_roja': 'erupcion_cutanea',
    'manchas_en_la_piel': 'erupcion_cutanea',
    'eritema': 'erupcion_cutanea',
    'fiebre_leve': 'fiebre',
    'temperatura_alta': 'fiebre',
    'escalofríos': 'escalofrios',
    'frio': 'escalofrios',

    # Urinarios / genitales
    'dolor_al_orinar': 'disuria',
    'ardor_al_orinar': 'disuria',
    'ganas_frecuentes_de_orinar': 'poliuria',
    'micciones_frecuentes': 'poliuria',
    'sangre_en_orina': 'hematuria',

    # Psicológicos / neurológicos
    'confusión': 'desorientacion',
    'olvidos': 'perdida_de_memoria',
    'desmayos': 'síncope',
    'mareos': 'mareo',
    'mareado': 'mareo',
    'visión_borrosa': 'vision_borrosa',
    'visión_doble': 'vision_borrosa',
}

# Busca síntomas válidos en el dataset usando coincidencia aproximada
def encontrar_sintomas_validos(sintomas_usuario, symptom_cols, cutoff=0.70):
    sintomas_validos = []
    coincidencias = {}
    for s in sintomas_usuario:
        match = difflib.get_close_matches(s, symptom_cols, n=1, cutoff=cutoff)
        if match:
            sintomas_validos.append(match[0])
            coincidencias[s] = match[0]
    return list(set(sintomas_validos)), coincidencias

# Solicita síntomas hasta que el usuario ingrese al menos uno válido
def solicitar_sintomas_validos(symptom_cols, sinonimos_personalizados):
    while True:
        entrada_usuario = input("Ingrese sus síntomas separados por comas: ").lower().strip()
        sintomas_usuario = re.split(r'[,;]\\s*', entrada_usuario)

        # Aplicar sinónimos dentro de frases
        sintomas_mapeados = []
        for s in sintomas_usuario:
            encontrado = False
            for clave, valor in sinonimos_personalizados.items():
                if clave in s:
                    sintomas_mapeados.append(valor)
                    encontrado = True
                    break
            if not encontrado:
                sintomas_mapeados.append(s.strip())

        # Buscar coincidencias con los síntomas del dataset
        sintomas_validos, coincidencias = encontrar_sintomas_validos(sintomas_mapeados, symptom_cols)

        if sintomas_validos:
            print("\\n Coincidencias iniciales encontradas en el dataset:")
            for original, match in coincidencias.items():
                print(f"- '{original}' fue interpretado como → '{match}'")
            return sintomas_validos, coincidencias
        else:
            print("\\n No se ingresaron síntomas válidos. Intente nuevamente.\\n")

# Ejecutamos la función: requiere al menos un síntoma válido antes de continuar
sintomas_validos, coincidencias = solicitar_sintomas_validos(symptom_cols, sinonimos_personalizados)

# Expandimos automáticamente los síntomas por grupo clínico si aplican
sintomas_expandidos = set(sintomas_validos)
for s in sintomas_validos:
    grupo = grupo_por_sintoma.get(s)
    if grupo:
        sintomas_expandidos.update(grupos_exclusivos.get(grupo, []))


# =========================================
# BLOQUE 7: FILTRADO INICIAL DE ENFERMEDADES POSIBLES Y AJUSTE POR GÉNERO
# =========================================
# Con los síntomas expandidos, filtramos las enfermedades que contienen al menos uno de ellos.
# También nos aseguramos de mantener enfermedades exclusivas del mismo género si hay síntomas coincidentes,
# y eliminar enfermedades irrelevantes del género opuesto si no hay coincidencia.

# Creamos un filtro lógico para quedarnos con enfermedades que tienen al menos un síntoma coincidente
filtro = np.logical_or.reduce([df[s] == 1 for s in sintomas_expandidos if s in df.columns])
enfermedades_posibles = df[filtro].copy()

# Conservar enfermedades exclusivas del mismo género solo si hay síntomas coincidentes
if genero == 'M':
    enfermedades_posibles = enfermedades_posibles[
        ~(
            enfermedades_posibles['nombre_de_la_enfermedad'].isin(enfermedades_exclusivas_mujer) &
            ~enfermedades_posibles[sintomas_validos].any(axis=1)
        )
    ]
else:
    enfermedades_posibles = enfermedades_posibles[
        ~(
            enfermedades_posibles['nombre_de_la_enfermedad'].isin(enfermedades_exclusivas_hombre) &
            ~enfermedades_posibles[sintomas_validos].any(axis=1)
        )
    ]

# Después del filtrado, eliminamos los síntomas exclusivos del género opuesto para las próximas fases
sintomas_validos = [s for s in sintomas_validos if s not in sintomas_excluir]

# Guardamos los síntomas ya preguntados (los que el usuario ingresó)
sintomas_ya_preguntados = set(sintomas_validos)

# Si alguno de los síntomas confirmados pertenece a un grupo (ej. fiebre), guardamos el grupo
for sintoma in sintomas_validos:
    grupo = grupo_por_sintoma.get(sintoma)
    if grupo:
        grupos_ya_confirmados.add(grupo)


# =========================================
# BLOQUE 8: FASE 1 – PREGUNTAS GUIADAS POR SÍNTOMAS FRECUENTES
# =========================================
# Esta fase consiste en hacer preguntas al usuario de forma estructurada.
# Se priorizan los síntomas más frecuentes en las enfermedades filtradas.
# Se hace un mínimo de preguntas para asegurar una base suficiente para el diagnóstico.

print("\n Fase 1: Preguntas guiadas por síntomas relacionados...\n")

preguntas_realizadas = 0           # Contador de preguntas realizadas
MAX_PREGUNTAS = 50                 # Máximo total de preguntas permitidas
MIN_PREGUNTAS = 15                 # Mínimo de preguntas obligatorias en esta fase

# Ordenamos los síntomas por frecuencia (los más comunes primero)
sintomas_frecuentes = enfermedades_posibles[symptom_cols].sum().sort_values(ascending=False).index.tolist()

# Iniciamos ciclo para hacer preguntas
for sintoma in sintomas_frecuentes:

    # No repetir síntomas ya preguntados o excluidos por género
    if sintoma in sintomas_ya_preguntados or sintoma in sintomas_excluir:
        continue

    # No volver a preguntar síntomas del mismo grupo si ya se confirmó alguno
    grupo = grupo_por_sintoma.get(sintoma)
    if grupo in grupos_ya_confirmados:
        continue

    # Pregunta binaria: ¿tiene este síntoma?
    respuesta = preguntar_binario(f"¿Presenta '{formato_legible(sintoma)}'? (s/n): ")
    user_vector[sintoma] = respuesta
    sintomas_ya_preguntados.add(sintoma)
    preguntas_realizadas += 1

    # Si el síntoma pertenece a un grupo clínico y fue confirmado, lo marcamos para evitar preguntar sus variantes
    if respuesta == 1 and grupo:
        grupos_ya_confirmados.add(grupo)

    # Si ya se hicieron suficientes preguntas obligatorias, salimos de la fase
    if preguntas_realizadas >= MIN_PREGUNTAS:
        break


# =========================================
# BLOQUE 9: FUNCIÓN PARA GENERAR LA SIGUIENTE PREGUNTA (Fase 2)
# =========================================
# Esta función determina cuál es el siguiente síntoma más relevante para preguntar.
# Se basa en los síntomas pendientes de las enfermedades más probables (top 10 por score).
# Si no hay síntomas prioritarios, selecciona uno genérico que no se haya preguntado.

def generar_siguiente_sintoma(resultado, user_vector, symptom_cols, sintomas_ya_preguntados, sintomas_excluir, grupos_ya_confirmados):
    # Estos grupos no deben preguntarse si ya hay un síntoma confirmado del grupo
    grupos_a_excluir_si_ya_confirmado = {'fiebre', 'tos', 'fatiga', 'presion_arterial'}
    sintomas_prioritarios = {}

    # Recorremos las 10 enfermedades más probables
    for enf in resultado[:10]:
        for sintoma in enf['pendientes']:
            if sintoma in sintomas_ya_preguntados or sintoma in sintomas_excluir:
                continue  # Evitar síntomas ya preguntados o excluidos por género

            grupo = grupo_por_sintoma.get(sintoma)
            # Si el grupo ya fue confirmado (ej. fiebre), evitamos sus variantes
            if grupo in grupos_a_excluir_si_ya_confirmado and any(user_vector[s] == 1 for s in grupos_exclusivos.get(grupo, [])):
                continue

            # Sumamos el score de la enfermedad para priorizar el síntoma
            sintomas_prioritarios[sintoma] = sintomas_prioritarios.get(sintoma, 0) + enf['score']

    # Si hay síntomas prioritarios, elegimos el de mayor puntuación
    if sintomas_prioritarios:
        siguiente_sintoma = sorted(sintomas_prioritarios.items(), key=lambda x: -x[1])[0][0]
        return siguiente_sintoma, True  # True: es relevante

    # Si no hay síntomas prioritarios, buscamos uno genérico aún no preguntado
    sintomas_restantes = [s for s in symptom_cols if s not in sintomas_ya_preguntados and s not in sintomas_excluir]
    if sintomas_restantes:
        return sintomas_restantes[0], False  # False: genérico (no optimizado)

    return None, False  # No quedan síntomas disponibles


# =========================================
# BLOQUE 10: FASE 2 – PREGUNTAS ADAPTATIVAS OPTIMIZADAS
# =========================================
# Aquí el sistema intenta afinar el diagnóstico con preguntas inteligentes.
# Se recalculan las coincidencias después de cada respuesta.
# Se permite detener esta fase si el usuario lo desea o si se alcanza un diagnóstico suficientemente confiable.

print("\n Fase 2: Preguntas adaptativas ...\n")

MIN_PREGUNTAS_OBLIGATORIAS = 30           # Mínimo de preguntas antes de ofrecer diagnóstico parcial
UMBRAL_PORCENTAJE_ENFERMEDAD = 0.7        # Mínimo 70% de síntomas cubiertos para aceptar un diagnóstico
preguntas_desde_ultima_confirmacion = 0   # Contador para evaluar cuándo volver a preguntar
confirmacion_umbral_clinico = 0           # Cuántas veces se superó el umbral mínimo
resultado = []                            # Lista para guardar los diagnósticos intermedios

while preguntas_realizadas < MAX_PREGUNTAS:
    # Generamos un vector de síntomas confirmados del usuario
    user_symptom_vector = np.array([user_vector[s] for s in symptom_cols])
    resultado = []

    # Recorremos todas las enfermedades filtradas
    for _, row in enfermedades_posibles.iterrows():
        enfermedad_vector = row[symptom_cols].values
        total_e = enfermedad_vector.sum()  # Total de síntomas que tiene esa enfermedad
        total_u = user_symptom_vector.sum()  # Total de síntomas que ha confirmado el usuario

        if total_e == 0 or total_u == 0:
            continue  # Saltar si no hay datos suficientes

        # Contamos cuántos síntomas coinciden entre usuario y enfermedad
        coincidencia = np.sum((user_symptom_vector == 1) & (enfermedad_vector == 1))
        porc_e = coincidencia / total_e     # % de síntomas de la enfermedad presentes en el usuario
        porc_u = coincidencia / total_u     # % de síntomas del usuario que coinciden con la enfermedad
        score = round(100 * (porc_e + porc_u) / 2, 1)  # Score final (promedio de ambos)

        # Síntomas de la enfermedad que el usuario aún no ha confirmado
        pendientes = [s for i, s in enumerate(symptom_cols)
                      if enfermedad_vector[i] == 1 and user_symptom_vector[i] == 0
                      and s not in sintomas_excluir and s not in sintomas_ya_preguntados]

        resultado.append({
            'nombre': row['nombre_de_la_enfermedad'],
            'coincidencia': int(coincidencia),
            'total_enfermedad': int(total_e),
            'score': score,
            'descripcion': row['breve_descripción'],
            'tratamiento': row['tratamiento'],
            'pendientes': pendientes
        })

    # Ordenamos enfermedades por score descendente
    resultado = sorted(resultado, key=lambda x: x['score'], reverse=True)
    if not resultado:
        print("\n No hay suficientes datos para continuar con la predicción.")
        break

    enfermedad_top = resultado[0]
    cobertura = enfermedad_top['coincidencia'] / enfermedad_top['total_enfermedad']

    # Diagnóstico suficientemente confiable: score >= 70 y cobertura >= 80%
    if enfermedad_top['score'] >= 70 and cobertura >= 0.8:
        print(f"\n Diagnóstico suficientemente confiable alcanzado con: {enfermedad_top['nombre']}")
        break

    # Diagnóstico fuerte pero síntomas aún pendientes (el usuario decide si continuar)
    if enfermedad_top['score'] >= 70 and preguntas_desde_ultima_confirmacion >= 10:
        print(f"\n Diagnóstico con alta coincidencia, pero con síntomas clínicos aún sin confirmar "
              f"({enfermedad_top['coincidencia']} de {enfermedad_top['total_enfermedad']}).")
        decision = input("¿Desea seguir respondiendo preguntas para afinar el diagnóstico? (s/n): ").strip().lower()
        if decision in ['n', 'no']:
            break
        preguntas_desde_ultima_confirmacion = 0

    # Generamos el siguiente síntoma a preguntar
    siguiente_sintoma, es_relevante = generar_siguiente_sintoma(
        resultado,
        user_vector,
        symptom_cols,
        sintomas_ya_preguntados,
        sintomas_excluir,
        grupos_ya_confirmados
    )

    if not siguiente_sintoma:
        print("\n No quedan síntomas para seguir preguntando.")
        break

    # Si el síntoma no es relevante, se hacen hasta 5 preguntas genéricas
    if not es_relevante:
        preguntas_genericas_realizadas = 0
        while preguntas_genericas_realizadas < 5:
            grupo = grupo_por_sintoma.get(siguiente_sintoma)
            respuesta = preguntar_binario(f"¿Presenta '{siguiente_sintoma.replace('_', ' ')}'? (s/n): ")
            user_vector[siguiente_sintoma] = respuesta
            sintomas_ya_preguntados.add(siguiente_sintoma)
            preguntas_realizadas += 1
            preguntas_genericas_realizadas += 1
            preguntas_desde_ultima_confirmacion += 1
            if respuesta == 1 and grupo:
                grupos_ya_confirmados.add(grupo)

            # Buscar el siguiente síntoma genérico
            siguiente_sintoma, es_relevante = generar_siguiente_sintoma(
                resultado,
                user_vector,
                symptom_cols,
                sintomas_ya_preguntados,
                sintomas_excluir,
                grupos_ya_confirmados
            )

            if not siguiente_sintoma:
                break

        # Preguntar al usuario si quiere continuar con más preguntas genéricas
        decision = input("¿Desea seguir respondiendo más preguntas generales? (s/n): ").strip().lower()
        if decision in ['n', 'no']:
            break
        else:
            continue

    # Pregunta relevante: se registra respuesta y se marcan síntomas del grupo como confirmados si aplica
    grupo = grupo_por_sintoma.get(siguiente_sintoma)
    respuesta = preguntar_binario(f"¿Presenta '{siguiente_sintoma.replace('_', ' ')}'? (s/n): ")
    user_vector[siguiente_sintoma] = respuesta
    sintomas_ya_preguntados.add(siguiente_sintoma)
    preguntas_realizadas += 1
    preguntas_desde_ultima_confirmacion += 1

    if respuesta == 1 and grupo:
        grupos_ya_confirmados.add(grupo)
        # Si se confirma un síntoma de un grupo exclusivo, confirmamos todo el grupo automáticamente
        for s in grupos_exclusivos.get(grupo, []):
            if s in symptom_cols and s not in sintomas_ya_preguntados:
                user_vector[s] = 1
                sintomas_ya_preguntados.add(s)

    # Si se llegó al mínimo obligatorio de preguntas, se activa evaluación clínica
    if preguntas_realizadas >= MIN_PREGUNTAS_OBLIGATORIAS:
        confirmacion_umbral_clinico += 1

    # Si se supera varias veces el umbral clínico, se pregunta si se quiere terminar
    if confirmacion_umbral_clinico >= 10:
        enfermedades_con_cobertura_aceptable = [
            e for e in resultado if e['coincidencia'] / e['total_enfermedad'] >= UMBRAL_PORCENTAJE_ENFERMEDAD
        ]
        if enfermedades_con_cobertura_aceptable:
            print("\n Se ha alcanzado un umbral razonable con al menos una enfermedad.")
            decision = input("¿Desea seguir respondiendo preguntas para afinar el diagnóstico? (s/n): ").strip().lower()
            if decision in ['n', 'no']:
                break
            confirmacion_umbral_clinico = 0

# =========================================
# BLOQUE 11: CÁLCULO DEL SCORE FINAL Y DIAGNÓSTICO COMBINADO (CLÁSICO + KNN)
# =========================================
# Aquí se calcula el resultado final del sistema. Se combinan dos enfoques:
# 1. Score clásico: basado en coincidencia de síntomas.
# 2. Score KNN: basado en similitud vectorial (Hamming).
# El resultado se muestra como una lista con los diagnósticos más probables.

print("\n*** Diagnóstico final basado en combinación de métodos ***")

# Primero, función de score clásico reutilizada
def calcular_score(enf_row, user_vector, symptom_cols):
    total_e = enf_row[symptom_cols].sum()
    coincidencia = sum(user_vector[c] == 1 and enf_row[c] == 1 for c in symptom_cols)
    total_u = sum(user_vector[c] for c in symptom_cols)
    if total_e == 0 or total_u == 0:
        return 0, 0, 0
    porc_e = coincidencia / total_e
    porc_u = coincidencia / total_u
    score = round(100 * (porc_e + porc_u) / 2, 1)
    return score, coincidencia, total_e

# Calculamos el score clásico para todas las enfermedades candidatas
scores_clasicos = {}
coincidencias_clasicas = {}
totales_enf_clasicas = {}
for _, row in enfermedades_posibles.iterrows():
    score, coincidencia, total_e = calcular_score(row, user_vector, symptom_cols)
    nombre = row['nombre_de_la_enfermedad']
    scores_clasicos[nombre] = score
    coincidencias_clasicas[nombre] = coincidencia
    totales_enf_clasicas[nombre] = total_e

# Calculamos el score por KNN con distancia Hamming (más eficiente para binarios)
X = enfermedades_posibles[symptom_cols + risk_cols].values
y = enfermedades_posibles['nombre_de_la_enfermedad'].values
knn = KNeighborsClassifier(n_neighbors=len(enfermedades_posibles), metric='hamming')
knn.fit(X, y)

# Convertimos el vector del usuario a arreglo compatible para comparación
user_input_vector = np.array([user_vector[col] for col in symptom_cols + risk_cols]).reshape(1, -1)

# Calculamos las distancias a todas las enfermedades usando KNN
distancias, indices = knn.kneighbors(user_input_vector, n_neighbors=len(enfermedades_posibles))

# Combinamos ambos scores en un resultado final ponderado
diagnosticos_combinados = []
for idx in indices[0]:
    fila = enfermedades_posibles.iloc[idx]
    nombre = fila['nombre_de_la_enfermedad']
    dist = distancias[0][list(indices[0]).index(idx)]
    score_knn = round(100 * (1 - dist), 1)
    score_clasico = scores_clasicos.get(nombre, 0)
    score_final = round(0.6 * score_clasico + 0.4 * score_knn, 1)  # 60% clásico + 40% KNN

    diagnosticos_combinados.append({
        'nombre': nombre,
        'score_final': score_final,
        'score_clasico': score_clasico,
        'score_knn': score_knn,
        'coincidencia': coincidencias_clasicas.get(nombre, 0),
        'total_enfermedad': totales_enf_clasicas.get(nombre, 0),
        'descripcion': fila['breve_descripción'],
        'tratamiento': fila['tratamiento']
    })

# Ordenamos los resultados por mayor score final
mejores = sorted(diagnosticos_combinados, key=lambda x: x['score_final'], reverse=True)[:5]

# Mostramos el top 5 de diagnósticos más probables
for r in mejores:
    print(f"\n{r['nombre']}: Score combinado = {r['score_final']}%")
    print(f"  - Score clásico: {r['score_clasico']}%")
    print(f"  - Similitud KNN: {r['score_knn']}%")
    if r['coincidencia'] and r['total_enfermedad']:
        print(f"  - Coincidencia de síntomas: {r['coincidencia']} de {r['total_enfermedad']}")
    print(f"  - Descripción: {r['descripcion']}")
    print(f"  - Tratamiento: {r['tratamiento']}")
    print("Este diagnóstico no sustituye a un médico, por favor acuda con un profesional.")


# =========================================
# BLOQUE 12: MOSTRAR ENFERMEDADES ADICIONALES CON COINCIDENCIA ACEPTABLE
# =========================================
# Aquí mostramos otras enfermedades con score razonable (>= 40%)
# Esto permite al usuario considerar más opciones, especialmente en diagnósticos ambiguos.

# Filtramos las enfermedades fuera del top 5 que tengan score >= 40%
otros_diagnosticos = sorted(diagnosticos_combinados, key=lambda x: x['score_final'], reverse=True)[5:]
otros_diagnosticos = [r for r in otros_diagnosticos if r['score_final'] >= 40]

# Si existen más enfermedades relevantes, las mostramos como lista adicional
if otros_diagnosticos:
    print("\n Otras enfermedades con coincidencia aceptable:")
    for r in otros_diagnosticos[:5]:  # Mostramos máximo 5 adicionales
        print(f"- {r['nombre']}: {r['score_final']}% ({r['coincidencia']} de {r['total_enfermedad']})")

    # Preguntamos si desea ver detalles extendidos de estas enfermedades
    ver_detalles = input("\n¿Desea ver detalles de estas enfermedades adicionales? (s/n): ").strip().lower()
    if ver_detalles in ['s', 'si']:
        for r in otros_diagnosticos[:5]:
            print(f"\n{r['nombre']}: Score combinado = {r['score_final']}%")
            print(f"  - Score clásico: {r['score_clasico']}%")
            print(f"  - Similitud KNN: {r['score_knn']}%")
            if r['coincidencia'] and r['total_enfermedad']:
                print(f"  - Coincidencia de síntomas: {r['coincidencia']} de {r['total_enfermedad']}")
            print(f"  - Descripción: {r['descripcion']}")
            print(f"  - Tratamiento: {r['tratamiento']}")
            print("Este diagnóstico no sustituye a un médico, por favor acuda con un profesional.")


# Este bloque se ejecuta al final del sistema y registra los resultados más importantes de la sesión diagnóstica.

# =========================================
# BLOQUE 13: GUARDAR RESULTADOS DE SESIÓN 
# =========================================

# Se genera un timestamp actual para la sesión
fecha = datetime.now()

# Se prepara un diccionario con los datos clave que se van a guardar por sesión
registro = {
    'id_sesion': f"S{fecha.strftime('%Y%m%d%H%M%S')}{random.randint(100,999)}",  # ID único basado en fecha y valor aleatorio
    'fecha_hora': datetime.now().isoformat(),  # Fecha y hora exacta en formato estándar ISO
    'score_clasico': scores_clasicos.get(mejores[0]['nombre'], 0),  # Score del método clásico para la enfermedad Top 1
    'score_knn': mejores[0]['score_knn'],  # Score del método KNN para la enfermedad Top 1
    'score_final': mejores[0]['score_final'],  # Score final combinado (60% clásico + 40% KNN)
    'preguntas_realizadas': preguntas_realizadas,  # Número de preguntas hechas al usuario en la sesión
    'sintomas_confirmados': sum(user_vector[s] for s in symptom_cols),  # Cuántos síntomas fueron confirmados por el usuario
    'top_1': mejores[0]['nombre'],  # Nombre de la enfermedad con mejor score final
    'score_top_1': mejores[0]['score_final']  # Repetición del score final de la mejor enfermedad (útil para reportes)
}

# Se define la ruta donde se guarda el archivo de registros acumulados
archivo_registro = 'CODING_SAMSUNG/chatbot_medico/Dataset/registro_sesiones.csv'

# Se obtiene la lista de campos (nombres de columnas) en el mismo orden del diccionario
campos = list(registro.keys())

# Se verifica si el archivo ya existe o si debe crearse por primera vez
archivo_nuevo = not os.path.exists(archivo_registro)

# Se abre el archivo en modo append ('a') para agregar la nueva sesión sin borrar las anteriores
with open(archivo_registro, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=campos)
    
    # Si es la primera vez, se escribe el encabezado con los nombres de las columnas
    if archivo_nuevo:
        writer.writeheader()
    
    # Se escribe la fila correspondiente a esta sesión
    writer.writerow(registro)

# Confirmación final al usuario
print("\n[✓] Sesión registrada correctamente en 'registro_sesiones.csv'.")
