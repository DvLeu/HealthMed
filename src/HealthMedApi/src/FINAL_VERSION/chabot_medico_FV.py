import pandas as pd
import numpy as np
import difflib
import re

# ===========================
# GRUPOS DE SÍNTOMAS EXCLUSIVOS
# ===========================
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

grupo_por_sintoma = {}
for grupo, sintomas in grupos_exclusivos.items():
    for s in sintomas:
        grupo_por_sintoma[s] = grupo

# ===========================
# FUNCIÓN DE PREGUNTA BINARIA
# ===========================
def preguntar_binario(pregunta):
    while True:
        respuesta = input(pregunta).strip().lower()
        if respuesta in ['s', 'si']:
            return 1
        elif respuesta in ['n', 'no']:
            return 0
        else:
            print("Entrada inválida. Responda con 's' o 'n'.")

def formato_legible(sintoma):
    return sintoma.replace('_', ' ').capitalize()

# ===========================
# CARGAR Y PROCESAR DATASET
# ===========================
df = pd.read_csv('Dataset_Enfermedades_Final.csv', encoding='utf-8', encoding_errors='replace')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# Forzar columnas binarias (síntomas) a enteros
for col in df.columns:
    if col not in ['nombre_de_la_enfermedad', 'breve_descripción', 'tratamiento']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)


risk_cols = ['hombre', 'mujer', 'obesidad', 'sobrepeso', 'desnutricion', 'niño', 'adolescente', 'adulto', 'adulto_mayor']
symptom_cols = [col for col in df.columns if col not in ['nombre_de_la_enfermedad', 'breve_descripción', 'tratamiento'] + risk_cols]

# ===========================
# DATOS DEL USUARIO
# ===========================
def pedir_float(msg):
    while True:
        try:
            valor = float(input(msg))
            if valor > 0:
                return valor
            print("Debe ser mayor que 0.")
        except ValueError:
            print("Entrada inválida.")

def pedir_int(msg):
    while True:
        try:
            valor = int(input(msg))
            if valor > 0:
                return valor
            print("Debe ser mayor que 0.")
        except ValueError:
            print("Entrada inválida.")

def pedir_genero():
    while True:
        g = input("Ingrese su género (M/F): ").strip().upper()
        if g in ['M', 'F']:
            return g
        print("Debe ingresar 'M' o 'F'.")

edad = pedir_int("Ingrese su edad: ")
genero = pedir_genero()
peso = pedir_float("Ingrese su peso en kg: ")
altura = pedir_float("Ingrese su altura en metros: ")

# ===========================
# IMC y CATEGORÍAS DE RIESGO
# ===========================
imc = peso / (altura ** 2)
print(f"Su IMC es de {imc:.1f}", end=", ")

user_vector = {s: 0 for s in symptom_cols + risk_cols}
grupos_ya_confirmados = set()

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

user_vector['hombre' if genero == 'M' else 'mujer'] = 1

if edad <= 12:
    user_vector['niño'] = 1
elif edad <= 18:
    user_vector['adolescente'] = 1
elif edad <= 59:
    user_vector['adulto'] = 1
else:
    user_vector['adulto_mayor'] = 1


# ===========================
# EXCLUSIÓN POR GÉNERO
# ===========================
sintomas_exclusivos_hombre = {
    'dolor_testicular', 'masa_testicular', 'disfuncion_erectil', 'problemas_prostaticos',
    'crecimiento_prostata', 'dificultad_eyaculacion', 'sangre_en_eyaculacion'
}
enfermedades_exclusivas_hombre = {
    'cáncer de próstata', 'cáncer de mama masculino', 'disfunción eréctil',
    'hiperplasia prostática benigna'
}
sintomas_exclusivos_mujer = {
    'sangrado_vaginal', 'flujo_vaginal', 'dolor_pelvico_ciclo_menstrual',
    'dolor_durante_relaciones_sexuales', 'ausencia_menstruacion', 'menstruacion_irregular',
    'menopausia', 'sindrome_ovario_poliquistico', 'amenorrea', 'endometriosis',
    'vaginismo', 'cancer_de_cuello_uterino'
}
enfermedades_exclusivas_mujer = {
    'cáncer de mama femenino', 'síndrome de ovario poliquístico', 'menopausia',
    'amenorrea', 'endometriosis', 'vaginismo', 'cáncer de cuello uterino'
}

# EXCLUIR SOLO LAS DEL GÉNERO OPUESTO
if genero == 'M':
    sintomas_excluir = sintomas_exclusivos_mujer
    enfermedades_excluir = enfermedades_exclusivas_mujer
else:
    sintomas_excluir = sintomas_exclusivos_hombre
    enfermedades_excluir = enfermedades_exclusivas_hombre

# ===========================
# ENTRADA DE SÍNTOMAS CON REPETICIÓN HASTA VALIDACIÓN
# ===========================

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

def encontrar_sintomas_validos(sintomas_usuario, symptom_cols, cutoff=0.70):
    sintomas_validos = []
    coincidencias = {}
    for s in sintomas_usuario:
        match = difflib.get_close_matches(s, symptom_cols, n=1, cutoff=cutoff)
        if match:
            sintomas_validos.append(match[0])
            coincidencias[s] = match[0]
    return list(set(sintomas_validos)), coincidencias

def solicitar_sintomas_validos(symptom_cols, sinonimos_personalizados):
    while True:
        entrada_usuario = input("Ingrese sus síntomas separados por comas: ").lower().strip()
        sintomas_usuario = re.split(r'[,;]\s*', entrada_usuario)

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

        sintomas_validos, coincidencias = encontrar_sintomas_validos(sintomas_mapeados, symptom_cols)

        if sintomas_validos:
            print("\n Coincidencias iniciales encontradas en el dataset:")
            for original, match in coincidencias.items():
                print(f"- '{original}' fue interpretado como → '{match}'")
            return sintomas_validos, coincidencias
        else:
            print("\n No se ingresaron síntomas válidos. Intente nuevamente.\n")

# Requiere al menos un síntoma válido antes de continuar
sintomas_validos, coincidencias = solicitar_sintomas_validos(symptom_cols, sinonimos_personalizados)

# Una vez obtenidos, aplicar el filtro con seguridad
filtro = np.logical_or.reduce([df[s] == 1 for s in sintomas_validos])
enfermedades_posibles = df[filtro].copy()


# Conservar enfermedades del mismo género siempre
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


# Ahora sí: excluir los síntomas del género opuesto para las siguientes fases (preguntas)
sintomas_validos = [s for s in sintomas_validos if s not in sintomas_excluir]

sintomas_ya_preguntados = set(sintomas_validos)
for sintoma in sintomas_validos:
    grupo = grupo_por_sintoma.get(sintoma)
    if grupo:
        grupos_ya_confirmados.add(grupo)

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


# ===========================
# FASE 1: PREGUNTAS GUIADAS
# ===========================
print("\n Fase 1: Preguntas guiadas por síntomas relacionados...\n")
preguntas_realizadas = 0
MAX_PREGUNTAS = 50
MIN_PREGUNTAS = 15

sintomas_frecuentes = enfermedades_posibles[symptom_cols].sum().sort_values(ascending=False).index.tolist()
for sintoma in sintomas_frecuentes:
    if sintoma in sintomas_ya_preguntados or sintoma in sintomas_excluir:
        continue

    grupo = grupo_por_sintoma.get(sintoma)
    if grupo in grupos_ya_confirmados:
        continue

    respuesta = preguntar_binario(f"¿Presenta '{sintoma.replace('_', ' ')}'? (s/n): ")
    user_vector[sintoma] = respuesta
    sintomas_ya_preguntados.add(sintoma)
    preguntas_realizadas += 1

    if respuesta == 1 and grupo:
        grupos_ya_confirmados.add(grupo)

    if preguntas_realizadas >= MIN_PREGUNTAS:
        break


def generar_siguiente_sintoma(resultado, user_vector, symptom_cols, sintomas_ya_preguntados, sintomas_excluir, grupos_ya_confirmados):
    grupos_a_excluir_si_ya_confirmado = {'fiebre', 'tos', 'fatiga', 'presion_arterial'}
    sintomas_prioritarios = {}

    for enf in resultado[:5]:
        for sintoma in enf['pendientes']:
            if sintoma in sintomas_ya_preguntados or sintoma in sintomas_excluir:
                continue
            grupo = grupo_por_sintoma.get(sintoma)
            if grupo in grupos_a_excluir_si_ya_confirmado and any(user_vector[s] == 1 for s in grupos_exclusivos.get(grupo, [])):
                continue
            sintomas_prioritarios[sintoma] = sintomas_prioritarios.get(sintoma, 0) + enf['score']

    if sintomas_prioritarios:
        siguiente_sintoma = sorted(sintomas_prioritarios.items(), key=lambda x: -x[1])[0][0]
        return siguiente_sintoma, True  # True: relevante

    sintomas_restantes = [s for s in symptom_cols if s not in sintomas_ya_preguntados and s not in sintomas_excluir]
    if sintomas_restantes:
        return sintomas_restantes[0], False  # False: genérico

    return None, False

# ===========================
# FASE 2: Preguntas adaptativas (optimizada y corregida)
# ===========================
print("\n Fase 2: Preguntas adaptativas ...\n")
MIN_PREGUNTAS_OBLIGATORIAS = 30
UMBRAL_PORCENTAJE_ENFERMEDAD = 0.7
preguntas_desde_ultima_confirmacion = 0
confirmacion_umbral_clinico = 0
resultado = []

while preguntas_realizadas < MAX_PREGUNTAS:
    user_symptom_vector = np.array([user_vector[s] for s in symptom_cols])
    resultado = []

    for _, row in enfermedades_posibles.iterrows():
        enfermedad_vector = row[symptom_cols].values
        total_e = enfermedad_vector.sum()
        total_u = user_symptom_vector.sum()

        if total_e == 0 or total_u == 0:
            continue

        coincidencia = np.sum((user_symptom_vector == 1) & (enfermedad_vector == 1))
        porc_e = coincidencia / total_e
        porc_u = coincidencia / total_u
        score = round(100 * (porc_e + porc_u) / 2, 1)

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

    resultado = sorted(resultado, key=lambda x: x['score'], reverse=True)
    if not resultado:
        print("\n No hay suficientes datos para continuar con la predicción.")
        break

    enfermedad_top = resultado[0]
    cobertura = enfermedad_top['coincidencia'] / enfermedad_top['total_enfermedad']

    if enfermedad_top['score'] >= 70 and cobertura >= 0.8:
        print(f"\n Diagnóstico suficientemente confiable alcanzado con: {enfermedad_top['nombre']}")
        break

    if enfermedad_top['score'] >= 70 and preguntas_desde_ultima_confirmacion >= 10:
        print(f"\n Diagnóstico con alta coincidencia, pero con síntomas clínicos aún sin confirmar "
              f"({enfermedad_top['coincidencia']} de {enfermedad_top['total_enfermedad']}).")
        decision = input("¿Desea seguir respondiendo preguntas para afinar el diagnóstico? (s/n): ").strip().lower()
        if decision in ['n', 'no']:
            break
        preguntas_desde_ultima_confirmacion = 0

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

        decision = input("¿Desea seguir respondiendo más preguntas generales? (s/n): ").strip().lower()
        if decision in ['n', 'no']:
            break
        else:
            continue

    grupo = grupo_por_sintoma.get(siguiente_sintoma)
    respuesta = preguntar_binario(f"¿Presenta '{siguiente_sintoma.replace('_', ' ')}'? (s/n): ")
    user_vector[siguiente_sintoma] = respuesta
    sintomas_ya_preguntados.add(siguiente_sintoma)
    preguntas_realizadas += 1
    preguntas_desde_ultima_confirmacion += 1

    if respuesta == 1 and grupo:
        grupos_ya_confirmados.add(grupo)

    if preguntas_realizadas >= MIN_PREGUNTAS_OBLIGATORIAS:
        confirmacion_umbral_clinico += 1
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


# ===========================
# RESULTADOS FINALES (Top 3)
# ===========================
print("\n*** Top 3 posibles diagnósticos ***")
diagnosticos_finales = []
for _, row in enfermedades_posibles.iterrows():
    score, coincidencia, total_e = calcular_score(row, user_vector, symptom_cols)
    if score == 0:
        continue
    diagnosticos_finales.append({
        'nombre': row['nombre_de_la_enfermedad'],
        'coincidencia': coincidencia,
        'total_enfermedad': total_e,
        'score': score,
        'descripcion': row['breve_descripción'],
        'tratamiento': row['tratamiento']
    })

top_3_resultados = sorted(diagnosticos_finales, key=lambda x: x['score'], reverse=True)[:3]

if not top_3_resultados:
    print("\n No hay suficientes coincidencias para un diagnóstico confiable.")
else:
    for r in top_3_resultados:
        print(f"\n{r['nombre']}: {r['score']}%")
        print(f"  - Coincidencia de síntomas: {r['coincidencia']} de {r['total_enfermedad']}")
        print(f"  - Descripción: {r['descripcion']}")
        print(f"  - Tratamiento: {r['tratamiento']}")
        print("Este diagnóstico no sustituye a un médico, por favor acuda con un profesional.")

    if all(r['score'] < 40 for r in top_3_resultados):
        print("\n Advertencia: la similitud general es baja. Ingrese más síntomas o consulte a un médico.")