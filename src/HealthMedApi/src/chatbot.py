from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import pandas as pd
import numpy as np
import difflib
import uuid
import re

# ===========================
# CLASES Pydantic para validacion
# ===========================
class DatosUsuario(BaseModel):
    edad: int
    genero: str
    peso: float
    altura: float
    sintomas: List[str]

class SintomaPregunta(BaseModel):
    sintoma: str
    grupo: Optional[str] = None
    es_relevante: bool = True

class RespuestaSintoma(BaseModel):
    sintoma: str
    respuesta: bool

class SesionChat(BaseModel):
    id_sesion: str
    datos_usuario: Optional[DatosUsuario] = None
    sintomas_confirmados: Dict[str, int] = {}
    sintomas_preguntados: List[str] = []
    grupos_confirmados: List[str] = []
    preguntas_realizadas: int = 0
    preguntas_desde_ultima_confirmacion: int = 0
    fase: int = 1  # 1: Preguntas guiadas, 2: Preguntas adaptativas

class ResultadoDiagnostico(BaseModel):
    enfermedades: List[Dict] = []
    mensaje: Optional[str] = None
    preguntas_realizadas: int = 0
    sintomas_confirmados: int = 0

# Variables globales
df = None
symptom_cols = []
risk_cols = []
sesiones = {}

# ===========================
# Grupos de síntomas y Sinónimos
# ===========================
grupos_exclusivos = {
    'fiebre': ['fiebre', 'fiebre_baja', 'fiebre_leve', 'fiebre_alta', 'fiebre_alta_o_hipotermia',
            'fiebre_intermitente', 'fiebre_nocturna', 'fiebre_en_casos_graves',
            'fiebre_persistente', 'fiebre_prolongada', 'fiebre_alta_y_prolongada', 'fiebre_alta_repentina'],
    'tos': ['tos', 'tos_seca', 'tos_con_flema', 'tos_con_expectoración',
            'tos_crónica', 'tos_persistente', 'tos_crónica_con_flemas', 'tos_leve'],
    'presion_arterial': ['presión_arterial_alta', 'presión_arterial_baja', 'hipertensión',
                        'hipotensión', 'hipertensión_arterial'],
    'fatiga': ['fatiga', 'fatiga_extrema', 'fatiga_diurna', 'fatiga_persistente', 'fatiga_crónica']
}

grupo_por_sintoma = {}
for grupo, sintomas in grupos_exclusivos.items():
    for sintoma in sintomas:
        grupo_por_sintoma[sintoma] = grupo

# Síntomas exclusivos por género
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

# NUEVO: Sinónimos para mejor reconocimiento de síntomas
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

# ===========================
# FUNCIONES AUXILIARES
# ===========================
def cargar_dataset(path: str):
    global df, symptom_cols, risk_cols
    df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    for col in df.columns:
        if col not in ['nombre_de_la_enfermedad', 'breve_descripción', 'tratamiento']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    risk_cols = ['hombre', 'mujer', 'obesidad', 'sobrepeso', 'desnutricion', 'niño', 'adolescente', 'adulto', 'adulto_mayor']
    symptom_cols = [col for col in df.columns if col not in ['nombre_de_la_enfermedad', 'breve_descripción', 'tratamiento'] + risk_cols]

def encontrar_sintomas_validos(sintomas_usuario, cutoff=0.70):
    """Encuentra coincidencias entre los síntomas ingresados y los del dataset."""
    sintomas_validos = []
    coincidencias = {}
    
    # Primero aplicar sinónimos
    sintomas_mapeados = []
    for s in sintomas_usuario:
        encontrado = False
        for clave, valor in sinonimos_personalizados.items():
            if clave in s:
                sintomas_mapeados.append(valor)
                coincidencias[s] = valor
                encontrado = True
                break
        if not encontrado:
            sintomas_mapeados.append(s.strip())
    
    # Luego buscar coincidencias con difflib
    for s in sintomas_mapeados:
        match = difflib.get_close_matches(s, symptom_cols, n=1, cutoff=cutoff)
        if match:
            sintomas_validos.append(match[0])
            coincidencias[s] = match[0]
    
    return list(set(sintomas_validos)), coincidencias

def calcular_score(enf_row, sintomas_confirmados):
    """Calcula el score de similitud entre los síntomas del usuario y una enfermedad."""
    coincidencia = sum(sintomas_confirmados.get(c) == 1 and enf_row[c] == 1 for c in symptom_cols)
    total_e = enf_row[symptom_cols].sum()
    total_u = sum(1 for c in symptom_cols if sintomas_confirmados.get(c) == 1)
    
    if total_e == 0 or total_u == 0 or coincidencia == 0:
        return 0, 0, 0
    
    porc_e = coincidencia / total_e
    porc_u = coincidencia / total_u
    score = round(100 * (porc_e + porc_u) / 2, 1)
    
    return score, coincidencia, total_e

# ===========================
# FUNCIONES PRINCIPALES
# ===========================
def iniciar_diagnostico(datos: DatosUsuario):
    """Inicia una nueva sesión de diagnóstico."""
    id_sesion = str(uuid.uuid4())
    imc = datos.peso / (datos.altura ** 2)
    
    # Inicializar vector de usuario
    user_vector = {s: 0 for s in symptom_cols + risk_cols}
    
    # Establecer factores de riesgo basados en IMC
    if imc < 18.5:
        user_vector['desnutricion'] = 1
    elif imc < 25:
        pass  # Peso normal
    elif imc < 30:
        user_vector['sobrepeso'] = 1
    else:
        user_vector['obesidad'] = 1
    
    # Establecer factores de riesgo basados en género
    user_vector['hombre' if datos.genero == 'M' else 'mujer'] = 1
    
    # Establecer factores de riesgo basados en edad
    if datos.edad <= 12:
        user_vector['niño'] = 1
    elif datos.edad <= 18:
        user_vector['adolescente'] = 1
    elif datos.edad <= 59:
        user_vector['adulto'] = 1
    else:
        user_vector['adulto_mayor'] = 1

    # Determinar exclusiones por género
    exclusiones = sintomas_exclusivos_hombre if datos.genero == 'M' else sintomas_exclusivos_mujer
    
    # Validar los síntomas iniciales
    sintomas_ingresados = [s.strip().lower() for s in datos.sintomas]
    sintomas_validos, _ = encontrar_sintomas_validos(sintomas_ingresados)
    sintomas_validos = [s for s in sintomas_validos if s not in exclusiones]
    
    # Agregar los síntomas al vector de usuario
    for s in sintomas_validos:
        user_vector[s] = 1
    
    # Guardar grupos confirmados
    grupos_confirmados = list(set(filter(None, [grupo_por_sintoma.get(s) for s in sintomas_validos])))
    
    # Crear la sesión
    sesion = SesionChat(
        id_sesion=id_sesion,
        datos_usuario=datos,
        sintomas_confirmados=user_vector,
        sintomas_preguntados=sintomas_validos,
        grupos_confirmados=grupos_confirmados,
        preguntas_realizadas=len(sintomas_validos)
    )
    
    sesiones[id_sesion] = sesion
    return sesion

def siguiente_pregunta(id_sesion: str):
    """Determina la siguiente pregunta a realizar en la fase actual."""
    if id_sesion not in sesiones:
        raise ValueError("Sesión no encontrada")
    
    sesion = sesiones[id_sesion]
    
    # Obtener exclusiones por género
    exclusivas = enfermedades_exclusivas_hombre if sesion.datos_usuario.genero == 'M' else enfermedades_exclusivas_mujer
    exclusiones_sintomas = sintomas_exclusivos_hombre if sesion.datos_usuario.genero == 'M' else sintomas_exclusivos_mujer
    
    # Comprobar que tenemos síntomas confirmados
    sintomas_confirmados = [s for s, v in sesion.sintomas_confirmados.items() if v == 1 and s in symptom_cols]
    if not sintomas_confirmados:
        raise ValueError("No hay síntomas confirmados")
    
    # Filtrar enfermedades basadas en síntomas confirmados
    filtro = np.logical_or.reduce([df[s] == 1 for s in sintomas_confirmados])
    enfermedades_posibles = df[filtro].copy()
    enfermedades_posibles = enfermedades_posibles[~enfermedades_posibles['nombre_de_la_enfermedad'].isin(exclusivas)]
    
    # FASE 1: Preguntas guiadas por síntomas comunes
    if sesion.fase == 1 and sesion.preguntas_realizadas < 15:
        sintomas_frecuentes = enfermedades_posibles[symptom_cols].sum().sort_values(ascending=False).index.tolist()
        for sintoma in sintomas_frecuentes:
            if sintoma in sesion.sintomas_preguntados or sintoma in exclusiones_sintomas:
                continue
                
            grupo = grupo_por_sintoma.get(sintoma)
            if grupo in sesion.grupos_confirmados:
                continue
                
            return SintomaPregunta(sintoma=sintoma, grupo=grupo, es_relevante=True)
        
        # Si hemos preguntado suficientes síntomas, pasar a fase 2
        sesion.fase = 2
    
    # FASE 2: Preguntas adaptativas basadas en información diagnóstica
    if sesion.fase == 2:
        # Calcular scores para las enfermedades posibles
        resultados = []
        for _, row in enfermedades_posibles.iterrows():
            score, coincidencia, total_e = calcular_score(row, sesion.sintomas_confirmados)
            if score == 0:
                continue
                
            pendientes = [s for s in symptom_cols 
                         if row[s] == 1 and sesion.sintomas_confirmados.get(s, 0) == 0
                         and s not in exclusiones_sintomas 
                         and s not in sesion.sintomas_preguntados]
            
            resultados.append({
                'nombre': row['nombre_de_la_enfermedad'],
                'score': score,
                'coincidencia': coincidencia,
                'total_enfermedad': total_e,
                'pendientes': pendientes
            })
        
        resultados = sorted(resultados, key=lambda x: x['score'], reverse=True)
        
        # Verificar si tenemos un diagnóstico confiable
        if resultados and resultados[0]['score'] >= 70 and resultados[0]['coincidencia'] / resultados[0]['total_enfermedad'] >= 0.8:
            # Diagnóstico suficientemente confiable
            raise ValueError("Diagnóstico confiable encontrado")
        
        # Generar la siguiente pregunta más informativa
        sintomas_prioritarios = {}
        for enf in resultados[:5]:  # Usar top 5 enfermedades
            for sintoma in enf['pendientes']:
                grupo = grupo_por_sintoma.get(sintoma)
                if grupo in sesion.grupos_confirmados:
                    continue
                sintomas_prioritarios[sintoma] = sintomas_prioritarios.get(sintoma, 0) + enf['score']
        
        if sintomas_prioritarios:
            siguiente_sintoma = sorted(sintomas_prioritarios.items(), key=lambda x: -x[1])[0][0]
            return SintomaPregunta(
                sintoma=siguiente_sintoma, 
                grupo=grupo_por_sintoma.get(siguiente_sintoma),
                es_relevante=True
            )
        
        # Si no hay síntomas prioritarios, usar método de máxima información
        mejor_sintoma = None
        mejor_puntaje = 1
        for sintoma in symptom_cols:
            if sintoma in sesion.sintomas_preguntados or sintoma in exclusiones_sintomas:
                continue
                
            grupo = grupo_por_sintoma.get(sintoma)
            if grupo in sesion.grupos_confirmados:
                continue
                
            prevalencia = enfermedades_posibles[sintoma].mean()
            puntaje = abs(0.5 - prevalencia)  # Más cercano a 0.5 = mejor discriminador
            if puntaje < mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_sintoma = sintoma
                
        if mejor_sintoma:
            return SintomaPregunta(
                sintoma=mejor_sintoma, 
                grupo=grupo_por_sintoma.get(mejor_sintoma),
                es_relevante=False
            )
    
    # Si llegamos aquí, no hay más preguntas
    raise ValueError("No hay más preguntas relevantes")

def responder_pregunta(id_sesion: str, respuesta: RespuestaSintoma):
    """Registra la respuesta a una pregunta sobre un síntoma."""
    if id_sesion not in sesiones:
        raise ValueError("Sesión no encontrada")
        
    sesion = sesiones[id_sesion]
    sintoma = respuesta.sintoma
    
    # Registrar la respuesta
    sesion.sintomas_confirmados[sintoma] = 1 if respuesta.respuesta else 0
    
    # Actualizar síntomas preguntados
    if sintoma not in sesion.sintomas_preguntados:
        sesion.sintomas_preguntados.append(sintoma)
    
    # Si la respuesta es positiva y hay un grupo, agregar a grupos confirmados
    if respuesta.respuesta:
        grupo = grupo_por_sintoma.get(sintoma)
        if grupo and grupo not in sesion.grupos_confirmados:
            sesion.grupos_confirmados.append(grupo)
            
    # Incrementar contador de preguntas
    sesion.preguntas_realizadas += 1
    sesion.preguntas_desde_ultima_confirmacion += 1
    
    # Si hemos hecho suficientes preguntas, cambiar a fase 2
    if sesion.fase == 1 and sesion.preguntas_realizadas >= 15:
        sesion.fase = 2
        
    return {"status": "success", "mensaje": "Respuesta registrada"}

def obtener_diagnostico(id_sesion: str):
    """Genera un diagnóstico basado en los síntomas confirmados por el usuario."""
    if id_sesion not in sesiones:
        raise ValueError("Sesión no encontrada")
        
    sesion = sesiones[id_sesion]
    
    # Verificar cantidad mínima de preguntas
    if sesion.preguntas_realizadas < 5:
        return ResultadoDiagnostico(
            mensaje="Se necesitan más preguntas para un diagnóstico adecuado",
            preguntas_realizadas=sesion.preguntas_realizadas
        )
    
    # Filtrar exclusiones por género
    exclusivas = enfermedades_exclusivas_hombre if sesion.datos_usuario.genero == 'M' else enfermedades_exclusivas_mujer
    
    # Calcular diagnóstico para cada enfermedad
    resultado = []
    sintomas_confirmados_count = 0
    
    for _, row in df.iterrows():
        if row['nombre_de_la_enfermedad'] in exclusivas:
            continue
            
        score, coincidencia, total_e = calcular_score(row, sesion.sintomas_confirmados)
        
        if coincidencia == 0:
            continue
            
        sintomas_confirmados_count = max(sintomas_confirmados_count, 
                                         sum(1 for s in symptom_cols if sesion.sintomas_confirmados.get(s) == 1))
        
        resultado.append({
            'nombre': row['nombre_de_la_enfermedad'],
            'coincidencia': int(coincidencia),
            'total_enfermedad': int(total_e),
            'score': score,
            'descripcion': row['breve_descripción'],
            'tratamiento': row['tratamiento']
        })
    
    # Ordenar por score y obtener los top 3
    resultado = sorted(resultado, key=lambda x: x['score'], reverse=True)[:3]
    
    # Generar mensaje según resultado
    mensaje = None
    if not resultado:
        mensaje = "No hay suficientes coincidencias para un diagnóstico confiable."
    elif resultado[0]['score'] < 60:
        mensaje = "Similitud baja con enfermedades conocidas. Consulte a un médico para un diagnóstico preciso."
    elif resultado[0]['coincidencia'] / resultado[0]['total_enfermedad'] < 0.7:
        mensaje = "Diagnóstico preliminar basado en coincidencias parciales. Consulte a un médico para confirmación."
    
    return ResultadoDiagnostico(
        enfermedades=resultado, 
        mensaje=mensaje,
        preguntas_realizadas=sesion.preguntas_realizadas,
        sintomas_confirmados=sintomas_confirmados_count
    )

def eliminar_sesion(id_sesion: str):
    """Elimina una sesión del diccionario de sesiones."""
    if id_sesion in sesiones:
        del sesiones[id_sesion]
        return {"status": "success", "mensaje": "Sesión eliminada"}
    else:
        raise ValueError("Sesión no encontrada")

# ===========================
# ENDPOINTS DE API
# ===========================
def api_iniciar_diagnostico(datos: DatosUsuario):
    try:
        sesion = iniciar_diagnostico(datos)
        return {
            "status": "success",
            "id_sesion": sesion.id_sesion,
            "sintomas_confirmados": len([s for s, v in sesion.sintomas_confirmados.items() if v == 1 and s in symptom_cols]),
            "sintomas_mapeados": {s: "confirmado" for s in sesion.sintomas_preguntados}
        }
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

def api_siguiente_pregunta(id_sesion: str):
    try:
        pregunta = siguiente_pregunta(id_sesion)
        return {
            "status": "success",
            "sintoma": pregunta.sintoma.replace("_", " "),
            "grupo": pregunta.grupo,
            "es_relevante": pregunta.es_relevante,
            "preguntas_realizadas": sesiones[id_sesion].preguntas_realizadas
        }
    except ValueError as e:
        if str(e) == "Diagnóstico confiable encontrado":
            return {
                "status": "complete",
                "mensaje": "Se ha encontrado un diagnóstico con alta confiabilidad"
            }
        return {"status": "error", "mensaje": str(e)}

def api_responder_pregunta(id_sesion: str, respuesta: RespuestaSintoma):
    try:
        resultado = responder_pregunta(id_sesion, respuesta)
        return {
            "status": "success",
            "preguntas_realizadas": sesiones[id_sesion].preguntas_realizadas
        }
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

def api_obtener_diagnostico(id_sesion: str):
    try:
        diagnostico = obtener_diagnostico(id_sesion)
        return {
            "status": "success",
            "enfermedades": diagnostico.enfermedades,
            "mensaje": diagnostico.mensaje,
            "preguntas_realizadas": diagnostico.preguntas_realizadas,
            "sintomas_confirmados": diagnostico.sintomas_confirmados
        }
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

def api_eliminar_sesion(id_sesion: str):
    try:
        resultado = eliminar_sesion(id_sesion)
        return resultado
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}