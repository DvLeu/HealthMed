# ============================
# MÓDULO NLP + RAG FINAL
# ============================

# Importamos el modelo generativo de OpenAI (vía LangChain)
from langchain_openai import ChatOpenAI

# Importamos herramientas para construir prompts personalizados
from langchain_core.prompts import ChatPromptTemplate

# Importamos el tipo Document, necesario para encapsular el contexto que se le pasa al modelo
from langchain_core.documents import Document

# Importamos el constructor de cadenas que permite unir modelo, contexto y prompt
from langchain.chains.combine_documents import create_stuff_documents_chain

# Importamos os en caso de usar variables de entorno como la API Key (no se requiere en este ejemplo)
import os

# ====================================
# CONTEXTO DEL GLOSARIO (solo texto)
# ====================================
# Se define un texto clínico confiable sobre hipertensión que será usado como base de conocimiento.
# Este contenido está embebido directamente, aunque en una implementación mayor podría leerse desde archivos externos.

hipertension_info = """
La hipertensión arterial, o presión arterial alta, es una condición en la que la fuerza de la sangre contra las paredes de las arterias es demasiado alta. Una lectura de presión arterial se expresa como dos números: la presión sistólica (el número superior) y la presión diastólica (el número inferior). Una presión normal es menor a 120/80 mmHg. La hipertensión se diagnostica generalmente cuando una persona tiene lecturas mayores a 140/90 mmHg de forma persistente.

Sistólica (primera): presión cuando el corazón late.

Diastólica (segunda): presión cuando el corazón se relaja.

Se considera hipertensión cuando la presión es igual o superior a 140/90 mmHg en dos mediciones distintas.

Síntomas comunes pueden incluir dolores de cabeza, visión borrosa, fatiga, mareos y zumbido en los oídos, aunque muchas personas no presentan síntomas (por eso se le llama el "asesino silencioso").

Factores de riesgo:
- Edad
- Obesidad
- Falta de ejercicio
- Dieta alta en sodio
- Consumo excesivo de alcohol
- Estrés crónico
- Antecedentes familiares

Complicaciones:
- Enfermedad cardíaca
- Infarto
- Accidente cerebrovascular
- Insuficiencia renal
- Pérdida de la visión

Tratamiento:
- Cambios en el estilo de vida: ejercicio, dieta DASH, menos sal, reducción de peso.
- Medicamentos: diuréticos, betabloqueantes, inhibidores de la ECA, bloqueadores de los canales de calcio, entre otros.

Hipertensión en niños y adolescentes
Se diagnostica comparando las cifras con las normales según edad, sexo y estatura. Requiere valoración pediátrica especializada.

Tipos de hipertensión
Primaria o esencial: Sin causa específica, se desarrolla con la edad.

Secundaria: Causada por enfermedades (renales, endocrinas) o medicamentos.

Gestacional: Aparece después de la semana 20 del embarazo.

Preeclampsia/Eclampsia: Hipertensión severa en el embarazo, puede causar daño multiorgánico.

Hipertensión en el embarazo
Tipos:

Hipertensión gestacional

Hipertensión crónica

Preeclampsia / Eclampsia / Síndrome HELLP

Posibles complicaciones:

Parto prematuro, bajo peso, desprendimiento de placenta, daño hepático o renal, convulsiones.

Control:

Monitoreo constante, cambios en la actividad física, medicación bajo supervisión, posible parto inducido.

Preguntas frecuentes:
- ¿La hipertensión es curable?
No, pero sí se puede controlar eficazmente con medicamentos y hábitos saludables.

- ¿Qué alimentos son buenos para la hipertensión?
Frutas, verduras, granos integrales, pescado, legumbres y productos bajos en sodio.

- ¿Cuándo debo consultar al médico?
Cuando tengas lecturas persistentes por arriba de 140/90 mmHg o presentes síntomas como dolor en el pecho, visión borrosa o mareos fuertes.

- recomendaciones
Controlar la presión al menos cada 2 años desde los 18 años.

En mayores de 40 años o personas con factores de riesgo: control anual o más frecuente.

Uso responsable de tensiómetros públicos (verificar tamaño del brazalete y postura correcta).

# Guía Resumida para Pacientes con Hipertensión Arterial

---

## Complicaciones de la Hipertensión Arterial mal tratada

- Ataque al corazón
- Embolia cerebral
- Problemas renales
- Problemas oculares
- Muerte

---

## Objetivos del Tratamiento

- **Presión arterial meta**:
  - General: < 140/90 mmHg
  - Personas con diabetes: < 130/85 mmHg
- **Colesterol total**: < 200 mg/dl
- **IMC**: < 25 kg/m²
- **Sodio**: < 2400 mg/día
- **Alcohol**: < 30 ml/día (la mitad en mujeres y hombres bajos)
- **Evitar completamente el tabaco**

---

## Tratamiento

### No farmacológico (Etapas 1 y 2)

- Alimentación saludable
- Reducción de sal
- Control de peso y colesterol
- Actividad física constante
- Evitar fumar y consumir alcohol

### Farmacológico

- Individualizado por el médico
- Considera efectos secundarios, interacciones y otras enfermedades
- **No automedicarse**

---

## Intervención médica según nivel de presión arterial

| Clasificación | Sistólica / Diastólica (mmHg) | Acción                                                 |
| ------------- | ----------------------------- | ------------------------------------------------------ |
| Óptima        | <120 / <80                    | Promoción de estilos saludables, detección cada 3 años |
| Normal        | 121-129 / 81-84               | Igual que anterior                                     |
| Fronteriza    | 130-139 / 85-89               | Estilos saludables, detección semestral                |
| Etapa 1       | 140-159 / 90-99               | Confirmación diagnóstica                               |
| Etapa 2       | 160-179 / 100-109             | Tratamiento integral                                   |
| Etapa 3       | >180 / >110                   | Tratamiento urgente                                    |

---

## Apoyo Emocional y Psicosocial

### Etapas del duelo

1. Negación
2. Enojo
3. Negociación
4. Depresión
5. Aceptación

### Recomendaciones

- Expresar emociones
- Fortalecer autoestima
- Buscar apoyo profesional si persisten síntomas > 6 meses
- Participar activamente en el tratamiento

---

## Alimentación Correcta

### Plato del Bien Comer

- Grupo 1: Frutas y verduras (ricos en potasio, fibra, antioxidantes)
- Grupo 2: Cereales, leguminosas y tubérculos (energía y proteínas)
- Grupo 3: Alimentos de origen animal (proteínas, moderar grasas)
- Grupo 4: Grasas y azúcares (restringir, preferir grasas vegetales)

### Potasio y Presión Arterial

- Consumir frutas y verduras ricas en potasio
- Ejemplos: plátano, melón, jitomate, acelgas, espinacas

### Sal y Sodio

- Reducir a < 6 g de sal/día (2.4 g de sodio)
- Leer etiquetas, evitar alimentos procesados
- Usar especias, ajo y cebolla en polvo

---

## Control del Peso y Colesterol

- Bajar de peso de forma gradual
- Comer más frutas, verduras, cereales integrales y lácteos bajos en grasa
- Limitar grasas saturadas, trans y colesterol

---

## Consumo de Alcohol y Tabaquismo

### Alcohol:

- Evitar o moderar
- No más de 30 ml al día (hombres), 15 ml (mujeres o talla baja)

### Tabaco:

- Dejar de fumar por completo
- Buscar apoyo y seguir estrategias para dejar el hábito

---

## Actividad Física

- Ejercicio aeróbico: caminar, bailar, nadar
- 30-45 minutos, 5 días por semana
- Comenzar gradualmente
- Evitar ejercicios anaeróbicos si hay hipertensión severa

---

## Recomendaciones Finales

- Conozca su condición
- Comparta información con su familia
- Acuda a sus citas médicas
- Ayude a otros pacientes
- Cuide su estado emocional y pida ayuda cuando la necesite

"""
# Se convierte el glosario a una lista de documentos (estructura requerida por LangChain)
document_chunks = [Document(page_content=hipertension_info)]

# ==========================================
# PROMPT PERSONALIZADO PARA EL MODELO
# ==========================================
# Se define cómo debe responder el asistente médico (tono, restricciones y contexto obligatorio)
prompt = ChatPromptTemplate.from_template("""
Eres un asistente médico especializado en hipertensión. Responde de forma clara, confiable y breve,
utilizando el contexto proporcionado. Si una pregunta no está relacionada con la hipertensión,
indícalo amablemente. Si el usuario plantea temas sensibles fuera del ámbito médico, responde con cortesía y evita abordarlos.

Contexto:
{context}

Pregunta:
{input}
""")

# ==========================================
# INSTANCIA DEL MODELO (GPT-4o-mini)
# ==========================================
# Se inicializa el modelo GPT con acceso vía LangChain (requiere configuración previa de API Key si se usa en producción)
llm = ChatOpenAI(model="gpt-4o-mini")

# ==========================================
# SE CREA LA CADENA COMPLETA DE RESPUESTA
# ==========================================
# Se une el modelo, el contexto (documentos) y el prompt en una cadena funcional
# Esta cadena será capaz de recibir preguntas y devolver respuestas condicionadas por el contenido del glosario

chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# ==========================================
# CONSULTA DE USUARIO Y RESPUESTA DEL LLM
# ==========================================
# Se simula una pregunta médica típica que el paciente podría hacer sobre su diagnóstico
respuesta = chain.invoke({
    "input": "¿Puedo dejar de tomar medicamento si ya me siento bien?",
    "context": document_chunks
})

# Se imprime en pantalla la respuesta generada por el modelo, basada en el contexto médico proporcionado
print(respuesta)
