# HealthMed Chat – Asistente Médico Especializado en Hipertensión

**HealthMed Chat** es una aplicación web interactiva que permite a los usuarios realizar preguntas sobre hipertensión arterial y recibir respuestas automáticas y confiables, generadas por un modelo de lenguaje respaldado por un sistema NLP con RAG (Retrieval-Augmented Generation).

## Características

- Interfaz web amigable y adaptable.
- Asistente conversacional entrenado con información clínica verificada sobre hipertensión.
- Procesamiento y generación de respuestas mediante LangChain y GPT (modelo `gpt-4o-mini`).
- Contexto médico embebido para asegurar precisión y relevancia en las respuestas.
- Renderizado de respuestas con formato (Markdown → HTML).
- Despliegue listo para conexión con backend vía API.

## Estructura del Proyecto

```
├── index.html          # Estructura HTML de la interfaz
├── style.css           # Estilos visuales del chat
├── script.js           # Lógica del cliente y conexión con backend
├── NLP_ROG_comentado.py# Backend: NLP + RAG con LangChain y contexto médico
```

## Requisitos

### Frontend
Solo necesitas un navegador moderno. El HTML se conecta automáticamente al backend mediante `fetch`.

### Backend (Python)

- Python 3.10+
- LangChain
- OpenAI (modelo `gpt-4o-mini`)
- FastAPI o cualquier servidor que reciba las preguntas en `/preguntar` (no incluido en este repositorio)

Instalación de dependencias sugeridas:

```bash
pip install langchain langchain-openai
```

## Cómo Funciona

1. El usuario escribe una pregunta en la interfaz.
2. Se envía a la API vía `fetch` POST.
3. El backend ejecuta `NLP_ROG_comentado.py`:
   - Inyecta el contexto médico (sobre hipertensión).
   - Procesa la pregunta con el modelo GPT y LangChain.
   - Devuelve una respuesta breve y precisa.
4. La respuesta se muestra con formato en la interfaz web.

## Ejemplo de uso

> Pregunta: ¿Puedo dejar de tomar medicamento si ya me siento bien?

> Respuesta del asistente: No debes suspender el tratamiento sin consultar a tu médico. Aunque te sientas bien, la hipertensión puede seguir afectando tu salud sin síntomas evidentes.

## Notas de Seguridad

- No es un sistema de diagnóstico médico.
- No almacena datos personales.
- No debe usarse en situaciones de emergencia médica.

## Autoría y Licencia

Proyecto académico para demostración de capacidades NLP + RAG.  
Puedes adaptar este sistema para otras condiciones médicas o expandir el glosario.  
Licencia MIT o Creative Commons según tu preferencia.