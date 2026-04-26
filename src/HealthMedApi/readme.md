# HealthMed — API de Diagnóstico Médico

Proyecto académico desarrollado en el Instituto Tecnológico de Veracruz (ene–may 2024).

Es una API REST que recibe síntomas de un paciente y devuelve un listado de posibles enfermedades ordenadas por similitud. No es para uso médico real.

---

## Cómo funciona

El usuario manda sus datos básicos (edad, género, peso, altura) y sus síntomas iniciales. La API abre una sesión y empieza a hacer preguntas de sí/no sobre síntomas adicionales. Al final devuelve las 3 enfermedades más probables con su score y tratamiento sugerido.

El proceso tiene dos fases:
- **Fase 1**: pregunta los síntomas más comunes entre las enfermedades candidatas
- **Fase 2**: recalcula scores y pregunta los síntomas que más ayudan a diferenciar entre las opciones top

El scoring combina qué tan bien coinciden los síntomas del usuario con los de cada enfermedad, desde ambas direcciones (cobertura de la enfermedad y cobertura del usuario).

---

## Endpoints

```
POST   /iniciar-diagnostico/         Inicia sesión con datos del paciente
GET    /siguiente-pregunta/{id}      Obtiene el siguiente síntoma a preguntar
POST   /responder-pregunta/{id}      Manda la respuesta (sí/no)
GET    /obtener-diagnostico/{id}     Obtiene el diagnóstico final
DELETE /eliminar-sesion/{id}         Cierra la sesión
```

La documentación interactiva está en `/docs` cuando el servidor está corriendo.

---

## Estructura del proyecto

```
src/HealthMedApi/src/
├── main.py           # App de FastAPI y rutas
├── chatbot.py        # Lógica de diagnóstico y modelos Pydantic
├── data/
│   └── Dataset_Enfermedades_Final.csv
└── requirements.txt

Modelo_diagnostico/   # Scripts de métricas y experimentos con KNN
Modelo_NLP_ROG/       # Prototipo con NLP e interfaz web
```

---

## Correr el proyecto

```bash
git clone https://github.com/DvLeu/HealthMed.git
cd HealthMed/src/HealthMedApi

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r src/requirements.txt

uvicorn src.main:app --reload
```

Abrir en el navegador: http://localhost:8000/docs

---

## Ejemplo de uso

```bash
curl -X POST "http://localhost:8000/iniciar-diagnostico/" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 28,
    "genero": "M",
    "peso": 75,
    "altura": 1.75,
    "sintomas": ["fiebre", "tos", "dolor de cabeza"]
  }'
```

---

## Stack

- FastAPI + Pydantic v2
- Pandas y NumPy para el procesamiento del dataset
- difflib para el mapeo aproximado de síntomas
- Uvicorn como servidor
