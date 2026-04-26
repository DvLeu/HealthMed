# main.py
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.chatbot import (
    DatosUsuario,
    RespuestaSintoma,
    SesionChat,
    SintomaPregunta,
    ResultadoDiagnostico,
    iniciar_diagnostico,
    siguiente_pregunta,
    responder_pregunta,
    obtener_diagnostico,
    eliminar_sesion,
    cargar_dataset,
)

DATASET_PATH = Path(__file__).parent / "data" / "Dataset_Enfermedades_Final.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_dataset(DATASET_PATH)
    yield


app = FastAPI(
    title="HealthMed — Medical Diagnosis API",
    description=(
        "Chatbot-style API that infers probable diagnoses from patient symptoms "
        "using rule-based filtering and KNN-inspired scoring. "
        "Visit /docs for interactive documentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
async def root():
    return {"message": "HealthMed API is running. Visit /docs for documentation."}


@app.post("/iniciar-diagnostico/", response_model=SesionChat, tags=["diagnostico"])
async def route_iniciar_diagnostico(datos: DatosUsuario):
    try:
        return iniciar_diagnostico(datos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/siguiente-pregunta/{id_sesion}", response_model=SintomaPregunta, tags=["diagnostico"])
async def route_siguiente_pregunta(id_sesion: str):
    try:
        return siguiente_pregunta(id_sesion)
    except ValueError as e:
        msg = str(e)
        if msg == "Sesión no encontrada":
            raise HTTPException(status_code=404, detail=msg)
        if msg in ("Diagnóstico confiable encontrado", "No hay más preguntas relevantes"):
            raise HTTPException(status_code=200, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@app.post("/responder-pregunta/{id_sesion}", tags=["diagnostico"])
async def route_responder_pregunta(id_sesion: str, respuesta: RespuestaSintoma):
    try:
        return responder_pregunta(id_sesion, respuesta)
    except ValueError as e:
        msg = str(e)
        if msg == "Sesión no encontrada":
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@app.get("/obtener-diagnostico/{id_sesion}", response_model=ResultadoDiagnostico, tags=["diagnostico"])
async def route_obtener_diagnostico(id_sesion: str):
    try:
        return obtener_diagnostico(id_sesion)
    except ValueError as e:
        msg = str(e)
        if msg == "Sesión no encontrada":
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@app.delete("/eliminar-sesion/{id_sesion}", tags=["diagnostico"])
async def route_eliminar_sesion(id_sesion: str):
    try:
        return eliminar_sesion(id_sesion)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
