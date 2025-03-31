# main.py
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
    cargar_dataset
)

app = FastAPI(
    title="API de Chatbot Médico",
    description="API para el diagnóstico médico basado en síntomas",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"mensaje": "Bienvenido a la API del Chatbot Médico. Visita /docs para la documentación interactiva."}

#? Cargar el dataset al iniciar
@app.on_event("startup")
async def startup_event():
    dataset_path = r"src\data\Dataset_Enfermedades_Final.csv"
    cargar_dataset(dataset_path)

@app.post("/iniciar-diagnostico/", response_model=SesionChat)
async def route_iniciar_diagnostico(datos: DatosUsuario):
    return iniciar_diagnostico(datos)

@app.get("/siguiente-pregunta/{id_sesion}", response_model=SintomaPregunta)
async def route_siguiente_pregunta(id_sesion: str):
    return siguiente_pregunta(id_sesion)

@app.post("/responder-pregunta/{id_sesion}")
async def route_responder_pregunta(id_sesion: str, respuesta: RespuestaSintoma):
    return responder_pregunta(id_sesion, respuesta)

@app.get("/obtener-diagnostico/{id_sesion}", response_model=ResultadoDiagnostico)
async def route_obtener_diagnostico(id_sesion: str):
    return obtener_diagnostico(id_sesion)

@app.delete("/eliminar-sesion/{id_sesion}")
async def route_eliminar_sesion(id_sesion: str):
    return eliminar_sesion(id_sesion)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
