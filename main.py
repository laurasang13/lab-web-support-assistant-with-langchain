"""
main.py — API FastAPI para el asistente de soporte.
Arrancar con: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

from agente import agente

app = FastAPI(
    title="Asistente de Soporte TiendaOnline",
    description="API de soporte al cliente con RAG, tools y memoria por sesión.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos de request/response ───────────────────────────────────────────────

class MensajeRequest(BaseModel):
    session_id: str
    mensaje: str

class MensajeResponse(BaseModel):
    session_id: str
    respuesta: str

class MensajeHistorial(BaseModel):
    rol: str       # "usuario" o "asistente"
    contenido: str

class HistorialResponse(BaseModel):
    session_id: str
    mensajes: list[MensajeHistorial]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=MensajeResponse)
def chat(request: MensajeRequest):
    """
    Envía un mensaje al asistente.
    El session_id mantiene el contexto entre turnos de la misma conversación.
    Dos session_id distintos tienen historiales completamente separados.
    """
    if not request.mensaje.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    config = {"configurable": {"thread_id": request.session_id}}

    resultado = agente.invoke(
        {"mensajes": [HumanMessage(content=request.mensaje)]},
        config=config,
    )

    # El último mensaje del estado es siempre la respuesta del asistente
    respuesta = resultado["mensajes"][-1].content
    return MensajeResponse(session_id=request.session_id, respuesta=respuesta)


@app.get("/chat/{session_id}/historial", response_model=HistorialResponse)
def obtener_historial(session_id: str):
    """
    Devuelve todos los mensajes de una sesión (bonus del lab).
    """
    config = {"configurable": {"thread_id": session_id}}
    estado = agente.get_state(config)

    if not estado or not estado.values:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró historial para la sesión '{session_id}'."
        )

    mensajes_raw = estado.values.get("mensajes", [])
    historial = []

    for m in mensajes_raw:
        if isinstance(m, HumanMessage):
            historial.append(MensajeHistorial(rol="usuario", contenido=m.content))
        elif isinstance(m, AIMessage) and m.content:
            # Filtra los mensajes intermedios que solo tienen tool_calls (sin texto)
            historial.append(MensajeHistorial(rol="asistente", contenido=m.content))

    return HistorialResponse(session_id=session_id, mensajes=historial)


@app.delete("/chat/{session_id}")
def cerrar_sesion(session_id: str):
    """
    Marca la sesión como cerrada.
    Nota: MemorySaver no permite borrado real; en producción usar PostgresCheckpointer.
    """
    return {
        "mensaje": f"Sesión '{session_id}' cerrada.",
        "nota": "Con MemorySaver el historial persiste en RAM hasta reiniciar el servidor.",
    }


@app.get("/health")
def health():
    """Comprueba que la API está levantada."""
    return {"status": "ok"}
