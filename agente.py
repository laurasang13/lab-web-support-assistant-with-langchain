"""
agente.py — Agente de soporte con LangGraph.
Combina RAG (ChromaDB), tools (pedidos/reembolsos) y memoria por sesión.
"""

import json
import operator
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage


# ── Estado del grafo ──────────────────────────────────────────────────────────

class EstadoSoporte(TypedDict):
    # operator.add hace que cada nodo AÑADA mensajes al historial en vez de reemplazarlo
    mensajes: Annotated[Sequence[BaseMessage], operator.add]


# ── Herramientas (Tools) ──────────────────────────────────────────────────────

@tool
def buscar_pedido(pedido_id: str) -> str:
    """Busca el estado de un pedido por su ID. Úsala siempre que el usuario
    mencione un número de pedido. Ejemplo: buscar_pedido('PED-1234')"""
    pedidos = {
        "PED-1234": {
            "estado": "enviado",
            "fecha_entrega": "15/06/2026",
            "total": 89.99,
            "articulos": ["Auriculares BT Pro", "Funda protectora"],
        },
        "PED-5678": {
            "estado": "en preparación",
            "fecha_entrega": "18/06/2026",
            "total": 45.50,
            "articulos": ["Teclado mecánico TKL"],
        },
        "PED-9999": {
            "estado": "entregado",
            "fecha_entrega": "01/06/2026",
            "total": 129.00,
            "articulos": ["Monitor 24\" Full HD"],
        },
    }
    pedido = pedidos.get(pedido_id.upper().strip())
    if pedido:
        return (
            f"Pedido {pedido_id.upper()}:\n"
            f"  Estado: {pedido['estado']}\n"
            f"  Entrega estimada: {pedido['fecha_entrega']}\n"
            f"  Total: {pedido['total']}€\n"
            f"  Artículos: {', '.join(pedido['articulos'])}"
        )
    return f"No se encontró el pedido '{pedido_id}'. Verifica que el ID sea correcto (formato PED-XXXX)."


@tool
def calcular_reembolso(total: float, porcentaje: float) -> str:
    """Calcula el importe de un reembolso parcial dado el total del pedido
    y el porcentaje a reembolsar. Ejemplo: calcular_reembolso(89.99, 50)"""
    if not (0 < porcentaje <= 100):
        return "El porcentaje debe estar entre 1 y 100."
    reembolso = round(total * porcentaje / 100, 2)
    return (
        f"Cálculo de reembolso:\n"
        f"  Total del pedido: {total}€\n"
        f"  Porcentaje: {porcentaje}%\n"
        f"  Importe a reembolsar: {reembolso}€"
    )


@tool
def escalar_a_humano(motivo: str) -> str:
    """Escala el caso a un agente humano cuando el problema es complejo o
    el cliente lo solicita explícitamente. Registra el motivo en un archivo."""
    import datetime
    caso = {
        "timestamp": datetime.datetime.now().isoformat(),
        "motivo": motivo,
    }
    with open("casos_escalados.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(caso, ensure_ascii=False) + "\n")
    return (
        f"Caso escalado correctamente. Un agente humano revisará tu caso "
        f"en un plazo máximo de 48 horas. Motivo registrado: '{motivo}'."
    )


# ── RAG: retriever sobre ChromaDB ─────────────────────────────────────────────

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ── Modelo y tools ────────────────────────────────────────────────────────────

tools = [buscar_pedido, calcular_reembolso, escalar_a_humano]
modelo = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
modelo_con_tools = modelo.bind_tools(tools)

# ── Nodos del grafo ───────────────────────────────────────────────────────────

def nodo_llm(estado: EstadoSoporte) -> dict:
    """Nodo principal: recupera contexto RAG y llama al LLM."""

    # Busca el último mensaje del usuario para hacer la búsqueda RAG
    ultimo_humano = next(
        (m.content for m in reversed(estado["mensajes"]) if isinstance(m, HumanMessage)),
        ""
    )

    # Recupera las políticas relevantes de ChromaDB
    docs = retriever.invoke(ultimo_humano)
    contexto_rag = "\n\n".join(d.page_content for d in docs)

    system = SystemMessage(content=f"""Eres un asistente de soporte al cliente amable, claro y preciso \
de TiendaOnline S.L.

Tienes acceso a estas herramientas:
- buscar_pedido: para consultar el estado de cualquier pedido por su ID
- calcular_reembolso: para calcular el importe de un reembolso parcial
- escalar_a_humano: cuando el problema requiere intervención humana

Usa las herramientas siempre que sea necesario. Si el usuario menciona un número \
de pedido, úsala aunque no lo pida explícitamente.

Responde preguntas sobre políticas usando el siguiente contexto extraído de \
nuestra base de conocimiento:

---
{contexto_rag}
---

Normas importantes:
- Si no tienes información suficiente, dilo claramente y ofrece escalar el caso.
- No inventes datos de pedidos, fechas ni importes.
- Responde siempre en español.
- Sé conciso pero completo.""")

    mensajes_completos = [system] + list(estado["mensajes"])
    respuesta = modelo_con_tools.invoke(mensajes_completos)
    return {"mensajes": [respuesta]}


def debe_continuar(estado: EstadoSoporte) -> str:
    """Decide si hay que ejecutar tools o finalizar."""
    ultimo = estado["mensajes"][-1]
    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "usar_tool"
    return END


# ── Construcción del grafo ────────────────────────────────────────────────────

def nodo_tools(estado: EstadoSoporte) -> dict:
    ultimo = estado["mensajes"][-1]
    resultados = []
    for tool_call in ultimo.tool_calls:
        tool_fn = {t.name: t for t in tools}[tool_call["name"]]
        resultado = tool_fn.invoke(tool_call["args"])
        resultados.append(ToolMessage(
            content=str(resultado),
            tool_call_id=tool_call["id"]
        ))
    return {"mensajes": resultados}

grafo = StateGraph(EstadoSoporte)
grafo.add_node("llm", nodo_llm)
grafo.add_node("tools", nodo_tools)
grafo.set_entry_point("llm")
grafo.add_conditional_edges(
    "llm",
    debe_continuar,
    {"usar_tool": "tools", END: END}
)
grafo.add_edge("tools", "llm")  # Tras ejecutar tool, vuelve al LLM

checkpointer = MemorySaver()
agente = grafo.compile(checkpointer=checkpointer)
