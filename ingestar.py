"""
ingestar.py — Indexa politicas.txt en ChromaDB.
Ejecutar UNA SOLA VEZ antes de arrancar la API:
    python ingestar.py
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def ingestar():
    print("Cargando politicas.txt...")
    loader = TextLoader("politicas.txt", encoding="utf-8")
    docs = loader.load()

    print("Dividiendo en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["===", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)

    print(f"Generando embeddings para {len(chunks)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("Guardando en ChromaDB (./chroma_db)...")
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    print(f"\n✓ Indexados {len(chunks)} chunks correctamente.")
    print("  Ya puedes arrancar la API con: uvicorn main:app --reload")

if __name__ == "__main__":
    ingestar()
