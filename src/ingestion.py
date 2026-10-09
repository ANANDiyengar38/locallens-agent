import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pypdf import PdfReader

# Directory Configurations
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
QDRANT_STORAGE_PATH = BASE_DIR / "qdrant_db"
COLLECTION_NAME = "local_research_collection"

def load_documents_from_folder(folder_path: Path) -> List[Document]:
    """Loads all .txt, .md, and .pdf documents from the target folder."""
    docs = []
    if not folder_path.exists():
        print(f"Directory {folder_path} does not exist.")
        return docs

    for file_path in folder_path.glob("**/*"):
        if file_path.suffix.lower() in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": file_path.name}))
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

        elif file_path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(file_path)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": file_path.name}))
            except Exception as e:
                print(f"Error reading PDF {file_path.name}: {e}")

    return docs


def get_embedding_model():
    """Initializes local Ollama embedding model (768 dimensions)."""
    return OllamaEmbeddings(model="nomic-embed-text")


def build_or_update_vector_db():
    """Extracts, chunks, and indexes all documents into local embedded Qdrant."""
    print(" [1/3] Loading documents from:", DOCS_DIR)
    raw_docs = load_documents_from_folder(DOCS_DIR)
    if not raw_docs:
        print("No documents found to ingest!")
        return None

    print(f" Found {len(raw_docs)} document(s). Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)
    print(f" [2/3] Created {len(chunks)} text chunks.")

    print(" [3/3] Embedding & writing to local Qdrant database...")
    embeddings = get_embedding_model()

    # Use local disk-backed Qdrant client (Zero Docker required)
    client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
    
    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    vector_store.add_documents(chunks)
    print(" Ingestion complete! Local vector store updated at:", QDRANT_STORAGE_PATH)
    return vector_store


def get_local_retriever():
    """Returns a retriever instance for querying the indexed docs."""
    client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
    embeddings = get_embedding_model()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    build_or_update_vector_db()

