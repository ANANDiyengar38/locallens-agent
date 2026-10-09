from langchain_core.tools import tool
from duckduckgo_search import DDGS
from .ingestion import get_local_retriever

@tool
def search_private_docs(query: str) -> str:
    """
    Search internal private documents, specifications, PDFs, and team notes.
    Use this tool whenever the user asks about confidential projects, internal specs, company policies, or uploaded files.
    """
    try:
        retriever = get_local_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No matching internal documents found."
        
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown file")
            results.append(f"[{i}] Source: {source}\nContent: {doc.page_content.strip()}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching private documents: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the public web using DuckDuckGo for live news, real-time facts, public documentation, or general web information.
    Use this tool when the query requires outside knowledge or recent information not found in internal documents.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No web results found."

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                snippet = r.get("body", "No Snippet")
                link = r.get("href", "")
                formatted.append(f"[{i}] {title}\nSummary: {snippet}\nURL: {link}")
            return "\n\n".join(formatted)
    except Exception as e:
        return f"Error searching the web: {str(e)}"

