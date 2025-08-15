from pathlib import Path
from datetime import datetime
import uuid
import json
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os

# Settings class definition
class Settings:
    # Local data & vector store
    docs_dir: Path = Path("./data")
    chroma_dir: Path = Path("./.chroma")
    chroma_collection: str = "vipo_bank_policies"

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Optional: Google fallback
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_llm_model: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-pro")

settings = Settings()

COLLECTION = settings.chroma_collection


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def load_pdfs(pdf_dir: str):
    base = Path(pdf_dir)
    # Gather and de-duplicate case-insensitively (Windows)
    candidates = list(base.rglob("*.pdf")) + list(base.rglob("*.PDF"))
    pdfs = []
    seen = set()
    for p in candidates:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            pdfs.append(p)
    print(f"Found {len(pdfs)} PDF files")
    for p in pdfs:
        # Try PyPDF first
        try:
            docs = PyPDFLoader(str(p)).load()
            for d in docs:
                d.metadata.update({"source": p.name, "path": str(p)})
            yield from docs
            continue
        except Exception as e:
            print(f"PyPDF failed for {p.name}: {e}")
        # Fallback: Unstructured
        try:
            docs = UnstructuredPDFLoader(str(p)).load()
            for d in docs:
                d.metadata.update({"source": p.name, "path": str(p)})
            yield from docs
        except Exception as e:
            print(f"Unstructured failed for {p.name}: {e}")


def run_ingest():
    # Prefer a local ./documents directory if present, else fallback to settings.pdf_dir
    documents_dir = Path("./documents")
    src_dir = str(documents_dir if documents_dir.exists() else Path(settings.pdf_dir))

    print("Loading PDFs from:", src_dir)
    docs = list(load_pdfs(src_dir))
    if not docs:
        print("No documents extracted from PDFs in", src_dir)
        return
    print(f"Loaded {len(docs)} pages; chunking…")
    chunks = chunk_documents(docs)

    # Use HuggingFace sentence transformer
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    print("Embedding with", model_id)
    embed = HuggingFaceEmbeddings(model_name=model_id)

    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=settings.chroma_dir,
        embedding_function=embed,
    )

    # Build enriched metadata per chunk
    texts = []
    metas = []
    json_records = []
    now_iso = datetime.now().isoformat()
    previous_for_doc: dict[str, str | None] = {}

    for c in chunks:
        doc_name = c.metadata.get("source") or c.metadata.get("doc_name") or "unknown.pdf"
        prev_id = previous_for_doc.get(doc_name)
        current_id = str(uuid.uuid4())

        enriched = {
            # Required schema
            "content": c.page_content,
            "date": now_iso,
            "current_chunk_id": current_id,
            "previous_chunk_id": prev_id,
            "access": "PUBLIC",
            "doc_name": doc_name,
            "source": "PDF",
        }

        texts.append(c.page_content)
        metas.append(enriched)
        json_records.append(enriched)

        # Update chain continuity per document
        previous_for_doc[doc_name] = current_id

    vectordb.add_texts(texts=texts, metadatas=metas)
    print("Ingest complete. Chroma dir:", settings.chroma_dir)

    # Write chunks JSON alongside the source documents for inspection
    try:
        out_dir = documents_dir if Path(src_dir) == documents_dir else Path(src_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks.json"
        out_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote chunk manifest:", out_path)
    except Exception as e:
        print("Warning: failed to write chunks.json:", e)


if __name__ == "__main__":
    run_ingest()

