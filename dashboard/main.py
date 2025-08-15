# app.py (FastAPI backend with integrated chunking + Chroma ingest)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import shutil
import json
import uuid
from typing import List
import os
import uvicorn
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============================================================================
# Settings
# =============================================================================
# Use resolve() to get clean absolute paths without ../ references
DOCUMENTS_DIR = Path("../documents").resolve()
CHROMA_DIR = Path("../.chroma").resolve()
CHROMA_COLLECTION = "vipo_bank_policies"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vipo Knowledge Base Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create static directory if it doesn't exist
Path("static").mkdir(exist_ok=True)

# Add route for favicon.ico
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

# =============================================================================
# Utility functions
# =============================================================================
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def load_pdf(path: Path):
    """Load a single PDF with fallback loader."""
    try:
        docs = PyPDFLoader(str(path)).load()
    except Exception as e:
        print(f"PyPDF failed for {path.name}: {e}")
        docs = UnstructuredPDFLoader(str(path)).load()
    for d in docs:
        d.metadata.update({"source": path.name, "path": str(path)})
    return docs

def get_chunk_count_for_file(filename: str) -> int:
    """Get the number of chunks for a specific file from ChromaDB"""
    try:
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed,
        )
        
        # Get all metadata and count matching documents manually
        # This is a workaround for ChromaDB versions that don't support where in count()
        try:
            # First try using get() with where parameter to get matching documents
            results = vectordb._collection.get(where={"doc_name": filename})
            if results and "ids" in results:
                return len(results["ids"])
        except Exception:
            # Fallback: Get all documents and count manually
            try:
                all_results = vectordb._collection.get()
                if all_results and "metadatas" in all_results and all_results["metadatas"]:
                    # Count documents where doc_name matches filename
                    count = sum(1 for meta in all_results["metadatas"] if meta and meta.get("doc_name") == filename)
                    return count
            except Exception:
                pass
        
        return 0
    except Exception as e:
        print(f"Warning: Could not get chunk count for {filename}: {e}")
        return 0

def ingest_files_to_chroma(file_paths: List[Path]):
    """Load, chunk, and store the given files into Chroma."""
    all_docs = []
    for path in file_paths:
        if path.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(path))
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            from langchain.schema import Document
            all_docs.append(Document(page_content=text, metadata={"source": path.name, "path": str(path)}))

    if not all_docs:
        return 0, 0

    chunks = chunk_documents(all_docs)

    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embed,
    )

    now_iso = datetime.now().isoformat()
    texts, metas, json_records = [], [], []
    prev_chunk_ids = {}

    for c in chunks:
        doc_name = c.metadata.get("source") or "unknown"
        prev_id = prev_chunk_ids.get(doc_name)
        current_id = str(uuid.uuid4())

        enriched_meta = {
            "content": c.page_content,
            "date": now_iso,
            "current_chunk_id": current_id,
            "previous_chunk_id": prev_id,
            "access": "PUBLIC",
            "doc_name": doc_name,
            "source": "PDF" if c.metadata.get("path", "").endswith(".pdf") else "TXT",
        }
        texts.append(c.page_content)
        metas.append(enriched_meta)
        json_records.append(enriched_meta)
        prev_chunk_ids[doc_name] = current_id

    vectordb.add_texts(texts=texts, metadatas=metas)

    # Update chunks.json manifest
    chunks_path = DOCUMENTS_DIR / "chunks.json"
    if chunks_path.exists():
        try:
            old_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        except:
            old_data = []
        old_data.extend(json_records)
        json_records = old_data
    chunks_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(file_paths), len(chunks)


# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and directly ingest files into Chroma."""
    try:
        saved_paths = []
        for file in files:
            if not file.filename.lower().endswith((".pdf", ".txt")):
                continue
            dest = DOCUMENTS_DIR / file.filename
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(dest)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid PDF/TXT files uploaded")

        file_count, chunk_count = ingest_files_to_chroma(saved_paths)

        return {
            "message": f"Uploaded and ingested {file_count} file(s) into vector store",
            "uploaded_files": [path.name for path in saved_paths],
            "chunks_added": chunk_count,
            "collection": CHROMA_COLLECTION
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/ingest error: {e}")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/stats")
async def get_stats():
    """Get document and vector database statistics"""
    try:
        # Count documents
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        txt_files = list(DOCUMENTS_DIR.glob("*.txt"))
        total_docs = len(pdf_files) + len(txt_files)
        
        # Calculate vector DB size and print files in the chromadb
        vector_db_size = 0
        if CHROMA_DIR.exists():
            for item in CHROMA_DIR.rglob("*"):
                if item.is_file():
                    vector_db_size += item.stat().st_size
        
        # Get REAL-TIME chunk count from ChromaDB (not from JSON file)
        total_chunks = 0
        try:
            embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            vectordb = Chroma(
                collection_name=CHROMA_COLLECTION,
                persist_directory=str(CHROMA_DIR),
                embedding_function=embed,
            )
            # Get actual count from ChromaDB
            total_chunks = vectordb._collection.count()
        except Exception as e:
            print(f"Warning: Could not get real-time chunk count: {e}")
            # Fallback to JSON file count
            chunks_file = DOCUMENTS_DIR / "chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                        total_chunks = len(chunks_data)
                except:
                    pass
        
        return {
            "total_documents": total_docs,
            "pdf_files": len(pdf_files),
            "txt_files": len(txt_files),
            "vector_db_size_mb": round(vector_db_size / (1024 * 1024), 2),
            "total_chunks": total_chunks,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """List all documents in the documents folder"""
    try:
        documents = []
        
        # PDF files
        for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
            stat = pdf_file.stat()
            documents.append({
                "name": pdf_file.name,
                "type": "PDF",
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # TXT files
        for txt_file in DOCUMENTS_DIR.glob("*.txt"):
            stat = txt_file.stat()
            documents.append({
                "name": txt_file.name,
                "type": "TXT",
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and remove its chunks from vector database"""
    try:
        file_path = DOCUMENTS_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get vector database
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed,
        )
        
        # Get chunk count before deletion for reporting
        chunks_before = get_chunk_count_for_file(filename)
        
        # Remove chunks associated with this file from ChromaDB
        try:
            # First try to get IDs of documents with matching doc_name
            results = vectordb._collection.get(where={"doc_name": filename})
            if results and "ids" in results and results["ids"]:
                # Delete by IDs instead of using where parameter
                vectordb._collection.delete(ids=results["ids"])
                print(f"Removed {len(results['ids'])} chunks for {filename} from ChromaDB")
                chunks_before = len(results["ids"])
                
                # Force persist to ensure changes are saved
                vectordb.persist()

                print(f"Persisted changes to ChromaDB")
            else:
                print(f"No chunks found for {filename} in ChromaDB")
        except Exception as e:
            # Fallback: Try to get all documents and filter manually
            try:
                all_results = vectordb._collection.get()
                if all_results and "ids" in all_results and "metadatas" in all_results:
                    # Find IDs where doc_name matches filename
                    ids_to_delete = [
                        all_results["ids"][i] for i, meta in enumerate(all_results["metadatas"])
                        if meta and meta.get("doc_name") == filename
                    ]
                    if ids_to_delete:
                        vectordb._collection.delete(ids=ids_to_delete)
                        print(f"Removed {len(ids_to_delete)} chunks for {filename} from ChromaDB (fallback method)")
                        chunks_before = len(ids_to_delete)
                        
                        # Force persist to ensure changes are saved
                        vectordb.persist()
                        print(f"Persisted changes to ChromaDB (fallback method)")
                    else:
                        print(f"No chunks found for {filename} in ChromaDB (fallback method)")
                else:
                    print(f"No documents found in ChromaDB")
            except Exception as e2:
                print(f"Warning: Could not remove chunks from ChromaDB: {e} / {e2}")
                chunks_before = 0
        
        # Delete the physical file
        file_path.unlink()
        
        # Update chunks.json manifest by removing entries for this file
        chunks_file = DOCUMENTS_DIR / "chunks.json"
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                # Filter out chunks for the deleted file
                filtered_chunks = [chunk for chunk in chunks_data if chunk.get("doc_name") != filename]
                
                # Write back the filtered data
                chunks_file.write_text(
                    json.dumps(filtered_chunks, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                print(f"Updated chunks.json - removed entries for {filename}")
                
            except Exception as e:
                print(f"Warning: Could not update chunks.json: {e}")
        
        # Try to clean up and reclaim space
        try:
            # Force a collection update to potentially reclaim space
            vectordb._collection.persist()
            print(f"Collection persisted after deletion")
        except Exception as e:
            print(f"Warning: Could not persist collection: {e}")
        
        return {
            "message": f"Successfully deleted {filename} and removed {chunks_before} chunks from vector database",
            "deleted_file": filename,
            "chunks_removed": chunks_before
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/api/reprocess")
async def reprocess_documents():
    """Manually reprocess all documents in the documents folder"""
    try:
        all_files = list(DOCUMENTS_DIR.glob("*.pdf")) + list(DOCUMENTS_DIR.glob("*.txt"))
        if not all_files:
            return {"message": "No documents found to process", "chunks_processed": 0}
        
        file_count, chunk_count = ingest_files_to_chroma(all_files)
        return {
            "message": f"Successfully reprocessed {chunk_count} chunks from {chunk_count} documents",
            "chunks_processed": chunk_count,
            "documents_processed": file_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocessing error: {str(e)}")



if __name__ == "__main__":
    print("🚀 Starting Vipo Knowledge Base Dashboard...")
    print(f"📁 Documents directory: {DOCUMENTS_DIR}")
    print(f"🗄️  ChromaDB directory: {CHROMA_DIR}")
    print("🌐 Dashboard available at: http://localhost:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090, reload=False)
