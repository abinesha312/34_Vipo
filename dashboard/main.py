
# app.py (FastAPI backend with integrated chunking + Chroma ingest + S3 storage only)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import json
import uuid
from typing import List
import os
import io
import tempfile
import uvicorn
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from botocore.exceptions import ClientError

# =============================================================================
# Settings
# =============================================================================
# Only use ChromaDB locally - all documents in S3
CHROMA_DIR = Path("./.chroma").resolve()
CHROMA_COLLECTION = "vipo_bank_policies"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# S3 settings
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "vipo-documents")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = "documents/"  # Folder prefix in S3 - must end with /

# Ensure ChromaDB directory exists
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vipo Knowledge Base Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist and use absolute path
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = Path(os.path.join(script_dir, "static"))
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Add route for favicon.ico
@app.get("/favicon.ico")
async def favicon():
    favicon_path = static_dir / "favicon.ico"
    if not favicon_path.exists():
        # Create a simple favicon if it doesn't exist
        with open(favicon_path, "wb") as f:
            f.write(b"\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x01\x00\x18\x00h\x03\x00\x00\x16\x00\x00\x00")
    return FileResponse(str(favicon_path), media_type="image/x-icon")

# =============================================================================
# S3 Client
# =============================================================================
def get_s3_client():
    """Get S3 client with credentials from environment variables, AWS CLI config, or instance profile"""
    try:
        # First try creating a client without explicit credentials
        # This will use credentials from environment variables, AWS config files, or instance profiles
        try:
            # Test if we can get caller identity
            test_client = boto3.client('sts', region_name=S3_REGION)
            test_client.get_caller_identity()
            print("Found AWS credentials from config or environment")
            return boto3.client('s3', region_name=S3_REGION)
        except Exception as e:
            print(f"Could not use existing AWS config: {e}")
        
        # Fall back to checking environment variables directly
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            print("WARNING: AWS credentials not found in environment variables.")
            print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            print("Using local file system as a fallback...")
            
            # Create a mock S3 client that uses local filesystem instead
            from unittest.mock import MagicMock
            mock_client = MagicMock()
            
            # Create documents directory if it doesn't exist
            documents_dir = Path("./documents").resolve()
            documents_dir.mkdir(parents=True, exist_ok=True)
            
            # Mock S3 operations to use local filesystem
            def mock_list_objects(**kwargs):
                bucket = kwargs.get('Bucket')
                prefix = kwargs.get('Prefix', '')
                base_dir = Path("./documents")
                files = list(base_dir.glob("*.*"))
                contents = []
                for file in files:
                    if file.name == "chunks.json":
                        continue
                    contents.append({
                        'Key': f"{prefix}{file.name}",
                        'Size': file.stat().st_size,
                        'LastModified': datetime.fromtimestamp(file.stat().st_mtime)
                    })
                return {'Contents': contents} if contents else {}
            
            def mock_get_object(**kwargs):
                key = kwargs.get('Key')
                filename = Path(key).name
                file_path = Path("./documents") / filename
                
                if not file_path.exists():
                    if filename == "chunks.json":
                        # Create empty chunks.json if it doesn't exist
                        file_path.write_text("[]")
                    else:
                        # For other files, raise NoSuchKey error
                        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'The specified key does not exist.'}}
                        raise ClientError(error_response, 'GetObject')
                
                class MockBody:
                    def read(self):
                        return file_path.read_bytes()
                
                return {'Body': MockBody()}
            
            def mock_put_object(**kwargs):
                key = kwargs.get('Key')
                body = kwargs.get('Body')
                filename = Path(key).name
                file_path = Path("./documents") / filename
                
                if isinstance(body, bytes):
                    file_path.write_bytes(body)
                else:
                    file_path.write_text(body)
                return {}
            
            def mock_download_file(bucket, key, filename):
                src_path = Path("./documents") / Path(key).name
                if src_path.exists():
                    import shutil
                    shutil.copy2(src_path, filename)
                else:
                    raise FileNotFoundError(f"File {src_path} not found")
            
            def mock_delete_object(**kwargs):
                key = kwargs.get('Key')
                filename = Path(key).name
                file_path = Path("./documents") / filename
                if file_path.exists():
                    file_path.unlink()
                return {}
            
            def mock_head_object(**kwargs):
                key = kwargs.get('Key')
                filename = Path(key).name
                file_path = Path("./documents") / filename
                if not file_path.exists():
                    error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
                    raise ClientError(error_response, 'HeadObject')
                return {}
            
            def mock_head_bucket(**kwargs):
                # Always succeed for local filesystem
                return {}
            
            # Assign mock methods
            mock_client.list_objects_v2 = mock_list_objects
            mock_client.get_object = mock_get_object
            mock_client.put_object = mock_put_object
            mock_client.download_file = mock_download_file
            mock_client.delete_object = mock_delete_object
            mock_client.head_object = mock_head_object
            mock_client.head_bucket = mock_head_bucket
            
            return mock_client
        
        # If we have credentials, use the real S3 client
        return boto3.client('s3', 
                           region_name=S3_REGION,
                           aws_access_key_id=aws_access_key,
                           aws_secret_access_key=aws_secret_key)
    except Exception as e:
        print(f"Error creating S3 client: {e}")
        raise

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

def load_pdf_from_s3(s3_client, bucket: str, key: str):
    """Load a PDF from S3 with fallback loader using temporary files."""
    filename = Path(key).name
    
    # Generate a unique temporary path
    import uuid
    temp_dir = tempfile.gettempdir()
    unique_filename = f"vipo_temp_{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join(temp_dir, unique_filename)
    
    try:
        print(f"Downloading {filename} to temporary file: {temp_path}")
        # Download to temp file
        s3_client.download_file(bucket, key, temp_path)
        
        # Try PyPDFLoader first
        try:
            print(f"Loading PDF with PyPDFLoader: {filename}")
            docs = PyPDFLoader(temp_path).load()
        except Exception as e:
            print(f"PyPDF failed for {filename}: {e}")
            print(f"Trying UnstructuredPDFLoader as fallback")
            docs = UnstructuredPDFLoader(temp_path).load()
            
        # Update metadata
        for d in docs:
            d.metadata.update({"source": filename, "path": key})
            
        print(f"Successfully loaded {len(docs)} pages from {filename}")
        return docs
        
    except Exception as e:
        print(f"Error loading PDF from S3: {e}")
        raise
    finally:
        # Always clean up the temp file
        try:
            if os.path.exists(temp_path):
                print(f"Cleaning up temporary file: {temp_path}")
                # Try multiple times with delays to handle Windows file locking
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        os.unlink(temp_path)
                        print(f"Successfully deleted temporary file on attempt {attempt+1}")
                        break
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            print(f"Failed to delete temp file on attempt {attempt+1}: {e}")
                            import time
                            time.sleep(1)  # Wait a bit before retrying
                        else:
                            print(f"Could not delete temporary file after {max_attempts} attempts: {e}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

def load_txt_from_s3(s3_client, bucket: str, key: str):
    """Load a text file from S3 directly into memory."""
    try:
        from langchain.schema import Document
        print(f"Loading text file from S3: {key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        text = response['Body'].read().decode('utf-8', errors='ignore')
        filename = Path(key).name
        print(f"Successfully loaded text file: {filename} ({len(text)} characters)")
        return [Document(page_content=text, metadata={"source": filename, "path": key})]
    except Exception as e:
        print(f"Error loading text from S3: {e}")
        raise

def get_chunk_count_for_file(s3_client, filename: str) -> int:
    """Get the number of chunks for a specific file from ChromaDB"""
    try:
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed,
        )
        
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
                    count = sum(1 for meta in all_results["metadatas"] if meta and meta.get("doc_name") == filename)
                    return count
            except Exception:
                pass
        
        return 0
    except Exception as e:
        print(f"Warning: Could not get chunk count for {filename}: {e}")
        return 0

def list_s3_documents(s3_client):
    """List all documents in the S3 bucket with prefix"""
    try:
        print(f"Listing documents in S3 bucket: {S3_BUCKET_NAME}, prefix: {S3_PREFIX}")
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        documents = []
        
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects in bucket")
            for item in response['Contents']:
                key = item['Key']
                # Skip the prefix directory itself and chunks.json
                if key.endswith('/') or key.endswith('chunks.json'):
                    print(f"Skipping {key} (directory or chunks.json)")
                    continue
                
                # Make sure the key is directly under the documents/ prefix
                # This ensures we only list files in the documents/ folder, not in subfolders
                if key.count('/') > 1:
                    print(f"Skipping {key} (in subfolder)")
                    continue
                    
                filename = Path(key).name
                file_type = "PDF" if filename.lower().endswith(".pdf") else "TXT" if filename.lower().endswith(".txt") else "Unknown"
                
                print(f"Adding document: {filename}, type: {file_type}, size: {item['Size']} bytes")
                documents.append({
                    "name": filename,
                    "type": file_type,
                    "size_mb": round(item['Size'] / (1024 * 1024), 2),
                    "modified": item['LastModified'].isoformat(),
                    "s3_key": key
                })
        else:
            print(f"No objects found in bucket with prefix {S3_PREFIX}")
                
        print(f"Total documents found: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error listing S3 documents: {e}")
        raise

def ingest_files_to_chroma(s3_client, file_keys: List[str]):
    """Load, chunk, and store the given S3 files into Chroma."""
    all_docs = []
    for key in file_keys:
        filename = Path(key).name
        
        if filename.lower().endswith(".pdf"):
            all_docs.extend(load_pdf_from_s3(s3_client, S3_BUCKET_NAME, key))
        elif filename.lower().endswith(".txt"):
            all_docs.extend(load_txt_from_s3(s3_client, S3_BUCKET_NAME, key))

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
            "s3_key": c.metadata.get("path")  # Store S3 key in metadata
        }
        texts.append(c.page_content)
        metas.append(enriched_meta)
        json_records.append(enriched_meta)
        prev_chunk_ids[doc_name] = current_id

    vectordb.add_texts(texts=texts, metadatas=metas)

    # Store chunks manifest in S3 or local file system
    try:
        # First try to get existing manifest if it exists
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}chunks.json")
            old_data = json.loads(response['Body'].read().decode('utf-8'))
            old_data.extend(json_records)
            json_records = old_data
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                raise
        except Exception as e:
            print(f"Warning: Error reading chunks.json: {e}")
            # If we're using local filesystem, try to read the file directly
            chunks_path = Path("./documents/chunks.json")
            if chunks_path.exists():
                try:
                    old_data = json.loads(chunks_path.read_text(encoding="utf-8"))
                    old_data.extend(json_records)
                    json_records = old_data
                except Exception as e2:
                    print(f"Warning: Could not read local chunks.json: {e2}")
        
        # Upload updated manifest to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{S3_PREFIX}chunks.json",
            Body=json.dumps(json_records, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"Updated chunks.json")
    except Exception as e:
        print(f"Warning: Could not update chunks.json: {e}")
        # If using local filesystem, try to write directly
        try:
            chunks_path = Path("./documents/chunks.json")
            chunks_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Updated local chunks.json")
        except Exception as e2:
            print(f"Warning: Could not update local chunks.json: {e2}")

    return len(file_keys), len(chunks)


# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), s3_client = Depends(get_s3_client)):
    """Upload files to S3 and ingest into Chroma."""
    try:
        uploaded_keys = []
        for file in files:
            if not file.filename.lower().endswith((".pdf", ".txt")):
                continue
                
            try:
                # Read file into memory
                file_content = await file.read()
                
                # Log file info
                print(f"Uploading file: {file.filename}, size: {len(file_content)} bytes")
                
                # Check if bucket exists
                try:
                    s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
                    print(f"Bucket {S3_BUCKET_NAME} exists and is accessible")
                except Exception as bucket_error:
                    print(f"Error checking bucket: {bucket_error}")
                    raise
                
                # Upload directly to S3 from memory
                s3_key = f"{S3_PREFIX}{file.filename}"
                print(f"Uploading to S3 key: {s3_key}")
                
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME, 
                    Key=s3_key, 
                    Body=file_content,
                    ContentType='application/pdf' if file.filename.lower().endswith('.pdf') else 'text/plain'
                )
                print(f"Successfully uploaded file to S3: {s3_key}")
                uploaded_keys.append(s3_key)
            except Exception as file_error:
                print(f"Error uploading file {file.filename}: {file_error}")
                raise

        if not uploaded_keys:
            raise HTTPException(status_code=400, detail="No valid PDF/TXT files uploaded")

        # Ingest the uploaded files
        print(f"Starting ingestion of {len(uploaded_keys)} files")
        try:
            file_count, chunk_count = ingest_files_to_chroma(s3_client, uploaded_keys)
            print(f"Ingestion complete: {file_count} files, {chunk_count} chunks")
        except Exception as ingest_error:
            print(f"Error during ingestion: {ingest_error}")
            raise

        return {
            "message": f"Uploaded and ingested {file_count} file(s) into vector store",
            "uploaded_files": [Path(key).name for key in uploaded_keys],
            "chunks_added": chunk_count,
            "collection": CHROMA_COLLECTION
        }

    except Exception as e:
        import traceback
        print(f"Upload/ingest error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload/ingest error: {e}")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    # Use absolute path to find the template
    import os
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates/index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/stats")
async def get_stats(s3_client = Depends(get_s3_client)):
    """Get document and vector database statistics"""
    try:
        # List documents from S3
        documents = list_s3_documents(s3_client)
        pdf_files = [doc for doc in documents if doc["type"] == "PDF"]
        txt_files = [doc for doc in documents if doc["type"] == "TXT"]
        
        # Calculate vector DB size
        vector_db_size = 0
        if CHROMA_DIR.exists():
            for item in CHROMA_DIR.rglob("*"):
                if item.is_file():
                    vector_db_size += item.stat().st_size
        
        # Get REAL-TIME chunk count from ChromaDB
        total_chunks = 0
        try:
            embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            vectordb = Chroma(
                collection_name=CHROMA_COLLECTION,
                persist_directory=str(CHROMA_DIR),
                embedding_function=embed,
            )
            total_chunks = vectordb._collection.count()
        except Exception as e:
            print(f"Warning: Could not get real-time chunk count: {e}")
            # Fallback to S3 JSON file count
            try:
                response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}chunks.json")
                chunks_data = json.loads(response['Body'].read().decode('utf-8'))
                total_chunks = len(chunks_data)
            except Exception:
                pass
        
        return {
            "total_documents": len(documents),
            "pdf_files": len(pdf_files),
            "txt_files": len(txt_files),
            "vector_db_size_mb": round(vector_db_size / (1024 * 1024), 2),
            "total_chunks": total_chunks,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/documents")
async def list_documents(s3_client = Depends(get_s3_client)):
    """List all documents in S3"""
    try:
        documents = list_s3_documents(s3_client)
        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str, s3_client = Depends(get_s3_client)):
    """Delete a document from S3 and remove its chunks from vector database"""
    try:
        # Find the S3 key for this filename
        s3_key = f"{S3_PREFIX}{filename}"
        
        # Check if file exists
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise HTTPException(status_code=404, detail="File not found in S3")
            raise
        
        # Get vector database
        embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed,
        )
        
        # Get chunk count before deletion for reporting
        chunks_before = get_chunk_count_for_file(s3_client, filename)
        
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
        
        # Delete the file from S3
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        
        # Update chunks.json manifest in S3
        try:
            # Get current manifest
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}chunks.json")
            chunks_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Filter out chunks for the deleted file
            filtered_chunks = [chunk for chunk in chunks_data if chunk.get("doc_name") != filename]
            
            # Upload updated manifest
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{S3_PREFIX}chunks.json",
                Body=json.dumps(filtered_chunks, ensure_ascii=False, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            print(f"Updated chunks.json in S3 - removed entries for {filename}")
            
        except Exception as e:
            print(f"Warning: Could not update chunks.json in S3: {e}")
        
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
async def reprocess_documents(s3_client = Depends(get_s3_client)):
    """Manually reprocess all documents in S3"""
    try:
        documents = list_s3_documents(s3_client)
        if not documents:
            return {"message": "No documents found to process", "chunks_processed": 0}
        
        # Extract S3 keys
        s3_keys = [doc["s3_key"] for doc in documents]
        
        file_count, chunk_count = ingest_files_to_chroma(s3_client, s3_keys)
        return {
            "message": f"Successfully reprocessed {chunk_count} chunks from {file_count} documents",
            "chunks_processed": chunk_count,
            "documents_processed": file_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocessing error: {str(e)}")

@app.get("/api/check-s3")
async def check_s3_connection(s3_client = Depends(get_s3_client)):
    """Check S3 connection and bucket existence"""
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"Bucket {S3_BUCKET_NAME} exists")
        
        # List objects to verify access
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=20)
        
        # Check if prefix directory exists
        prefix_exists = False
        objects = []
        if 'Contents' in response:
            objects = [item['Key'] for item in response['Contents']]
            print(f"Found objects in bucket: {objects}")
            for key in objects:
                if key == S3_PREFIX:
                    prefix_exists = True
                    break
        
        # Create prefix directory if it doesn't exist
        if not prefix_exists:
            print(f"Creating prefix directory: {S3_PREFIX}")
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=S3_PREFIX,
                Body=''
            )
            print(f"Created prefix directory: {S3_PREFIX}")
        
        # Check for chunks.json file
        chunks_json_exists = False
        chunks_json_key = f"{S3_PREFIX}chunks.json"
        for key in objects:
            if key == chunks_json_key:
                chunks_json_exists = True
                break
        
        # Create empty chunks.json if it doesn't exist
        if not chunks_json_exists:
            print(f"Creating empty chunks.json file")
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=chunks_json_key,
                Body='[]',
                ContentType='application/json'
            )
            print(f"Created empty chunks.json file")
        
        # List only documents in the documents/ folder
        docs_response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        document_files = []
        if 'Contents' in docs_response:
            for item in docs_response['Contents']:
                key = item['Key']
                if key != S3_PREFIX and not key.endswith('chunks.json'):
                    document_files.append({
                        'key': key,
                        'size': item['Size'],
                        'modified': item['LastModified'].isoformat()
                    })
        
        return {
            "status": "success",
            "message": f"Successfully connected to S3 bucket: {S3_BUCKET_NAME}",
            "region": S3_REGION,
            "bucket": S3_BUCKET_NAME,
            "prefix": S3_PREFIX,
            "prefix_exists": prefix_exists,
            "chunks_json_exists": chunks_json_exists,
            "document_files": document_files,
            "all_objects": objects
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        if error_code == '404':
            return {
                "status": "error",
                "message": f"Bucket {S3_BUCKET_NAME} does not exist",
                "error": error_msg
            }
        elif error_code == '403':
            return {
                "status": "error",
                "message": f"Access denied to bucket {S3_BUCKET_NAME}",
                "error": error_msg
            }
        else:
            return {
                "status": "error",
                "message": f"S3 error: {error_code}",
                "error": error_msg
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "Failed to connect to S3",
            "error": str(e)
        }

if __name__ == "__main__":
    print("Starting Vipo Knowledge Base Dashboard...")
    print(f"S3 Bucket: {S3_BUCKET_NAME}/{S3_PREFIX}")
    print(f"ChromaDB directory: {CHROMA_DIR}")
    
    # Check if AWS credentials are available from any source
    aws_creds_available = False
    try:
        # Test if we can get caller identity
        test_client = boto3.client('sts')
        test_client.get_caller_identity()
        aws_creds_available = True
        print("\nAWS credentials found in config or environment.")
        print(f"Using S3 bucket: {S3_BUCKET_NAME}")
    except Exception:
        aws_creds_available = False
    
    if not aws_creds_available:
        print("\nWARNING: AWS credentials not found!")
        print("To use S3 storage, please set the following environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION (optional, defaults to us-east-1)")
        print("  - S3_BUCKET_NAME (optional, defaults to vipo-documents)")
        print("\nFalling back to local file system for document storage.")
        print("Documents will be stored in ./documents/ directory.")
        
        # Create documents directory
        Path("./documents").mkdir(parents=True, exist_ok=True)
    
    print("\nDashboard available at: http://localhost:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090, reload=False)# # app.py (FastAPI backend with integrated chunking + Chroma ingest)

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import HTMLResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# from datetime import datetime
# import shutil
# import json
# import uuid
# from typing import List
# import os
# import uvicorn
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # =============================================================================
# # Settings
# # =============================================================================
# # Use resolve() to get clean absolute paths without ../ references
# DOCUMENTS_DIR = Path("../documents").resolve()
# CHROMA_DIR = Path("../.chroma").resolve()
# CHROMA_COLLECTION = "vipo_bank_policies"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# # Ensure directories exist
# DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
# CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# app = FastAPI(title="Vipo Knowledge Base Dashboard", version="2.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Create static directory if it doesn't exist
# Path("static").mkdir(exist_ok=True)

# # Add route for favicon.ico
# @app.get("/favicon.ico")
# async def favicon():
#     return FileResponse("static/favicon.ico", media_type="image/x-icon")

# # =============================================================================
# # Utility functions
# # =============================================================================
# def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return splitter.split_documents(documents)

# def load_pdf(path: Path):
#     """Load a single PDF with fallback loader."""
#     try:
#         docs = PyPDFLoader(str(path)).load()
#     except Exception as e:
#         print(f"PyPDF failed for {path.name}: {e}")
#         docs = UnstructuredPDFLoader(str(path)).load()
#     for d in docs:
#         d.metadata.update({"source": path.name, "path": str(path)})
#     return docs

# def get_chunk_count_for_file(filename: str) -> int:
#     """Get the number of chunks for a specific file from ChromaDB"""
#     try:
#         embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#         vectordb = Chroma(
#             collection_name=CHROMA_COLLECTION,
#             persist_directory=str(CHROMA_DIR),
#             embedding_function=embed,
#         )
        
#         # Get all metadata and count matching documents manually
#         # This is a workaround for ChromaDB versions that don't support where in count()
#         try:
#             # First try using get() with where parameter to get matching documents
#             results = vectordb._collection.get(where={"doc_name": filename})
#             if results and "ids" in results:
#                 return len(results["ids"])
#         except Exception:
#             # Fallback: Get all documents and count manually
#             try:
#                 all_results = vectordb._collection.get()
#                 if all_results and "metadatas" in all_results and all_results["metadatas"]:
#                     # Count documents where doc_name matches filename
#                     count = sum(1 for meta in all_results["metadatas"] if meta and meta.get("doc_name") == filename)
#                     return count
#             except Exception:
#                 pass
        
#         return 0
#     except Exception as e:
#         print(f"Warning: Could not get chunk count for {filename}: {e}")
#         return 0

# def ingest_files_to_chroma(file_paths: List[Path]):
#     """Load, chunk, and store the given files into Chroma."""
#     all_docs = []
#     for path in file_paths:
#         if path.suffix.lower() == ".pdf":
#             all_docs.extend(load_pdf(path))
#         elif path.suffix.lower() == ".txt":
#             text = path.read_text(encoding="utf-8", errors="ignore")
#             from langchain.schema import Document
#             all_docs.append(Document(page_content=text, metadata={"source": path.name, "path": str(path)}))

#     if not all_docs:
#         return 0, 0

#     chunks = chunk_documents(all_docs)

#     embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vectordb = Chroma(
#         collection_name=CHROMA_COLLECTION,
#         persist_directory=str(CHROMA_DIR),
#         embedding_function=embed,
#     )

#     now_iso = datetime.now().isoformat()
#     texts, metas, json_records = [], [], []
#     prev_chunk_ids = {}

#     for c in chunks:
#         doc_name = c.metadata.get("source") or "unknown"
#         prev_id = prev_chunk_ids.get(doc_name)
#         current_id = str(uuid.uuid4())

#         enriched_meta = {
#             "content": c.page_content,
#             "date": now_iso,
#             "current_chunk_id": current_id,
#             "previous_chunk_id": prev_id,
#             "access": "PUBLIC",
#             "doc_name": doc_name,
#             "source": "PDF" if c.metadata.get("path", "").endswith(".pdf") else "TXT",
#         }
#         texts.append(c.page_content)
#         metas.append(enriched_meta)
#         json_records.append(enriched_meta)
#         prev_chunk_ids[doc_name] = current_id

#     vectordb.add_texts(texts=texts, metadatas=metas)

#     # Update chunks.json manifest
#     chunks_path = DOCUMENTS_DIR / "chunks.json"
#     if chunks_path.exists():
#         try:
#             old_data = json.loads(chunks_path.read_text(encoding="utf-8"))
#         except:
#             old_data = []
#         old_data.extend(json_records)
#         json_records = old_data
#     chunks_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")

#     return len(file_paths), len(chunks)


# # =============================================================================
# # API Endpoints
# # =============================================================================
# @app.post("/api/upload")
# async def upload_files(files: List[UploadFile] = File(...)):
#     """Upload and directly ingest files into Chroma."""
#     try:
#         saved_paths = []
#         for file in files:
#             if not file.filename.lower().endswith((".pdf", ".txt")):
#                 continue
#             dest = DOCUMENTS_DIR / file.filename
#             with open(dest, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
#             saved_paths.append(dest)

#         if not saved_paths:
#             raise HTTPException(status_code=400, detail="No valid PDF/TXT files uploaded")

#         file_count, chunk_count = ingest_files_to_chroma(saved_paths)

#         return {
#             "message": f"Uploaded and ingested {file_count} file(s) into vector store",
#             "uploaded_files": [path.name for path in saved_paths],
#             "chunks_added": chunk_count,
#             "collection": CHROMA_COLLECTION
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Upload/ingest error: {e}")


# @app.get("/", response_class=HTMLResponse)
# async def dashboard():
#     with open("templates/index.html", "r", encoding="utf-8") as f:
#         return HTMLResponse(content=f.read())


# @app.get("/api/stats")
# async def get_stats():
#     """Get document and vector database statistics"""
#     try:
#         # Count documents
#         pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
#         txt_files = list(DOCUMENTS_DIR.glob("*.txt"))
#         total_docs = len(pdf_files) + len(txt_files)
        
#         # Calculate vector DB size and print files in the chromadb
#         vector_db_size = 0
#         if CHROMA_DIR.exists():
#             for item in CHROMA_DIR.rglob("*"):
#                 if item.is_file():
#                     vector_db_size += item.stat().st_size
        
#         # Get REAL-TIME chunk count from ChromaDB (not from JSON file)
#         total_chunks = 0
#         try:
#             embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#             vectordb = Chroma(
#                 collection_name=CHROMA_COLLECTION,
#                 persist_directory=str(CHROMA_DIR),
#                 embedding_function=embed,
#             )
#             # Get actual count from ChromaDB
#             total_chunks = vectordb._collection.count()
#         except Exception as e:
#             print(f"Warning: Could not get real-time chunk count: {e}")
#             # Fallback to JSON file count
#             chunks_file = DOCUMENTS_DIR / "chunks.json"
#         if chunks_file.exists():
#             try:
#                 with open(chunks_file, 'r', encoding='utf-8') as f:
#                     chunks_data = json.load(f)
#                     total_chunks = len(chunks_data)
#             except:
#                 pass
        
#         return {
#             "total_documents": total_docs,
#             "pdf_files": len(pdf_files),
#             "txt_files": len(txt_files),
#             "vector_db_size_mb": round(vector_db_size / (1024 * 1024), 2),
#             "total_chunks": total_chunks,
#             "last_updated": datetime.now().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# @app.get("/api/documents")
# async def list_documents():
#     """List all documents in the documents folder"""
#     try:
#         documents = []
        
#         # PDF files
#         for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
#             stat = pdf_file.stat()
#             documents.append({
#                 "name": pdf_file.name,
#                 "type": "PDF",
#                 "size_mb": round(stat.st_size / (1024 * 1024), 2),
#                 "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
#             })
        
#         # TXT files
#         for txt_file in DOCUMENTS_DIR.glob("*.txt"):
#             stat = txt_file.stat()
#             documents.append({
#                 "name": txt_file.name,
#                 "type": "TXT",
#                 "size_mb": round(stat.st_size / (1024 * 1024), 2),
#                 "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
#             })
        
#         # Sort by modification time (newest first)
#         documents.sort(key=lambda x: x["modified"], reverse=True)
        
#         return {"documents": documents}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

# @app.delete("/api/documents/{filename}")
# async def delete_document(filename: str):
#     """Delete a document and remove its chunks from vector database"""
#     try:
#         file_path = DOCUMENTS_DIR / filename
        
#         if not file_path.exists():
#             raise HTTPException(status_code=404, detail="File not found")
        
#         # Get vector database
#         embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#         vectordb = Chroma(
#             collection_name=CHROMA_COLLECTION,
#             persist_directory=str(CHROMA_DIR),
#             embedding_function=embed,
#         )
        
#         # Get chunk count before deletion for reporting
#         chunks_before = get_chunk_count_for_file(filename)
        
#         # Remove chunks associated with this file from ChromaDB
#         try:
#             # First try to get IDs of documents with matching doc_name
#             results = vectordb._collection.get(where={"doc_name": filename})
#             if results and "ids" in results and results["ids"]:
#                 # Delete by IDs instead of using where parameter
#                 vectordb._collection.delete(ids=results["ids"])
#                 print(f"Removed {len(results['ids'])} chunks for {filename} from ChromaDB")
#                 chunks_before = len(results["ids"])
                
#                 # Force persist to ensure changes are saved
#                 vectordb.persist()

#                 print(f"Persisted changes to ChromaDB")
#             else:
#                 print(f"No chunks found for {filename} in ChromaDB")
#         except Exception as e:
#             # Fallback: Try to get all documents and filter manually
#             try:
#                 all_results = vectordb._collection.get()
#                 if all_results and "ids" in all_results and "metadatas" in all_results:
#                     # Find IDs where doc_name matches filename
#                     ids_to_delete = [
#                         all_results["ids"][i] for i, meta in enumerate(all_results["metadatas"])
#                         if meta and meta.get("doc_name") == filename
#                     ]
#                     if ids_to_delete:
#                         vectordb._collection.delete(ids=ids_to_delete)
#                         print(f"Removed {len(ids_to_delete)} chunks for {filename} from ChromaDB (fallback method)")
#                         chunks_before = len(ids_to_delete)
                        
#                         # Force persist to ensure changes are saved
#                         vectordb.persist()
#                         print(f"Persisted changes to ChromaDB (fallback method)")
#                     else:
#                         print(f"No chunks found for {filename} in ChromaDB (fallback method)")
#                 else:
#                     print(f"No documents found in ChromaDB")
#             except Exception as e2:
#                 print(f"Warning: Could not remove chunks from ChromaDB: {e} / {e2}")
#                 chunks_before = 0
        
#         # Delete the physical file
#         file_path.unlink()
        
#         # Update chunks.json manifest by removing entries for this file
#         chunks_file = DOCUMENTS_DIR / "chunks.json"
#         if chunks_file.exists():
#             try:
#                 with open(chunks_file, 'r', encoding='utf-8') as f:
#                     chunks_data = json.load(f)
                
#                 # Filter out chunks for the deleted file
#                 filtered_chunks = [chunk for chunk in chunks_data if chunk.get("doc_name") != filename]
                
#                 # Write back the filtered data
#                 chunks_file.write_text(
#                     json.dumps(filtered_chunks, ensure_ascii=False, indent=2),
#                     encoding="utf-8"
#                 )
#                 print(f"Updated chunks.json - removed entries for {filename}")
                
#             except Exception as e:
#                 print(f"Warning: Could not update chunks.json: {e}")
        
#         # Try to clean up and reclaim space
#         try:
#             # Force a collection update to potentially reclaim space
#             vectordb._collection.persist()
#             print(f"Collection persisted after deletion")
#         except Exception as e:
#             print(f"Warning: Could not persist collection: {e}")
        
#         return {
#             "message": f"Successfully deleted {filename} and removed {chunks_before} chunks from vector database",
#             "deleted_file": filename,
#             "chunks_removed": chunks_before
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# @app.post("/api/reprocess")
# async def reprocess_documents():
#     """Manually reprocess all documents in the documents folder"""
#     try:
#         all_files = list(DOCUMENTS_DIR.glob("*.pdf")) + list(DOCUMENTS_DIR.glob("*.txt"))
#         if not all_files:
#             return {"message": "No documents found to process", "chunks_processed": 0}
        
#         file_count, chunk_count = ingest_files_to_chroma(all_files)
#         return {
#             "message": f"Successfully reprocessed {chunk_count} chunks from {chunk_count} documents",
#             "chunks_processed": chunk_count,
#             "documents_processed": file_count
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Reprocessing error: {str(e)}")



# if __name__ == "__main__":
#     print("🚀 Starting Vipo Knowledge Base Dashboard...")
#     print(f"📁 Documents directory: {DOCUMENTS_DIR}")
#     print(f"🗄️  ChromaDB directory: {CHROMA_DIR}")
#     print("🌐 Dashboard available at: http://localhost:8090")
#     uvicorn.run(app, host="0.0.0.0", port=8090, reload=False)
