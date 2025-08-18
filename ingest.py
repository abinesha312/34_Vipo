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
import tempfile
import boto3
from botocore.exceptions import ClientError

# Settings class definition
class Settings:
    # Local data & vector store
    docs_dir: Path = Path("./data")
    chroma_dir: Path = Path("./.chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "vipo_bank_policies")

    # S3 settings
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "vipo-documents")
    s3_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_prefix: str = os.getenv("S3_PREFIX", "documents/")  # Folder prefix in S3 - must end with /

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Optional: Google fallback
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_llm_model: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-pro")

settings = Settings()

COLLECTION = settings.chroma_collection

def get_s3_client():
    """Get S3 client with credentials from environment variables, AWS CLI config, or instance profile"""
    try:
        # First try creating a client without explicit credentials
        # This will use credentials from environment variables, AWS config files, or instance profiles
        try:
            # Test if we can get caller identity
            test_client = boto3.client('sts', region_name=settings.s3_region)
            test_client.get_caller_identity()
            print("Found AWS credentials from config or environment")
            return boto3.client('s3', region_name=settings.s3_region)
        except Exception as e:
            print(f"Could not use existing AWS config: {e}")
            
        # Fall back to checking environment variables directly
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            print("WARNING: AWS credentials not found in environment variables.")
            print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            print("Using local file system as a fallback...")
            return None
        
        # If we have credentials, use the real S3 client
        return boto3.client('s3', 
                           region_name=settings.s3_region,
                           aws_access_key_id=aws_access_key,
                           aws_secret_access_key=aws_secret_key)
    except Exception as e:
        print(f"Error creating S3 client: {e}")
        return None

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def load_pdf_from_s3(s3_client, bucket: str, key: str):
    """Load a PDF from S3 with fallback loader using temporary files."""
    filename = Path(key).name
    
    # Generate a unique temporary path
    temp_dir = tempfile.gettempdir()
    unique_filename = f"vipo_temp_{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join(temp_dir, unique_filename)
    
    try:
        print(f"Downloading {filename} from S3 to temporary file: {temp_path}")
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

def load_pdfs(pdf_dir: str):
    """Load PDFs from local directory (fallback method)"""
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

def list_s3_documents(s3_client):
    """List all documents in the S3 bucket with prefix"""
    try:
        print(f"Listing documents in S3 bucket: {settings.s3_bucket_name}, prefix: {settings.s3_prefix}")
        response = s3_client.list_objects_v2(Bucket=settings.s3_bucket_name, Prefix=settings.s3_prefix)
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
                
                if file_type == "PDF":
                    print(f"Adding document: {filename}, type: {file_type}, size: {item['Size']} bytes")
                    documents.append({
                        "name": filename,
                        "type": file_type,
                        "size_mb": round(item['Size'] / (1024 * 1024), 2),
                        "modified": item['LastModified'].isoformat(),
                        "s3_key": key
                    })
        else:
            print(f"No objects found in bucket with prefix {settings.s3_prefix}")
                
        print(f"Total documents found: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error listing S3 documents: {e}")
        return []

def run_ingest():
    # Try to get S3 client
    s3_client = get_s3_client()
    
    if s3_client:
        # Use S3 as the source
        print("Using S3 for document source")
        documents = list_s3_documents(s3_client)
        if not documents:
            print("No documents found in S3")
            return
            
        all_docs = []
        for doc in documents:
            if doc["type"] == "PDF":
                try:
                    pdf_docs = load_pdf_from_s3(s3_client, settings.s3_bucket_name, doc["s3_key"])
                    all_docs.extend(pdf_docs)
                except Exception as e:
                    print(f"Error processing {doc['name']}: {e}")
        
        if not all_docs:
            print("No documents were successfully loaded from S3")
            return
            
        print(f"Loaded {len(all_docs)} pages from S3; chunking…")
    else:
        # Fallback to local files
        documents_dir = Path("./documents")
        src_dir = str(documents_dir if documents_dir.exists() else Path(settings.docs_dir))
        
        print("Loading PDFs from local directory:", src_dir)
        all_docs = list(load_pdfs(src_dir))
        if not all_docs:
            print("No documents extracted from PDFs in", src_dir)
            return
        print(f"Loaded {len(all_docs)} pages; chunking…")
    
    # Continue with chunking and processing
    chunks = chunk_documents(all_docs)

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
            "s3_key": c.metadata.get("path")  # Store S3 key in metadata
        }

        texts.append(c.page_content)
        metas.append(enriched)
        json_records.append(enriched)

        # Update chain continuity per document
        previous_for_doc[doc_name] = current_id

    vectordb.add_texts(texts=texts, metadatas=metas)
    print("Ingest complete. Chroma dir:", settings.chroma_dir)

    # Write chunks JSON
    try:
        if s3_client:
            # Store in S3
            try:
                # First try to get existing manifest if it exists
                try:
                    response = s3_client.get_object(Bucket=settings.s3_bucket_name, Key=f"{settings.s3_prefix}chunks.json")
                    old_data = json.loads(response['Body'].read().decode('utf-8'))
                    old_data.extend(json_records)
                    json_records = old_data
                except ClientError as e:
                    if e.response['Error']['Code'] != 'NoSuchKey':
                        raise
                
                # Upload updated manifest to S3
                s3_client.put_object(
                    Bucket=settings.s3_bucket_name,
                    Key=f"{settings.s3_prefix}chunks.json",
                    Body=json.dumps(json_records, ensure_ascii=False, indent=2).encode('utf-8'),
                    ContentType='application/json'
                )
                print(f"Updated chunks.json in S3")
            except Exception as e:
                print(f"Warning: Could not update chunks.json in S3: {e}")
                # Fall back to local file
                documents_dir = Path("./documents")
                documents_dir.mkdir(parents=True, exist_ok=True)
                out_path = documents_dir / "chunks.json"
                out_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Wrote chunk manifest locally: {out_path}")
        else:
            # Store locally
            documents_dir = Path("./documents")
            documents_dir.mkdir(parents=True, exist_ok=True)
            out_path = documents_dir / "chunks.json"
            out_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote chunk manifest locally: {out_path}")
    except Exception as e:
        print("Warning: failed to write chunks.json:", e)


if __name__ == "__main__":
    run_ingest()
# from pathlib import Path
# from datetime import datetime
# import uuid
# import json
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pathlib import Path
# import os

# # Settings class definition
# class Settings:
#     # Local data & vector store
#     docs_dir: Path = Path("./data")
#     chroma_dir: Path = Path("./.chroma")
#     chroma_collection: str = "vipo_bank_policies"

#     # OpenAI
#     openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
#     openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

#     # Optional: Google fallback
#     google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
#     google_llm_model: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-pro")

# settings = Settings()

# COLLECTION = settings.chroma_collection


# def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
#     """Split documents into chunks using RecursiveCharacterTextSplitter"""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return text_splitter.split_documents(documents)


# def load_pdfs(pdf_dir: str):
#     base = Path(pdf_dir)
#     # Gather and de-duplicate case-insensitively (Windows)
#     candidates = list(base.rglob("*.pdf")) + list(base.rglob("*.PDF"))
#     pdfs = []
#     seen = set()
#     for p in candidates:
#         key = str(p.resolve()).lower()
#         if key not in seen:
#             seen.add(key)
#             pdfs.append(p)
#     print(f"Found {len(pdfs)} PDF files")
#     for p in pdfs:
#         # Try PyPDF first
#         try:
#             docs = PyPDFLoader(str(p)).load()
#             for d in docs:
#                 d.metadata.update({"source": p.name, "path": str(p)})
#             yield from docs
#             continue
#         except Exception as e:
#             print(f"PyPDF failed for {p.name}: {e}")
#         # Fallback: Unstructured
#         try:
#             docs = UnstructuredPDFLoader(str(p)).load()
#             for d in docs:
#                 d.metadata.update({"source": p.name, "path": str(p)})
#             yield from docs
#         except Exception as e:
#             print(f"Unstructured failed for {p.name}: {e}")


# def run_ingest():
#     # Prefer a local ./documents directory if present, else fallback to settings.pdf_dir
#     documents_dir = Path("./documents")
#     src_dir = str(documents_dir if documents_dir.exists() else Path(settings.pdf_dir))

#     print("Loading PDFs from:", src_dir)
#     docs = list(load_pdfs(src_dir))
#     if not docs:
#         print("No documents extracted from PDFs in", src_dir)
#         return
#     print(f"Loaded {len(docs)} pages; chunking…")
#     chunks = chunk_documents(docs)

#     # Use HuggingFace sentence transformer
#     model_id = "sentence-transformers/all-MiniLM-L6-v2"
#     print("Embedding with", model_id)
#     embed = HuggingFaceEmbeddings(model_name=model_id)

#     vectordb = Chroma(
#         collection_name=COLLECTION,
#         persist_directory=settings.chroma_dir,
#         embedding_function=embed,
#     )

#     # Build enriched metadata per chunk
#     texts = []
#     metas = []
#     json_records = []
#     now_iso = datetime.now().isoformat()
#     previous_for_doc: dict[str, str | None] = {}

#     for c in chunks:
#         doc_name = c.metadata.get("source") or c.metadata.get("doc_name") or "unknown.pdf"
#         prev_id = previous_for_doc.get(doc_name)
#         current_id = str(uuid.uuid4())

#         enriched = {
#             # Required schema
#             "content": c.page_content,
#             "date": now_iso,
#             "current_chunk_id": current_id,
#             "previous_chunk_id": prev_id,
#             "access": "PUBLIC",
#             "doc_name": doc_name,
#             "source": "PDF",
#         }

#         texts.append(c.page_content)
#         metas.append(enriched)
#         json_records.append(enriched)

#         # Update chain continuity per document
#         previous_for_doc[doc_name] = current_id

#     vectordb.add_texts(texts=texts, metadatas=metas)
#     print("Ingest complete. Chroma dir:", settings.chroma_dir)

#     # Write chunks JSON alongside the source documents for inspection
#     try:
#         out_dir = documents_dir if Path(src_dir) == documents_dir else Path(src_dir)
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / "chunks.json"
#         out_path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2), encoding="utf-8")
#         print("Wrote chunk manifest:", out_path)
#     except Exception as e:
#         print("Warning: failed to write chunks.json:", e)


# if __name__ == "__main__":
#     run_ingest()

