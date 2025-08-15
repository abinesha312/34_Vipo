#!/usr/bin/env python3
"""
Test script to verify embeddings are stored in ChromaDB and show documents.
Run this after ingestion to verify everything works.
"""

from pathlib import Path
from settings import settings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def show_documents():
    """Show all documents stored in the vector database"""
    print("📚 Showing all documents in ChromaDB...")
    
    try:

        
        # Connect to existing DB
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(
            collection_name="vipo_bank_policies",
            persist_directory=settings.chroma_dir,
            embedding_function=embed,
        )
        
        # Get all documents
        collection = vectordb._collection
        count = collection.count()
        print(f"📊 Total documents in collection: {count}")
        
        if count > 0:
            # Get all documents with metadata
            results = collection.get(limit=count)
            
            for i, (content, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"\n{'='*60}")
                print(f"📄 Document {i+1}")
                print(f"📁 File: {metadata.get('doc_name', 'Unknown')}")
                print(f"🆔 Chunk ID: {metadata.get('current_chunk_id', 'N/A')}")
                print(f"📅 Date: {metadata.get('date', 'N/A')}")
                print(f"🔗 Source: {metadata.get('source', 'N/A')}")
                print(f"🔓 Access: {metadata.get('access', 'N/A')}")
                print(f"📝 Content Preview:")
                print(f"   {content[:300]}...")
                if len(content) > 300:
                    print(f"   ... (truncated, total length: {len(content)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error showing documents: {e}")
        return False


def show_summary():
    """Show summary info about the vector database"""
    print("\n🗄️  Vector Database Summary...")
    
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Connect to existing DB
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(
            collection_name="vipo_bank_policies",
            persist_directory=settings.chroma_dir,
            embedding_function=embed,
        )
        
        # Get collection info
        collection = vectordb._collection
        count = collection.count()
        print(f"📊 Total documents in collection: {count}")
        
        # Show some metadata
        if count > 0:
            results = collection.get(limit=1)
            if results['metadatas']:
                sample_meta = results['metadatas'][0]
                print(f"📋 Sample metadata keys: {list(sample_meta.keys())}")
                print(f"📄 Sample doc_name: {sample_meta.get('doc_name', 'N/A')}")
                print(f"🆔 Sample chunk_id: {sample_meta.get('current_chunk_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting vector DB info: {e}")
        return False


def search_documents(query, top_k=5):
    """Search for similar documents based on query"""
    print(f"\n🔍 Searching for: '{query}'")
    print(f"📊 Returning top {top_k} results...")
    
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Connect to existing DB
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(
            collection_name="vipo_bank_policies",
            persist_directory=settings.chroma_dir,
            embedding_function=embed,
        )
        
        # Search for similar documents
        results = vectordb.similarity_search(query, k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} similar documents:")
            for i, doc in enumerate(results):
                print(f"\n{'='*60}")
                print(f"📄 Result {i+1}")
                print(f"📁 File: {doc.metadata.get('doc_name', 'Unknown')}")
                print(f"🆔 Chunk ID: {doc.metadata.get('current_chunk_id', 'N/A')}")
                print(f"📅 Date: {doc.metadata.get('date', 'N/A')}")
                print(f"🔗 Source: {doc.metadata.get('source', 'N/A')}")
                print(f"🔓 Access: {doc.metadata.get('access', 'N/A')}")
                print(f"📝 Content:")
                print(f"   {doc.page_content}")
                if len(doc.page_content) > 500:
                    print(f"   ... (truncated, total length: {len(doc.page_content)} chars)")
        else:
            print("❌ No similar documents found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error searching documents: {e}")
        return False


def search_with_reranking(query, top_k=5):
    """Search with cross-encoder re-ranking mechanism"""
    print(f"\n🔄 Searching with re-ranking: '{query}'")
    print(f"📊 Returning top {top_k} re-ranked results...")
    
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        
        # Connect to existing DB
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(
            collection_name="vipo_bank_policies",
            persist_directory=settings.chroma_dir,
            embedding_function=embed,
        )
        
        # Build retriever with re-ranking
        base_retriever = vectordb.as_retriever(search_kwargs={"k": max(top_k * 2, 10)})
        
        # Cross-encoder re-ranking
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base", 
            model_kwargs={"device": "cpu"}
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
        reranking_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever, 
            base_compressor=compressor
        )
        
        # Get re-ranked documents
        results = reranking_retriever.get_relevant_documents(query)
        
        if results:
            print(f"✅ Found {len(results)} re-ranked documents:")
            for i, doc in enumerate(results):
                print(f"\n{'='*60}")
                print(f"📄 Re-ranked Result {i+1}")
                print(f"📁 File: {doc.metadata.get('doc_name', 'Unknown')}")
                print(f"🆔 Chunk ID: {doc.metadata.get('current_chunk_id', 'N/A')}")
                print(f"📅 Date: {doc.metadata.get('date', 'N/A')}")
                print(f"🔗 Source: {doc.metadata.get('source', 'N/A')}")
                print(f"🔓 Access: {doc.metadata.get('access', 'N/A')}")
                print(f"📝 Content:")
                print(f"   {doc.page_content}")
                if len(doc.page_content) > 500:
                    print(f"   ... (truncated, total length: {len(doc.page_content)} chars)")
        else:
            print("❌ No re-ranked documents found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with re-ranking search: {e}")
        return False


def main():
    """Show all documents in ChromaDB and allow searching"""
    print("🚀 Vipo Document Viewer & Search")
    print("=" * 50)
    
    # Check if ChromaDB exists
    chroma_path = Path(settings.chroma_dir)
    if not chroma_path.exists():
        print("❌ ChromaDB directory not found. Run ingestion first:")
        print("   python -m vipo.ingest")
        return
    
    print(f"📁 ChromaDB found at: {chroma_path.absolute()}")
    
    # Show summary
    if show_summary():
        print()
        
        # Interactive search
        while True:
            print("\n" + "=" * 50)
            print("🔍 Search Options:")
            print("1. Basic similarity search")
            print("2. Search with cross-encoder re-ranking")
            print("3. Show all documents")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                query = input("Enter your search query: ").strip()
                if query:
                    top_k = input("How many results? (default 5): ").strip()
                    try:
                        top_k = int(top_k) if top_k else 5
                    except ValueError:
                        top_k = 5
                    search_documents(query, top_k)
                else:
                    print("❌ Query cannot be empty")
                    
            elif choice == "2":
                query = input("Enter your search query: ").strip()
                if query:
                    top_k = input("How many results? (default 5): ").strip()
                    try:
                        top_k = int(top_k) if top_k else 5
                    except ValueError:
                        top_k = 5
                    search_with_reranking(query, top_k)
                else:
                    print("❌ Query cannot be empty")
                    
            elif choice == "3":
                show_documents()
                print("\n" + "=" * 50)
                print("✅ Document display complete!")
                
            elif choice == "4":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    else:
        print("❌ Failed to show summary. Check the errors above.")


if __name__ == "__main__":
    main()
