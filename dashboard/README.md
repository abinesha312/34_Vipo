# 🏦 Vipo Knowledge Base Dashboard

A modern web dashboard for managing your Vipo RAG system's knowledge base. Upload, process, and manage banking policy documents with real-time statistics and a beautiful interface.

## ✨ Features

- **📊 Real-time Statistics**: View document counts, chunk counts, and vector database size
- **📁 Drag & Drop Upload**: Upload PDF documents directly through the web interface
- **🔄 Auto-processing**: Documents are automatically chunked and embedded into ChromaDB
- **🗑️ Document Management**: View, delete, and manage uploaded documents
- **🎨 Modern UI**: Beautiful, responsive interface with real-time updates
- **⚡ Fast Processing**: Direct integration with LangChain for efficient document processing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd dashboard
pip install -r requirements.txt
```

### 2. Start the Dashboard

```bash
python main.py
```

### 3. Open Your Browser

Navigate to: **http://localhost:8001**

## 🔧 How It Works

### Document Processing Pipeline

1. **Upload**: PDF files are uploaded to the `../documents/` folder
2. **Loading**: Documents are loaded using PyPDFLoader with UnstructuredPDFLoader fallback
3. **Chunking**: Text is split into chunks using RecursiveCharacterTextSplitter
4. **Embedding**: Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2
5. **Storage**: Embeddings are stored in ChromaDB with rich metadata
6. **Manifest**: A `chunks.json` file is generated for inspection

### Metadata Schema

Each chunk includes:

- `content`: The actual text content
- `date`: ISO timestamp of processing
- `current_chunk_id`: Unique UUID for the chunk
- `previous_chunk_id`: Link to previous chunk for continuity
- `access`: Access level (PUBLIC)
- `doc_name`: Original filename
- `source`: Source type (PDF)
- `path`: File path
- `page`: Page number

## 📁 Directory Structure

```
dashboard/
├── main.py              # FastAPI backend
├── templates/
│   └── index.html      # Dashboard frontend
├── requirements.txt     # Python dependencies
└── README.md           # This file

../documents/           # PDF storage (relative path)
../.chroma/            # ChromaDB storage (relative path)
```

## 🌐 API Endpoints

- `GET /` - Dashboard HTML interface
- `GET /api/stats` - Get knowledge base statistics
- `POST /api/upload` - Upload and process documents
- `POST /api/reprocess` - Reprocess all documents
- `GET /api/documents` - List all documents
- `DELETE /api/documents/{filename}` - Delete a document

## 🔄 Integration with Vipo

The dashboard works seamlessly with your existing Vipo RAG system:

- **Same ChromaDB**: Uses the same vector database as your RAG app
- **Same Embeddings**: Uses the same sentence-transformers model
- **Same Chunking**: Uses the same RecursiveCharacterTextSplitter
- **Same Metadata**: Maintains the same chunk metadata schema

## 🎯 Use Cases

- **Document Management**: Upload new banking policies and regulations
- **Knowledge Base Updates**: Reprocess documents when policies change
- **System Monitoring**: Monitor the size and health of your vector database
- **Content Inspection**: View processed chunks and metadata

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the port in `main.py` (line 205)
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Permission Errors**: Ensure write access to `../documents/` and `../.chroma/`

### Performance Tips

- **Large PDFs**: The system handles large documents but may take time
- **Memory Usage**: Monitor memory usage with very large document collections
- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` in `chunk_documents()` function

## 🔮 Future Enhancements

- **Batch Processing**: Process multiple documents in parallel
- **Progress Tracking**: Real-time progress bars for document processing
- **Document Preview**: View document content before processing
- **Search Interface**: Search through processed chunks
- **User Authentication**: Secure access to the dashboard

## 📞 Support

For issues or questions:

1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure proper file permissions
4. Check that the documents and chroma directories exist

---

**Happy Document Processing! 🚀**
