# Vipo - Banking Policy Assistant

A tool for processing, storing, and querying banking policy documents using AWS services and vector search.

## Overview

Vipo is a document processing system that extracts information from banking policy PDFs, chunks them into smaller pieces, and stores them in a vector database for semantic search. It uses AWS services like S3 for storage and Bedrock for embeddings, with local fallbacks when needed.

## Features

- Upload and process PDF banking policy documents
- Extract text and create searchable chunks
- Store documents in AWS S3 or local filesystem
- Generate embeddings using AWS Bedrock or HuggingFace
- Search across documents using semantic similarity
- Web dashboard for document management
- API endpoints for integration with other systems

## Getting Started

### Prerequisites

- Python 3.9+
- AWS account (optional, for S3 storage and Bedrock)
- PDF documents to process

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/vipo.git
   cd vipo
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up AWS credentials (optional):

   ```
   # Windows
   set_aws_credentials.bat

   # Linux/Mac
   source set_aws_credentials.sh
   ```

### Configuration

Create a `.env` file with your settings:

```
# AWS Settings
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=vipo-documents
S3_PREFIX=documents/

# Bedrock Settings (optional)
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.titan-embed-text-v1

# Vector DB Settings
CHROMA_COLLECTION=vipo_bank_policies
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Usage

### Processing Documents

1. Place your PDF files in the `documents` folder or upload them through the dashboard

2. Run the ingestion script:

   ```
   python ingest.py
   ```

3. This will:
   - Extract text from PDFs
   - Split into manageable chunks
   - Generate embeddings
   - Store in ChromaDB
   - Save metadata to S3 or locally

### Running the Dashboard

Start the dashboard to manage your documents:

```
cd dashboard
python main.py
```

Access the dashboard at http://localhost:8090

### Running the Chainlit App

For the conversational interface:

```
python app.py
```

Access the chat interface at http://localhost:9000

## Architecture

- **Document Storage**: AWS S3 or local filesystem
- **Vector Database**: ChromaDB
- **Embeddings**: AWS Bedrock or HuggingFace
- **Dashboard**: FastAPI + HTML/JS
- **Chat Interface**: Chainlit

## Deployment

### Local Development

Run components individually as described in the Usage section.

### AWS Deployment

1. Set up an EC2 instance with the required dependencies
2. Configure Nginx as a reverse proxy (see `vipo.conf`)
3. Set up systemd services for dashboard and chat app
4. Configure SSL with a self-signed certificate or Let's Encrypt

For detailed deployment instructions, see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

## File Structure

```
vipo/
├── app.py                 # Chainlit chat application
├── ingest.py              # Document processing script
├── dashboard/
│   ├── main.py            # FastAPI dashboard application
│   └── templates/         # Dashboard HTML templates
├── documents/             # Local document storage
├── .chroma/               # Vector database storage
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Troubleshooting

### AWS Credentials

If you're having issues with AWS credentials:

1. Check your environment variables
2. Verify your AWS CLI configuration
3. Use the provided scripts to set credentials
4. Check AWS permissions for your user

### Connection Issues

If you can't connect to the dashboard or chat interface:

1. Verify the services are running
2. Check port access in your security groups
3. Ensure Nginx is properly configured
4. Check for firewall restrictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- LangChain for the document processing pipeline
- ChromaDB for vector storage
- AWS for cloud infrastructure
- FastAPI and Chainlit for web interfaces
