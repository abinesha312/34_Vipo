# Vipo - Banking Policy Document Assistant

Vipo is a Retrieval-Augmented Generation (RAG) application focused on bank-related policies and documents.

## Features

- Document ingestion and processing (PDF, TXT)
- Vector storage with ChromaDB
- Re-ranking for improved search relevance
- Web dashboard for document management
- Chainlit chat interface for policy questions

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vipo.git
cd vipo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Environment Variables:
   - Copy `env.template` to `.env`
   - Add your OpenAI API key and other configuration

```bash
cp env.template .env
# Edit .env with your API keys and settings
```

### Running the Application

1. Process documents:
```bash
python ingest.py
```

2. Start the dashboard:
```bash
python dashboard/main.py
```

3. Start the Chainlit interface:
```bash
chainlit run app.py
```

## Dashboard

Access the dashboard at http://localhost:8090 to:
- View document statistics
- Upload new documents
- Delete documents
- Reprocess all documents

## Chainlit Interface

Access the Chainlit interface at http://localhost:9000 to:
- Ask questions about banking policies
- Get answers from your document collection

## AWS Deployment

For AWS deployment instructions, see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)

## Security Note

Never commit API keys or sensitive information to the repository. Always use environment variables for sensitive data.
