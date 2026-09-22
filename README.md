# RAG Project

A Retrieval-Augmented Generation (RAG) project for asking questions against uploaded PDF documents using LangChain, Chroma, and Mistral models.

## Features

- Upload a PDF in the Streamlit app
- Split the document into chunks
- Generate embeddings and store them in a local Chroma vector database
- Retrieve relevant context using MMR retrieval
- Answer questions using a Mistral LLM grounded in document content
- Run a lightweight terminal-based RAG flow as well

## Project Structure

```text
RAG Project/
├── app.py                 # Streamlit UI for PDF upload and Q&A
├── main.py                # Terminal-based RAG chat loop
├── rag.py                 # Reusable RAG logic module
├── create_db.py           # PDF-to-vector-store creation script
├── requirements.txt       # Python dependencies
├── .env                   # Local environment variables (not committed)
├── chroma_db/             # Persisted vector database
├── document_loaders/      # PDF/doc related samples and loader scripts
├── retrievers/            # Retrieval strategies
├── vector_store/          # Vector store helpers
└── README.md              # Project documentation
```

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain
- ChromaDB
- Mistral AI models and embeddings
- PyPDF for PDF loading
- python-dotenv for environment management

## Prerequisites

1. Python installed
2. A Mistral API key
3. A virtual environment recommended

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "RAG Project"
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

You may also choose to store other keys or project settings here if needed.

## Run the App

### Streamlit web app

```bash
streamlit run app.py
```

This opens a browser UI where you can:

- upload a PDF
- create a Chroma vector database
- ask questions based on the uploaded content

### Terminal-based RAG chat

```bash
python main.py
```

This runs a simple command-line chat interface using the stored vector database.

### Build the vector database manually

```bash
python create_db.py
```

This script creates embeddings from a PDF and persists them in the local Chroma database directory.

## How It Works

1. The PDF is loaded with PyPDFLoader
2. Text is split into chunks using RecursiveCharacterTextSplitter
3. Chunk embeddings are generated with Mistral embeddings
4. The embeddings are stored in ChromaDB
5. A user query is matched against stored chunks using MMR retrieval
6. The retrieved context is passed to a Mistral LLM to generate the answer

## Notes

- The app expects the Chroma database directory to exist and be consistent with the code configuration.
- Some files in the project use `chroma_db` while others reference `chroma-db`; make sure the directory name matches across the script you are running.
- Keep your `.env` file private and do not commit it to source control.

## Common Issues

### Module not found errors

Make sure dependencies are installed in the active virtual environment:

```bash
pip install -r requirements.txt
```

### Missing Mistral API key

Ensure your `.env` file contains a valid `MISTRAL_API_KEY` before running the app.

### Database not found

Create the vector database first by uploading a PDF in the Streamlit app or by running the database creation script.

## License

This project is intended for learning and local experimentation. Add your preferred license if you plan to distribute it.
