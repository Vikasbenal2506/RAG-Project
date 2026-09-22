#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def create_vectorstore(pdf_path):
    """
    Create a vector store from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
    """
    # Load PDF
    data = PyPDFLoader(pdf_path)
    docs = data.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2"
    )

    # Store in Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma-db"
    )
    
    return vector_store


if __name__ == "__main__":
    # Test: create vectorstore from sample PDF
    create_vectorstore("document_loaders/deeplearning.pdf")