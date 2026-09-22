from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

data = PyPDFLoader("document_loaders/GRU.pdf")

docs = data.load()

chunks = text_splitter.split_documents(docs)

print(chunks[0].page_content)

# print(docs[14])