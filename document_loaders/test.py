from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=25,
    chunk_overlap=1
)

data = TextLoader("document_loaders/notes.txt")

docs = data.load()

chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print()
    print()

print(len(docs))