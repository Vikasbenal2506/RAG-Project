from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context: {context}
Question: {question}
""",
        ),
    ]
)


def ask_question(query):
    """
    Ask a question about the uploaded book.
    
    Returns:
        tuple: (answer_text, retrieved_documents)
    """
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Combine document content into context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate prompt and get response
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)
    
    return response.content, docs
