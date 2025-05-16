import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import openai

def setup_openai():
    """Set up OpenAI API credentials"""
    openai.api_key = "Enter your open ai or any other api key here"

def create_documents():
    """Create sample documents for demonstration"""
    text = """RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based and generative approaches. 
    It uses a vector database to store and retrieve relevant documents, then feeds them to a language model for generation.
    This allows for more accurate and context-aware responses compared to pure generative models.
    
    Long Context models, on the other hand, are language models designed to handle longer context windows. 
    While traditional models might have a context window of 4096 tokens, Long Context models can process much longer texts.
    This is achieved through various techniques like sliding windows or specialized architectures.
    
    The key difference is that RAG can access external knowledge through retrieval, while Long Context models rely on their internal context window."""
    
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents([text])
    return docs

def demonstrate_rag():
    """Demonstrate RAG approach"""
    print("\n=== RAG Approach ===")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    docs = create_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Create RAG chain
    llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
    )
    
    # Ask a question
    question = "What is the main difference between RAG and Long Context models?"
    print(f"Question: {question}")
    print("\nAnswer:")
    print(qa_chain.run(question))

def demonstrate_long_context():
    """Demonstrate Long Context approach"""
    print("\n=== Long Context Approach ===")
    
    # Create a long context prompt
    docs = create_documents()
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create a prompt with the long context
    prompt = f"""Here is some context about RAG and Long Context models:
    {context}
    
    What is the main difference between RAG and Long Context models?"""
    
    # Use OpenAI to generate response
    llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
    response = llm(prompt)
    print("\nAnswer:")
    print(response)

def main():
    setup_openai()
    
    print("Demonstrating RAG vs Long Context approaches...")
    
    # Demonstrate RAG approach
    demonstrate_rag()
    
    # Demonstrate Long Context approach
    demonstrate_long_context()

if __name__ == "__main__":
    main()
