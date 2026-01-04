# genai-rag-hr-policy
GenAI RAG mini project using LangChain

# GenAI RAG Mini Project – HR Policy Assistant

## 📌 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain.
It answers HR policy questions by retrieving relevant policy text and generating grounded responses using an LLM.

## 🧠 Architecture
User Question  
→ Retriever (Vector Search)  
→ Relevant Policy Chunks  
→ Prompt Template  
→ LLM  
→ Final Answer  

## 🛠️ Tech Stack
- Python
- LangChain
- Vector Embeddings
- Jupyter Notebook
- RAG Architecture

## 📂 Project Structure

#hr_policy.txt
Employees are entitled to 20 paid leaves per year.
Unused leaves cannot be carried forward.
Work from home is allowed two days per week.

#loader.py
from langchain_core.documents import Document
from pathlib import Path
def load_documents():
    data_path = Path(__file__).parent.parent / "data" / "hr_policy.txt"
    with open(data_path, "r", encoding ="utf-8") as f:
        text = f.read()
        doc = Document(
            page_content = text,
            metadata={"source": "HR Policy"}
        )
        return[doc]

        #embeddings.py
        from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #retriever.py
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
def build_retriever(documents, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20
    )
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs = {"k":2})

    #rag_chain.py
    from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
def docs_to_context(docs):
    return "\n".join(doc.page_content for doc in docs)
def fake_llm(prompt:str) -> str:
    if "20 paid leaves" in prompt:
        return "Employees are entitled to 20 paid leaves per year."
        return "I don't know."

def build_rag_chain(retriever):
    llm = RunnableLambda(fake_llm)

    prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = """
    Answer the question using ONLY the context below.
    If the answer is not present, say "I dont know".

    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    )
    return(
    {
    "context": retriever | docs_to_context,
    "question": RunnableLambda(lambda x: x)
    }
    | prompt
    | llm
    )

   
