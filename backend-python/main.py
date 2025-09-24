# main.py
import os
import io
import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict
from operator import itemgetter

# --- CONFIGURATION ---
from config import settings

# --- CORE DEPENDENCIES ---
import uvicorn
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- DOCUMENT PROCESSING ---
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- LANGCHAIN IMPORTS ---
# LLMs, Embeddings, and Vector Stores
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Advanced Retrieval (Hybrid Search, Reranking)
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from sentence_transformers.cross_encoder import CrossEncoder

# Conversational Memory & Chains
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

# --- IN-MEMORY APPLICATION STORE ---
# This dictionary will hold our models, retrievers, and chat histories
app_store = {}

# --- Pydantic Models ---
class DocumentAnalysis(BaseModel):
    summary: str
    action_items: List[str]
    assigned_role: str

class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(description="A unique identifier for the conversation session.")

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# --- FastAPI Lifespan Manager (for Startup and Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up application...")
    # Initialize models and load them into the app_store
    app_store["llm"] = Ollama(model=settings.LLM_MODEL)
    app_store["embeddings"] = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
    app_store["reranker"] = CrossEncoder(settings.RERANKER_MODEL)
    app_store["chat_sessions"] = {} # To store memory for each session

    # Load existing vector store and create retrievers
    index_file = os.path.join(settings.FAISS_PATH, "index.faiss")
    if os.path.exists(index_file):
        print(f"Loading existing vector store from {settings.FAISS_PATH}...")
        vector_store = FAISS.load_local(
            settings.FAISS_PATH,
            app_store["embeddings"],
            allow_dangerous_deserialization=True
        )
        app_store["vector_store"] = vector_store
        # Rebuild the BM25 retriever from the documents in the vector store
        docs_from_vectorstore = vector_store.docstore._dict.values()
        app_store["bm25_retriever"] = BM25Retriever.from_documents(docs_from_vectorstore)
        print("✅ Retrievers are ready.")
    else:
        app_store["vector_store"] = None
        app_store["bm25_retriever"] = None
        print("⚠️ No vector store found. A new one will be created on first upload.")

    yield

    print("Shutting down application...")
    app_store.clear()

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Advanced RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
app.mount("/files", StaticFiles(directory=settings.UPLOAD_DIRECTORY), name="files")

# --- DEPENDENCY FUNCTIONS ---
def get_llm(): return app_store["llm"]
def get_embeddings(): return app_store["embeddings"]
def get_reranker(): return app_store["reranker"]

def get_retrievers():
    if not app_store.get("vector_store") or not app_store.get("bm25_retriever"):
        raise HTTPException(status_code=404, detail="Knowledge Base is empty. Please upload a document.")
    return {
        "vector": app_store["vector_store"].as_retriever(search_kwargs={"k": 10}),
        "keyword": app_store["bm25_retriever"]
    }

# --- HELPER FUNCTIONS ---
def process_and_chunk_text(file_content: bytes, filename: str) -> List[Document]:
    # (This function remains the same as your original version)
    docs = []
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension == ".pdf":
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                docs.append(Document(page_content=page_text, metadata={"source": filename, "page": i + 1}))
    elif file_extension in [".txt", ".md"]:
        text = file_content.decode("utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": filename}))
    elif file_extension in [".png", ".jpg", ".jpeg"]:
         image = Image.open(io.BytesIO(file_content))
         text = pytesseract.image_to_string(image)
         if text.strip():
             docs.append(Document(page_content=text, metadata={"source": filename}))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

    if not docs:
        raise HTTPException(status_code=400, detail=f"Could not extract any text from '{filename}'.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_documents(docs)

async def trigger_n8n_webhooks(urls: List[str], data: dict):
    # (This function remains the same as your original version)
    if not urls: return
    async with httpx.AsyncClient() as client:
        tasks = [client.post(url, json=data, timeout=10) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"❌ Error triggering n8n webhook {urls[i]}: {res}")
            else:
                print(f"✅ n8n webhook triggered successfully: {urls[i]}")

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/api/upload-and-process", response_model=DocumentAnalysis)
async def upload_and_process_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), llm=Depends(get_llm), embeddings=Depends(get_embeddings)):
    file_path = os.path.join(settings.UPLOAD_DIRECTORY, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        docs = process_and_chunk_text(content, file.filename)
        # Use the first few chunks for a quicker analysis
        analysis_text = " ".join([doc.page_content for doc in docs[:4]])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # LLM analysis for summary, actions, role
    try:
        # Define prompts for each analysis task
        summary_prompt = ChatPromptTemplate.from_template(
            "Provide a concise, professional summary (around 100-150 words) of the following document content: \n\n{document}"
        )
        actions_prompt = ChatPromptTemplate.from_template(
            "Extract the 3 to 5 most important, actionable tasks from the following document. Present them as a bulleted list. If no clear action items exist, respond with 'None'. \n\n{document}"
        )
        role_prompt = ChatPromptTemplate.from_template(
            "Read the document and determine the single most relevant employee role to handle it. Choose ONLY from this list: [Finance Manager, Customer Manager, Safety Manager, HR Coordinator, Legal Counsel, Rolling Stock Engineer]. Respond with ONLY the role name. Document: \n\n{document}"
        )

        # Create chains by piping prompts into the LLM
        summary_chain = summary_prompt | llm
        actions_chain = actions_prompt | llm
        role_chain = role_prompt | llm

        # Asynchronously run all analysis chains
        summary_result, actions_result, role_result = await asyncio.gather(
            summary_chain.ainvoke({"document": analysis_text}),
            actions_chain.ainvoke({"document": analysis_text}),
            role_chain.ainvoke({"document": analysis_text})
        )

        # Process the action items string into a clean list
        action_items_list = [
            line.strip().lstrip('-* ').strip() for line in actions_result.split('\n') 
            if line.strip() and "none" not in line.lower()
        ]
        
        # Create the final analysis object with real data
        analysis = DocumentAnalysis(
            summary=summary_result.strip(),
            action_items=action_items_list,
            assigned_role=role_result.strip().replace("'", "").replace('"', '')
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")

    # --- ADVANCED INGESTION PIPELINE ---
    # Update or create the vector store and retrievers
    vector_store = app_store.get("vector_store")
    if vector_store is None:
        app_store["vector_store"] = FAISS.from_documents(docs, embeddings)
    else:
        vector_store.add_documents(docs)

    # For BM25, we rebuild it with all documents from the vector store's docstore
    all_docs = app_store["vector_store"].docstore._dict.values()
    app_store["bm25_retriever"] = BM25Retriever.from_documents(all_docs)

    app_store["vector_store"].save_local(settings.FAISS_PATH)
    print(f"✅ Successfully processed, embedded, and indexed '{file.filename}'.")

    background_tasks.add_task(trigger_n8n_webhooks, urls=settings.N8N_WEBHOOK_URLS, data=analysis.model_dump())
    return analysis

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_knowledge_base(request: ChatRequest, llm=Depends(get_llm), reranker=Depends(get_reranker), retrievers=Depends(get_retrievers)):
    # --- 1. SET UP CONVERSATIONAL MEMORY ---
    if request.session_id not in app_store["chat_sessions"]:
        app_store["chat_sessions"][request.session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key='answer'
        )
    memory = app_store["chat_sessions"][request.session_id]

    # --- 2. CREATE THE ADVANCED RETRIEVAL PIPELINE ---
    # Hybrid Search: Combine vector and keyword retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retrievers["keyword"], retrievers["vector"]],
        weights=[0.5, 0.5] # Give equal weight to keyword and vector search
    )

    # Reranking and Context Compression
    compressor = CrossEncoderReranker(model=reranker, top_n=4) # Return top 4 most relevant docs
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # --- 3. DEFINE THE CONVERSATIONAL RAG CHAIN ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent assistant. Answer the user's question based on the provided context. If you don't know the answer from the context, say so. After the answer, list the sources you used.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    # Load conversation history
    def load_history(inputs):
        return memory.load_memory_variables(inputs)["chat_history"]

    # The final chain that combines retrieval, history, and generation
    conversational_rag_chain = (
        RunnablePassthrough.assign(chat_history=load_history)
        | {
            "context": itemgetter("input") | compression_retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
          }
        | document_chain
    )

    try:
        # --- 4. INVOKE THE CHAIN AND MANAGE MEMORY ---
        response = await conversational_rag_chain.ainvoke({"input": request.query})
        
        # Save the current interaction to memory
        memory.save_context({"input": request.query}, {"answer": response})

        # Extract sources from the retrieved context
        retrieved_docs = compression_retriever.get_relevant_documents(request.query)
        sources = sorted(list(set(f"{doc.metadata.get('source', 'N/A')}" + (f" (Page {doc.metadata['page']})" if 'page' in doc.metadata else "") for doc in retrieved_docs)))

        return ChatResponse(answer=response, sources=sources)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during chat retrieval: {e}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)