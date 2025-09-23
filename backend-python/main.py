import os
import io
import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict

# --- Core Dependencies ---
import uvicorn
import httpx 
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Document Processing ---
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from langchain.docstore.document import Document

# --- LangChain Imports ---
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- CONFIGURATION ---
FAISS_PATH = "vector_store.faiss"
UPLOAD_DIRECTORY = "uploaded_files"
EMBEDDING_MODEL = "mxbai-embed-large:335m"
LLM_MODEL = "llama3"
# ## n8n INTEGRATION ##: Your n8n webhook URLs as a list
N8N_WEBHOOK_URLS = [
    "http://localhost:5678/webhook/2ffa7f0f-c3a0-4fd4-9f50-e1395c034762",
    "http://localhost:5678/webhook/4deb2b15-fea5-4d3f-b4e8-b9795290aac1" 
]

# In-memory store for our vector database and models
db_store = {}

# --- Pydantic Models ---
class DocumentAnalysis(BaseModel):
    summary: str = Field(description="A concise, professional summary of the document (around 100-150 words).")
    action_items: List[str] = Field(description="A bulleted list of the 3-5 most important, actionable tasks from the document.")
    assigned_role: str = Field(description="The single most relevant employee role to handle this document. Choose from: [Finance Manager, Customer Manager, Safety Manager,].")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up application...")
    db_store["llm"] = Ollama(model=LLM_MODEL)
    db_store["embeddings"] = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # More robust check for the actual index file
    index_file = os.path.join(FAISS_PATH, "index.faiss")

    if os.path.exists(index_file):
        print(f"Loading existing vector store from {FAISS_PATH}...")
        db_store["vector_store"] = FAISS.load_local(
            FAISS_PATH, 
            db_store["embeddings"], 
            allow_dangerous_deserialization=True 
        )
    else:
        db_store["vector_store"] = None
        print("No vector store found. A new one will be created on the first upload.")
    
    yield
    
    print("Shutting down application...")
    db_store.clear()

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="KMRL InsightEngine Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIRECTORY), name="files")

# --- DEPENDENCY ---
def get_vector_store():
    if db_store.get("vector_store") is None:
        raise HTTPException(status_code=404, detail="Knowledge Base is empty. Please upload a document first.")
    return db_store["vector_store"]

# --- HELPER FUNCTIONS ---
def process_and_chunk_text(file_content: bytes, filename: str) -> List[Document]:
    """Extracts text and splits it into chunks with detailed metadata."""
    docs = []
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension == ".pdf":
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                docs.append(Document(
                    page_content=page_text, 
                    metadata={"source": filename, "page": i + 1}
                ))
    elif file_extension in [".txt", ".md"]:
        text = file_content.decode("utf-8", errors="ignore")
        docs.append(Document(
            page_content=text, 
            metadata={"source": filename}
        ))
    else: # Handle images
        try:
            image = Image.open(io.BytesIO(file_content))
            text = pytesseract.image_to_string(image)
            if text.strip():
                docs.append(Document(
                    page_content=text, 
                    metadata={"source": filename}
                ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image '{filename}': {e}")

    if not docs:
        raise HTTPException(status_code=400, detail=f"Could not extract any text from '{filename}'.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs

async def trigger_n8n_webhooks(urls: List[str], data: dict):
    if not urls:
        print("⚠️ n8n Webhook URL list is empty. Skipping trigger.")
        return

    async def post_to_webhook(client, url, payload):
        try:
            print(f"🚀 Triggering n8n webhook: {url}")
            response = await client.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"✅ n8n webhook triggered successfully: {url}")
        except httpx.RequestError as e:
            print(f"❌ Error triggering n8n webhook {url}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred for webhook {url}: {e}")

    async with httpx.AsyncClient() as client:
        tasks = [post_to_webhook(client, url, data) for url in urls]
        await asyncio.gather(*tasks)

# --- API ENDPOINTS ---
@app.post("/api/upload-and-process", response_model=DocumentAnalysis)
async def upload_and_process_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 1. Process the document into chunks with metadata
    try:
        docs = process_and_chunk_text(content, file.filename)
        analysis_text = " ".join([doc.page_content for doc in docs[:4]])
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # 2. Perform LLM analysis
    llm = db_store["llm"]
    try:
        summary_prompt = ChatPromptTemplate.from_template("Provide a concise, professional summary (around 100-150 words) of the following document content: \n\n{document}")
        summary_chain = summary_prompt | llm
        summary_result = await summary_chain.ainvoke({"document": analysis_text})

        actions_prompt = ChatPromptTemplate.from_template("Extract the 3 to 5 most important, actionable tasks from the following document. Present them as a bulleted list. If no clear action items exist, say 'None'. \n\n{document}")
        actions_chain = actions_prompt | llm
        actions_result = await actions_chain.ainvoke({"document": analysis_text})
        action_items_list = [line.strip() for line in actions_result.split('\n') if line.strip() and "none" not in line.lower()]

        role_prompt = ChatPromptTemplate.from_template("Read the document and determine the single most relevant employee role. Choose ONLY from this list: [Finance Manager, Sales Manager, Customer Care Manager, Safety Manager, HR Coordinator, Legal Counsel, Procurement Specialist, Rolling Stock Engineer, Station Controller]. Respond with ONLY the role name. Document: \n\n{document}")
        role_chain = role_prompt | llm
        role_result = await role_chain.ainvoke({"document": analysis_text})
        
        analysis = DocumentAnalysis(
            summary=summary_result.strip(),
            action_items=action_items_list,
            assigned_role=role_result.strip().replace("'", "").replace('"', '')
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")

    # 3. Embed and save the document chunks
    vector_store = db_store.get("vector_store")
    if vector_store is None:
        db_store["vector_store"] = FAISS.from_documents(docs, db_store["embeddings"])
    else:
        vector_store.add_documents(docs)

    db_store["vector_store"].save_local(FAISS_PATH)
    print(f"✅ Successfully processed and embedded '{file.filename}'.")
    
    background_tasks.add_task(trigger_n8n_webhooks, urls=N8N_WEBHOOK_URLS, data=analysis.model_dump())
    return analysis

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_knowledge_base(request: ChatRequest, vector_store: FAISS = Depends(get_vector_store)):
    llm = db_store["llm"]
    
    prompt = ChatPromptTemplate.from_template("""
        Answer the user's question based ONLY on the provided context from various documents.
        Synthesize the information from all relevant sources to give a comprehensive answer.
        If the information is not in the context, clearly state that.
        After the answer, list all the unique source documents you used, including page numbers if available.

        Context:
        {context}
        
        Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 7}), llm=llm
    )
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        response = await retrieval_chain.ainvoke({"input": request.query})
        
        # Extract unique sources from the context
        sources = []
        if "context" in response and response["context"]:
            unique_sources = set()
            for doc in response["context"]:
                source_str = doc.metadata.get("source", "Unknown")
                if "page" in doc.metadata:
                    source_str += f" (Page {doc.metadata['page']})"
                unique_sources.add(source_str)
            sources = sorted(list(unique_sources))

        return ChatResponse(answer=response["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat retrieval: {e}")

@app.get("/api/documents", response_model=Dict[str, int])
def get_indexed_documents(vector_store: FAISS = Depends(get_vector_store)):
    """Returns a list of unique source documents and the number of chunks for each."""
    if not hasattr(vector_store, 'docstore'):
        return {}
    
    source_counts = {}
    for doc_id, doc in vector_store.docstore._dict.items():
        source = doc.metadata.get("source", "Unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        
    return source_counts

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)