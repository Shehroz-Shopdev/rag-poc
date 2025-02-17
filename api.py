from fastapi import FastAPI, Query, HTTPException  # Added HTTPException
from dotenv import load_dotenv
from rag import chat, rag_system
from pydantic import BaseModel
import requests
from utils.generate_url import generate_base_url
# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Dropbox RAG Chatbot API")

class ScrapRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return {"message": "Welcome to the Dropbox RAG Chatbot API!"}

@app.get("/query")
def query_chatbot(question: str = Query(..., title="User Query")):
    """Handles chatbot queries"""
    response = chat(question)
    return {"response": response.content if response else "No response"}

@app.post("/scrap")
async def scrap_url(request: ScrapRequest):
    try:
        base_url = generate_base_url(request.url)
        if not base_url:
            raise HTTPException(status_code=400, detail="Invalid URL provided")
        
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        
        documents = rag_system.process_file(data)  
        rag_system.vector_store.add_documents(documents)
        
        # rag_system.vector_store.save_local(rag_system.vector_store_path)
        
        return {"status": "success", "message": "Data added to knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)