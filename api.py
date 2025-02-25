import os
from fastapi import FastAPI, Query, HTTPException  
from dotenv import load_dotenv
from typing import List, Dict
from langchain_core.messages import HumanMessage
from rag import RAG, chat, rag_system  
from generate_base_url import generate_base_url
from pydantic import BaseModel
import requests

load_dotenv()

app = FastAPI(title="Wowcher RAG Chatbot API")


class ScrapRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    question: str
    conversation_id: str  
    deal_id: int

@app.get("/")
def home():
    return {"message": "Welcome to the Wowcher Chatbot API!"}

@app.post("/query")
def query_chatbot(chat_request: ChatRequest):
    """Handles chatbot queries"""
    try:
        response = chat(chat_request.question, chat_request.conversation_id, chat_request.deal_id)

        if response:   
            return {
                "response": response,
            }
        else:
            return {
                "response": "No response",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
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
                
        return {"status": "success", "message": "Data added to knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)