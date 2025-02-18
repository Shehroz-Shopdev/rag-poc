import os
from fastapi import FastAPI, Query, HTTPException  # Added HTTPException
from dotenv import load_dotenv
from typing import List, Dict
from langchain_core.messages import HumanMessage
from dropbox_rag import DropboxRAG, chat, setup_dropbox_updates, rag_system  # Import your chatbot logic
from pydantic import BaseModel
from urllib.parse import urlparse  # Added urlparse
import requests

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Dropbox RAG Chatbot API")

def generate_base_url(page_url):
    base_url = "https://public-api.wowcher.co.uk/v1/product-detail-page-deal/"
    
    parsed_url = urlparse(page_url)
    path_parts = parsed_url.path.strip("/").split("/")
    
    if len(path_parts) >= 5:
        region_or_deal_type = path_parts[1]  # "deal" or region like "london"
        category = path_parts[2]  # e.g., "electricals", "travel", "beauty"
        subcategory = path_parts[3]  # e.g., "laptops-macbooks", "spa"
        product_id = path_parts[4]  # Numeric product ID

        # Determine AJAX request category
        if region_or_deal_type == "deal":
            # National deals and travel deals
            if category == "travel":
                ajax_category = "travel"
                page_size = 9  # Travel uses pageSize=9
            else:
                ajax_category = "national-deal"
                page_size = 14  # Default for national deals
        else:
            # Regional deals (e.g., London)
            ajax_category = region_or_deal_type
            page_size = 14  

        base_url = f"{base_url}{ajax_category}/{category}/{subcategory}/{product_id}?page=0&pageSize={1}&offset=0"
        return base_url
    else:
        return None 

class ScrapRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    question: str
    conversation_id: str  

@app.get("/")
def home():
    return {"message": "Welcome to the Wowcher Chatbot API!"}

@app.post("/query")
def query_chatbot(chat_request: ChatRequest):
    """Handles chatbot queries"""
    try:
        response = chat(chat_request.question, chat_request.conversation_id)

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