import os
import json
import io
import hashlib
import datetime
import schedule
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import dropbox
from dropbox.files import FileMetadata, FolderMetadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from bs4 import BeautifulSoup
import pandas as pd

if not os.environ.get("GROQ_API_KEY"):
    import getpass
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = ChatGroq(model_name="llama3-8b-8192")

@dataclass
class DropboxFile:
    path: str
    name: str
    content_hash: str
    modified_time: datetime.datetime

class DropboxRAG:
    def __init__(
        self,
        access_token: str,
        vector_store_path: str = "faiss_index",
        cache_dir: str = "dropbox_cache"
    ):
        self.dbx = dropbox.Dropbox(access_token)
        self.vector_store_path = vector_store_path
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = self.load_or_create_vector_store()
    
    def load_or_create_vector_store(self) -> FAISS:
        """Enhanced vector store with better similarity settings"""
        dummy_doc = Document(
            page_content="Initial dummy document",
            metadata={"source": "dummy"}
        )
        if os.path.exists(self.vector_store_path):
            return FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        return FAISS.from_documents(
            [dummy_doc],
            self.embeddings)
    
    def list_files(self, folder_path: str = "") -> List[DropboxFile]:
        """List files in the specified Dropbox folder"""
        files = []
        try:
            result = self.dbx.files_list_folder(folder_path, recursive=True)
            while True:
                for entry in result.entries:
                    if isinstance(entry, FileMetadata):
                        files.append(DropboxFile(
                            path=entry.path_display,
                            name=entry.name,
                            content_hash=entry.content_hash,
                            modified_time=entry.server_modified
                        ))
                
                if not result.has_more:
                    break
                    
                result = self.dbx.files_list_folder_continue(result.cursor)
        
        except dropbox.exceptions.ApiError as e:
            print(f"Error listing files: {str(e)}")
        
        return files
    
    def download_file(self, file_path: str) -> str:
        """Download file from Dropbox and return local path"""
        local_path = os.path.join(self.cache_dir, os.path.basename(file_path))
        try:
            metadata, response = self.dbx.files_download(file_path)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return local_path
            
        except dropbox.exceptions.ApiError as e:
            print(f"Error downloading {file_path}: {str(e)}")
            return None

    def process_file(self, data) -> List[Document]:
        """Enhanced file processing with HTML content handling"""
        try:
                    
            documents = []
            
            if 'mainDeal' in data:
                deal = data['mainDeal']

                description_text = ""
                if deal.get('description'):
                    soup = BeautifulSoup(deal['description'], 'html.parser')
                    
                    sections = []
                    
                    if soup.find('h2'):
                        sections.append(soup.find('h2').get_text(strip=True))
                    
                    for heading in soup.find_all('h3'):
                        section_text = [heading.get_text(strip=True)]
                        
                        next_element = heading.find_next_sibling()
                        while next_element and next_element.name != 'h3':
                            if next_element.name == 'ul':
                                bullets = [f"• {li.get_text(strip=True)}" 
                                         for li in next_element.find_all('li')]
                                section_text.append("\n".join(bullets))
                            elif next_element.name == 'p':
                                section_text.append(next_element.get_text(strip=True))
                            next_element = next_element.find_next_sibling()
                        
                        sections.append("\n".join(section_text))
                    
                    for p in soup.find_all('p', recursive=False):
                        sections.append(p.get_text(strip=True))
                    
                    description_text = "\n\n".join(sections)

                product_features_text = ""
                if deal.get('highlights'):
                    for highlight in deal['highlights']:
                        soup = BeautifulSoup(highlight, 'html.parser')
                        heading = soup.find('b')
                        if heading:
                            heading_text = heading.get_text(strip=True)
                            heading.decompose()
                            detail_text = soup.get_text(strip=True)
                            product_features_text += f"{heading_text}: {detail_text}\n"

                terms_text = ""
                if deal.get('terms'):
                    valid_terms = [term for term in deal['terms'] if term.strip()]
                    terms_text = "\n".join(f"{term}" for term in valid_terms)

                sections = {
                    'title': deal.get('title', ''),
                    'headline': deal.get('headline', ''),
                    'description': description_text,
                    'product_features': product_features_text,
                    'terms': terms_text
                }
                
                for section_name, content in sections.items():
                    if content:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        
                        chunks = text_splitter.split_text(content)
                        
                        for chunk in chunks:
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    'source': 'wowcher',
                                    'section': section_name,
                                    'deal_id': deal.get('id'),
                                    'category': deal.get('category').get('name')
                                }
                            )
                            documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def query(self, query: str, k: int = 3) -> List[Document]:
        """Enhanced query method with better context retrieval"""
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=None  
        )
    
    def update_knowledge_base(
        self,
        folder_path: str = "",
        force_update: bool = False
    ) -> Tuple[int, int]:
        """Update knowledge base with new or modified files"""
        files = self.list_files(folder_path)
        
        metadata_path = os.path.join(self.cache_dir, 'metadata.txt')
        existing_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    path, hash_value = line.strip().split('\t')
                    existing_metadata[path] = hash_value
        
        updated_files = 0
        processed_files = 0
        
        new_metadata = {}
        for file in files:
            needs_update = (
                force_update or
                file.path not in existing_metadata or
                existing_metadata[file.path] != file.content_hash
            )
            
            if needs_update:
                try:
                    # Download and process file
                    local_path = self.download_file(file.path)
                    print(local_path)
                    if local_path:
                        new_docs = self.process_file(local_path)
                        print(f"Documents to add: {new_docs}")
                        self.vector_store.add_documents(new_docs)
                        
                        new_metadata[file.path] = file.content_hash
                        updated_files += 1
                        print(f"Updated: {file.name}")
                    
                        
                except Exception as e:
                    print(f"Error processing {file.name}: {str(e)}")
            else:
                new_metadata[file.path] = existing_metadata[file.path]
            
            processed_files += 1
        
        with open(metadata_path, 'w') as f:
            for path, hash_value in new_metadata.items():
                f.write(f"{path}\t{hash_value}\n")
        
        self.vector_store.save_local(self.vector_store_path)
        
        return processed_files, updated_files
    
    def query(self, query: str, k: int = 2) -> List[Document]:
        """Query the knowledge base"""
        return self.vector_store.similarity_search(query, k=k)

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
   
rag_system = DropboxRAG(DROPBOX_ACCESS_TOKEN)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information from the knowledge base."""
    retrieved_docs = rag_system.query(query, k=2)
    print(f"Retrieved documents: {retrieved_docs}")
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    retrieve_tool = {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Retrieve information from the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for"
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    llm_with_tools = llm.bind_tools([retrieve])
    
    last_message = state["messages"][-1].content.lower()
    retrieval_triggers = [
        "explain", "what", "how", "where", "when", "who",
        "tell me about", "describe", "find", "search",
        "information", "details", "show"
    ]
    
    should_retrieve = any(
        trigger in last_message 
        for trigger in retrieval_triggers
    ) or len(last_message.split()) >= 3
    
    if should_retrieve:
        query = last_message
        retrieved_info = retrieve(query)
        
        system_prompt = (
            "You are a helpful assistant. Using the following retrieved information, "
            "provide a natural and informative response to the user's query. "
            f"Retrieved information: {retrieved_info}\n\n"
            "Remember to be concise and focus on relevant details."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ]
        
        response = llm.invoke(messages)
    else:
        response = llm.invoke(state["messages"])
    
    return {"messages": [response]}

def generate(state: MessagesState):
    """Generate answer using conversation history."""
    conversation_context = []
    retrieved_info = None
    
    for message in state["messages"]:
        if isinstance(message, AIMessage):
            if hasattr(message, 'content') and isinstance(message.content, tuple):
                retrieved_info = message.content[0]
            else:
                conversation_context.append(message)
        else:
            conversation_context.append(message)
    
    system_message_content = (
        "You are a helpful assistant providing information based on a knowledge base. "
        "Provide accurate, relevant responses based on the conversation history. "
        "Be concise and specific in your answers."
    )
    
    if retrieved_info:
        system_message_content += f"\n\nRetrieved Context:\n{retrieved_info}"
    
    prompt = [SystemMessage(content=system_message_content)] + conversation_context
    response = llm.invoke(prompt)
    
    return {"messages": [response]}

def create_chat_graph():
    """Create the structured chat graph."""
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieve]))
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_respond")
    
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"}
    )
    
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()

def chat(message: str, conversation_history: List = None):
    """Chat function that maintains history and processes responses."""
    if conversation_history is None:
        conversation_history = []
    
    messages = conversation_history + [HumanMessage(content=message)]
    
    graph = create_chat_graph()
    responses = []
    
    for step in graph.stream(
        {"messages": messages},
        stream_mode="values"
    ):
        if "messages" in step and step["messages"]:
            responses.append(step["messages"][-1])
    
    if responses:
        return responses[-1]
    return None

def setup_dropbox_updates(folder_path: str = "", update_interval: int = 1):
    total, updated = rag_system.update_knowledge_base(folder_path)

    
    # Schedule periodic updates
    # schedule.every(update_interval).hours.do(update_job)
    
    # # Run scheduler in a separate thread
    # import threading
    # def run_scheduler():
    #     while True:
    #         schedule.run_pending()
    #         time.sleep(60)
    
    # scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    # scheduler_thread.start()

# def main():
#     """Main function to run the chatbot with Dropbox integration"""
#     print("\nInitializing Dropbox RAG Chatbot...")
    
#     # Setup Dropbox monitoring (optionally specify a folder_path)
#     setup_dropbox_updates(folder_path="", update_interval=1)
    
#     print("\nWelcome to the Dropbox RAG Chatbot! Type 'exit' to end the conversation.")
#     print("-----------------------------------------------------------")

#     conversation_history = []
    
#     while True:
#         user_input = input("\nYou: ").strip()
        
#         if user_input.lower() in ['exit', 'quit', 'bye']:
#             print("\nGoodbye!")
#             break
        
#         if not user_input:
#             continue
        
#         try:
#             response = chat(user_input, conversation_history)
            
#             if response:
#                 print(f"\nAssistant: {response.content}")
#                 conversation_history.extend([
#                     HumanMessage(content=user_input),
#                     response
#                 ])
                
#         except Exception as e:
#             print(f"\nError: {str(e)}")
#             print("Please try again.")

# if __name__ == "__main__":
#     main()