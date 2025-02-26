import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from bs4 import BeautifulSoup
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from qdrant_client import models
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
import matplotlib.pyplot as plt


llm = init_chat_model("gpt-4o-mini", model_provider="openai")

memory = MemorySaver()


class RAG:
    def __init__(
        self,
        collection_name: str = "wowcher_deals",
    ):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = self.create_vector_store()
        self.llm_graph_transformer = LLMGraphTransformer(llm=llm)
        self.knowledge_graph = nx.DiGraph()
    
    def create_vector_store(self) -> Qdrant:
        """Enhanced vector store with better similarity settings"""
        dummy_doc = Document(
            page_content="Wowcher Limited is the second largest British e-commerce deal of the day site in the United Kingdom and Ireland. ",
            metadata={"source": "wikipedia"}
        )

        return Qdrant.from_documents(
            documents=[dummy_doc],
            embedding=self.embeddings,
            location=":memory:",
            collection_name=self.collection_name,
        )
    
    
    def create_documents(self, data) -> List[Document]:
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

                # Enhanced terms processing
                terms_text = ""
                if deal.get('terms'):
                    cleaned_terms = []
                    for term in deal['terms']:
                        if not term.strip():
                            continue
                        
                        # Parse and clean any HTML
                        soup = BeautifulSoup(term, 'html.parser')
                        clean_term = soup.get_text(strip=True)
                        
                        # Skip empty terms after HTML cleaning
                        if not clean_term:
                            continue
                        
                        cleaned_terms.append(clean_term)
                    
                    terms_text = "\n".join(cleaned_terms)
                base_metadata = {
                    'source': 'wowcher',
                    'deal_id': deal.get('id'),
                    'category': deal.get('category', {}).get('name'),
                    'subcategory': deal.get('subCategory', {}).get('name'),
                    'price': deal.get('price'),
                    'original_price': deal.get('originalPrice'),
                    'discount_percentage': deal.get('discountPercentage'),
                    'totalBought': deal.get('totalBought'),
                    'totalRemaining': deal.get('totalRemaining')
                }
                sections = {
                    'title': deal.get('title', ''),
                    'headline': deal.get('headline', ''),
                    'description': description_text,
                    'product_features': product_features_text,
                    'terms_and_conditions': terms_text
                }
                
                for section_name, content in sections.items():
                    if content:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=2000,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        
                        chunks = text_splitter.split_text(content)
                        
                        for chunk in chunks:
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    **base_metadata,
                                    'source': 'wowcher',
                                    'section': section_name,
                                }
                            )
                            documents.append(doc)
            
            self.update_graph(self.llm_graph_transformer.convert_to_graph_documents(documents))
            return documents
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return []

    def update_graph(self, documents):
        print(f"graph documents: {documents}")
        for graph_doc in documents:
            for node in graph_doc.nodes:
                self.knowledge_graph.add_node(node.id, node_type=node.type)
            
            for edge in graph_doc.relationships:
                self.knowledge_graph.add_edge(edge.source.id, edge.target.id, relation=edge.type)
        # self.visualize_graph(self.knowledge_graph)


    # def visualize_graph(self, knowledge_graph):
    #     plt.figure(figsize=(12, 8))
        
    #     pos = nx.spring_layout(knowledge_graph)  # Layout for visualization
    #     nx.draw(knowledge_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)

    #     edge_labels = nx.get_edge_attributes(knowledge_graph, 'relation')
    #     nx.draw_networkx_edge_labels(knowledge_graph, pos, edge_labels=edge_labels, font_size=8)
        
    #     plt.title("Knowledge Graph Visualization")
    #     plt.show()


    def query(self, query: str, deal_id: Optional[str] = None, k: int = 5) -> List[Document]:
        """Enhanced query method with better context retrieval"""
        if deal_id:
            return self.vector_store.similarity_search(
                query,
                k=k,
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.deal_id",
                            match=models.MatchValue(
                                value=int(deal_id)
                            ),
                        ),
                    ]
                ),
            )
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=None  
        )
   
rag_system = RAG()

def retrieve(query: str, deal_id: Optional[str] = None):
    """Retrieve information from the knowledge base."""
    print(f"query: {query}")
    retrieved_docs = rag_system.query(query, deal_id, k=5)
    print(f"Retrieved documents: {retrieved_docs}")
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def contextualize_query(current_query: str, state: MessagesState) -> str:
    """
    Contextualizes the current query based on conversation history.
    """
    recent_messages = state["messages"][-3:]  
    
    system_prompt = (
        "Given the conversation history and current query, create a contextualized search query. "
        "If the current query contains pronouns or references to previous topics, "
        "incorporate the relevant context into a clear, specific query."
    )
    
    context_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Conversation history:\n" + 
            "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" 
                      for m in recent_messages[:-1]]) +
            f"\n\nCurrent query: {current_query}\n\nGenerate a contextualized search query.")
    ]
    
    contextualized = llm.invoke(context_messages)
    return contextualized.content

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    
    llm_with_tools = llm.bind_tools([retrieve])
    last_message = state["messages"][-1].content.lower()

    deal_id = None
    if "///deal_id:" in last_message:
        try:
            deal_id = last_message.split("///deal_id:")[1].split()[0].strip()
            last_message = last_message.replace(f"///deal_id:{deal_id}", "").strip()
        except:
            pass
    
    retrieval_triggers = [
        "explain", "what", "how", "where", "when", "who",
        "tell me about", "describe", "find", "search",
        "information", "details", "show", "why", "how"
    ]
    
    should_retrieve = any(
        trigger in last_message 
        for trigger in retrieval_triggers
    ) or len(last_message.split()) >= 3
    
    if should_retrieve:
        query = last_message
        if not deal_id:
            query = contextualize_query(last_message, state)
        serialized, retrieved_info = retrieve(query, deal_id)

        
        if len(retrieved_info) == 0 or (len(retrieved_info) == 1 and retrieved_info[0].metadata.get("source") == "wikipedia"):
            return {"messages": [AIMessage(content="Sorry, I couldn't find relevant information for your query.")]}

        system_prompt = (
            "You are a helpful chatbot that have been trained on products from Wowcher which is an e-commerce deal of the day site. Using the following retrieved information, "
            "provide a natural and informative response to the user's query. "
            f"Retrieved information: {serialized}\n\n"
            "Remember to be concise and focus on relevant details. "
            "Also consider the previous conversation context when responding."
        )
        
        previous_messages = [
            msg for msg in state["messages"] 
            if not isinstance(msg, SystemMessage)
        ]

        print(f"retrieved_info: {retrieved_info}")
        
        messages = [
            SystemMessage(content=system_prompt)
        ] + previous_messages

        
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

    return graph_builder.compile(checkpointer=memory)

def chat(message: str, thread_id: str, deal_id: Optional[str] = None):
    """Chat function that maintains history and processes responses."""
    graph = create_chat_graph()

    if deal_id:
        message += f"///deal_id:{deal_id}"
    

    response = graph.invoke(
        {"messages": HumanMessage(content=message)}, 
        {"configurable": {"thread_id": thread_id}} 
    )
    
    if response:
        return response["messages"][-1].content
    return None


