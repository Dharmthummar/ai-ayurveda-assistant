from fastapi import FastAPI, WebSocket, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import base64
import json
import asyncio
from typing import Dict, Any
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Local imports
from models import PrakritiModel
from utils import (
    analyze_facial_features,
    get_dosha_description,
    get_dosha_recommendations,
    DOSHA_CHARACTERISTICS,
    body_outline_svg
)

app = FastAPI()
os.environ["GROQ_API_KEY"] = ""

# Initialize templates at the global level
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Initialize Prakriti model
prakriti_model = PrakritiModel()

# Global variables for chatbot
db = None
qa_chain = None
embeddings = None

def proc_doc():
    """Process and prepare document database for chatbot"""
    global embeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings"
    )
    
    db_directory = "chroma_db"
    if os.path.exists(db_directory):
        print("Using existing vector database")
        return
    
    print("Creating new vector database...")
    loader = DirectoryLoader('data', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    print(f"Split into {len(texts)} text chunks")
    
    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector DB created and saved to {db_directory}")

def initialize_qa_chain():
    """Initialize the Question-Answering chain"""
    global db, qa_chain, embeddings
    
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="NeuML/pubmedbert-base-embeddings"
        )
    
    db_directory = "chroma_db"
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})  # Increased from 4 to 6 for more context
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    print("Initializing Groq LLM...")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mistral-saba-24b",
        temperature=0.3,  # Slightly increased for more creative responses
        max_tokens=8192,  # Increased from 4096 to 8192
        model_kwargs={
            "top_p": 0.9,  # Increased from 0.7 to 0.9
            "presence_penalty": 0.0,  # Reduced from 0.1 to 0.0
            "frequency_penalty": 0.0  # Reduced from 0.1 to 0.0
        }
    )
    
    prompt_template = """
You are an expert Ayurvedic knowledge assistant with deep understanding of herbal remedies and traditional medicine. Your purpose is to provide helpful information from Ayurvedic texts.

When answering questions:
1. Provide comprehensive information about the topic from the context
2. Include relevant Sanskrit terms with translations when available
3. Quote specific passages from the texts when helpful
4. Provide practical applications and usage guidelines when appropriate
5. If the question is about herbal remedies, provide detailed information about relevant herbs, their properties, and uses

Context: {context}
Question: {question}

GUIDELINES:
- Use information from the provided context
- Include source citations when possible
- Be helpful and informative even if the question is broad
- Maintain a respectful and professional tone
"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    try:
        proc_doc()
        initialize_qa_chain()
        print("Application initialized successfully")
    except Exception as e:
        print(f"Error during initialization: {e}")

# API Endpoints
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat functionality"""
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            query = json.loads(data)["query"]

            if qa_chain is None:
                await websocket.send_json({"error": "The chatbot is still initializing."})
                continue

            response = await qa_chain.ainvoke({"query": query})
            answer = response['result']

            sources = []
            for doc in response['source_documents']:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'Unknown')
                })

            await websocket.send_json({"answer": answer, "sources": sources})

        except Exception as e:
            await websocket.send_json({"error": str(e)})

@app.get("/prakriti-analysis", response_class=HTMLResponse)
async def prakriti_analysis(request: Request):
    """Render page for prakriti analysis"""
    return templates.TemplateResponse("webcam.html", {
        "request": request,
        "dosha_characteristics": DOSHA_CHARACTERISTICS
    })

@app.post("/analyze-features")
async def analyze_features(data: Dict[str, Any]):
    """Analyze facial features from submitted images"""
    try:
        images = {}
        for feature, img_data in data["images"].items():
            header, encoded = img_data.split(",", 1)
            images[feature] = base64.b64decode(encoded)

        result = analyze_facial_features(
            images.get("face"),
            images.get("eyes"),
            images.get("mouth"),
            images.get("skin"),
            images.get("profile", None)
        )
        
        # Add detailed dosha characteristics
        if "dominant_dosha" in result:
            dosha = result["dominant_dosha"]
            result["detailed_characteristics"] = DOSHA_CHARACTERISTICS[dosha]
            
            # Add modern health correlations
            modern_conditions = {
                "vata": [
                    "Osteoporosis", "Chronic Fatigue", "Insomnia",
                    "Leaky Gut", "IBS", "Thyroid/Adrenal instability"
                ],
                "pitta": [
                    "Hypertension", "Inflammatory Bowel Disease (IBD)",
                    "Acid Reflux", "GERD", "Ulcers", "Estrogen Dominance"
                ],
                "kapha": [
                    "Obesity", "Diabetes", "Cardiovascular issues",
                    "Sluggish Digestion", "Hypothyroidism", "Insulin Resistance"
                ]
            }
            
            result["health_correlations"] = modern_conditions[dosha]
            
            # Add metabolic type
            metabolic_types = {
                "vata": "Fast Metabolism (Ectomorph)",
                "pitta": "Balanced Metabolism (Mesomorph)",
                "kapha": "Slow Metabolism (Endomorph)"
            }
            
            result["metabolic_type"] = metabolic_types[dosha]
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render index page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Monetization Endpoints
@app.post("/create-subscription")
async def create_subscription():
    """Endpoint for subscription creation"""
    return {"status": "success", "message": "Subscription created"}

@app.get("/premium-features")
async def premium_features():
    """Return available premium features"""
    return {
        "features": [
            "Personalized Ayurvedic consultations",
            "Detailed dosha tracking over time",
            "Customized diet and lifestyle plans",
            "Priority support"
        ],
        "pricing": {
            "monthly": "$9.99",
            "yearly": "$99.99 (16% savings)"
        }
    }

def start_app():
    """Start the FastAPI application"""
    nest_asyncio.apply()
    
    # Create necessary directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")
    
    os.makedirs(os.path.join(base_dir, "chroma_db"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    
    # Configure FastAPI
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    try:
        # Set up ngrok tunnel
        ngrok.set_auth_token("")
        ngrok_tunnel = ngrok.connect(8080)
        print('Public URL:', ngrok_tunnel.public_url)
        
        # Start the server
        uvicorn.run(app, host="127.0.0.1", port=8080)
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    start_app()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            user_message = json.loads(data)["message"]
            
            # Process user message
            if qa_chain is None:
                initialize_qa_chain()
            
            # Get response from QA chain
            result = qa_chain({"query": user_message})
            answer = result["result"]
            source_docs = result.get("source_documents", [])
            
            # Format the answer with proper HTML
            # Convert markdown-style headers to HTML headers
            formatted_answer = answer
            
            # Replace headers (### Title) with proper HTML h3 tags
            formatted_answer = re.sub(r'### (.*?)(\n|$)', r'<h3>\1</h3>\2', formatted_answer)
            formatted_answer = re.sub(r'## (.*?)(\n|$)', r'<h2>\1</h2>\2', formatted_answer)
            formatted_answer = re.sub(r'# (.*?)(\n|$)', r'<h1>\1</h1>\2', formatted_answer)
            
            # Replace bold text (**text**) with <strong> tags
            formatted_answer = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_answer)
            
            # Replace italic text (*text*) with <em> tags
            formatted_answer = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', formatted_answer)
            
            # Replace numbered lists
            lines = formatted_answer.split('\n')
            in_list = False
            for i in range(len(lines)):
                # Check for numbered list items (1. Item)
                if re.match(r'^\d+\.\s', lines[i]):
                    if not in_list:
                        lines[i] = '<ol>\n<li>' + lines[i][lines[i].find('.')+1:].strip() + '</li>'
                        in_list = True
                    else:
                        lines[i] = '<li>' + lines[i][lines[i].find('.')+1:].strip() + '</li>'
                # Check for bullet list items
                elif re.match(r'^\*\s', lines[i]) or re.match(r'^\-\s', lines[i]):
                    if not in_list:
                        lines[i] = '<ul>\n<li>' + lines[i][1:].strip() + '</li>'
                        in_list = True
                    else:
                        lines[i] = '<li>' + lines[i][1:].strip() + '</li>'
                elif in_list and lines[i].strip() == '':
                    if i > 0 and re.match(r'^\d+\.\s', lines[i-1]):
                        lines[i-1] += '\n</ol>'
                    else:
                        lines[i-1] += '\n</ul>'
                    in_list = False
            
            formatted_answer = '\n'.join(lines)
            
            # Wrap paragraphs
            paragraphs = []
            current_paragraph = []
            
            for line in formatted_answer.split('\n'):
                if line.strip() == '':
                    if current_paragraph:
                        paragraphs.append('<p>' + ' '.join(current_paragraph) + '</p>')
                        current_paragraph = []
                elif not (line.startswith('<h') or line.startswith('<ul') or line.startswith('<ol') or 
                         line.startswith('<li') or line.startswith('</ul') or line.startswith('</ol')):
                    current_paragraph.append(line)
            
            if current_paragraph:
                paragraphs.append('<p>' + ' '.join(current_paragraph) + '</p>')
            
            formatted_answer = '\n'.join(paragraphs)
            
            # Format sources in a cleaner way
            sources_text = ""
            if source_docs:
                sources_text = "<div class='sources-section'><h4>Sources:</h4><ul>"
                seen_sources = set()
                for doc in source_docs:
                    source = doc.metadata.get("source", "")
                    page = doc.metadata.get("page", "")
                    source_key = f"{source}_{page}"
                    
                    if source_key not in seen_sources and source:
                        seen_sources.add(source_key)
                        page_info = f" (Page {page})" if page else ""
                        sources_text += f"<li>{source}{page_info}</li>"
                sources_text += "</ul></div>"
            
            # Format the final response with better HTML structure
            final_answer = f"<div class='answer-section'>{formatted_answer}</div>"
            if sources_text:
                final_answer += sources_text
            
            await websocket.send_text(json.dumps({"message": final_answer}))
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {str(e)}")
        try:
            await websocket.send_text(json.dumps({"message": f"<p>I'm sorry, but I encountered an error: {str(e)}</p>"}))
        except:
            pass