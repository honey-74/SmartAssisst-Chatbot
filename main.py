from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Setup Gemini LLM
genai.configure(api_key="MY-API-KEY")
llm_model = genai.GenerativeModel('gemini-3.8-flash')

# Setup ChromaDB & Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="faq_knowledge_base")

# Initialize the app exactly once
app = FastAPI()

# Add CORS right after initializing the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>SmartAssist Chatbot</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f4f4f9;">
        <h2 style="text-align: center;">SmartAssist Customer Support</h2>
        <div id="chatbox" style="height: 400px; background: white; border: 1px solid #ccc; border-radius: 8px; overflow-y: auto; padding: 15px; margin-bottom: 10px;"></div>
        <div style="display: flex; gap: 10px;">
            <input type="text" id="userInput" style="flex-grow: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px;" placeholder="Ask about shipping, returns, etc...">
            <button onclick="sendMessage()" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Send</button>
        </div>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('userInput');
                const chatbox = document.getElementById('chatbox');
                const msg = input.value;
                if(!msg) return;
                
                chatbox.innerHTML += `<p style="margin: 5px 0;"><b>You:</b> ${msg}</p>`;
                input.value = '';
                chatbox.scrollTop = chatbox.scrollHeight;
                
                const response = await fetch('http://127.0.0.1:8000/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                
                const data = await response.json();
                chatbox.innerHTML += `<p style="margin: 5px 0; color: #0056b3;"><b>SmartAssist:</b> ${data.reply}</p>`;
                if (data.escalate) {
                    chatbox.innerHTML += `<p style="color:red; font-style: italic; margin: 5px 0;">Transferring you to a human agent now...</p>`;
                }
                chatbox.scrollTop = chatbox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
async def chat(req: ChatRequest):
    print(f"➡️ REACHED BACKEND! User said: {req.message}")
    user_msg = req.message.lower()
    
    # 1. Intent Classification / Escalation Logic
    escalation_keywords = ["human", "agent", "angry", "manager"]
    escalate = any(word in user_msg for word in escalation_keywords)
    
    # 2. Safe RAG Retrieval
    vector = embedding_model.encode(req.message).tolist()
    results = collection.query(query_embeddings=[vector], n_results=2)
    
    if results and results.get("documents") and len(results["documents"]) > 0 and len(results["documents"][0]) > 0:
        context = " ".join(results["documents"][0])
    else:
        context = "No specific company knowledge base context found."
    
    # 3. LLM Generation
    prompt = f"You are a helpful customer support bot. Answer the user's question using ONLY this context: {context}. Question: {req.message}"
    response = llm_model.generate_content(prompt)
    return {"reply": response.text, "escalate": escalate}