from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Gemini Client with a placeholder for public safety
client = genai.Client(api_key="YOUR_API_KEY_HERE")

# Setup ChromaDB & Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="faq_knowledge_base")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>SmartAssist Chatbot</title></head>
        <body style="font-family: Arial; padding: 40px;">
            <h2>SmartAssist Customer Support</h2>
            <div id="chat" style="border: 1px solid #ccc; height: 300px; overflow-y: scroll; padding: 10px; margin-bottom: 10px;"></div>
            <input type="text" id="msg" placeholder="Type your message..." style="width: 80%; padding: 8px;" />
            <button onclick="send()" style="padding: 8px 15px;">Send</button>
            <script>
                async function send() {
                    let m = document.getElementById('msg').value;
                    let chat = document.getElementById('chat');
                    chat.innerHTML += "<p><b>You:</b> " + m + "</p>";
                    document.getElementById('msg').value = "";
                    
                    let res = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: m})
                    });
                    let data = await res.json();
                    chat.innerHTML += "<p><b>SmartAssist:</b> " + data.reply + "</p>";
                    chat.scrollTop = chat.scrollHeight;
                }
            </script>
        </body>
    </html>
    """

@app.post("/chat")
async def chat(req: ChatRequest):
    print(f"➡️ REACHED BACKEND! User said: {req.message}")
    user_msg = req.message.lower()
    
    escalation_keywords = ["human", "agent", "angry", "manager"]
    escalate = any(word in user_msg for word in escalation_keywords)
    
    vector = embedding_model.encode(req.message).tolist()
    results = collection.query(query_embeddings=[vector], n_results=2)
    
    if results and results.get("documents") and len(results["documents"]) > 0 and len(results["documents"][0]) > 0:
        context = " ".join(results["documents"][0])
    else:
        context = "No specific company knowledge base context found."
    
    prompt = f"You are a helpful customer support bot. Answer the user's question using ONLY this context: {context}. Question: {req.message}"
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt
    )
    
    return {"reply": response.text, "escalate": escalate}
