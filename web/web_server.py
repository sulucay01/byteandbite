#!/usr/bin/env python3
"""
Byte & Bite — Web UI + API proxy to RAG backend
Fix: no default min_rating; filters remain optional.
"""
import time
import os
import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000")

MODELS = [
    ("llama3.1:8b-instruct-q4_K_M", "llama31_8b"),
    ("mistral:7b-instruct-q4_K_M", "mistral7b"),
]

<<<<<<< Updated upstream
def chat_rag_backend(query: str, *, city=None, state=None, cuisines=None, diets=None, min_rating=None, max_price=None, top_k=10):
    url = f"{RAG_BACKEND_URL}/chat"
    params = {"q": query, "top_k": top_k}
    if city: params["city"] = city
    if state: params["state"] = state
    if cuisines: 
        for c in cuisines: params.setdefault("cuisines", []).append(c)
    if diets:
        for d in diets: params.setdefault("diets", []).append(d)
    if min_rating is not None: params["min_rating"] = min_rating
    if max_price  is not None: params["max_price"]  = max_price
    r = requests.get(url, params=params, timeout=600)
=======
# GÜNCELLENDİ: Stream için backend çağrısı güncellendi
def chat_rag_backend(messages: list, *, model_tag: str, city=None, state=None, cuisines=None, diets=None, min_rating=None, max_price=None, top_k=10):
    url = f"{RAG_BACKEND_URL}/chat"
    
    # Construct payload for POST request
    payload = {
        "messages": messages,
        "model": model_tag,
        "filters": {
            "city": city,
            "state": state,
            "cuisines": cuisines,
            "diets": diets,
            "min_rating": min_rating,
            "max_price": max_price,
            "top_k": top_k
        }
    }
    
    # Use POST with stream=True (backend handles contextualization)
    r = requests.post(url, json=payload, timeout=600, stream=True)
>>>>>>> Stashed changes
    r.raise_for_status()
    return r.json()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Byte & Bite</title>
<link rel="icon" type="image/jpeg" href="/logo">
<style>
/* styles condensed */
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,sans-serif;background:linear-gradient(135deg,#1e3257 0%,#fd7370 100%);min-height:100vh;padding:20px}
.container{max-width:900px;margin:0 auto;background:#fff;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);padding:30px;display:flex;flex-direction:column;height:calc(100vh - 40px);max-height:900px}
.logo-container{text-align:center;margin-bottom:20px}
.logo{max-width:150px;max-height:150px;margin-bottom:15px}
h1{color:#1e3257;margin-bottom:10px;text-align:center;font-size:32px;font-weight:700}
.subtitle{color:#666;text-align:center;margin-bottom:20px;font-size:14px}
.header-controls{display:flex;gap:10px;margin-bottom:12px;align-items:center;flex-wrap:wrap}
.header-controls label{color:#333;font-weight:500;margin-right:8px}
.header-controls select,.header-controls input{padding:10px;border:2px solid #e0e0e0;border-radius:8px;font-size:14px;font-family:inherit;transition:border-color .3s}
.header-controls select:focus,.header-controls input:focus{outline:none;border-color:#fd7370}
.clear-btn{padding:10px 20px;background:#fd7370;color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
.chat-container{flex:1;overflow-y:auto;padding:20px;background:#f8f9fa;border-radius:12px;margin-bottom:20px;display:flex;flex-direction:column;gap:16px}
.message{display:flex;gap:12px}
.message.user{flex-direction:row-reverse}
.message-avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:16px}
.message.user .message-avatar{background:linear-gradient(135deg,#1e3257 0%,#fd7370 100%);color:#fff}
.message.assistant .message-avatar{background:#fd7370;color:#fff}
.message-content{flex:1;padding:12px 16px;border-radius:12px;max-width:75%;line-height:1.5;background:#fff;border:1px solid #e0e0e0}
.message.user .message-content{background:linear-gradient(135deg,#1e3257 0%,#fd7370 100%);color:#fff;border:none}
.message-meta{font-size:11px;color:#888;margin-top:4px}
.input-container{display:flex;gap:10px;align-items:flex-end}
textarea{width:100%;padding:12px;border:2px solid #e0e0e0;border-radius:8px;font-size:16px;resize:none;min-height:50px;max-height:120px}
.send-btn{padding:12px 24px;background:linear-gradient(135deg,#1e3257 0%,#fd7370 100%);color:#fff;border:none;border-radius:8px;font-weight:600;cursor:pointer}
</style>
</head>
<body>
<div class="container">
  <div class="logo-container"><img src="/logo" alt="Byte & Bite Logo" class="logo"></div>
  <h1>Byte & Bite</h1>
  <p class="subtitle">Ask questions about food, restaurants, and dining</p>

  <div class="header-controls">
    <label>Model:</label>
    <select id="model">
      <option value="llama3.1:8b-instruct-q4_K_M">Llama 3.1 8B</option>
      <option value="mistral:7b-instruct-q4_K_M">Mistral 7B</option>
    </select>
    <button type="button" class="clear-btn" id="clearBtn">Clear Chat</button>
  </div>


  <div class="chat-container" id="chatContainer"></div>

  <div class="input-container">
    <textarea id="messageInput" placeholder="Type your message here..." rows="1"></textarea>
    <button type="button" id="sendBtn" class="send-btn">Send</button>
  </div>
</div>

<script>
const chatContainer = document.getElementById('chatContainer');
const messageInput  = document.getElementById('messageInput');
const sendBtn       = document.getElementById('sendBtn');
const clearBtn      = document.getElementById('clearBtn');
const modelSelect   = document.getElementById('model');
const cityInput     = document.getElementById('cityInput');
const cuisineInput  = document.getElementById('cuisineInput');
const minRatingInput= document.getElementById('minRatingInput');
const topKInput     = document.getElementById('topKInput');

let conversationHistory = [];

messageInput.addEventListener('input', function(){
  this.style.height = 'auto';
  this.style.height = this.scrollHeight + 'px';
});
messageInput.addEventListener('keydown', (e)=>{
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', ()=>{
  if (confirm('Clear chat?')) { chatContainer.innerHTML=''; conversationHistory=[]; messageInput.focus(); }
});

function addMessage(role, content, meta={}){
  const el = document.createElement('div');
  el.className = `message ${role}`;
  const av = document.createElement('div'); av.className='message-avatar'; av.textContent = role === 'user' ? 'You' : 'AI';
  const body = document.createElement('div'); body.className='message-content'; body.textContent = content;
  if (meta.latency || meta.model){
    const md = document.createElement('div'); md.className='message-meta';
    const arr=[]; if (meta.model) arr.push(`Model: ${meta.model}`); if (meta.latency) arr.push(`Latency: ${meta.latency}s`);
    md.textContent = arr.join(' • '); body.appendChild(md);
  }
  el.appendChild(av); el.appendChild(body); chatContainer.appendChild(el);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addErrorMessage(content){
  addMessage('assistant', 'Error: ' + content);
}

async function sendMessage(){
  const message = messageInput.value.trim();
  if (!message || sendBtn.disabled) return;

  addMessage('user', message);
  conversationHistory.push({role:'user', content: message});
  messageInput.value=''; messageInput.style.height='auto';
  sendBtn.disabled = true; messageInput.disabled = true; modelSelect.disabled = true;

  try{
    const start = Date.now();
    const filters = {};
    if (cityInput) filters.city = cityInput.value || null;
    if (cuisineInput) filters.cuisines = (cuisineInput.value || '').trim() ? [cuisineInput.value.trim()] : null;
    if (minRatingInput) filters.min_rating = minRatingInput.value ? parseFloat(minRatingInput.value) : null;
    filters.top_k = topKInput ? parseInt(topKInput.value || '10', 10) : 10;
    
    const payload = {
      model: modelSelect.value,
      messages: conversationHistory,
      filters: filters
    };
    const res = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    const latency = ((Date.now() - start)/1000).toFixed(2);

    if (data.error) addErrorMessage(data.error);
    else {
      addMessage('assistant', data.answer, {model: data.model_name, latency});
      conversationHistory.push({role:'assistant', content: data.answer});
    }
  }catch(err){ addErrorMessage(err.message); }
  finally{ sendBtn.disabled=false; messageInput.disabled=false; modelSelect.disabled=false; messageInput.focus(); }
}
messageInput.focus();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/logo')
@app.route('/favicon.ico')
def logo():
    return send_from_directory(os.path.dirname(__file__), 'byte&bite.jpeg', mimetype='image/jpeg')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.json or {}
        model_tag = data.get('model')
        messages  = data.get('messages', [])
        filters   = data.get('filters', {})

        valid_models = [m[0] for m in MODELS]
        if model_tag not in valid_models:
            return jsonify({"error": f"Invalid model. Must be one of: {', '.join(valid_models)}"}), 400
        if not messages or messages[-1].get('role') != 'user':
            return jsonify({"error": "Last message must be from user"}), 400

        user_query = messages[-1].get('content', '')
        if not user_query:
            return jsonify({"error": "Empty user message"}), 400

        city       = filters.get('city')
        state      = filters.get('state')
        cuisines   = filters.get('cuisines') or None
        diets      = filters.get('diets') or None
        min_rating = filters.get('min_rating', None)
        max_price  = filters.get('max_price', None)
        top_k      = int(filters.get('top_k', 10))

<<<<<<< Updated upstream
        start_time   = time.time()
        rag_response = chat_rag_backend(
            user_query, city=city, state=state, cuisines=cuisines, diets=diets,
=======
        # GÜNCELLENDİ: Backend'den stream generator'ı al
        rag_generator = chat_rag_backend(
            messages, model_tag=model_tag, city=city, state=state, cuisines=cuisines, diets=diets,
>>>>>>> Stashed changes
            min_rating=min_rating, max_price=max_price, top_k=top_k
        )
        latency = time.time() - start_time
        model_short = next(m[1] for m in MODELS if m[0] == model_tag)
        answer = rag_response.get('answer', 'No answer received')

        return jsonify({
            "answer": answer,
            "model_name": model_short,
            "latency_sec": round(latency, 3),
            "answer_words": len(answer.split()),
            "sources": rag_response.get('used_sources', [])
        })
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"RAG backend error: {str(e)}. Make sure the RAG backend (app.py) is running on {RAG_BACKEND_URL}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({"models": [{"tag": m[0], "name": m[1]} for m in MODELS]})

if __name__ == '__main__':
    print("Web UI: http://localhost:5000  |  Backend:", RAG_BACKEND_URL)
    app.run(host='0.0.0.0', port=5000, debug=True)
