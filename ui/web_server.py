#!/usr/bin/env python3
"""
Byte & Bite — Web UI + API proxy to RAG backend
Fix: no default min_rating; filters remain optional.

GÜNCELLENDİ:
1. Backend'i (app.py) stream=True ile çağırır.
2. /api/chat endpoint'i, cevabı JSON olarak değil, text/plain olarak stream eder.
3. JavaScript, stream'i alıp token-token gösterecek şekilde güncellendi.
"""
import time
import os
import requests
# GÜNCELLENDİ: Flask'tan 'Response' eklendi
from flask import Flask, request, jsonify, render_template_string, send_from_directory, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RAG_BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000")

MODELS = [
    ("llama3.1:8b-instruct-q4_K_M", "llama31_8b"),
    ("mistral:7b-instruct-q4_K_M", "mistral7b"),
]

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
    r = requests.post(url, json=payload, timeout=(10, 600), stream=True)
    r.raise_for_status()
    # Ham içeriği stream etmek için generator döndür
    return r.iter_content(chunk_size=1024) 

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
.message-content{flex:1;padding:12px 16px;border-radius:12px;max-width:75%;line-height:1.5;background:#fff;border:1px solid #e0e0e0; white-space: pre-wrap;} /* GÜNCELLENDİ: Plain text formatlaması için pre-wrap eklendi */
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
// (JS GÜNCELLENDİ: Streaming için)
const chatContainer = document.getElementById('chatContainer');
const messageInput  = document.getElementById('messageInput');
const sendBtn       = document.getElementById('sendBtn');
const clearBtn      = document.getElementById('clearBtn');
const modelSelect   = document.getElementById('model');

// Not: Filtre input'ları (city, cuisine, rating, topk) HTML'inizde yoktu,
// ama JS kodunuzda referans veriliyordu. 
// Hata almamak için bu referansları (cityInput vb.) şimdilik kaldırdım.
// Eğer bu input'lar varsa, 'filters' nesnesini oluşturduğum yerin altına
// eski kodunuzdaki gibi ekleyebilirsiniz.

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

// GÜNCELLENDİ: Mesajı eklerken, içeriği güncelleyebilmek için
// 'message-content' div'ini döndürür.
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
  return body; // GÜNCELLENDİ: İçerik elementini döndür
}

function addErrorMessage(content){
  addMessage('assistant', 'Error: ' + content);
}

// GÜNCELLENDİ: Streaming (Fetch API Reader) kullanmak için
// 'async function' ve tüm içerik güncellendi.
async function sendMessage(){
  const message = messageInput.value.trim();
  if (!message || sendBtn.disabled) return;

  addMessage('user', message);
  conversationHistory.push({role:'user', content: message});
  messageInput.value=''; messageInput.style.height='auto';
  sendBtn.disabled = true; messageInput.disabled = true; modelSelect.disabled = true;
  
  let fullResponse = ''; // Stream bittiğinde history'e eklemek için

  try{
    const start = Date.now();
    
    // Not: Orijinal kodunuzda cityInput, cuisineInput vb. vardı.
    // HTML'de olmadıkları için hata vermesin diye kaldırdım.
    // Gerekirse buraya ekleyin:
    const filters = {
        // city: cityInput.value || null,
        // cuisines: (cuisineInput.value || '').trim() ? [cuisineInput.value.trim()] : null,
        // min_rating: minRatingInput.value ? parseFloat(minRatingInput.value) : null,
        // top_k: topKInput ? parseInt(topKInput.value || '10', 10) : 10
    };
    
    const payload = {
      model: modelSelect.value,
      messages: conversationHistory,
      filters: filters
    };
    
    const res = await fetch('/api/chat', { 
        method:'POST', 
        headers:{'Content-Type':'application/json'}, 
        body: JSON.stringify(payload) 
    });

    // Latency (time to first byte)
    const latency = ((Date.now() - start)/1000).toFixed(2);
    
    // Model adını (kısa) al
    const model_short = modelSelect.options[modelSelect.selectedIndex].text;
    
    if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `Request failed with status ${res.status}`);
    }

    // GÜNCELLENDİ: Stream'i okumak
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    
    // Boş bir AI mesajı oluştur ve content elementini yakala
    const aiMsgContent = addMessage('assistant', '...', {model: model_short, latency});
    aiMsgContent.textContent = ''; // '...' metnini temizle

    while (true) {
        const { value, done } = await reader.read();
        if (done) break; // Stream bitti
        
        const chunk = decoder.decode(value);
        aiMsgContent.textContent += chunk; // Gelen veriyi doğrudan ekle
        fullResponse += chunk; // Tam yanıtı biriktir
        
        // Yeni içerik geldikçe en alta kaydır
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Stream bittiğinde, tam yanıtı history'e ekle
    conversationHistory.push({role:'assistant', content: fullResponse});

  }catch(err){ 
      addErrorMessage(err.message); 
      // Hata olursa, son (başarısız) AI mesajını history'den çıkar
      conversationHistory.pop();
  }
  finally{ 
      sendBtn.disabled=false; 
      messageInput.disabled=false; 
      modelSelect.disabled=false; 
      messageInput.focus(); 
  }
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

# GÜNCELLENDİ: Artık JSON değil, stream edilmiş Response döndürüyor.
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

        # GÜNCELLENDİ: Backend'den stream generator'ı al
        rag_generator = chat_rag_backend(
            messages, model_tag=model_tag, city=city, state=state, cuisines=cuisines, diets=diets,
            min_rating=min_rating, max_price=max_price, top_k=top_k
        )
        
        # GÜNCELLENDİ: Generator'ı doğrudan client'a stream et
        return Response(rag_generator, mimetype='text/plain', headers={"X-Accel-Buffering": "no"})
    
    except requests.exceptions.RequestException as e:
        # Backend (app.py) çalışmıyorsa bu hata döner
        error_msg = f"RAG backend error: {str(e)}. Make sure the RAG backend (app.py) is running on {RAG_BACKEND_URL}"
        return Response(error_msg, status=500, mimetype='text/plain')
    except Exception as e:
        # Diğer sunucu hataları
        return Response(f"Server error: {str(e)}", status=500, mimetype='text/plain')

@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({"models": [{"tag": m[0], "name": m[1]} for m in MODELS]})

if __name__ == '__main__':
    print(f"Web UI: http://0.0.0.0:{port}  |  Backend: {RAG_BACKEND_URL}")
    port = int(os.environ.get("PORT", "8080"))
    app.run(host='0.0.0.0', port=port, debug=False)
