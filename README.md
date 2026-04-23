# 🎬 AutoStream – Social-to-Lead AI Agent

A conversational AI agent built for **AutoStream**, a fictional SaaS company offering automated video editing tools for content creators. This agent detects user intent, answers product questions using RAG, and captures qualified leads through tool execution. 
**Stack:** Python · LangGraph · GPT-4o-mini · RAG

---

## 📁 Project Structure

```
autostream_agent/
├── agent.py                        # Main agent logic (LangGraph)
├── requirements.txt                # All dependencies
├── .env                            # Your API key (not committed)
├── .gitignore
├── README.md
└── knowledge_base/
    └── autostream_kb.json          # RAG knowledge base (pricing, plans, policies)
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key
Open the `.env` file and replace the placeholder:
```
GOOGLE_API_KEY=AIzaSy-your-actual-key-here
```

### 5. Run the Agent
```bash
python agent.py
```

---

## 💬 Sample Conversation

```
You: Hi there!
🤖 AutoStream: Hey! Welcome to AutoStream 👋 How can I help you today?

You: What plans do you offer?
🤖 AutoStream: We have two plans:
  • Basic – $29/month: 10 videos/month, 720p resolution, email support
  • Pro – $79/month: Unlimited videos, 4K, AI captions, 24/7 support
  Want to know more about either?

You: The Pro plan sounds great, I want to sign up for my YouTube channel
🤖 AutoStream: Awesome! Let's get you started. What's your name?

You: Shreshti Mittal
🤖 AutoStream: Nice to meet you, John! What's your email address?

You: shreshti@example.com
🤖 AutoStream: Perfect! And which platform do you mainly create content for?

You: YouTube
✅ LEAD CAPTURED: Shreshti Mittal, shreshti@example.com, YouTube
🤖 AutoStream: 🎉 You're all set, Shreshti! Our team will reach out shortly. Welcome to AutoStream!
```

---

## 🏗️ Architecture Explanation (~200 words)

This agent is built using **LangGraph**, a framework that models conversational workflows as a directed state graph. Each node in the graph performs a specific function, and edges (including conditional ones) route the conversation based on the current state.

**Why LangGraph over AutoGen?**  
LangGraph offers fine-grained control over the conversation flow using a typed state object. This is essential for a lead-capture workflow where we need to track partial data (name → email → platform) across multiple turns without losing context. AutoGen is powerful for multi-agent collaboration, but LangGraph is simpler and more predictable for single-agent, multi-step pipelines.

**How State is Managed:**  
A typed `AgentState` dictionary is passed through every node in the graph. It stores the full message history (using LangGraph's `add_messages` reducer to prevent duplicates), detected intent, collected lead fields, and flags like `collecting_lead` and `lead_captured`. This state persists across all conversation turns in the session, enabling the agent to remember what it has already asked and what it still needs — without re-asking or losing track.

**RAG Implementation:**  
The knowledge base (`autostream_kb.json`) is loaded once at startup, formatted into a plain-text context, and injected into the system prompt. This gives the LLM grounded, accurate answers without hallucinating pricing or features.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following integration approach would be used:

### Architecture
```
WhatsApp User
     ↓  (sends message)
WhatsApp Business API (Meta Cloud API)
     ↓  (webhook POST request)
Your Server (FastAPI / Flask endpoint)
     ↓  (calls agent)
LangGraph Agent
     ↓  (returns response)
Your Server
     ↓  (sends reply via WhatsApp API)
WhatsApp User
```

### Step-by-Step

1. **Register a WhatsApp Business App** on [Meta for Developers](https://developers.facebook.com/) and get a phone number + access token.

2. **Set up a Webhook Endpoint** using FastAPI:
```python
from fastapi import FastAPI, Request
from agent import build_graph, AgentState
from langchain_core.messages import HumanMessage

app = FastAPI()
sessions = {}  # In-memory session store (use Redis in production)

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    phone = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    user_msg = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    # Retrieve or create session
    if phone not in sessions:
        sessions[phone] = {
            "messages": [], "intent": None,
            "lead_name": None, "lead_email": None,
            "lead_platform": None, "lead_captured": False, "collecting_lead": False
        }

    sessions[phone]["messages"].append(HumanMessage(content=user_msg))
    graph = build_graph()
    sessions[phone] = graph.invoke(sessions[phone])

    # Send reply back via WhatsApp API
    reply = sessions[phone]["messages"][-1].content
    send_whatsapp_message(phone, reply)
    return {"status": "ok"}
```

3. **Register the webhook URL** in your Meta App Dashboard under WhatsApp → Configuration.

4. **Session Persistence:** Use Redis or a database (e.g., PostgreSQL) to store `AgentState` per phone number so sessions survive server restarts.

5. **Deploy** the FastAPI app on a server with HTTPS (required by Meta) — e.g., Railway, Render, or AWS EC2 with an SSL certificate.

---

## 🧪 Evaluation Checklist

| Criteria | Status |
|---|---|
| Intent detection (greeting / inquiry / high_intent) | ✅ |
| RAG from local knowledge base | ✅ |
| Lead collection (name → email → platform) | ✅ |
| Tool called only after all 3 fields collected | ✅ |
| State retained across 5–6 turns | ✅ |
| WhatsApp deployment explanation | ✅ |

---

## 🔑 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph |
| LLM | Groq |model="llama-3.3-70b-versatile"
| RAG | JSON knowledge base + system prompt injection |
| State Management | LangGraph `AgentState` (TypedDict) |
| Environment | python-dotenv |
