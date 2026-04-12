"""
AutoStream Conversational AI Agent
Built with LangGraph + OpenAI GPT-4o-mini
Social-to-Lead Agentic Workflow
"""

import json
import os
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ─────────────────────────────────────────────
# 1. LOAD KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base() -> str:
    """Load and format the knowledge base into a readable string for the LLM."""
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")
    with open(kb_path, "r") as f:
        kb = json.load(f)

    formatted = f"""
COMPANY: {kb['company']} – {kb['tagline']}

PRICING PLANS:
- Basic Plan: ${kb['plans']['basic']['price_monthly']}/month
  Features: {', '.join(kb['plans']['basic']['features'])}

- Pro Plan: ${kb['plans']['pro']['price_monthly']}/month
  Features: {', '.join(kb['plans']['pro']['features'])}

COMPANY POLICIES:
- Refund Policy: {kb['policies']['refund']}
- Support: {kb['policies']['support']}
- Free Trial: {kb['policies']['trial']}
- Cancellation: {kb['policies']['cancellation']}

FAQ:
"""
    for faq in kb["faq"]:
        formatted += f"- Q: {faq['question']}\n  A: {faq['answer']}\n"

    return formatted


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API call to capture a qualified lead."""
    print(f"\n{'='*50}")
    print(f"✅ LEAD CAPTURED SUCCESSFULLY!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. LANGGRAPH STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Full conversation history
    intent: Optional[str]                      # "greeting" | "inquiry" | "high_intent"
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool                      # True when actively collecting lead info


# ─────────────────────────────────────────────
# 4. LLM SETUP
# ─────────────────────────────────────────────

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

SYSTEM_PROMPT = f"""You are AutoStream's friendly and helpful sales assistant.
AutoStream is a SaaS product that provides automated video editing tools for content creators.

Here is your knowledge base – ALWAYS answer questions using ONLY this data:
{KNOWLEDGE_BASE}

YOUR BEHAVIOR RULES:
1. For greetings or small talk → respond warmly and briefly.
2. For product/pricing questions → answer accurately from the knowledge base above.
3. When a user clearly wants to sign up, try a plan, or shows buying intent → you MUST:
   - Acknowledge their interest enthusiastically
   - Ask for their Name first (one question at a time)
   - Then ask for Email
   - Then ask for their Creator Platform (e.g., YouTube, Instagram, TikTok)
   - Collect ONE piece of info per message. Do not ask for multiple things at once.
4. Do NOT make up any pricing, features, or policies not in the knowledge base.
5. Be concise and conversational, like a real chat agent.
"""


# ─────────────────────────────────────────────
# 5. INTENT DETECTION NODE
# ─────────────────────────────────────────────

def detect_intent(state: AgentState) -> AgentState:
    """Classify the latest user message into an intent category."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    intent_prompt = f"""Classify the following user message into exactly one of these intents:
- "greeting"         → casual hello, how are you, thanks, etc.
- "inquiry"          → asking about pricing, features, plans, policies
- "high_intent"      → user wants to sign up, try, buy, or start using the product

User message: "{last_user_msg}"

Reply with ONLY one word: greeting, inquiry, or high_intent"""

    response = llm.invoke([HumanMessage(content=intent_prompt)])
    detected = response.content.strip().lower()

    if detected not in ["greeting", "inquiry", "high_intent"]:
        detected = "inquiry"  # default fallback

    return {**state, "intent": detected}


# ─────────────────────────────────────────────
# 6. LEAD COLLECTION NODE
# ─────────────────────────────────────────────

def collect_lead_info(state: AgentState) -> AgentState:
    """Extract lead details (name/email/platform) from the latest user message."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # Determine what we're currently missing and try to extract it
    extract_prompt = f"""The user said: "{last_user_msg}"

We are collecting lead information step by step.
Current status:
- Name collected: {state.get('lead_name') or 'NOT YET'}
- Email collected: {state.get('lead_email') or 'NOT YET'}
- Platform collected: {state.get('lead_platform') or 'NOT YET'}

Based on the user's message, extract ONLY the next missing piece of information.
Reply in JSON format with exactly these keys: {{"name": null, "email": null, "platform": null}}
Set the value only for the field you found, keep others null.
If nothing relevant was found, return all nulls."""

    response = llm.invoke([HumanMessage(content=extract_prompt)])
    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted = json.loads(raw.strip())
    except Exception:
        extracted = {"name": None, "email": None, "platform": None}

    new_state = {**state}
    if extracted.get("name") and not state.get("lead_name"):
        new_state["lead_name"] = extracted["name"]
    if extracted.get("email") and not state.get("lead_email"):
        new_state["lead_email"] = extracted["email"]
    if extracted.get("platform") and not state.get("lead_platform"):
        new_state["lead_platform"] = extracted["platform"]

    return new_state


# ─────────────────────────────────────────────
# 7. RESPONSE GENERATION NODE
# ─────────────────────────────────────────────

def generate_response(state: AgentState) -> AgentState:
    """Generate the agent's reply based on current state."""

    # Check if we have all lead info → trigger tool
    if (
        state.get("collecting_lead")
        and state.get("lead_name")
        and state.get("lead_email")
        and state.get("lead_platform")
        and not state.get("lead_captured")
    ):
        # Trigger the mock lead capture tool
        result = mock_lead_capture(
            state["lead_name"],
            state["lead_email"],
            state["lead_platform"]
        )
        response_text = (
            f"🎉 You're all set, {state['lead_name']}! "
            f"Our team will reach out to your {state['lead_platform']} channel at {state['lead_email']} shortly. "
            f"Welcome to AutoStream – let's make your content shine! ✨"
        )
        ai_msg = AIMessage(content=response_text)
        return {
            **state,
            "messages": state["messages"] + [ai_msg],
            "lead_captured": True,
            "collecting_lead": False,
        }

    # Build conversation history for the LLM
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # If we're in lead collection mode, add a reminder
    if state.get("collecting_lead"):
        collected = []
        missing = []
        if state.get("lead_name"):
            collected.append(f"Name: {state['lead_name']}")
        else:
            missing.append("Name")
        if state.get("lead_email"):
            collected.append(f"Email: {state['lead_email']}")
        else:
            missing.append("Email")
        if state.get("lead_platform"):
            collected.append(f"Platform: {state['lead_platform']}")
        else:
            missing.append("Platform (YouTube, Instagram, TikTok, etc.)")

        reminder = f"\n[INTERNAL NOTE – do not reveal: Collected so far: {collected}. Still need: {missing[0] if missing else 'nothing'}. Ask for ONLY the next missing item.]"
        messages_for_llm.append(SystemMessage(content=reminder))

    response = llm.invoke(messages_for_llm)
    ai_msg = AIMessage(content=response.content)

    # If intent was high_intent, mark collecting_lead as True
    new_collecting = state.get("collecting_lead", False)
    if state.get("intent") == "high_intent":
        new_collecting = True

    return {
        **state,
        "messages": state["messages"] + [ai_msg],
        "collecting_lead": new_collecting,
    }


# ─────────────────────────────────────────────
# 8. ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """Decide next node after intent detection."""
    if state.get("collecting_lead") or state.get("intent") == "high_intent":
        return "collect_lead"
    return "generate_response"


# ─────────────────────────────────────────────
# 9. BUILD THE LANGGRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("collect_lead", collect_lead_info)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "collect_lead": "collect_lead",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("collect_lead", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 10. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🎬 AutoStream AI Assistant  ")
    print("  Powered by LangGraph + GPT-4o-mini")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 55)

    app = build_graph()

    # Initial state
    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("\nThanks for chatting with AutoStream! 👋")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        state = app.invoke(state)

        # Print the latest AI response
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\n🤖 AutoStream: {msg.content}")
                break

        # Exit if lead was just captured
        if state.get("lead_captured"):
            print("\n[Session complete – lead successfully captured!]")
            break


if __name__ == "__main__":
    main()