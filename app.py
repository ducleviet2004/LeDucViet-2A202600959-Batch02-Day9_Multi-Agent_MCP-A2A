import asyncio
import os
import time
import streamlit as st
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal Multi-Agent Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: radial-gradient(circle at top right, #f8f9fa, #e9ecef);
    }
    h1 {
        color: #1e293b;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subheader {
        color: #64748b;
        font-family: 'Inter', sans-serif;
        margin-bottom: 25px;
    }
    .agent-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
    }
    .badge-active {
        background-color: #22c55e;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-inactive {
        background-color: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allowed_html=True)

# App Title
st.markdown("<h1>⚖️ Legal Multi-Agent Assistant</h1>", unsafe_allowed_html=True)
st.markdown("<div class='subheader'>Production-grade Agentic RAG Multi-Agent Assistant powered by LangGraph & A2A Protocol</div>", unsafe_allowed_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar config
st.sidebar.title("⚙️ System Settings")

customer_agent_url = st.sidebar.text_input(
    "Customer Agent Endpoint",
    value=os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
)

st.sidebar.markdown("### 🤖 Agents Status")


# Helper function to check if an agent is running
def check_agent(url, name):
    try:
        resp = httpx.get(f"{url}/.well-known/agent.json", timeout=1.0)
        if resp.status_code == 200:
            return f"<span class='badge-active'>Online</span>"
    except Exception:
        pass
    return f"<span class='badge-inactive'>Offline</span>"


# Show active agent status lights in Sidebar
registry_status = "Online"
try:
    httpx.get("http://localhost:10000/agents", timeout=1.0)
except Exception:
    registry_status = "Offline"
    
registry_badge = "<span class='badge-active'>Online</span>" if registry_status == "Online" else "<span class='badge-inactive'>Offline</span>"

st.sidebar.markdown(f"**Service Registry**: {registry_badge}", unsafe_allowed_html=True)
st.sidebar.markdown(f"**Customer Agent**: {check_agent(customer_agent_url, 'Customer')}", unsafe_allowed_html=True)
st.sidebar.markdown(f"**Law Agent**: {check_agent('http://localhost:10101', 'Law')}", unsafe_allowed_html=True)
st.sidebar.markdown(f"**Tax Agent**: {check_agent('http://localhost:10102', 'Tax')}", unsafe_allowed_html=True)
st.sidebar.markdown(f"**Compliance Agent**: {check_agent('http://localhost:10103', 'Compliance')}", unsafe_allowed_html=True)
st.sidebar.markdown(f"**RAG Agent**: {check_agent('http://localhost:10104', 'RAG')}", unsafe_allowed_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Sample Questions")
samples = [
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý như thế nào?",
    "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024?",
    "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện?",
    "If a company breaks a contract and avoids taxes, what are the consequences?"
]
for q in samples:
    if st.sidebar.button(q, key=q):
        st.session_state.messages.append({"role": "user", "content": q})
        st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# Core calling client
async def call_legal_assistant(question: str):
    async with httpx.AsyncClient(timeout=300.0) as http_client:
        card_url = f"{customer_agent_url}/.well-known/agent.json"
        card_resp = await http_client.get(card_url)
        card_resp.raise_for_status()

        from a2a.types import AgentCard, Message, Part, Role, TextPart, SendMessageRequest, MessageSendParams as MSP
        from a2a.client import A2AClient
        from uuid import uuid4

        agent_card = AgentCard.model_validate(card_resp.json())
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
            message_id=str(uuid4()),
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MSP(message=message),
        )

        response = await client.send_message(request)
        
        result_text = ""
        if hasattr(response, "root"):
            root = response.root
            if hasattr(root, "result"):
                result = root.result
                if hasattr(result, "artifacts") and result.artifacts:
                    for artifact in result.artifacts:
                        for part in artifact.parts:
                            p = part.root if hasattr(part, "root") else part
                            if hasattr(p, "text"):
                                result_text += p.text
                elif hasattr(result, "parts") and result.parts:
                    for part in result.parts:
                        p = part.root if hasattr(part, "root") else part
                        if hasattr(p, "text"):
                            result_text += p.text
        return result_text


# Render chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn ở đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    st.rerun()

# Run the query if the last message is from user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Đang gửi câu hỏi tới Customer Agent và điều phối xử lý qua giao thức A2A (có thể mất 15-30 giây)...*")
        
        start_time = time.time()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(call_legal_assistant(user_query))
            
            latency = time.time() - start_time
            
            # Show response with latency metrics
            full_response = f"{response_text}\n\n---\n*⏱️ Latency: {latency:.2f} giây*"
            message_placeholder.markdown(full_response)
            
            # Store in session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            err_msg = f"❌ **Lỗi kết nối**: {e}\n\n*Hãy chắc chắn rằng bạn đã khởi chạy hệ thống bằng `./start_all.ps1` và tất cả agent hiển thị Online ở sidebar.*"
            message_placeholder.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
