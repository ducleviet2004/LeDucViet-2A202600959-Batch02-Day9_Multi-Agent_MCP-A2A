import streamlit as st
import os
import sys
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# Thêm thư mục gốc dự án vào path để import
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv()

from src.task10_generation import generate_with_citation

# Cấu hình Page
st.set_page_config(
    page_title="DrugLaw RAG - Trợ Lý Pháp Luật Ma Túy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS (Glassmorphism, Dark Mode, Modern Fonts & Glow effects)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #E2E8F0;
    }
    
    /* Dark mode background */
    .stApp {
        background: radial-gradient(circle at top right, #0F172A, #020617);
    }
    
    /* Glassmorphism sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Title Glow */
    .title-glow {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(6, 182, 212, 0.2);
        margin-bottom: 5px;
    }
    
    /* Message Cards */
    .chat-bubble-user {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin-bottom: 15px;
        color: #F1F5F9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .chat-bubble-bot {
        background: rgba(8, 47, 73, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin-bottom: 15px;
        color: #F1F5F9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Source list styling */
    .source-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        font-size: 0.85rem;
    }
    
    .source-badge {
        background: #0284C7;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Hover micro-animations */
    button[kind="secondary"] {
        transition: all 0.3s ease;
    }
    button[kind="secondary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);
    }
</style>
""", unsafe_allow_html=True)


def get_source_files():
    """Lấy danh sách các file tài liệu đã được chuẩn hóa."""
    std_dir = PROJECT_DIR / "data" / "standardized"
    legal_files = []
    news_files = []
    
    if (std_dir / "legal").exists():
        legal_files = [f.name for f in (std_dir / "legal").glob("*.md")]
    if (std_dir / "news").exists():
        news_files = [f.name for f in (std_dir / "news").glob("*.md")]
        
    return legal_files, news_files


def rewrite_query_with_memory(query: str, chat_history: list) -> str:
    """
    Sử dụng LLM viết lại câu hỏi dựa trên lịch sử để truy vấn RAG hiệu quả (Query Rewriting).
    """
    if not chat_history:
        return query

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # 1. Thử sử dụng Gemini viết lại câu truy vấn
    if gemini_key and gemini_key != "your-gemini-api-key-here" and gemini_key != "GEMINI_API_KEY":
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            
            history_str = ""
            for msg in chat_history[-3:]:
                history_str += f"{msg['role']}: {msg['content']}\n"
                
            prompt = f"""Dưới đây là lịch sử cuộc trò chuyện và câu hỏi tiếp theo của người dùng.
Hãy viết lại câu hỏi tiếp theo thành một câu truy vấn độc lập, rõ ràng bằng tiếng Việt để tìm kiếm tài liệu pháp luật.
Không trả lời câu hỏi, chỉ trả về câu truy vấn được viết lại.

Lịch sử trò chuyện:
{history_str}
Câu hỏi tiếp theo: {query}

Câu truy vấn độc lập:"""

            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            pass

    # 2. Fallback sử dụng OpenAI nếu có key hợp lệ
    if openai_key and openai_key != "sk-xxx" and openai_key != "sk-your-actual-key-here" and not openai_key.startswith("sk-proj-06qEI"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            history_str = ""
            for msg in chat_history[-3:]:
                history_str += f"{msg['role']}: {msg['content']}\n"
                
            prompt = f"""Dưới đây là lịch sử cuộc trò chuyện và câu hỏi tiếp theo của người dùng.
Hãy viết lại câu hỏi tiếp theo thành một câu truy vấn độc lập, rõ ràng bằng tiếng Việt để tìm kiếm tài liệu pháp luật.
Không trả lời câu hỏi, chỉ trả về câu truy vấn được viết lại.

Lịch sử trò chuyện:
{history_str}
Câu hỏi tiếp theo: {query}

Câu truy vấn độc lập:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten
        except Exception:
            return query
            
    return query



# --- GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='title-glow'>⚖️ DrugLaw RAG</div>", unsafe_allow_html=True)
    st.markdown("*Hệ thống Tra cứu & Trợ lý Pháp luật Ma túy thông minh*")
    st.markdown("---")
    
    # Hiển thị trạng thái các Vector Store
    st.subheader("🛠️ Cấu hình hệ thống")
    st.info("⚡ Vector Store: **ChromaDB**\n\n📌 Model: **MiniLM-L6-v2**")
    
    # Hiển thị danh sách tài liệu
    st.subheader("📚 Thư viện tài liệu")
    legal_files, news_files = get_source_files()
    
    with st.expander(f"📑 Văn bản pháp luật ({len(legal_files)})"):
        for f in legal_files:
            st.markdown(f"- `{f.replace('.md', '')}`")
            
    with st.expander(f"📰 Bài báo tin tức ({len(news_files)})"):
        for f in news_files:
            st.markdown(f"- `{f.replace('.md', '')}`")
            
    st.markdown("---")
    if st.button("🧹 Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()


# --- GIAO DIỆN CHAT CHÍNH ---
st.markdown("<h2 style='margin-top: -30px;'>💬 Chatbot Trợ Lý Pháp Luật</h2>", unsafe_allow_html=True)

# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-bubble-user">
            <b>👤 Bạn:</b><br>{msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bubble-bot">
            <b>⚖️ Trợ Lý:</b><br>{msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị nguồn của câu trả lời trước đó nếu có
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Xem nguồn tài liệu đối chiếu"):
                for src in msg["sources"]:
                    source_name = src.get("metadata", {}).get("source", "Nguồn")
                    doc_type = src.get("metadata", {}).get("type", "news")
                    badge = "VĂN BẢN LUẬT" if doc_type == "legal" else "BÁO CHÍ"
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-badge">{badge}</span> <b>{source_name}</b> (Độ trùng khớp: {src.get('score', 0):.2f})<br>
                        <i>"{src.get('content', '')[:300]}..."</i>
                    </div>
                    """, unsafe_allow_html=True)

# Xử lý nhập câu hỏi mới
if prompt := st.chat_input("Nhập câu hỏi của bạn về luật ma túy hoặc showbiz tại đây..."):
    # Hiển thị câu hỏi của User ngay lập tức
    st.markdown(f"""
    <div class="chat-bubble-user">
        <b>👤 Bạn:</b><br>{prompt}
    </div>
    """, unsafe_allow_html=True)
    
    # Lưu vào lịch sử chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("⚖️ Đang phân tích pháp luật và tìm kiếm nguồn tài liệu..."):
        # Viết lại câu hỏi sử dụng Memory nếu cần
        rewritten_query = rewrite_query_with_memory(prompt, st.session_state.messages[:-1])
        
        # Gọi RAG pipeline
        result = generate_with_citation(rewritten_query)
        answer = result["answer"]
        sources = result["sources"]
        
        # Hiển thị câu trả lời của Bot
        st.markdown(f"""
        <div class="chat-bubble-bot">
            <b>⚖️ Trợ Lý:</b><br>{answer}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            with st.expander("🔍 Xem nguồn tài liệu đối chiếu"):
                for src in sources:
                    source_name = src.get("metadata", {}).get("source", "Nguồn")
                    doc_type = src.get("metadata", {}).get("type", "news")
                    badge = "VĂN BẢN LUẬT" if doc_type == "legal" else "BÁO CHÍ"
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-badge">{badge}</span> <b>{source_name}</b> (Độ trùng khớp: {src.get('score', 0):.2f})<br>
                        <i>"{src.get('content', '')[:300]}..."</i>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Lưu vào lịch sử chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        
    st.rerun()
