"""
Streamlit Chatbot UI for Harel Insurance RAG System
Run with: streamlit run test/chatbot_ui.py
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

import streamlit as st
from src.rag import RAGPipeline, RAGConfig
from src.rag.answer_generator import GeneratorConfig

# Page config
st.set_page_config(
    page_title="Harel Insurance Chatbot",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for RTL support (Hebrew)
st.markdown("""
<style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stChatInput { direction: rtl; }
    .source-box { 
        background-color: #f0f2f6; 
        border-radius: 5px; 
        padding: 10px; 
        margin: 5px 0;
        direction: rtl;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    """Initialize RAG pipeline (cached)."""
    config = RAGConfig(
        retrieval_top_k=50,
        rerank_top_k=15,
        final_context_k=10,
        use_reranker=True,
        use_verification=False,
        use_auto_domain=False,
        generator_config=GeneratorConfig(
            provider="nebius",
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        ),
    )
    return RAGPipeline(config=config)

# Title
st.title("🏦 הראל ביטוח - צ'אטבוט")
st.markdown("שאל שאלות על ביטוחי הראל וקבל תשובות מבוססות מסמכים")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 מקורות"):
                for src in message["sources"]:
                    st.markdown(f"• [{src['file']}](https://www.harel-group.co.il/documents/{src['file']}) - עמוד {src['page']}")

# Chat input
if prompt := st.chat_input("הקלד שאלה..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("מחפש תשובה..."):
            try:
                pipeline = get_pipeline()
                response = pipeline.query(prompt, domain_filter=None)
                
                answer = response.answer
                sources = []
                for citation in response.citations:
                    source_file = citation.source_file
                    if source_file.startswith("data/"):
                        source_file = source_file.split("/")[-1]
                    sources.append({
                        "file": source_file,
                        "page": citation.page_num or 1
                    })
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("📚 מקורות"):
                        for src in sources:
                            st.markdown(f"• [{src['file']}](https://www.harel-group.co.il/documents/{src['file']}) - עמוד {src['page']}")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"שגיאה: {str(e)}")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ אודות")
    st.markdown("""
    צ'אטבוט זה משיב על שאלות בנושא ביטוחי הראל.
    
    **תחומים:**
    - ביטוח רכב
    - ביטוח דירה
    - ביטוח בריאות
    - ביטוח חיים
    - ביטוח נסיעות
    - ביטוח עסקי
    """)
    
    if st.button("🗑️ נקה היסטוריה"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"📊 **הודעות:** {len(st.session_state.messages)}")

