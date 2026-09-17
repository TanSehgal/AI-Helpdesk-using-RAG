"""Streamlit frontend for the AI Campus Helpdesk.

Talks to the FastAPI backend over HTTP. Keeps chat history in
st.session_state (no database needed for this prototype).
"""
import os
import sys

import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

st.set_page_config(page_title="AI Campus Helpdesk", page_icon="🎓", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar ---
with st.sidebar:
    st.header("About this project")
    st.markdown(
        "This is a **research prototype** chatbot for a fictional university, "
        "**Greenfield Institute of Technology (GIT)**. All campus information "
        "shown here is sample data created for this project, not a real "
        "institution."
    )

    st.subheader("What is RAG?")
    st.markdown(
        "**Retrieval-Augmented Generation (RAG)** retrieves relevant text chunks "
        "from a knowledge base before asking the LLM to answer. This grounds "
        "answers in real documents instead of the model's memory, reducing "
        "hallucination."
    )

    st.subheader("Multi-Agent Architecture")
    st.markdown(
        "A **Router Agent** reads your question and decides which department "
        "it belongs to. The question is then handed to a specialized "
        "**Department Agent**, which searches only that department's "
        "knowledge base and generates a grounded answer."
    )

    st.subheader("Available Departments")
    st.markdown(
        "- **Academic** — courses, attendance, exams, syllabus\n"
        "- **Hostel** — applications, rooms, mess, fees\n"
        "- **Administration** — fees, scholarships, certificates\n"
        "- **Student Services** — library, ID cards, transport, IT support"
    )

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main area ---
st.title("🎓 AI Campus Helpdesk")
st.caption(
    "Ask a campus-related question. Answers are generated using Retrieval-Augmented "
    "Generation over a fictional university's knowledge base — this is a research "
    "prototype, not an official university system."
)

question = st.text_input("Ask a question", placeholder="e.g. What documents are required for hostel admission?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Routing your question and retrieving answer..."):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
            if response.status_code == 200:
                data = response.json()
                st.session_state.chat_history.append({"question": question, "response": data, "error": None})
            else:
                detail = response.json().get("detail", response.text)
                st.session_state.chat_history.append({"question": question, "response": None, "error": detail})
        except requests.exceptions.ConnectionError:
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "response": None,
                    "error": "Could not connect to the backend API. Is it running? "
                    "Start it with: uvicorn app.main:app --reload",
                }
            )
        except requests.exceptions.Timeout:
            st.session_state.chat_history.append(
                {"question": question, "response": None, "error": "The request timed out. The LLM may be slow to respond."}
            )
elif ask_clicked:
    st.warning("Please enter a question before clicking Ask.")

st.divider()

for entry in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 Student:** {entry['question']}")

    if entry["error"]:
        st.error(f"Error: {entry['error']}")
    else:
        data = entry["response"]
        st.markdown(f"**🤖 Answer:** {data['answer']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Department", data["department"])
        col2.metric("Agent", data["agent"])
        col3.metric("Response time", f"{data['response_time']}s")

        if not data.get("grounded", True):
            st.info(
                "ℹ️ This answer could not be fully grounded in the campus knowledge base. "
                "Please verify with the relevant department."
            )

        if data["sources"]:
            with st.expander(f"📄 Sources ({len(data['sources'])})"):
                for src in data["sources"]:
                    st.markdown(f"- `{src}`")
                for doc in data["retrieved_documents"]:
                    st.caption(f"**{doc['document_name']}** ({doc['department']})")
                    st.text(doc["text_snippet"])
        else:
            st.caption("No source documents were retrieved for this answer.")

    st.divider()

st.caption(
    "⚠️ Disclaimer: This chatbot answers based only on the fictional sample campus "
    "knowledge base loaded into this prototype. It may not have information on every "
    "topic, and will say so rather than inventing an answer."
)
