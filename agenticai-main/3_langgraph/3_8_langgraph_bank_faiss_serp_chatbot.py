# pip install gradio python-dotenv langgraph langchain-community langchain-huggingface langchain-openai requests

import os
import requests
import gradio as gr
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# --------------------------------------------------
# 1. Define State
# --------------------------------------------------
class ChatState(TypedDict):
    query: str
    answer: str
    source: str   # "banking-db" or "web"
    match: float

# --------------------------------------------------
# 2. Setup Embeddings + FAISS + LLM
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS directory (ASSUMED TO EXIST)
faiss_dir = "c://code//agenticai//3_langgraph//banking_faiss"

vectordb = FAISS.load_local(
    faiss_dir,
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------
# 3. Vector Search Node (Primary)
# --------------------------------------------------
def banking_faq_search(state: ChatState) -> ChatState:
    results = vectordb.similarity_search_with_score(
        state["query"], k=1
    )

    if not results:
        state["answer"] = ""
        state["source"] = "none"
        state["match"] = 0.0
        return state

    doc, distance = results[0]

    # FAISS distance → similarity
    similarity = 1 - distance

    # Confidence threshold
    if similarity >= 0.65:
        state["answer"] = doc.metadata.get("Response", "")
        state["source"] = "banking-db"
        state["match"] = similarity
    else:
        state["answer"] = ""
        state["source"] = "web"
        state["match"] = similarity

    return state

# --------------------------------------------------
# 4. SERP + LLM Node (Fallback)
# --------------------------------------------------
def web_fallback(state: ChatState) -> ChatState:
    if state["source"] == "banking-db":
        return state

    params = {
        "q": f"{state['query']} banking",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": 3
    }

    data = requests.get(
        "https://serpapi.com/search",
        params=params,
        timeout=10
    ).json()

    snippets = [
        r.get("snippet", "")
        for r in data.get("organic_results", [])[:3]
    ]

    context = "\n".join(snippets)

    prompt = f"""
You are a banking assistant.
Answer the user query using the information below.

Query:
{state['query']}

Web information:
{context}

Answer clearly and safely.
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content
    state["source"] = "web"
    return state

# --------------------------------------------------
# 5. Build LangGraph
# --------------------------------------------------
graph = StateGraph(ChatState)

graph.add_node("faq_search", banking_faq_search)
graph.add_node("web_fallback", web_fallback)

graph.set_entry_point("faq_search")
graph.add_edge("faq_search", "web_fallback")
graph.add_edge("web_fallback", END)

chatbot = graph.compile()

# --------------------------------------------------
# 6. Gradio UI
# --------------------------------------------------
def chat_fn(message, history):
    result = chatbot.invoke({"query": message})
    return (
        f"{result['answer']}\n\n"
        f"(Source: {result['source']}, "
        f"Confidence of FAQ Search: {result['match']:.2f})"
    )

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Banking Chatbot (FAISS + SERP Fallback)",
    examples=[
        "How can I open a new bank account?",
        "What documents are needed for KYC?",
        "What is the current savings account interest rate?"
    ]
)

if __name__ == "__main__":
    demo.launch()
