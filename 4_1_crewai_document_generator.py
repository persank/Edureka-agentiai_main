# pip install langgraph langchain-community langchain-huggingface
# pip install faiss-cpu gradio pandas python-dotenv

import os
import pandas as pd
import gradio as gr
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --------------------------------------------------
# 1. Define State
# --------------------------------------------------
class ChatState(TypedDict):
    query: str
    answer: str

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
csv_path = "c://code//agenticai//3_langgraph//Dataset_banking_chatbot.csv"
df = pd.read_csv(csv_path, encoding="latin-1")

documents = [
    Document(
        page_content=row["Query"],
        metadata={
            "Query": row["Query"],
            "Response": row["Response"]
        }
    )
    for _, row in df.iterrows()
]

# --------------------------------------------------
# 3. Embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 4. Setup FAISS (Create if not exists)
# --------------------------------------------------
faiss_dir = "c://code//agenticai//3_langgraph//banking_faiss"

if os.path.exists(faiss_dir):
    print("Loading existing FAISS index...")
    vectordb = FAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new FAISS index...")
    vectordb = FAISS.from_documents(
        documents,
        embeddings
    )
    vectordb.save_local(faiss_dir)
    print("FAISS index created and saved.")

# --------------------------------------------------
# 5. LangGraph node
# --------------------------------------------------
def retrieve_answer(state: ChatState) -> ChatState:
    results = vectordb.similarity_search_with_score(
        state["query"], k=1
    )

    if not results:
        state["answer"] = "Sorry, I could not find an answer."
        return state

    doc, distance = results[0]

    # Convert distance to similarity
    similarity = 1 - distance

    response = doc.metadata.get("Response", "")

    state["answer"] = (
        f"{response}\n\n"
        f"(Confidence: {similarity:.2f})"
    )
    return state

# --------------------------------------------------
# 6. Build LangGraph
# --------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_answer)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", END)
runnable = graph.compile()

# --------------------------------------------------
# 7. Gradio handler
# --------------------------------------------------
def chat_fn(message, history):
    return runnable.invoke({"query": message})["answer"]

# --------------------------------------------------
# 8. Gradio UI
# --------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Banking FAQ Chatbot (FAISS Vector Search)",
    examples=[
        "How can I open a new bank account?",
        "What documents are required to open an account?",
        "How do I check my balance?"
    ],
)

if __name__ == "__main__":
    demo.launch()
