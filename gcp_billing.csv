# pip install crewai langchain-openai langchain-huggingface langchain-chroma chromadb gradio pandas python-dotenv requests

import os
import smtplib
import requests
import pandas as pd
import gradio as gr

from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from crewai import Agent, Task, Crew
from crewai.tools import tool

load_dotenv(override=True)

CSV_PATH = r"C:\code\agenticai\4_crewai\ticket_helpdesk_labeled_multi_languages_english_spain_french_german.csv"
CHROMA_DIR = r"C:\code\agenticai\4_crewai\helpdesk_chroma"

GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NTFY_TOPIC = os.getenv("NTFY_URGENT_TICKETS_TOPIC")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =======================================================================
# TRANSLATION (ONE-TIME), because tickets are also in different languages
# =======================================================================

def translate_to_english(text: str, language: str) -> str:
    if language == "en":
        return text

    prompt = f"""
Translate the following text to English.
Preserve meaning. Do not summarize.

Text:
{text}
"""
    return llm.invoke(prompt).content.strip()

# =========================
# VECTOR DB
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    collection_name="helpdesk_tickets",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# =========================
# BUILD DB IF EMPTY
# =========================

df = pd.read_csv(CSV_PATH)

if vectordb._collection.count() == 0:
    print("Building Chroma DB...")

    texts, metadatas = [], []

    for _, row in df.iterrows():
        translated = translate_to_english(row["text"], row["language"])
        texts.append(translated)
        metadatas.append({
            "queue": row["queue"],
            "priority": int(row["priority"]),
            "language": row["language"],
            "subject": row["subject"]
        })

    vectordb.add_texts(texts=texts, metadatas=metadatas)

    print(f"Indexed {len(texts)} tickets.")
else:
    print("Chroma DB already exists.")

# =========================
# TOOLS
# =========================

@tool("VectorSearch")
def vector_search(query: str) -> dict:
    """Search helpdesk tickets and return priority"""
    results = vectordb.similarity_search(query, k=1)

    if not results:
        return {"found": False}

    doc = results[0]
    return {
        "found": True,
        "priority": int(doc.metadata["priority"]),
        "queue": doc.metadata["queue"],
        "matched_text": doc.page_content[:200]
    }


@tool("SendNotify")
def send_notify(message: str) -> str:
    """Send urgent ntfy push notification"""
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    requests.post(url, data=message.encode())
    return "Urgent issue notified to support team."


@tool("SendEmail")
def send_email(message: str) -> str:
    """Send Gmail email for medium priority"""

    from_email = "ekahate@gmail.com"
    to_email = "newdelthis@gmail.com"

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = "Helpdesk Ticket (Priority 2)"
    msg.attach(MIMEText(message, "plain"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_email, GMAIL_APP_PASSWORD)
    server.send_message(msg)
    server.quit()

    return "Support email sent."


@tool("ChatAck")
def chat_ack() -> str:
    """Low priority acknowledgement"""
    return "We have noted your request and will get back to you shortly."

# =========================
# AGENT (CREWAI STYLE)
# =========================

router_agent = Agent(
    role="Helpdesk Routing Agent",
    goal="Route customer tickets based on historical priority",
    backstory=(
        "You are a strict support router.\n"
        "RULES:\n"
        "1. ALWAYS call VectorSearch first\n"
        "2. If priority == 1 → SendNotify\n"
        "3. If priority == 2 → SendEmail\n"
        "4. If priority == 3 → ChatAck\n"
        "5. Do not invent priorities"
    ),
    llm=llm,
    tools=[vector_search, send_notify, send_email, chat_ack],
    verbose=True,
)

# =========================
# TASK FLOW
# =========================

def handle_ticket(user_query: str):
    if not user_query:
        return "Please enter your issue."

    routing_task = Task(
        description=f"""
User issue:
"{user_query}"

Steps:
1. Call VectorSearch(query="{user_query}")
2. Read returned priority
3. Route using correct tool
4. Return final customer-facing response
""",
        expected_output="Final support response",
        agent=router_agent
    )

    crew = Crew(
        agents=[router_agent],
        tasks=[routing_task],
        verbose=True
    )

    return str(crew.kickoff())

# =========================
# GRADIO UI
# =========================

with gr.Blocks(title="CrewAI Helpdesk") as demo:
    gr.Markdown("# CrewAI Helpdesk Support")
    gr.Markdown("Multilingual → VectorDB → Priority-based routing")

    query = gr.Textbox(
        label="Describe your issue",
        placeholder="My wireless mouse stopped working"
    )

    btn = gr.Button("Submit Ticket", variant="primary")

    output = gr.Textbox(lines=10, show_copy_button=True)

    btn.click(handle_ticket, query, output)

if __name__ == "__main__":
    demo.launch()
