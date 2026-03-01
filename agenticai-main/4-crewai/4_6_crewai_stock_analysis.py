# pip install gradio requests python-dotenv crewai

import gradio as gr
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import tool
import os

load_dotenv(override=True)

MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")

# =========================
# TOOL: REAL MARKET NEWS
# =========================

@tool
def fetch_market_news(stock_symbol: str) -> str:
    """
    Fetch real, recent market news for a stock symbol using MarketAux API.
    """
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": stock_symbol,
        "language": "en",
        "limit": 5,
        "api_token": MARKETAUX_API_KEY
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json().get("data", [])

    if not data:
        return "No recent news found."

    news_text = []
    for item in data:
        news_text.append(
            f"- {item.get('title')} ({item.get('source')})"
        )

    return "\n".join(news_text)

# =========================
# AGENTS
# =========================

news_agent = Agent(
    role="Market News Analyst",
    goal="Analyze stock using real market news",
    backstory="You analyze only provided news data. You never invent events.",
    tools=[fetch_market_news],
    llm="gpt-4o-mini",
    verbose=True
)

financial_agent = Agent(
    role="Financial Analyst",
    goal="Provide high-level financial reasoning without fabricating numbers",
    backstory="You reason qualitatively unless verified data is provided.",
    llm="gpt-4o-mini",
    verbose=True
)

# =========================
# MAIN FUNCTION
# =========================

def analyze_stock(stock_symbol: str):
    if not stock_symbol:
        return "Please enter a stock symbol."

    # ---- Tasks ----

    news_task = Task(
        description=(
            f"Use the tool to fetch REAL news for {stock_symbol}.\n"
            "Summarize sentiment based ONLY on that news.\n"
            "Do not add external facts."
        ),
        expected_output="News-based sentiment analysis grounded in fetched headlines",
        agent=news_agent
    )

    finance_task = Task(
        description=(
            f"Provide a QUALITATIVE financial perspective on {stock_symbol}.\n"
            "If numbers are unknown, explicitly say so."
        ),
        expected_output="High-level financial reasoning with uncertainty",
        agent=financial_agent
    )

    crew = Crew(
        agents=[news_agent, financial_agent],
        tasks=[news_task, finance_task],
        verbose=True
    )

    return str(crew.kickoff())

# =========================
# GRADIO UI
# =========================

with gr.Blocks(title="Stock Analysis") as demo:
    gr.Markdown("#Stock Analysis")
    gr.Markdown(
        "Uses **real market news via MarketAux** combined with qualitative analysis."
    )

    stock = gr.Textbox(label="Stock Symbol", value="AAPL")
    run = gr.Button("Analyze")
    output = gr.Textbox(lines=25, show_copy_button=True)

    run.click(analyze_stock, stock, output)

if __name__ == "__main__":
    demo.launch()
