import streamlit as st
import httpx
import asyncio
import ollama
import torch
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Check for GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# Constants
SUMMARIZER_MODEL = "deepseek-r1:1.5b"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_CASE_TEXT_LENGTH = 9500
MAX_CONCURRENT_REQUESTS = 50  # Reduce to avoid rate limits

executor = ThreadPoolExecutor(max_workers=20)  # Increased parallel processing

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def fetch_url(url, client):
    """Fetches a URL using an async client with controlled concurrency."""
    async with semaphore:
        try:
            response = await client.get(url, headers=HEADERS, follow_redirects=True)
            return response.text if response.status_code == 200 else None
        except httpx.RequestError:
            return None

async def search_indiankanoon(query, client):
    """Search Indian Kanoon for cases based on a query using persistent client."""
    search_url = f"https://indiankanoon.org/search/?formInput={query}"
    html = await fetch_url(search_url, client)

    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")  # Faster parser
    return [
        {"title": link.text.strip(), "url": "https://indiankanoon.org" + link["href"]}
        for link in soup.select(".result_title a")
    ]

async def scrape_case(url, client):
    """Scrapes case details from Indian Kanoon using persistent client."""
    html = await fetch_url(url, client)

    if not html:
        return {"title": "Unknown", "text": "Failed to fetch case details.", "url": url}

    soup = BeautifulSoup(html, "lxml")

    # Optimized parsing: select_one is faster than looping
    fragment = soup.select_one(".expanded_headline .fragment")
    if not fragment:
        return {"title": "Unknown", "text": "No case text found.", "url": url}

    case_text = " ".join(p.get_text(separator=" ", strip=True) for p in fragment.find_all("p"))[:MAX_CASE_TEXT_LENGTH]
    return {"title": "Unknown", "text": case_text, "url": url}

def summarize_text(text):
    """Summarizes case text using the LLM model on GPU."""
    prompt = f"""
    You are an Indian legal AI assistant. Summarize the following court case with high accuracy includes insights 
    for lawyers and dates, times, keypoints, issue, takeaway, etc.

    Case Text: {text}...
    """

    try:
        response = ollama.chat(
            model=SUMMARIZER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"device": device},
            stream=True
        )
        return response["message"]["content"].split("</think>", 1)[-1].strip()
    except Exception as e:
        return f"Error in summarization: {str(e)}"

async def process_case(case, client):
    """Processes a single case asynchronously without redundant fetching."""
    case_data = await scrape_case(case["url"], client)

    if not case_data["text"] or "Failed to fetch" in case_data["text"]:
        return {"title": case["title"], "summary": "Summary unavailable due to missing case text."}

    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(executor, summarize_text, case_data["text"])
    return {"title": case["title"], "summary": summary}

async def fetch_and_process_cases(query):
    """Fetches case references and summarizes them asynchronously using a shared HTTP client."""
    async with httpx.AsyncClient(timeout=10) as client:
        cases = await search_indiankanoon(query, client)
        if not cases:
            return None

        tasks = [process_case(case, client) for case in cases]
        return await asyncio.gather(*tasks)

def run_async_task(query):
    """Runs an async task inside a synchronous function for Streamlit."""
    return asyncio.run(fetch_and_process_cases(query))

# Streamlit UI
st.title("Indian Legal AI - Case Reference & Summarization")

query = st.text_input("Enter your legal query:")
if st.button("Search"):
    with st.spinner("Fetching results..."):
        results = run_async_task(query)  # Run async function synchronously

    if not results:
        st.error("No results found.")
    else:
        summaries = [f"### {case['title']}\n\n{case['summary']}\n\n" for case in results]
        st.write("### Overall Insights for Lawyers:")
        st.info("\n".join(summaries))
