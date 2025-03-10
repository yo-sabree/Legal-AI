import streamlit as st
import requests
from bs4 import BeautifulSoup
import ollama

SUMMARIZER_MODEL = "deepseek-r1:1.5b"

def search_indiankanoon(query):
    """Search Indian Kanoon for cases related to the query."""
    search_url = f"https://indiankanoon.org/search/?formInput={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    # Using a more reliable selector for links to cases
    for link in soup.select("a[href*='/doc/']")[:10]:  
        url = "https://indiankanoon.org" + link["href"]
        title = link.get_text(strip=True)
        results.append({"title": title, "url": url})

    return results

def scrape_case(url):
    """Scrape the case text from the given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"title": "Unknown", "text": "Failed to fetch case details.", "url": url}

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract case title
    title = soup.find("title").get_text(strip=True) if soup.find("title") else "Unknown"

    # Extract main case text
    paragraphs = []
    for fragment in soup.find_all("div", class_="judgments"):
        for p in fragment.find_all("p"):
            paragraphs.append(p.get_text(separator=" ", strip=True))

    case_text = " ".join(paragraphs) if paragraphs else "No case text found."

    return {"title": title, "text": case_text[:10000], "url": url}

def summarize_text(text):
    """Summarize the extracted case text using the Ollama AI model."""
    prompt = f"""
    Summarize the following legal case, ensuring to include:
    - *Key Dates* (case filed, judgment date).
    - *Laws, Acts, Articles referenced*.
    - *Main legal issue*.
    - *Court's decision & reasoning*.
    - *Relevance to future similar cases*.
    - *Practical insights for lawyers handling similar cases*.
    - *Do not add texts like  Let me know if you need further elaboration or clarification or  Here is a structured legal summary of the case based on your query.*

    Case Text: {text}
    """

    response = ollama.chat(model=SUMMARIZER_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Streamlit UI
st.title("Indian Legal AI - Case Reference & Summarization")

query = st.text_input("Enter your legal query:")
if st.button("Search"):
    with st.spinner("Fetching results..."):
        cases = search_indiankanoon(query)

        if not cases:
            st.error("No results found.")
        else:
            structured_cases = []
            all_summaries = ""

            for case in cases:
                case_data = scrape_case(case["url"])
                structured_cases.append(case_data)

                with st.spinner(f"Summarizing: {case_data['title']}"):
                    summary = summarize_text(case_data["text"])
                    all_summaries += f"### {case_data['title']}\n\n{summary}\n\n"

            st.write("### Overall Insights/Summary:")
            st.info(all_summaries)
