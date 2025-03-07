import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_docs")

embedding_model = SentenceTransformer("BAAI/bge-small-en")

def retrieve_relevant_docs(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n\n".join(results["documents"][0]) if results["documents"] else ""

def generate_legal_answer(query, context):
    prompt = f"Context: {context}\n\nQuery: {query}\n\nProvide a structured legal response with citations."
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].split("</think>", 1)[-1].strip()

st.title("Legal RAG AI")
query = st.text_area("Enter your legal question")

if st.button("Get Answer"):
    with st.spinner("Retrieving relevant laws..."):
        context = retrieve_relevant_docs(query)
        if context:
            answer = generate_legal_answer(query, context)
            st.success("Response:")
            st.write(answer)
        else:
            st.error("No relevant legal information found.")