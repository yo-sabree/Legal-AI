import os
import fitz
import chromadb
import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_from_pdfs(pdf_folder):
    texts, metadata = [], []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, pdf_file))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    texts.append(text)
                    metadata.append({"source": pdf_file, "page": page_num})
    return texts, metadata

pdf_folder = "E:/BRiX/Legal AI"
texts, metadata = extract_text_from_pdfs(pdf_folder)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_docs")

embedding_model = SentenceTransformer("BAAI/bge-small-en").to(device)

embeddings = embedding_model.encode(texts, batch_size=16, convert_to_numpy=True, device=device)

for idx, (text, meta) in enumerate(zip(texts, metadata)):
    collection.add(ids=[str(idx)], embeddings=[embeddings[idx].tolist()], metadatas=[meta], documents=[text])

print("PDFs processed and stored in vector DB")