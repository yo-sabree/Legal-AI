import streamlit as st
import pdfplumber
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, PegasusForConditionalGeneration, PegasusTokenizer
import spacy

MODEL_NAME = "law-ai/InLegalBERT"
EXPLAIN_MODEL = "akhilm97/pegasus_indian_legal"

tokenizer_legalbert = AutoTokenizer.from_pretrained(MODEL_NAME)
model_legalbert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nlp_legalbert = pipeline("text-classification", model=model_legalbert, tokenizer=tokenizer_legalbert)

tokenizer_explain = PegasusTokenizer.from_pretrained(EXPLAIN_MODEL)
model_explain = PegasusForConditionalGeneration.from_pretrained(EXPLAIN_MODEL)

spacy_model = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip() if text else "No text found in the PDF."

def summarize_contract(text):
    inputs = tokenizer_explain(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_explain.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    return tokenizer_explain.decode(summary_ids[0], skip_special_tokens=True)

def extract_dates(text):
    doc = spacy_model(text)
    return [ent.text for ent in doc.ents if ent.label_ == "DATE"] or ["No dates found."]

def check_compliance(contract_text):
    classification = nlp_legalbert(contract_text)
    
    if classification[0]['label'] == 'Compliant':
        return "✅ This contract is compliant with Indian laws."
    
    issue_prompt = f"Identify compliance issues in this contract under Indian laws: {contract_text}"
    correction_prompt = f"Suggest legal corrections and missing clauses for this contract under Indian laws: {contract_text}"
    
    inputs_issue = tokenizer_explain(issue_prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids_issue = model_explain.generate(inputs_issue["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    issue_explanation = tokenizer_explain.decode(summary_ids_issue[0], skip_special_tokens=True)
    
    inputs_correction = tokenizer_explain(correction_prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids_correction = model_explain.generate(inputs_correction["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    correction_suggestion = tokenizer_explain.decode(summary_ids_correction[0], skip_special_tokens=True)

    return f"❌ **Non-Compliant**\n\n📌 **Issues:** {issue_explanation}\n\n📜 **Legal Corrections:** {correction_suggestion}"

st.title("📜 Indian Legal AI Assistant")
st.sidebar.header("Upload a Contract PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    contract_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Contract Text", contract_text, height=300)

    option = st.sidebar.selectbox(
        "Select a Legal AI Feature",
        ["Check Compliance", "Summarize Contract", "Obligation & Deadline Tracking"]
    )

    if st.sidebar.button("Run Analysis"):
        if contract_text == "No text found in the PDF.":
            st.error("Uploaded PDF contains no text. Please try another file.")
        else:
            if option == "Check Compliance":
                result = check_compliance(contract_text)
            elif option == "Summarize Contract":
                result = summarize_contract(contract_text)
            elif option == "Obligation & Deadline Tracking":
                result = extract_dates(contract_text)

            st.subheader("Result")
            st.write(result)
