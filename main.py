import streamlit as st
import pdfplumber
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, PegasusForConditionalGeneration, \
    PegasusTokenizer, AutoModelForQuestionAnswering
import spacy
from datetime import datetime

MODEL_NAME = "law-ai/InLegalBERT"
EXPLAIN_MODEL = "akhilm97/pegasus_indian_legal"
QA_MODEL = "mrm8488/bert-mini-finetuned-squadv2"

tokenizer_legalbert = AutoTokenizer.from_pretrained(MODEL_NAME)
model_legalbert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nlp_legalbert = pipeline("text-classification", model=model_legalbert, tokenizer=tokenizer_legalbert)

tokenizer_explain = PegasusTokenizer.from_pretrained(EXPLAIN_MODEL)
model_explain = PegasusForConditionalGeneration.from_pretrained(EXPLAIN_MODEL)

tokenizer_qa = AutoTokenizer.from_pretrained(QA_MODEL)
model_qa = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
nlp_qa = pipeline("question-answering", model=model_qa, tokenizer=tokenizer_qa)

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
    today = datetime.today()
    dates = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            try:
                parsed_date = datetime.strptime(ent.text, "%d %B %Y")
                days_remaining = (parsed_date - today).days
                dates.append(
                    f"{ent.text} (⏳ {days_remaining} days remaining)" if days_remaining > 0 else f"{ent.text} (⚠️ Deadline Passed)")
            except ValueError:
                dates.append(ent.text)
    return dates if dates else ["No dates found."]


def check_compliance(contract_text):
    # Limit text length for classification
    short_text = contract_text[:512]

    # Run classification using LegalBERT
    classification = nlp_legalbert(short_text)

    if classification[0]['label'] == 'Compliant':
        return "✅ This contract is compliant with Indian laws."

    # Split contract text into smaller chunks (if > 1024 tokens)
    chunk_size = 1024
    contract_chunks = [contract_text[i:i+chunk_size] for i in range(0, len(contract_text), chunk_size)]

    issue_explanations = []
    correction_suggestions = []

    for chunk in contract_chunks:
        issue_prompt = f"Identify compliance issues in this contract under Indian laws:\n\n{chunk}"
        correction_prompt = f"Suggest legal corrections and missing clauses for this contract under Indian laws:\n\n{chunk}"

        inputs_issue = tokenizer_explain(issue_prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids_issue = model_explain.generate(inputs_issue["input_ids"], max_length=256, num_beams=5, early_stopping=True)
        issue_explanations.append(tokenizer_explain.decode(summary_ids_issue[0], skip_special_tokens=True))

        inputs_correction = tokenizer_explain(correction_prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids_correction = model_explain.generate(inputs_correction["input_ids"], max_length=256, num_beams=5, early_stopping=True)
        correction_suggestions.append(tokenizer_explain.decode(summary_ids_correction[0], skip_special_tokens=True))

    return f"❌ **Non-Compliant**\n\n📌 **Issues:** {' '.join(issue_explanations)}\n\n📜 **Legal Corrections:** {' '.join(correction_suggestions)}"



def legal_qa(question, context):
    response = nlp_qa(question=question, context=context)
    return response['answer']


st.title("📜 Indian Legal AI Assistant")
st.sidebar.header("Upload a Contract PDF")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    contract_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Contract Text", contract_text, height=300)

    option = st.sidebar.selectbox("Select a Legal AI Feature",
                                  ["Check Compliance", "Summarize Contract", "Obligation & Deadline Tracking",
                                   "Legal Q/A Chatbot"])

    if option == "Legal Q/A Chatbot":
        question = st.text_input("Ask a legal question about this contract:")
        if st.button("Get Answer"):
            if question and contract_text:
                answer = legal_qa(question, contract_text)
                st.subheader("Answer")
                st.write(answer)
            else:
                st.error("Please enter a question and upload a valid contract.")
    else:
        if st.sidebar.button("Run Analysis"):
            if contract_text == "No text found in the PDF.":
                st.error("Uploaded PDF contains no text. Please try another file.")
            else:
                result = {
                    "Check Compliance": check_compliance(contract_text),
                    "Summarize Contract": summarize_contract(contract_text),
                    "Obligation & Deadline Tracking": extract_dates(contract_text)
                }.get(option, "Invalid Selection")
                st.subheader("Result")
                st.write(result)
