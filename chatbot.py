import streamlit as st
import torch
from openai import OpenAI
from transformers import BertTokenizer, BertForSequenceClassification
import os
from dotenv import load_dotenv


# Set up OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load model
model = BertForSequenceClassification.from_pretrained("fine-tuned-legal-bert-v1")
tokenizer = BertTokenizer.from_pretrained("fine-tuned-legal-bert-v1")

#Use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Clause classification
def classify_clause(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1)
    return preds.item()

# Call LLM model for clause analysis
def llm_clause_analysis(classification_label, clause):
    message = (
        f"Below is a contract clause classified as '{classification_label}':\n\n"
        f"'{clause}'"
    )
    system_prompt = """You are a legal advisor. Provide a concise, cohesive explanation linking the clause, its classification, and the listed risks. Use the exact template below:
        **Clause**: <clause>
        
        **Classification**: <label>
        
        **Key risks**: <In 5–8 bullet points max, flag any risks in the clauses>
        
        **Mitigations**: <Solutions for risks>"""

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=system_prompt,
        input=message,
    )
    return response.output_text

# Define a combined function
def clause_analyzer(clause):
    classification_result = classify_clause(clause)
    classification_label = "Audit Clause" if classification_result else "Not an Audit Clause"
    llm_response = llm_clause_analysis(classification_label, clause)
    return llm_response

# Initialize chat history
st.set_page_config(page_title="Legal Clause Analyzer", page_icon="⚖️")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Automated Contract Clause Assessment & Advice")

# Render history
for msg in st.session_state.messages:
    with st.chat_message("user" if "question" in msg else "assistant"):
        st.markdown(msg.get("question", msg.get("answer", "")))

# Chat input
user_input = st.chat_input("Paste a contract clause to get an automated risk score and concrete advice...")
if user_input:
    # show user message
    st.session_state.messages.append({"question": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # run model
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer = clause_analyzer(user_input)
        st.markdown("\n" + answer)

    # save assistant message
    st.session_state.messages.append({"answer": "\n" + answer})
