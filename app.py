import streamlit as st

# Import both chatbot versions
from chatbot import generate_response as ft_generate_response
from rag_chatbot import setup_rag_chain

st.set_page_config(page_title="Unified Chatbot", layout="centered")
st.title("🧠 GenAI Chatbot Interface")
st.write("Choose your chatbot type below and ask a question:")

# Dropdown to select chatbot type
chatbot_type = st.selectbox(
    "Choose the Chatbot Type:",
    ("Fine-tuned Chatbot", "RAG Chatbot (LangChain)")
)

# User input
query = st.text_input("Your Question:")

# Response logic based on chatbot type
if query:
    if chatbot_type == "Fine-tuned Chatbot":
        st.subheader("🤖 Fine-Tuned Model Response")
        response = ft_generate_response(query)
        st.write(response)

    elif chatbot_type == "RAG Chatbot (LangChain)":
        st.subheader("📚 RAG Response from LangChain")
        chain = setup_rag_chain()
        response = chain.run(query)
        st.write(response)

