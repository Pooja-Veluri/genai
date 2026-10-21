from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


st.title("🤖 AI Q&A Bot")
st.markdown("Ask me anything! Type 'exit' or 'quit' to end the conversation.")
query = st.chat_input("Ask me anything...")
if query:
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.text)
