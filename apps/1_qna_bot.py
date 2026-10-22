from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("🤖 AI Q&A Bot")
st.markdown("Ask me anything! Type 'exit' or 'quit' to end the conversation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask me anything...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    response = llm.invoke(query)
    st.chat_message("assistant").markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
