"""LLM
Tool - google search tool
agent
memory
streaming
web interface"""

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq  # groq llm
from langchain_tavily import TavilySearch  # google search
from langchain.agents import create_agent  # agents
from langgraph.checkpoint.memory import InMemorySaver  # memory
import streamlit as st
# import uuid

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = TavilySearch()
tools = [search]

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.history = []
# memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="you're an amazing AI agent and can search google",
)
# Building web interface

st.subheader("QuickAnswer - Answer at the speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask me anything")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages",
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].text
            space.write(message)

        st.session_state.history.append({"role": "ai", "content": message})
