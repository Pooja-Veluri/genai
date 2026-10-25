from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()


# 1. Initialize the local LLM
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

# 2. Create the web search tool
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
)

# 3. Create the LangChain agent
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="""
    You are a web research assistant.

    Instructions:
    1. For factual questions about the real world,
       current events, or recent information, use the
       web search tool before answering.
    2. Summarize the search results clearly.
    3. Include relevant source titles and URLs.
    4. Never invent URLs or claim you searched if you did not.
    5. If sources disagree, explain the uncertainty.
    6. If the search results are insufficient, say so.
    """,
)

# 4. Run the agent interactively
print("Web Search Agent is ready!")
print("Type 'exit' to stop.\n")

while True:
    query = input("You: ").strip()

    if query.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not query:
        continue

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})

        print("\nAgent:")
        print(response["messages"][-1].content)
        print()

    except Exception as exc:
        print(f"\nError: {exc}\n")
