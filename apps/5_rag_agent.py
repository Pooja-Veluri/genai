from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# load documents
loader = PyPDFLoader("/Users/poojaveluri/Documents/genai/data/medical_report.pdf")
docs = loader.load()

# chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = splitter.split_documents(docs)

# convert chunk to embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# store embeddings in vector db
vector_store = InMemoryVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    persist_directory="./vector_rag_agent_db",
)


llm = ChatGroq(model="openai/gpt-oss-120b")


# creating tool
@tool
def retrieve_context(query: str):
    """
    Retrieve documents relavant to the query from knowledge base
    """
    docs = vector_store.similarity_search(query=query, k=4)
    context = " "
    for doc in docs:
        context = doc.page_content + "\n\n"
    return context


# define system prompt
system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the 'retrieve_context' tool for questions requiring external knowledge.
        If you don't have information for the asked question in the knowledge base tell 'Sorry! i don't have sufficient information'"""


# save memory
memory = InMemorySaver()

# agent creation = llm, tool, system_prompt
agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt,
    checkpointer=memory,
)


while True:
    query = input("User :")
    if query.lower() == "quit":
        break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "1"}},
    )
    response = result["messages"][-1].text
    print("AI :", response)
