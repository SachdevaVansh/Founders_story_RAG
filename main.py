import streamlit as st
import requests
import numpy as np
import wikipedia
import os

from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from euriai.langchain import EuriaiChatModel
from dotenv import load_dotenv

# -------------------------------
# Load API key
# -------------------------------
load_dotenv()
api_key = os.getenv("EURI_API_KEY")

# -------------------------------
# Embedding wrapper
# -------------------------------
def generate_embeddings(text):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {"input": text, "model": "text-embedding-3-small"}
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    embedding = np.array(data['data'][0]['embedding'])
    return embedding

class EuriEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [generate_embeddings(t).tolist() for t in texts]

    def embed_query(self, text):
        return generate_embeddings(text).tolist()

embeddings = EuriEmbeddings()

# -------------------------------
# LLM setup
# -------------------------------
chat_model = EuriaiChatModel(
    api_key=api_key,
    model="gpt-4.1-nano"
)

# -------------------------------
# Tools
# -------------------------------
def summarizer_tool(text):
    resp = chat_model.invoke([
        {"role": "system", "content": "Summarize this text"},
        {"role": "user", "content": f"Summarize:\n{text}"}
    ])
    return resp.content

def wikipedia_tool(query):
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {str(e)}"

def rag_tool(query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=False
    )
    return qa_chain.run(query)

def translate_tool(text_and_language):
    try:
        parts = text_and_language.split("||")
        text = parts[0].strip()
        target_language = parts[1].strip()
    except:
        return "Invalid input. Format: 'Hello world || French'"

    resp = chat_model.invoke([
        {"role": "system", "content": "You translate the text"},
        {"role": "user", "content": f"Translate '{text}' from English to {target_language}"}
    ])
    return resp.content

# -------------------------------
# Retriever
# -------------------------------
loader = TextLoader("data/founder.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

faiss_index = FAISS.from_documents(chunks, embeddings)
retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

tools = [
    Tool(name="Summarizer", func=summarizer_tool, description="Summarizes text"),
    Tool(name="Wikipedia", func=wikipedia_tool, description="Fetches info from Wikipedia"),
    Tool(name="RAG", func=rag_tool, description="Answer from founder.txt using RAG"),
    Tool(name="Translate", func=translate_tool, description="Translate text into target language"),
]

# -------------------------------
# Specialized Agents
# -------------------------------
from langchain.agents import initialize_agent

class SpecializedAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = chat_model
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=False,
            handle_parsing_errors=True
        )
        self.system_prompt = system_prompt

    def run(self, user_input):
        try:
            return self.agent.run(f"{self.system_prompt}\n{user_input}")
        except Exception as e:
            return f"Error: {str(e)}"

class RouterAgent:
    def __init__(self, agents):
        self.llm = chat_model
        self.agents = agents

    def route(self, user_input):
        system_prompt = (
            "You are a router agent. Given a user query, decide which agent should handle it: "
            "'Researcher' or 'Teacher'. Reply with only the name."
        )
        prompt = f"{system_prompt}\nUser query: {user_input}\nAgent:"
        ai_message = self.llm.invoke(prompt)
        agent_name = ai_message.content.strip().split()[0].capitalize()
        if agent_name not in self.agents:
            agent_name = "Teacher"
        return agent_name

    def chat(self, user_input):
        agent_name = self.route(user_input)
        response = self.agents[agent_name].run(user_input)
        return agent_name, response

agents = {
    "Researcher": SpecializedAgent(
        "Researcher",
        "You are a research assistant. Be factual, concise, and use RAG for founder/startup questions."
    ),
    "Teacher": SpecializedAgent(
        "Teacher",
        "You are a teacher. Explain clearly, teach concepts, and only use RAG for founder/startup questions."
    )
}

router = RouterAgent(agents)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Multi-Agent Chat", page_icon="🤖", layout="wide")
st.title("🤖 Connected Multi-Agent Chat")
st.write("Agents available: **Researcher** and **Teacher**. The router decides automatically.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question here...")
if user_input:
    agent, response = router.chat(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append((f"{agent} Agent", response))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(f"**{speaker}:** {msg}")
