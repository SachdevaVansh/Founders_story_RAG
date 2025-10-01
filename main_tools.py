import requests 
import numpy as np
import wikipedia 
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings 
from euriai.langchain import EuriaiChatModel

from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation

from euriai.langchain import create_chat_model

from dotenv import load_dotenv

load_dotenv()
import os 

api_key=os.getenv("EURI_API_KEY")

def generate_embeddings(text):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    embedding = np.array(data['data'][0]['embedding'])
    return embedding

class EuriEmbeddings(Embeddings):
    def embed_documents(self,texts):
        return [generate_embeddings(t).tolist() for t in texts]
    
    def embed_query(self,text):
        return generate_embeddings(text).tolist()

embeddings=EuriEmbeddings()

chat_model = EuriaiChatModel(
    api_key=api_key,
    model="gpt-4.1-nano")

# Simple tools 
def summarizer_tool(text):
    return chat_model.invoke([
        {"role":"system","content":"Summarzize this text"},
        {"role":"user","content":f"Summarzie :\n{text}"}

    ])

def wikipedia_tool(query):
    try:
        summary=wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Error while fetching Wikipedia: {str(e)}"

def rag_tool(query):
    """Answer the questions about the founder from the founder's story document using RAG."""
    qa_chain=RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=False)
    return qa_chain.run(query)

# Create a memory buffer for our conversation
def translate_tool(text_and_language):
    """Translate the given text from one language to another.
    Example input: "Hello World || French"
    """
    try:
        parts=text_and_language.split("||")
        text=parts[0].strip()
        target_language=parts[1].strip()
    except Exception as e:
        return "Invalid input. Please provide a text and a target language separated by '||'."

    prompt=[
        {"role":"system","content":"You translate the text"},
        {"role":"user","content":f"Translate '{text}' from English to {target_language}"}
    ]

    return chat_model.invoke(prompt)

## Creating the retriever 
loader= TextLoader("data/founder.txt",encoding="utf-8")
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
chunks=text_splitter.split_documents(docs)

faiss_index=FAISS.from_documents(chunks,embeddings)

retriever=faiss_index.as_retriever(search_kwargs={"k": 3})

## Defining the tools 
tools=[
    Tool(
        name="Summarizer",
        func=summarizer_tool,
        description="Summarizes long texts into concise summaries."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_tool,
        description="Fetches information from Wikipedia."
    ),
    Tool(
        name="RAG",
        func=rag_tool,
        description=("Answer questions about the founder from the founder's story document using RAG."
                    "This tool searches the local founder_story.txt document for answers. "
                    "For all other topics, use other tools or your own knowledge.")
    ),
    Tool(
        name="Translate",
        func=translate_tool,
        description="Translates text from one language to another."
    )
]

# Specializing Agents( Researcher, Teacher)
class SpecializedAgent:
    def __init__(self,name,system_prompt):
        self.name=name
        self.memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        self.llm=chat_model
        self.agent=initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True)
        self.system_prompt=system_prompt

    def run(self,user_input):
        try:
            return self.agent.run(f"{self.system_prompt}\n{user_input}")
        except Exception as e:
            return f"Error while running the agent: {str(e)}"

## Router Agent 
class RouterAgent:
    def __init__(self,agents):
        self.llm=chat_model
        self.agents=agents ## Dict: name---> SpecializedAgent

    def route(self,user_input):
        system_prompt=(
            "You are a router agent. Given a user query, decide which agent should handle it: "
            "'Researcher' (for factual, research, or data questions) or 'Teacher' (for explanations, learning, or teaching). "
            "Reply with only the agent's name: Researcher or Teacher."
        )

        prompt=f"{system_prompt}\n User query:{user_input}\n Agent:"

        ai_message = self.llm.invoke(prompt)  # returns AIMessage
        agent_name = ai_message.content.strip().split()[0].capitalize()

        if agent_name not in self.agents:
            agent_name="Teacher"
        return agent_name 

    def chat(self,user_input):
        agent_name=self.route(user_input)
        # Using the agent's we are running the agent's run method here 
        response=self.agents[agent_name].run(user_input)
        return f"[Router -> {agent_name}\n {response}]"

## Lets create Specialized agents 
agents={
    "Researcher":SpecializedAgent(
        name="Researcher", 
        system_prompt="You are a research assistant. Be factutal and concise. Use tools if needed.You have access to the local data and can use the RAG tool to answer questions if the user asks something about the founder's story or startups. For other topics use other tools or your own knowledge."),
        "Teacher":SpecializedAgent(
        name="Teacher",
        system_prompt="You are a teacher. You can explain complex topics, teach new skills, or learn from others. Use tools if needed. Use the RAG tool only when the user asks something about the founder's story or startups. For other topics use other tools or your own knowledge.")
}

## Create the Router Agent
router=RouterAgent(agents)

## CLI Loop

print("\n Connected multi-Agent Chat (type 'exit' to quit):\n")
print("Agents: Researcher, Teacher (choosen automatically by the router) \n")

while True:
    user_input=input("You: ")
    if user_input.lower() in ("exit","quit"):
        break
    response=router.chat(user_input)
    print(f"{response}\n")

