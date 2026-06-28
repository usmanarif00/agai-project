from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from duckduckgo_search import DDGS

# -----------------------
# LLM
# -----------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"]
)

# -----------------------
# EMBEDDINGS + VECTOR DB
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader = TextLoader("software_manual.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------
# STATE
# -----------------------
class AgentState(TypedDict):
    original_query: str
    optimized_query: str
    documents: List[str]
    generation: str
    loop_count: int
    hallucination: str

# -----------------------
# NODES
# -----------------------
def rewrite(state):
    res = llm.invoke(f"Rewrite into keywords: {state['original_query']}")
    return {"optimized_query": res.content}

def retrieve(state):
    docs = retriever.invoke(state["optimized_query"])
    return {"documents": [d.page_content for d in docs]}

def generate(state):
    context = "\n\n".join(state["documents"])
    res = llm.invoke(f"""
Use ONLY context:
{context}

Question: {state['optimized_query']}
""")
    return {"generation": res.content}


def hallucination(state):
    res = llm.invoke(f"""
Check answer correctness.

Context:
{state['documents']}

Answer:
{state['generation']}

Return only: passed or failed
""")

    result = "passed" if "passed" in res.content.lower() else "failed"

    return {
        "hallucination": result,
        "loop_count": state["loop_count"] + 1  
    }

# -----------------------
# ROUTING
# -----------------------
def route(state):
    if state["hallucination"] == "passed":
        return END
    if state["loop_count"] >= 2:
        return END
    return "rewrite"

# -----------------------
# GRAPH BUILD
# -----------------------
workflow = StateGraph(AgentState)

workflow.add_node("rewrite", rewrite)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("hallucination", hallucination)

workflow.set_entry_point("rewrite")

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "hallucination")

workflow.add_conditional_edges(
    "hallucination",
    route,
    {
        "rewrite": "rewrite",
        END: END
    }
)

app = workflow.compile()

# -----------------------
# RUN FUNCTION
# -----------------------
def run_pipeline(user_query):
    state = {
        "original_query": user_query,
        "optimized_query": "",
        "documents": [],
        "generation": "",
        "loop_count": 0,
        "hallucination": ""
    }

    result = app.invoke(state)
    return result["generation"]
