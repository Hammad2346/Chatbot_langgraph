from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))


embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
print("Vector database accessed complete")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


class State(TypedDict):
    question: str
    documents: List[str]
    answer: str
    history: List[dict]


def retriever_node(state: State):
    results = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in results]}


def generate_response_node(state: State):
    context = "\n".join(state["documents"])
    history = state.get("history", [])
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = f"""You are a helpful assistant. Answer based on the context and conversation history.
    Context:
    {context}
    Conversation History:
    {history_text}
    Question: {state['question']}"""
    response = llm.invoke(prompt)
    return {"answer": response.content}

def summarize_node(state: State):
    history = state.get("history", [])
    if len(history) < 10:
        return {"history": history}
    
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = f"""Summarize this conversation in 2-3 sentences, keeping the key facts and context.
    Conversation:
    {history_text}
    Return only the summary, no explanation."""
    
    response = llm.invoke(prompt)
    return {"history": [{"role": "system", "content": f"Previous conversation summary: {response.content}"}]}


graph = StateGraph(State)
graph.add_node("summarize_node", summarize_node)
graph.add_node("retriever_node", retriever_node)
graph.add_node("generate_response_node", generate_response_node)
graph.set_entry_point("summarize_node")
graph.add_edge("summarize_node", "retriever_node")
graph.add_edge("retriever_node", "generate_response_node")
graph.add_edge("generate_response_node", END)
app = graph.compile()
history=[]
while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break
    
    result = app.invoke({"question": question, "documents": [], "answer": "", "history": history})
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    
    print(f"\nAnswer: {result['answer']}")