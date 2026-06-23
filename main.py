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
print("Vector database created successfully")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


class State(TypedDict):
    question: str
    documents: List[str]
    answer: str


def retriever_node(state: State):
    results = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in results]}


def generate_response_node(state: State):
    context = "\n".join(state["documents"])
    prompt = f"Answer the question based on this context:\n{context}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}


graph = StateGraph(State)
graph.add_node("retriever_node", retriever_node)
graph.add_node("generate_response_node", generate_response_node)
graph.set_entry_point("retriever_node")
graph.add_edge("retriever_node", "generate_response_node")
graph.add_edge("generate_response_node", END)
app = graph.compile()

while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break
    result = app.invoke({"question": question})
    print(f"\nAnswer: {result['answer']}")