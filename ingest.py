from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

file_name="Hammad_Arif_Resume_latest"
loader = PyPDFLoader(f"docs/{file_name}.pdf")
docs = loader.load()
raw_text = "\n".join([d.page_content for d in docs])
print("PDF loaded")
through_md=os.getenv("through_md", "false").lower() == "true"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if (through_md):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""Convert this resume text into clean markdown format.
    Use proper headers (##) for each section like Skills, Projects, Education, Experience.
    Keep all the content, don't summarize or remove anything.
    Return only the markdown, no explanation.

    Resume text:
    {raw_text}"""

    response = llm.invoke(prompt)
    md = response.content

    print("Markdown generated")

    if md.startswith("```"):
        md = md.split("\n", 1)[1]  
    if md.endswith("```"):
        md = md.rsplit("```", 1)[0]
    md = md.strip()

    os.makedirs("data", exist_ok=True)
    with open(f"data/{file_name}.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved to data/{file_name}.md")

    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]


    with open(f"data/{file_name}.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(md_text)
    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:", chunk.metadata) 
else:
    splitter=SemanticChunker(embeddings=embeddings)
    chunks=splitter.split_documents(docs)
    print("chunks created.")
    
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("Vectorstore created successfully")