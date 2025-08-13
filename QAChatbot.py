from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import gradio as gr
import os


# -------------------------------
# Configs
# -------------------------------
HUGGINGFACEHUB_API_TOKEN = ""  # Replace with your HuggingFace token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Choose your LLM model from Hugging Face (e.g., Flan-T5, Mistral, Zephyr, etc.)
LLM_MODEL_NAME = "google/flan-t5-large"

# Embedding model (can be replaced with other sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------
# Load and Process PDF
# -------------------------------
def load_and_index_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Store in FAISS vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# -------------------------------
# Build the QA Chain
# -------------------------------
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceHub(
        repo_id=LLM_MODEL_NAME,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# -------------------------------
# Gradio Interface
# -------------------------------
uploaded_vectorstore = None
qa_chain = None

def upload_pdf(file):
    global uploaded_vectorstore, qa_chain
    uploaded_vectorstore = load_and_index_pdf(file.name)
    qa_chain = create_qa_chain(uploaded_vectorstore)
    return "PDF uploaded and processed. You can now ask questions."

def answer_question(question):
    if qa_chain is None:
        return "Please upload a PDF first."
    result = qa_chain({"query": question})
    return result["result"]

with gr.Blocks() as demo:
    gr.Markdown("## PDF Question Answering Chatbot (LangChain + HuggingFace)")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("Process PDF")

    with gr.Row():
        question_input = gr.Textbox(label="Ask a question about the document")
        answer_output = gr.Textbox(label="Answer")

    upload_button.click(upload_pdf, inputs=pdf_file, outputs=answer_output)
    question_input.submit(answer_question, inputs=question_input, outputs=answer_output)

demo.launch()
