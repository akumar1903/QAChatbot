# QAChatbot
📄 PDF Question Answering Chatbot (LangChain + HuggingFace + Gradio)
This project is a local PDF-based Question Answering (QA) chatbot built with:

LangChain – for document loading, splitting, embeddings, and retrieval

HuggingFace Hub – for LLMs and embeddings

FAISS – for local vector storage and fast similarity search

Gradio – for a simple web-based interface

It allows you to upload a PDF, process it into vector embeddings, and ask natural language questions about the document.

🚀 Features
Upload any PDF and process it into searchable text chunks.

Store embeddings locally using FAISS.

Retrieve the most relevant chunks for your query.

Answer questions using a HuggingFace LLM (e.g., google/flan-t5-large).

Simple and interactive Gradio UI.

📦 Installation
1️⃣ Clone this repository
bash
Copy
Edit
git clone https://github.com/yourusername/pdf-qa-chatbot.git
cd pdf-qa-chatbot
2️⃣ Install dependencies
bash
Copy
Edit
pip install langchain langchain-community gradio faiss-cpu sentence-transformers pypdf
🔑 API Key Setup
You’ll need a HuggingFace API token to access the model.

Get your token from: https://huggingface.co/settings/tokens

Add it to your environment:

bash
Copy
Edit
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
Or replace in code:

python
Copy
Edit
HUGGINGFACEHUB_API_TOKEN = "your_token_here"
⚙️ Configuration
You can change the LLM and embedding models in the script:

python
Copy
Edit
LLM_MODEL_NAME = "google/flan-t5-large"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
▶️ Usage
Run the script:

bash
Copy
Edit
python app.py
Open your browser at http://127.0.0.1:7860.

💻 How It Works
PDF Upload – Loads and extracts text from the uploaded PDF using PyPDFLoader.

Text Splitting – Splits text into overlapping chunks with RecursiveCharacterTextSplitter.

Embeddings – Creates vector embeddings with HuggingFaceEmbeddings.

FAISS Index – Stores embeddings locally for fast retrieval.

Question Answering – Uses RetrievalQA to fetch relevant chunks and pass them to the HuggingFace LLM.

Gradio Interface – Handles PDF upload, processing, and answering questions.

🖼 Example
Upload: sample.pdf

Ask: "What is the main topic of this document?"

Answer: The chatbot will return a concise answer based on the PDF content.

📌 Notes
For large PDFs, processing time may be longer.

You can swap the LLM with other HuggingFace models (e.g., mistralai/Mistral-7B-Instruct-v0.2).

This runs fully locally except for HuggingFace API calls.

📜 License
MIT License – feel free to use and modify.
