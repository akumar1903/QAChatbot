# PDF Question Answering Chatbot (LangChain + HuggingFace)

This project is an interactive **PDF Question Answering Chatbot** built with **LangChain**, **HuggingFace models**, and **Gradio**. 
It allows users to upload a PDF file, processes the document into chunks, creates embeddings, and uses a language model to answer user queries based on the PDF content.

---

## Features
- **PDF Loading**: Reads and extracts text from PDFs using `PyPDFLoader`.
- **Text Splitting**: Uses `RecursiveCharacterTextSplitter` to break text into overlapping chunks for better context.
- **Embeddings**: Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Store**: Stores embeddings in a **FAISS** vector database for efficient retrieval.
- **Retrieval QA**: Uses a HuggingFace LLM (e.g., `flan-t5-large`) to answer questions from the document.
- **Gradio Interface**: Simple UI to upload PDFs and ask questions.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

2. **Create a virtual environment and activate it**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set HuggingFace API Token**
Replace `HUGGINGFACEHUB_API_TOKEN` in the script or set it as an environment variable:
```bash
export HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

---

## Dependencies
- `langchain`
- `langchain_community`
- `faiss-cpu`
- `gradio`
- `sentence-transformers`
- `pypdf`

Install all dependencies with:
```bash
pip install langchain langchain_community faiss-cpu gradio sentence-transformers pypdf
```

---

## Usage
Run the chatbot locally:
```bash
python app.py
```

This will start a **Gradio** web interface where you can:
1. Upload a PDF document.
2. Ask questions related to its content.

---

## Project Structure
```
.
├── app.py              # Main application script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Configuration
You can modify the following in the script:
- `LLM_MODEL_NAME`: Change the HuggingFace LLM model (e.g., `mistralai/Mistral-7B-Instruct-v0.1`).
- `EMBEDDING_MODEL_NAME`: Change the embeddings model.

---

## Example
Upload a **research paper** and ask:
> "What is the main conclusion of this paper?"

The chatbot retrieves relevant sections and answers using the chosen LLM.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgements
- [LangChain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [Gradio](https://www.gradio.app/)

