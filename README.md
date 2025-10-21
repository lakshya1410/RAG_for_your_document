# 📄 Advanced RAG System - Chat with Your Documents

A Retrieval-Augmented Generation (RAG) application for uploading documents and querying them using natural language. Built with Streamlit, LangChain, and Groq AI.

## 🌟 Features

- Multi-format support (PDF, CSV, TXT)
- Configurable text chunking with overlap
- Local vector storage with ChromaDB
- Semantic search using Sentence Transformers
- AI-powered responses via Groq's Gemma2-9b-it
- Source attribution and conversation history



<img width="680" height="308" alt="RAG Pipeline Architecture" src="https://github.com/user-attachments/assets/d03f9f13-8cfc-4e62-85e3-dbe16293808c" />

```
Document Upload → Text Splitting → Embeddings → Vector Store → Similarity Search → LLM Context
```

##  Installation

```bash
# Clone the repository
git clone https://github.com/lakshya1410/RAG_for_your_document.git
cd RAG_for_your_document

# Install dependencies
pip install -r requirements.txt

# Create .env file with your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

Get your free Groq API key from [console.groq.com](https://console.groq.com/)

## 🚀 Usage

```bash
streamlit run main.py
```

1. Upload documents (PDF, CSV, TXT) via the sidebar
2. Configure chunk size (default: 1000) and overlap (default: 100)
3. Click "Submit & Process"
4. Ask questions and get AI-generated answers with source attribution

## � Key Technologies

- **Streamlit** - Web interface
- **LangChain** - RAG framework
- **Groq (Gemma2-9b-it)** - LLM
- **Sentence Transformers** - Embeddings
- **ChromaDB** - Vector storage

## 🛠️ Programmatic Usage

```python
from rag_pipeline import process_files, ask_question

# Process documents
with open('document.pdf', 'rb') as f:
    process_files([f], chunk_size=1000, chunk_overlap=100)

# Ask questions
answer, sources = ask_question("What is the main topic?", k=3)
```

## 🐛 Troubleshooting

- **"GROQ_API_KEY not found"**: Create `.env` file with valid API key
- **"Vector store is empty"**: Upload and process documents first
- **"Error processing file"**: Ensure supported format (PDF, CSV, TXT) and valid encoding


