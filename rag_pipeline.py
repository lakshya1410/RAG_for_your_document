import os
import json
import tempfile
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

# Ensure folder exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Initialize sentence transformer embeddings (free)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(files, chunk_size=1000, chunk_overlap=100):
    docs = []
    print(f"Processing {len(files)} files...")
    
    for file in files:
        try:
            file_ext = os.path.splitext(file.name)[-1].lower()
            print(f"Processing file: {file.name} (type: {file_ext})")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                # Reset file pointer to beginning in case it was read before
                file.seek(0)
                file_content = file.read()
                
                if not file_content:
                    print(f"Warning: File {file.name} is empty")
                    continue
                    
                # Handle both bytes and string content
                if isinstance(file_content, str):
                    file_content = file_content.encode('utf-8')
                    
                tmp.write(file_content)
                tmp_path = tmp.name
            
            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_ext == ".csv":
                loader = CSVLoader(tmp_path)
            elif file_ext == ".txt":
                # Try different encodings for text files
                for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                    try:
                        loader = TextLoader(tmp_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    print(f"Could not decode text file {file.name} with any encoding")
                    os.unlink(tmp_path)
                    continue
            else:
                print(f"Unsupported file type: {file_ext}")
                os.unlink(tmp_path)
                continue
                
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} documents from {file.name}")
            docs.extend(loaded_docs)
            
            # Clean up temp file
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            continue

    if not docs:
        raise ValueError("No documents were successfully loaded")

    print(f"Total documents loaded: {len(docs)}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        raise ValueError("No chunks were created from the documents")
    
    print(f"Created {len(chunks)} chunks")
    
    # Validate chunks have content
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not valid_chunks:
        raise ValueError("All chunks are empty")
    
    print(f"Valid chunks with content: {len(valid_chunks)}")

    # Store vectors
    try:
        vectorstore = Chroma.from_documents(valid_chunks, embeddings, persist_directory=VECTOR_STORE_PATH)
        print(f"Successfully created vector store with {len(valid_chunks)} chunks")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def ask_question(query, k=3):
    try:
        # Check if vector store exists
        if not os.path.exists(VECTOR_STORE_PATH):
            raise ValueError("No vector store found. Please upload and process documents first.")
        
        vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        
        # Check if vector store has any documents
        try:
            # Try to get at least one document to verify the store isn't empty
            test_results = vectordb.similarity_search("test", k=1)
            if not test_results:
                raise ValueError("Vector store is empty. Please upload and process documents first.")
        except Exception as e:
            if "no docs" in str(e).lower() or "empty" in str(e).lower():
                raise ValueError("Vector store is empty. Please upload and process documents first.")
            # If it's another error, continue (might be a benign error)

        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set your Groq API key in the .env file.")
            
        llm = ChatGroq(
            model="gemma2-9b-it",
            groq_api_key=api_key,
            temperature=0
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa({"query": query})

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Save history
        log_result(query, answer, sources)

        return answer, [doc.metadata for doc in sources]
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(error_msg)
        return error_msg, []

def log_result(query, answer, sources):
    entry = {
        "query": query,
        "answer": answer,
        "sources": [doc.metadata for doc in sources]
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []