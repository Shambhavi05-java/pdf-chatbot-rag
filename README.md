# PDF RAG Chatbot | LangChain, Google Gemini, FAISS & Streamlit

A conversational **Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF documents and ask questions about their content. The application uses semantic search to retrieve relevant document information and Google Gemini to generate context-aware answers.

Built using Python, LangChain, Google Gemini, FAISS, Sentence Transformers, and Streamlit.

## Features

- **Multiple PDF Uploads:** Upload and process multiple PDF documents as knowledge sources.
- **PDF Text Extraction:** Extract text and page-level information using PyMuPDF.
- **OCR Support:** Optionally extract text from scanned PDF pages using Tesseract OCR.
- **Text Chunking:** Divide extracted document content into smaller chunks for efficient retrieval.
- **Text Embeddings:** Generate numerical vector representations using Sentence Transformers.
- **TF-IDF Fallback:** Provide an alternative embedding implementation when Sentence Transformers initialization fails.
- **Vector Database:** Store document embeddings using FAISS.
- **Semantic Search:** Retrieve relevant document chunks using similarity search and configurable Top-K retrieval.
- **Conversational Memory:** Maintain context across multiple questions using LangChain's ConversationBufferMemory.
- **Google Gemini Integration:** Generate context-aware responses using Google's Gemini models.
- **Source References:** Display retrieved document snippets, filenames, and page numbers.
- **Visual PDF Highlighting:** Display original PDF page images with bounding boxes highlighting extracted text regions.
- **Interactive Streamlit UI:** Upload documents, ask questions, view responses, and manage conversation history.
- **Environment Configuration:** Store API credentials and model configuration in environment variables.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| LLM | Google Gemini |
| LLM Framework | LangChain |
| Embedding Model | Sentence Transformers |
| Fallback Embeddings | TF-IDF |
| Vector Database | FAISS |
| PDF Processing | PyMuPDF |
| OCR | Tesseract OCR (optional) |
| Image Processing | Pillow |
| Environment Management | Python dotenv |

## How Retrieval-Augmented Generation (RAG) Works

The application follows a document retrieval and answer-generation pipeline.

1. **Document Upload:** Users upload one or more PDF files through the Streamlit interface.
2. **Text Extraction:** PyMuPDF extracts text blocks and their page coordinates from PDF documents.
3. **OCR Processing:** For pages with little or no extractable text, optional Tesseract OCR can extract text from scanned pages.
4. **Text Chunking:** Extracted content is divided into smaller chunks to make retrieval more efficient.
5. **Embedding Generation:** Sentence Transformers convert document chunks into numerical vector representations.
6. **Vector Storage:** FAISS stores the embeddings and enables efficient similarity search.
7. **Semantic Retrieval:** When a user asks a question, the application retrieves the most relevant document chunks.
8. **Answer Generation:** Retrieved context and conversation history are passed to Google Gemini through LangChain.
9. **Source Display:** The application displays the generated answer along with retrieved source snippets and visual PDF page highlighting when source documents are returned.

## Project Architecture

```text
pdf-chatbot-rag/
│
├── app.py
│
├── src/
│   ├── config.py
│   ├── processor.py
│   ├── embedding.py
│   └── chat.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Components

**1. app.py — Streamlit Application**
- Provides the interactive user interface.
- Handles PDF uploads and document processing.
- Displays chatbot responses and conversation history.
- Displays source snippets and highlighted PDF page images.
- Provides controls to clear conversation history and document chunks.

**2. src/processor.py — PDF Processing**
- Extracts text blocks using PyMuPDF.
- Supports optional OCR using Tesseract.
- Splits extracted content into manageable chunks.
- Stores page numbers, bounding boxes, and page images as document metadata.

**3. src/embedding.py — Embedding and Retrieval**
- Generates embeddings using Sentence Transformers.
- Provides a TF-IDF fallback if Sentence Transformers initialization fails.
- Creates and manages the FAISS vector store.
- Retrieves relevant document chunks using similarity search.

**4. src/chat.py — Chat Management**
- Integrates Google Gemini using LangChain.
- Implements ConversationalRetrievalChain.
- Maintains conversation context using ConversationBufferMemory.
- Returns generated answers and retrieved source documents.
- Includes a direct Gemini generation fallback if the retrieval chain fails.

**5. src/config.py — Configuration**
- Manages application configuration.
- Loads environment variables.
- Configures the Gemini API key, model name, embedding model, chunk size, chunk overlap, and retrieval settings.

## Installation and Setup

### Prerequisites

- Python 3.11 recommended
- Git
- Google Gemini API key
- Tesseract OCR (optional, for scanned PDF support)

### 1. Clone the repository

```bash
git clone https://github.com/Shambhavi05-java/pdf-chatbot-rag.git

cd pdf-chatbot-rag
```

### 2. Create a virtual environment

For Windows:

```powershell
py -3.11 -m venv venv

.\venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
python3.11 -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project's root directory.

Add the following configuration:

```env
GEMINI_API_KEY=your_google_gemini_api_key
MODEL_NAME=your_supported_gemini_model
```

Replace the placeholders with your actual Gemini API key and a model supported by your Google Gemini API access.

Get your API key from:

[Google AI Studio](https://aistudio.google.com/)

**Security Note:** Never upload your `.env` file or expose your API key publicly. The `.gitignore` file excludes sensitive environment configuration.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## How to Use

1. Launch the Streamlit application.
2. Upload one or more PDF documents using the sidebar.
3. Click the **Process Documents** button.
4. Wait for document extraction, chunking, and embedding generation.
5. Enter a question related to the uploaded documents.
6. Receive an AI-generated response based on the retrieved document context.
7. Ask follow-up questions to continue the conversation.
8. Expand the source references to view retrieved text snippets and highlighted PDF page regions.
9. Use **Clear Conversation** to reset chat history.
10. Use **Clear File Chunks** to remove processed document chunks and the in-memory vector store.

## Example Questions

After uploading a PDF, try asking:

- Summarize this document in five bullet points.
- What are the main topics discussed in this PDF?
- Explain the important concepts covered in the document.
- What does the document say about a particular topic?
- Explain the previous answer in more detail.

## Configuration

Application parameters are managed through `src/config.py` and environment variables.

Important configuration parameters include:

| Parameter | Description |
|---|---|
| GEMINI_API_KEY | Google Gemini API authentication |
| MODEL_NAME | Gemini model used for response generation |
| EMBEDDING_MODEL | Sentence Transformers embedding model |
| CHUNK_SIZE | Maximum configured text chunk size |
| CHUNK_OVERLAP | Overlap between text chunks |
| TOP_K | Number of relevant chunks retrieved |

Refer to `src/config.py` for the actual default values.

## Key Learning Outcomes

Through this project, I explored:

- Retrieval-Augmented Generation (RAG) architecture.
- Large Language Model integration using Google Gemini.
- Document processing and text extraction using PyMuPDF.
- Semantic embeddings and vector representations.
- Similarity search using FAISS.
- Conversational retrieval chains and memory in LangChain.
- OCR-based text extraction.
- Building interactive AI applications using Streamlit.
- Managing API keys and configuration through environment variables.

## Future Improvements

- Add persistent vector storage instead of keeping embeddings only in memory.
- Improve source attribution and answer-grounding validation.
- Add support for more document formats.
- Add document-level filtering and improved retrieval strategies.
- Introduce automated evaluation of retrieval relevance and answer quality.

## Attribution

This repository is a fork of the original project by [dksrinath](https://github.com/dksrinath).

Original repository:

https://github.com/dksrinath/pdf-chatbot-rag

My contributions include:

- Setting up the local development environment using Python 3.11.
- Configuring Google Gemini API integration and model settings.
- Improving chat response handling and fallback behavior.
- Testing PDF processing, document retrieval, and conversational functionality.
- Updating project configuration and documentation.

## Author

**Shambhavi Bhargava**

GitHub: [Shambhavi05-java](https://github.com/Shambhavi05-java)

Project Repository: [PDF RAG Chatbot](https://github.com/Shambhavi05-java/pdf-chatbot-rag)

---

*This project is intended for learning and demonstrating document-based question answering using Retrieval-Augmented Generation.*