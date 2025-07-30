# Extract Tables from PDF using Claude (OCR enabled)

A Python application that extracts content from PDFs and enables intelligent question-answering using AI models and vector search.

## Core Concepts

### PDF Processing Pipeline
The system processes PDFs in two phases:
1. **Content Extraction**: Uses Claude 4 to convert PDF pages (processed in pairs) into structured Markdown
2. **Text Chunking**: Splits extracted content into manageable pieces for efficient retrieval

### Vector-Based Retrieval
- **Embeddings**: Converts text chunks into numerical vectors using `sentence-transformers/all-mpnet-base-v2`
- **FAISS Index**: Creates a searchable vector database for fast similarity matching
- **Retrieval**: Finds the most relevant document chunks for each question

### Question-Answering Chain
Uses LangChain's LCEL (LangChain Expression Language) to create a RAG (Retrieval-Augmented Generation) pipeline:
```
Question → Retrieve Context → Generate Answer → Parse Output
```

## Key Components

### `extract_pdf(pdf_base64, api_key)`
- Sends base64-encoded PDF pages to Claude 4
- Extracts all content including tables, text, and OCR'd images
- Returns structured Markdown preserving original formatting

### `read_pdf_in_pairs(file_path)`
- Splits PDFs into 2-page chunks using PyMuPDF
- Converts each pair to base64 for API transmission
- Reduces memory usage and API payload size

### `create_vector_store(texts)`
- Generates embeddings for text chunks
- Builds FAISS index for semantic search
- Automatically detects CUDA/CPU for optimal performance

### `create_question_answer_chain(vector_store)`
- Creates retrieval chain using top-3 similar chunks
- Uses Ollama's LLama2 for local answer generation
- Implements custom prompt template for context-aware responses

## Workflow

1. **PDF Input**: Load PDF file from `Data/sample-tables.pdf`
2. **Caching**: Check for previously extracted content in `Data/extracted_pdf.txt`
3. **Extraction**: Process PDF pairs through Claude 4 if not cached
4. **Sanitization**: Clean special characters (checkboxes, symbols)
5. **Vectorization**: Create searchable embeddings
6. **Query Processing**: Answer questions using retrieved context

## Dependencies

- **AI Models**: Anthropic Claude 4, HuggingFace embeddings, Ollama LLama2
- **Vector Search**: FAISS for similarity search
- **PDF Handling**: PyMuPDF for document processing
- **Framework**: LangChain for orchestration

## Usage

```python
# Set API key and run
main(your_anthropic_api_key)

# Example query
query = "How did Respondent B respond to the question about UK citizenship?"
```

The system combines cloud-based extraction (Claude) with local inference (LLama2) for a hybrid approach balancing accuracy and privacy.