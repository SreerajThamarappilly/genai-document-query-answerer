# GenAI Document Query Answering System

This repository contains a complete **Document Query Answering** solution.  The system ingests a PDF document (such as a loan application, closing disclosure, or credit report), extracts and processes its text, stores semantic representations in a vector database, and allows users to ask questions that are answered using only the document’s content. 

## Project Overview

The assignment requires four core tasks:

1. **Parse and Extract Text** – Read all text from the PDF, preserving both structured content (tables, headers) and unstructured paragraphs.
2. **Create Document Embeddings** – Generate vector embeddings for different sections of the document using an NLP model (e.g. Sentence‑Transformers or an OpenAI embedding model).
3. **Semantic Search with a Vector DB** – Store the embeddings in a vector database (FAISS, Pinecone, or Weaviate) and perform similarity search to find relevant sections.
4. **Query Answering** – For a given question, retrieve the most relevant chunks and use an LLM to generate an answer based on those chunks.

The solution implements these steps through a modular pipeline:

- **PDF Parsing** – `src/pdf_parser.py` uses PyMuPDF to extract text and metadata from each page; if no text is found, it optionally falls back to OCR via Tesseract.  It heuristically classifies blocks as paragraphs, tables, or forms to aid downstream processing.
- **Chunking** – `src/chunker.py` splits sections into smaller overlapping chunks for embedding.  It respects paragraph boundaries and uses sentence splitting to avoid cutting sentences mid‑way.
- **Embedding** – `src/embedder.py` loads a Sentence‑Transformer model (MiniLM by default) to convert chunks into high‑dimensional vectors.  It can alternatively call OpenAI’s embedding API when configured via environment variables.
- **Vector Store** – `src/vector_store.py` wraps a FAISS index for storing and querying embeddings.  It optionally supports approximate nearest‑neighbour search via HNSW and can persist the index to disk.  Index type, persistence, and HNSW parameters are configurable via `.env`.
- **Query Engine** – `src/query_engine.py` orchestrates the pipeline: it embeds the user’s query, searches the vector store, applies a relevance threshold (gatekeeping), constructs a prompt with the retrieved chunks, and uses an LLM (GPT‑4 by default) to produce the answer.  It cleanly separates concerns, making it straightforward to swap out models or vector backends.
- **Streamlit UI** – `app/app.py` provides a lightweight web interface where users can upload a PDF, enter questions, and optionally view a 2‑D embedding visualisation.  Streamlit state is used to persist the loaded index across interactions.

## 🗂 Directory Structure

```bash
genai_doc_qa/
├── app/ # Streamlit front‑end
│ └── app.py
├── src/ # Core Python modules
│ ├── pdf_parser.py # Parses PDFs into structured sections
│ ├── chunker.py # Splits sections into overlapping chunks
│ ├── embedder.py # Generates embeddings via Sentence‑Transformers or OpenAI
│ ├── vector_store.py # FAISS-based vector index (supports persistence & HNSW)
│ ├── query_engine.py # Orchestrates parsing, embedding, search, and LLM
│ ├── config.py # Central configuration loaded from environment variables
│ ├── logger.py # Configures logging with timestamps and levels
│ └── utils/
│   └── visualize.py # Plots embedding clusters using PCA
├── tests/ # Unit tests covering chunking, retrieval, and gatekeeping
│ └── test_system.py
├── Dockerfile # Container image definition
├── docker-compose.yml # Orchestrates container(s), defines env variables and ports
├── requirements.txt # Pinned dependencies for reproducible installs
├── .env # Actual environment configuration
├── .env.example # Template for environment configuration
└── README.md # This documentation file
```

## Design Choices & Approaches

### Parsing & Chunking

We use **PyMuPDF** for parsing because it provides high‑level access to a PDF’s text, font, and layout information.  For scanned pages, we integrate **pytesseract** to perform OCR.  The parser wraps each block in a `Section` dataclass storing its text, page number, and a simple classification (paragraph, table, or form), facilitating downstream operations.

The **TextChunker** enforces a configurable maximum chunk size (default 800 characters) with overlap to preserve context across boundaries.  This splitting strategy balances retrieval precision and embedding size; smaller chunks yield more accurate retrieval but increase embedding overhead, while larger chunks may dilute context.  Overlap ensures that sentences spanning two chunks remain available to the model.

### Embedding

By default, the project uses **`sentence-transformers/all-MiniLM-L6-v2`**, a compact yet powerful model producing 384‑dimensional vectors.  It offers a good trade‑off between performance and semantic accuracy.  The embedder normalises vectors to unit length so that cosine similarity can be computed with inner products.  The system can switch to OpenAI’s embedding API by setting `USE_OPENAI=True` and providing an API key in the `.env` file.

### Vector Database

We chose **FAISS** for the vector store due to its speed and ease of use.  The default index uses **inner‑product similarity** (equivalent to cosine for normalised vectors).  Alternatively, you can enable an **HNSW** index for approximate search by setting `INDEX_TYPE=hnsw` and tuning `HNSW_M` and `HNSW_EF_CONSTRUCTION` in `.env`.  Persistence is supported by toggling `PERSIST_INDEX=True`; this stores the index and metadata to `INDEX_DIR` so that documents do not need to be re‑embedded on each run.

### Retrieval & LLM

The **QueryEngine** embeds the user’s query, retrieves the top‑k most similar document chunks, and applies a relevance threshold to implement a basic **gatekeeper**.  This prevents irrelevant questions from reaching the LLM, saving cost and reducing hallucinations.  The retrieved context is passed into a prompt instructing the LLM (e.g. GPT‑4) to answer using only the provided excerpts.  You can disable LLM calls entirely (for debugging) by setting `USE_LLM=False` in `.env`.

### Embedding Visualization (PCA)

To validate embedding quality and chunking strategy, we project high-dimensional SBERT embeddings (384-D) into 2D using PCA. This visualization demonstrates natural semantic clustering of document sections (e.g., headers, paragraphs, page-level groupings), providing confidence that semantic search will retrieve relevant context. PCA is used strictly for interpretability and debugging; retrieval uses full-dimensional vectors.

### Modularity & Patterns

The codebase follows the **single responsibility principle**: each class handles one concern (parsing, chunking, embedding, indexing, answering).  This allows for easy swapping or extension: e.g. adding a `PdfPlumberParser` for advanced table extraction or a `PGVectorStore` for database‑backed persistence.  Dependency injection via `config.py` decouples components from specific models or index types.  The design also makes it straightforward to implement additional patterns, such as a factory method to select the embedding model based on configuration.

### Performance & Complexity

Parsing runs in **O(p × b)** time where *p* is the number of pages and *b* the average number of blocks per page.  Chunking is linear in the total text length.  Embedding is **O(n × d)** where *n* is the number of chunks and *d* the embedding dimension.  FAISS’s flat index performs exact similarity search in **O(N)** but is extremely fast for up to millions of vectors; enabling HNSW reduces search to logarithmic complexity at the cost of slight accuracy loss.  The LLM call is the dominant time component (1–3 seconds per query for GPT‑4).  Memory usage scales roughly with the number of chunks times the embedding dimension.

### Scaling Considerations

The current implementation loads one document at a time, but we can extend it to a **multi‑document index** by storing document identifiers in each `Section` and filtering search results by doc ID.  For large document sets, persist the FAISS index (`PERSIST_INDEX=True`) and consider using **PGVector** or **Weaviate** for distributed storage.  Embedding generation can be parallelised or offloaded to GPUs to improve throughput.  To support dozens of concurrent queries, expose the retrieval + LLM pipeline as a REST API (FastAPI) and use a queue or caching layer to reuse answers.

### Accuracy & Precision

Semantic retrieval combined with an LLM dramatically reduces hallucinations because answers are grounded in retrieved text.  However, the quality of results depends on the embedding model and the chunk granularity.  Using a domain‑specific model (e.g. Legal‑BERT for legal documents or FinBERT for financial data) may improve retrieval fidelity.  We have to always monitor recall and precision by evaluating sample queries against known answers.  Overlap size and relevance threshold are critical hyperparameters for balancing completeness vs. noise; adjust these in `.env` to tune performance.

## Testing Procedure

Unit tests in `tests/test_system.py` validate core components:

- **Chunking** – Ensures that the overlap logic works as expected and that long paragraphs are split into multiple chunks.
- **VectorStore** – Confirms that nearest‑neighbour search returns the correct section for a given query vector.
- **QueryEngine Gatekeeping** – Checks that relevant queries return answers and irrelevant queries are rejected when below the similarity threshold.

Run tests with:

```bash
pytest -q
```

It is possible to add further tests for PDF parsing (e.g. verifying that tables are detected) or multi‑document scenarios. Continuous integration pipelines should run these tests on each commit to catch regressions.

## Installation & Usage

### Local Setup (Windows 11 / Linux / macOS)

1. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/SreerajThamarappilly/genai-document-query-answerer.git genai_doc_qa
cd genai_doc_qa
python -m venv .venv
.venv\Scripts\activate    # On Windows (use `source .venv/bin/activate` on Linux/macOS)
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Configure environment variables:

- Copy .env.example to .env and set OPENAI_API_KEY if you plan to use OpenAI.
- Adjust other variables (e.g. USE_OPENAI, PERSIST_INDEX, INDEX_TYPE, LLM_MODEL_NAME) as needed.

1. Run tests:

```bash
pytest -q
```

5. Start the Streamlit app:

```bash
streamlit run app/app.py
```

Navigate to http://localhost:8501, upload one of the sample PDFs (1003.pdf, CP-Sample.pdf, or credit_report.pdf) and ask a question.

### Docker & Docker‑Compose

To ensure consistent deployments, a Dockerfile and docker-compose.yml are provided. Build and run the container as follows:

```bash
docker-compose build
docker-compose up
```

The service will be available at http://localhost:8501. Environment variables defined in .env (or passed via the env_file entry in docker-compose.yml) are loaded into the container.

## Environment Variables

The .env.example file documents all configurable settings. Key variables include:

- OPENAI_API_KEY – API key for OpenAI services; leave blank if not using OpenAI.
- USE_OPENAI – When True, use OpenAI embeddings instead of the local model.
- USE_LLM – Disable to skip LLM calls (useful for debugging retrieval results).
- CHUNK_SIZE / CHUNK_OVERLAP – Control chunk splitting granularity.
- TOP_K – Number of chunks to retrieve per query.
- RELEVANCE_THRESHOLD – Minimum cosine similarity to consider a chunk relevant (gatekeeping)
- PERSIST_INDEX / INDEX_DIR – Save/load the FAISS index to avoid recomputing embeddings.
- INDEX_TYPE – Use flat for exact search or hnsw for approximate search.

Adhering to environment variables allows us to tune the system without modifying code. Always keep the .env file secret.

## Evaluation / Demo Queries (Expected Output Examples)

| PDF | Example Question | Expected Answer Snippet (1–2 lines) |
|-----|------------------|--------------------------------------|
| 1003.pdf | What does Section 1 cover? | Should reference borrower/application info, not Section 5 |
| 1003.pdf | What does Section 5 cover? | Should reference declarations/questions about funding/history |
| credit_report.pdf | What is the credit score / score range? | Should show score/score band if present |
| CP-Sample.pdf | What is the policy effective date? | Should show the effective date line |

> Tip for evaluation: The Streamlit app supports quick testing by uploading each PDF and running the questions above.
