# FinRAG — RAG Pipeline with Evaluation Framework

A production-grade Retrieval-Augmented Generation system built on Apple's 2023 SEC 10-K filing,
with a full evaluation layer measuring retrieval and answer quality across different configurations.

## Architecture

Document Ingestion → Chunking → Embeddings → FAISS Vector Store → RAG Chain → RAGAS Evaluation → MLflow Tracking

## Tech Stack

- LLM: LLaMA 3.3 70B via Groq
- Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)
- Vector Store: FAISS
- RAG Framework: LangChain
- Evaluation: RAGAS
- Experiment Tracking: MLflow
- Frontend: Streamlit
- Backend: FastAPI

## Quickstart

1. Clone the repo
2. Create virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate
4. Install dependencies: pip install -r requirements.txt
5. Add your Groq API key to .env file: GROQ_API_KEY=your_key_here
6. Build vector store: python retrieval/vectorstore.py
7. Launch dashboard: streamlit run dashboard/app.py

## Project Structure

finrag/
├── ingestion/        # PDF loading and chunking
├── retrieval/        # FAISS vector store and search
├── pipeline/         # RAG chain with LLaMA
├── evaluation/       # RAGAS metrics and golden dataset
├── dashboard/        # Streamlit chat UI
├── api/              # FastAPI endpoints
└── data/             # PDF documents

## Evaluation Results

| Experiment     | Chunk Size | Faithfulness | Answer Relevancy | Context Precision | Answer Correctness |
|----------------|------------|--------------|------------------|-------------------|--------------------|
| Baseline       | 512        | 0.82         | 0.75             | 0.70              | 0.66               |
| Larger Chunks  | 1024       | 0.78         | 0.80             | 0.65              | 0.71               |
| Smaller Chunks | 256        | 0.85         | 0.70             | 0.75              | 0.62               |

## Key Findings

- Smaller chunks improve faithfulness but reduce answer correctness
- Larger chunks improve answer relevancy but hurt context precision
- Baseline 512 chunk size offers the best overall balance
- Keyword boosting alongside semantic search significantly improves retrieval recall

## Sample Questions

- What was Apple's total revenue in 2023?
- How much did Apple spend on R&D?
- What was Apple's gross margin in 2023?
- How much revenue did the Services segment generate?
