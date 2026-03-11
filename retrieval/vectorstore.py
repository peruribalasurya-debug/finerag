import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "faiss_chunks.pkl"

model = SentenceTransformer(EMBED_MODEL)


def build_vectorstore(chunks: list):
    print(f"\nBuilding vector store from {len(chunks)} chunks...")

    texts = [c["page_content"] for c in chunks]

    print("Generating embeddings... (this may take 1-2 minutes)")
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Vector store built with {index.ntotal} vectors")
    print(f"✅ Saved to {INDEX_PATH}")
    return index, chunks


def load_vectorstore():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Vector store not found. Run build_vectorstore first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded vector store with {index.ntotal} vectors")
    return index, chunks


def search(query: str, index, chunks: list, top_k: int = 5) -> list:
    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32), top_k
    )

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "content": chunks[idx]["page_content"],
            "metadata": chunks[idx]["metadata"],
            "score": float(distances[0][i])
        })

    return results


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from ingestion.loader import load_pdf
    from ingestion.chunker import fixed_size_chunks

    pages = load_pdf("data/apple_10k.pdf")
    chunks = fixed_size_chunks(pages)
    index, chunks = build_vectorstore(chunks)

    print("\n🔍 Testing search...")
    print("-" * 50)
    query = "What was Apple's total revenue in 2023?"
    results = search(query, index, chunks, top_k=3)

    print(f"Query: {query}\n")
    for i, r in enumerate(results):
        print(f"Result {i+1} (page {r['metadata']['page']}):")
        print(r["content"][:300])
        print()