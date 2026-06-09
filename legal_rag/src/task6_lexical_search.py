"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import json
import numpy as np

# Load corpus từ vector store file
VECTORSTORE_FILE = Path(__file__).parent.parent / "data" / "vectorstore.json"
CORPUS: list[dict] = []

try:
    if VECTORSTORE_FILE.exists():
        CORPUS = json.loads(VECTORSTORE_FILE.read_text(encoding="utf-8"))
except Exception:
    pass


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global CORPUS
    try:
        if VECTORSTORE_FILE.exists():
            CORPUS = json.loads(VECTORSTORE_FILE.read_text(encoding="utf-8"))
        else:
            # Fallback tải corpus từ ChromaDB
            import chromadb
            chroma_dir = Path(__file__).parent.parent / "data" / "chroma_db"
            if chroma_dir.exists():
                client = chromadb.PersistentClient(path=str(chroma_dir))
                collection = client.get_collection("news_and_legal")
                chroma_data = collection.get(include=["documents", "metadatas"])
                if chroma_data and chroma_data.get("documents"):
                    CORPUS = []
                    for doc, meta in zip(chroma_data["documents"], chroma_data["metadatas"]):
                        CORPUS.append({
                            "content": doc,
                            "metadata": meta
                        })
    except Exception:
        pass


    if not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx].get("metadata", {})
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
