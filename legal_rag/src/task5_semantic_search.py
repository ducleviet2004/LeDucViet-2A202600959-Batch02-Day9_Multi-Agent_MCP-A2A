"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

VECTORSTORE_FILE = Path(__file__).parent.parent / "data" / "vectorstore.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity từ ChromaDB (fallback sang JSON).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # 1. Thử truy vấn qua ChromaDB làm phương án chính
    chroma_dir = Path(__file__).parent.parent / "data" / "chroma_db"
    if chroma_dir.exists():
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            # Kiểm tra xem collection tồn tại
            collection = client.get_collection("news_and_legal")
            
            # Encode câu truy vấn
            model = SentenceTransformer(EMBEDDING_MODEL)
            query_vector = model.encode(query).tolist()
            
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            formatted_results = []
            if results and results.get("documents") and results["documents"]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for doc, meta, dist in zip(documents, metadatas, distances):
                    # Trong Chroma với khoảng cách cosine, score = 1.0 - distance
                    score = 1.0 - dist
                    formatted_results.append({
                        "content": doc,
                        "score": float(score),
                        "metadata": meta
                    })
                # Trả về kết quả đã được sắp xếp từ Chroma
                return formatted_results
        except Exception as e:
            # Fallback sang JSON nếu Chroma lỗi
            pass

    # 2. Phương án fallback sử dụng file JSON cục bộ
    if not VECTORSTORE_FILE.exists():
        return []

    # Load chunks
    try:
        chunks = json.loads(VECTORSTORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not chunks:
        return []

    # Load model and encode query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode(query)

    results = []
    for chunk in chunks:
        if "embedding" not in chunk:
            continue
        chunk_vector = np.array(chunk["embedding"])
        score = np.dot(query_vector, chunk_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector) + 1e-9)

        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {})
        })

    # Sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]



if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
