"""
Phase 1: Catalog + RAG
----------------------
Turns catalog.json into a searchable vector index (Chroma) so the agent
can find products by MEANING, not just exact keyword matches, and can
never recommend a product that doesn't actually exist.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Free, local embedding model - no API key needed for this part.
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = chromadb.PersistentClient(path=CHROMA_DIR)


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    """Run once at startup (or whenever catalog.json changes)."""
    catalog = load_catalog()
    collection = _client.get_or_create_collection(
        name="products", embedding_function=_embed_fn
    )
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    ids, docs, metadatas = [], [], []
    for p in catalog:
        pid = str(p["id"])
        if pid in existing_ids:
            continue
        ids.append(pid)
        docs.append(f"{p['name']}. {p['description']}. Category: {p['category']}.")
        metadatas.append(
            {
                "name": p["name"],
                "price": p["price"],
                "category": p["category"],
                "stock": p["stock"],
                "rating": p.get("rating", 0),
            }
        )
    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
    print(f"[rag_store] Index ready. {collection.count()} products indexed.")


def search(query: str, top_k: int = 5, budget: float | None = None, category: str | None = None):
    collection = _client.get_or_create_collection(
        name="products", embedding_function=_embed_fn
    )
    where = {"category": category} if category else None

    results = collection.query(query_texts=[query], n_results=top_k, where=where)

    matches = []
    if not results["ids"][0]:
        return matches

    for i, pid in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        if budget is not None and meta["price"] > budget:
            continue
        matches.append(
            {
                "id": pid,
                "name": meta["name"],
                "price": meta["price"],
                "category": meta["category"],
                "stock": meta["stock"],
                "rating": meta["rating"],
                "match_score": round(1 - distance, 3),
            }
        )
    return matches


if __name__ == "__main__":
    build_index()
    print(search("something for dry skin", budget=1500))