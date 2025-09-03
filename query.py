import os, json, numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_meta():
    id_list = json.loads((INDEX_DIR / "idmap.json").read_text(encoding="utf-8"))
    # id -> (meta, text)
    meta = {}
    with open(INDEX_DIR / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta[obj["id"]] = {"meta": obj["meta"], "text": obj["text"]}
    return id_list, meta

def search(query: str, k=5, filter_tag=None):
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    id_list, meta = load_meta()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # normalize_embeddings=True to match IndexFlatIP cosine setup
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    scores, idxs = index.search(q, k*10)  # overfetch then filter (useful if you filter by tag)
    idxs = idxs[0]
    scores = scores[0]

    results = []
    for i, s in zip(idxs, scores):
        if i == -1:
            continue
        rid = id_list[i]
        entry = meta.get(rid)
        if not entry:
            continue
        if filter_tag and entry["meta"].get("tag") != filter_tag:
            continue
        results.append({"score": float(s), "id": rid, **entry})
        if len(results) == k:
            break
    return results

if __name__ == "__main__":
    print("Loaded model:", EMBED_MODEL_NAME)
    while True:
        q = input("\nAsk: ").strip()
        if not q:
            break
        hits = search(q, k=5)  # or search(q, k=5, filter_tag="personal")
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            snippet = h["text"][:300].replace("\n", " ")
            print(f"\n#{i}  score={h['score']:.4f}  source={meta.get('source')}  type={meta.get('type')} chunk={meta.get('chunk')}")
            print(f"    {snippet}...")

