# src/answer_bedrock.py
import os, sys, json
from pathlib import Path

import numpy as np
import faiss
import boto3
from sentence_transformers import SentenceTransformer
from botocore.exceptions import ClientError

# ---- Config ----
BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "index"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # must match ingestion
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID   = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")  # change if needed

TOP_K = 3
SNIPPET_CHARS = 600  # how many chars per chunk to send to the model

# ---- Helpers ----
def load_index_and_meta():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    id_list = json.loads((INDEX_DIR / "idmap.json").read_text(encoding="utf-8"))
    meta = {}
    with open(INDEX_DIR / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta[obj["id"]] = {"meta": obj["meta"], "text": obj["text"]}
    return index, id_list, meta

def retrieve(question: str, k=TOP_K):
    # embed query
    model = SentenceTransformer(EMBED_MODEL_NAME)
    q = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    index, id_list, meta = load_index_and_meta()
    ntotal = index.ntotal
    if ntotal == 0:
        return []

    k_eff = min(k, ntotal)
    scores, idxs = index.search(q, k_eff)

    hits = []
    for i, s in zip(idxs[0], scores[0]):
        if i == -1:
            continue
        rid = id_list[i]
        entry = meta.get(rid)
        if not entry:
            continue
        hits.append({"score": float(s), "id": rid, **entry})
    return hits

def build_context(hits, per_snippet_chars=SNIPPET_CHARS):
    ctx_lines = []
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["text"].split())[:per_snippet_chars]
        src = h["meta"].get("source")
        ctx_lines.append(f"[{i}] {snippet}\n(source: {src})")
    return "\n\n".join(ctx_lines)

def ask_bedrock_converse(question: str, context: str, model_id: str = MODEL_ID, region: str = AWS_REGION):
    client = boto3.client("bedrock-runtime", region_name=region)

    system_instruction = (
        "Answer strictly using the provided CONTEXT. "
        "If the answer is not explicitly in the context, reply exactly: \"I don't know from the provided context.\" "
        "Be concise and cite sources like [1], [2] where appropriate."
    )
    user_prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}"

    try:
        resp = client.converse(
            modelId=model_id,
            # <<<<<<<<<<  system prompt goes here (NOT as a message role)  >>>>>>>>>>
            system=[{"text": system_instruction}],
            messages=[
                {"role": "user", "content": [{"text": user_prompt}]}
            ],
            inferenceConfig={"maxTokens": 400, "temperature": 0.0, "topP": 1.0},
        )
        return resp["output"]["message"]["content"][0]["text"]
    except ClientError as e:
        # Helpful hints for common issues
        if e.response.get("Error", {}).get("Code") == "ValidationException":
            return (
                "Bedrock ValidationException. Tips:\n"
                "- Use top-level `system=[...]` instead of a 'system' message role.\n"
                "- Ensure MODEL_ID is available in your AWS region.\n"
                "- Check your AWS credentials/permissions for bedrock:InvokeModel.\n"
                f"Raw error: {e}"
            )
        return f"Bedrock call failed: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/answer_bedrock.py "Your question"')
        sys.exit(1)

    question = sys.argv[1]
    hits = retrieve(question, k=TOP_K)

    if not hits:
        print("No retrieved chunks. Did you run ingestion and place files under data/?")
        sys.exit(0)

    print("Top hits used as context:")
    for i, h in enumerate(hits, 1):
        print(f" [{i}] {h['meta'].get('source')} (chunk {h['meta'].get('chunk')}), score={h['score']:.4f}")

    context = build_context(hits)
    answer = ask_bedrock_converse(question, context)

    print("\n=== Answer ===\n" + str(answer) + "\n")

    print("=== Sources ===")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['meta'].get('source')} (chunk {h['meta'].get('chunk')})")

