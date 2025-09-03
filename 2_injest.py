import os, json, uuid, re, numpy as np
from pathlib import Path
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

import pdfplumber
import pandas as pd

# -------- CONFIG --------
BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_DIR       = BASE_DIR / "data"
PDF_DIR        = DATA_DIR / "pdfs"
EXCEL_DIR      = DATA_DIR / "excels"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
INDEX_DIR      = BASE_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, runs fast on CPU
# PDFs / transcripts use chunking; Excel/CSV are row-wise (1 row = 1 chunk)
CHUNK_SIZE       = 1200
CHUNK_OVERLAP    = 200
MIN_CHARS        = 60    # drop very tiny text chunks (noise)
MAX_VALUE_LEN    = 200   # clip long cell values when building row text
TOPIC_TAG        = "personal"  # for simple metadata filtering if you want

# Optional limits to avoid huge embeddings on giant sheets
MAX_ROWS_PER_SHEET = None   # e.g., 20000 to cap; None = no cap
MAX_ROWS_PER_CSV   = None

# -------- TEXT NORMALIZATION --------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    # de-hyphenate line breaks like "inves-\n tment" -> "investment"
    s = re.sub(r"-\s*\n\s*", "", s)
    # normalize whitespace
    s = s.replace("\u00A0", " ")           # non-breaking space
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    return s.strip()

# -------- READER: PDFs (text-based only) --------
def read_pdf_text(path: Path) -> str:
    texts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    return normalize_text("\n".join(texts))

# -------- READER: transcripts (.txt) --------
def read_transcript_text(path: Path) -> str:
    return normalize_text(Path(path).read_text(encoding="utf-8", errors="ignore"))

# -------- CHUNKER: for PDFs / transcripts --------
def simple_chunk(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHARS:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# -------- ROW-WISE BUILDERS: Excel / CSV --------
def row_text_from_series(sr: pd.Series) -> str:
    """
    Turn a DataFrame row into a compact, search-friendly text like:
    'Name: Gunjan Moto | Dept: Sales | City: Pune | Year: 2018'
    """
    parts = []
    for col, val in sr.items():
        if pd.isna(val):
            continue
        v = str(val).strip()
        if not v:
            continue
        if len(v) > MAX_VALUE_LEN:
            v = v[:MAX_VALUE_LEN] + "…"
        parts.append(f"{col}: {v}")
    return " | ".join(parts)

def iter_excel_rows(path: Path):
    """
    Yield per-row records for .xlsx (all sheets).
    Each yield = {'id','text','meta':{'type':'excel','source', 'sheet','row_index','tag'}}
    """
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] Could not open Excel {path}: {e}")
        return

    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet, dtype=str)  # keep as strings
        except Exception as e:
            print(f"[WARN] Could not parse sheet {sheet} in {path}: {e}")
            continue

        if df.empty:
            continue

        df = df.fillna("")  # avoid NaN strings
        rows = df.iterrows()
        count = 0
        for idx, row in rows:
            txt = row_text_from_series(row)
            txt = normalize_text(txt)
            if len(txt) < MIN_CHARS:
                continue
            rid = str(uuid.uuid4())
            yield {
                "id": rid,
                "text": txt,
                "meta": {
                    "source": str(path),
                    "type": "excel",
                    "sheet": str(sheet),
                    "row_index": int(idx),   # 0-based index from pandas
                    "tag": TOPIC_TAG
                }
            }
            count += 1
            if MAX_ROWS_PER_SHEET and count >= MAX_ROWS_PER_SHEET:
                break

def iter_csv_rows(path: Path):
    """
    Yield per-row records for CSV.
    """
    try:
        df = pd.read_csv(path, dtype=str)  # keep as strings
    except Exception as e:
        print(f"[WARN] Could not read CSV {path}: {e}")
        return

    if df.empty:
        return

    df = df.fillna("")
    count = 0
    for idx, row in df.iterrows():
        txt = row_text_from_series(row)
        txt = normalize_text(txt)
        if len(txt) < MIN_CHARS:
            continue
        rid = str(uuid.uuid4())
        yield {
            "id": rid,
            "text": txt,
            "meta": {
                "source": str(path),
                "type": "excel",
                "sheet": "csv",
                "row_index": int(idx),
                "tag": TOPIC_TAG
            }
        }
        count += 1
        if MAX_ROWS_PER_CSV and count >= MAX_ROWS_PER_CSV:
            break

# -------- MAIN BUILD --------
def main():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    records = []

    print("Loading & chunking data...")

    # PDFs (text-based)
    for p in tqdm(sorted(PDF_DIR.glob("**/*.pdf")), desc="PDFs"):
        try:
            text = read_pdf_text(p)
            chunks = simple_chunk(text)
            for i, c in enumerate(chunks):
                rid = str(uuid.uuid4())
                records.append({
                    "id": rid,
                    "text": c,
                    "meta": {"source": str(p), "type": "pdf", "chunk": i, "tag": TOPIC_TAG}
                })
        except Exception as e:
            print(f"[WARN] PDF failed {p}: {e}")

    # Excel (.xlsx) row-wise
    for p in tqdm(sorted(EXCEL_DIR.glob("**/*.xlsx")), desc="Excel rows"):
        try:
            for rec in iter_excel_rows(p):
                records.append(rec)
        except Exception as e:
            print(f"[WARN] Excel rows failed {p}: {e}")

    # CSV row-wise
    for p in tqdm(sorted(EXCEL_DIR.glob("**/*.csv")), desc="CSV rows"):
        try:
            for rec in iter_csv_rows(p):
                records.append(rec)
        except Exception as e:
            print(f"[WARN] CSV rows failed {p}: {e}")

    # Transcripts (.txt)
    for p in tqdm(sorted(TRANSCRIPT_DIR.glob("**/*.txt")), desc="Transcripts"):
        try:
            text = read_transcript_text(p)
            chunks = simple_chunk(text)
            for i, c in enumerate(chunks):
                rid = str(uuid.uuid4())
                records.append({
                    "id": rid,
                    "text": c,
                    "meta": {"source": str(p), "type": "transcript", "chunk": i, "tag": TOPIC_TAG}
                })
        except Exception as e:
            print(f"[WARN] Transcript failed {p}: {e}")

    if not records:
        print("No content found. Put files under data/ and re-run.")
        return

    print(f"Embedding {len(records)} chunks/rows with {EMBED_MODEL_NAME}...")
    texts = [r["text"] for r in records]
    vecs  = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine/IP: normalize here
    ).astype("float32")

    dim = vecs.shape[1]

    print("Building FAISS index (cosine via inner product)...")
    index = faiss.IndexFlatIP(dim)  # exact, high-recall
    index.add(vecs)

    # persist FAISS
    tmp = INDEX_DIR / "faiss.index.tmp"
    faiss.write_index(index, str(tmp))    # write atomically
    tmp.replace(INDEX_DIR / "faiss.index")

    # persist metadata + id map (ids aligned with FAISS row order)
    idmap = [r["id"] for r in records]
    with open(INDEX_DIR / "idmap.json", "w", encoding="utf-8") as f:
        json.dump(idmap, f, ensure_ascii=False)

    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "meta": r["meta"], "text": r["text"]}, ensure_ascii=False) + "\n")

    # quick stats
    pdf_n   = sum(1 for r in records if r["meta"]["type"] == "pdf")
    exl_n   = sum(1 for r in records if r["meta"]["type"] == "excel")
    trans_n = sum(1 for r in records if r["meta"]["type"] == "transcript")
    print(f"Done. Index saved under index/ (pdf={pdf_n}, excel_rows={exl_n}, transcripts={trans_n}, total={len(records)})")

if __name__ == "__main__":
    main()

