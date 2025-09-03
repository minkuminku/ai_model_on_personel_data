I used mac to perform these steps, modify steps accordingly if you use diiferent operating system.
# 1) Make a project folder
mkdir faiss_rag && cd faiss_rag

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # (fish: source .venv/bin/activate.fish)

# 3) Upgrade pip (often avoids install issues)
python -m pip install --upgrade pip setuptools wheel


# 4 ) pip install faiss-cpu sentence-transformers numpy pandas pdfplumber openpyxl tqdm

# 5 ) mkdir -p data/pdfs data/excels data/transcripts index src


Copy files to below locations
copy src/2_ingest.py
copy src/query.py
copy src/ans_bedrock.py

Copy Data files to below locations 

data/transcripts should have data1.txt and data2.txt or any other file you like to add.

data/excels can have any excel files that you want to process.
