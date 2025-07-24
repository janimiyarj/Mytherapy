import pandas as pd
import os
import faiss
import json
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("data/train.csv").dropna()

# Ensure you're extracting clean context
documents = df["Context"].astype(str).tolist()

# Embed documents
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=True)

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index to disk
os.makedirs("rag/vectors", exist_ok=True)
faiss.write_index(index, "rag/vectors/faiss_index.bin")

# Save metadata mapping (id -> document)
with open("rag/vectors/metadata.json", "w") as f:
    json.dump({str(i): doc for i, doc in enumerate(documents)}, f)

print("✅ FAISS index and metadata created.")
