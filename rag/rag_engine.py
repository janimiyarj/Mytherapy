import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, 'vectors', 'faiss_index.bin')
META_PATH = os.path.join(BASE_DIR, 'vectors', 'metadata.json')

# Load once (at startup)
try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print("❌ Error loading FAISS index or metadata:", e)
    index = None
    metadata = {}
    embed_model = None

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    if not index or not embed_model:
        return []

    # Embed query
    query_embedding = embed_model.encode([query])

    # Search top_k
    scores, indices = index.search(np.array(query_embedding).astype('float32'), top_k)

    # Return matching docs from metadata
    return [metadata[str(idx)] for idx in indices[0] if str(idx) in metadata]
