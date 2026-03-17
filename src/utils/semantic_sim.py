from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_chunks(text, chunk_size=200):
    text = text.split()
    return [" ".join(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size)]

def semantic_similarity(pred, gt):
    chunks1 = get_chunks(pred)
    chunks2 = get_chunks(gt)

    emb1 = model.encode(chunks1, convert_to_tensor=True)
    emb2 = model.encode(chunks2, convert_to_tensor=True)

    # Average embeddings across chunks
    avg1 = emb1.mean(dim=0)
    avg2 = emb2.mean(dim=0)

    similarity = util.cos_sim(avg1, avg2)
    return similarity.item()