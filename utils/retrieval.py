import requests 
import os 
import faiss
import pickle
import numpy as np

from utils.chunking import create_chunks
from utils.embedding import generate_embeddings

def create_faiss_index():
    if os.path.exists("faiss_store/index.faiss"):
        index=faiss.read_index("faiss_store/index.faiss")
        with open("faiss_store/chunk_mapping.pkl",'rb') as f:
            chunk_mapping=pickle.load(f)
    else:
        with open("data/founder.txt",'r',encoding="utf-8") as f:
            text=f.read()

        chunks=create_chunks(text)
        chunk_mapping=[]

        index=faiss.IndexFlatL2(1536)

        for chunk in chunks:
            emb=generate_embeddings(chunk)
            index.add(np.array([emb]).astype("float32"))
            chunk_mapping.append(chunk)

        os.makedirs("faiss_store",exist_ok=True)
        faiss.write_index(index,"faiss_store/index.faiss")

        with open("faiss_store/chunk_mapping.pkl",'wb') as f:
            pickle.dump(chunk_mapping,f)
    return index,chunk_mapping

def retrieve_top_k_chunks(query,index,chunk_mapping,k=3):
    query_vec=generate_embeddings(query)
    D,I=index.search(np.array([query_vec]).astype("float32"),k)
    return [chunk_mapping[i] for i in I[0]]