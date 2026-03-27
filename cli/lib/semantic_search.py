from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies

import re
import numpy as np
import os
import logging
import json

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace")
        encoded_text = self.model.encode([text]) 
        return encoded_text[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        document_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            document_list.append(f"{doc["title"]}: {doc["description"]}")
        
        self.embeddings = self.model.encode(document_list, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedded_query = self.generate_embedding(query)
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(embedded_query, embedding)
            similarities.append((similarity_score, self.documents[i]))
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        return [{"score": result[0], "title": result[1].get("title"), "description": result[1].get("description")} for result in similarities[:limit]]

#--------------------------------------------------------------------------------------------------------

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        
    def build_chunk_embeddings(self, documents):
        all_chunks = []
        chunk_metadata = []
        
        self.documents = documents
        for i, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"]:
                continue
            else:
                chunk = semantic_chunk(doc["description"], max_chunk_size = 4, overlap = 1)
                all_chunks.extend(chunk)
                for chunk_idx, C in enumerate(chunk):
                    chunk_metadata.append({"movie_idx": i, 
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunk)
                    })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        self.chunk_metadata = chunk_metadata
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int=10):
        embedded_query = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embedded_query, chunk_embedding)
            chunk_score = {"chunk_idx": self.chunk_metadata[i].get("chunk_idx"),
            "movie_idx": self.chunk_metadata[i].get("movie_idx"),
            "score": similarity}
            chunk_scores.append(chunk_score)
        
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score.get("movie_idx")
            score = chunk_score.get("score")
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = chunk_score
        
        sorted_movie_scores = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)
        sorted_movie_scores = sorted_movie_scores[:limit]
        
        results = []
        for chunk_score in sorted_movie_scores:
            movie_idx = chunk_score["movie_idx"]
            document = self.documents[movie_idx]
            results.append({
                "id": document["id"],
                "title": document["title"],
                "document": document["description"][:100],
                "score": round(chunk_score["score"], 4),
                "metadata": chunk_score
            })
        
        return results
#-----------------------------------------------------------------------------------------------------------

def simple_chunk(text, chunk_size, overlap):
    split_text = text.split()
    chunks = []
    for i in range(0, len(split_text), chunk_size - overlap):
        chunk= " ".join(split_text[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def semantic_chunk(text, max_chunk_size, overlap):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        return [sentences[0].strip()]
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences) - max_chunk_size + 1, max_chunk_size - overlap):
        chunk = " ".join(sentences[i:i+max_chunk_size])
        chunks.append(chunk)
    return chunks

def verify_model():
    ss = SemanticSearch()
    MODEL = ss.model
    MAX_LENGTH = MODEL.max_seq_length

    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("The vectors are not the same length")
    dot_product = np.dot(vec1, vec2)
    mag_vec1 = np.linalg.norm(vec1)
    mag_vec2 = np.linalg.norm(vec2)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0.0
    return dot_product / (mag_vec1 * mag_vec2)