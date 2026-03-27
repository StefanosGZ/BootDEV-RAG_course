from lib.keyword_search import preprocess_text, tokenize_text
from collections import Counter
from lib.search_utils import BM25_K1, CACHE_DIR, BM25_B
import os
import pickle
import math

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")    

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens: 
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        tokens = tokenize_text(term)
        if not tokens:
            return []
        return sorted(self.index.get(tokens[0], set()))

    def build(self, data):
        for m in data:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")
    
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
    
    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Index not found. Run 'build' first.")
        elif not os.path.exists(self.docmap_path):
            raise FileNotFoundError("Docmap not found. Run 'build' first.")
        elif not os.path.exists(self.term_frequencies_path):
            raise FileNotFoundError("Term frequencies not found. Run 'build' first.")
        elif not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError("Doc lengths not found. Run 'build' first.")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
        if not tokens:
            return 0
        doc_counter = self.term_frequencies.get(doc_id, Counter())        
        return doc_counter.get(tokens[0], 0)

    def get_bm25_idf(self, term:str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id:int, term:str, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_lengths()
        doc_length = self.doc_lengths.get(doc_id)
        length_norm = length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __get_avg_doc_lengths(self) -> float:
        sum_doc_lengths = sum(self.doc_lengths.values())
        num_of_docs = len(self.doc_lengths)
        if num_of_docs == 0:
            return 0.0
        return sum_doc_lengths/num_of_docs
    
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf

    def bm25_search(self, query, limit):
        query = tokenize_text(query)
        scores = {doc_id: 0 for doc_id in self.docmap}
        for doc_id in self.docmap.keys():
            for token in query:
                scores[doc_id] += self.bm25(doc_id, token)
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return scores[:limit]