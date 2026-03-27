import string

from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopowords, BM25_K1, BM25_B

def bm25_idf_command(idx, term:str) -> float:
    return float(idx.get_bm25_idf(term))

def bm25_tf_command(idx, doc_id: int, term:str, k1=BM25_K1, b = BM25_B):
    return idx.get_bm25_tf(doc_id, term, k1, b)


def search_command(idx, query) -> list[dict]:
    tokens = tokenize_text(query)
    seen_ids = set()
    results = []

    for token in tokens:
        for doc_id in idx.get_documents(token):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            results.append(idx.docmap[doc_id])
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                return results

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    stopwords = load_stopowords()
    stemmer = PorterStemmer()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            token = stemmer.stem(token)
            valid_tokens.append(token)
    return valid_tokens
