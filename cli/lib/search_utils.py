import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopowords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = f.read().splitlines()
    return stopwords