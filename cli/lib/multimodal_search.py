import logging

from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

class MultiModalSearch():
    def __init__(self, documents=None, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img):
        image = Image.open(img)
        encoded_image = self.model.encode([image])
        return encoded_image[0]

    def search_with_image(self, img):
        img_embedding = self.embed_image(img)
        
        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(img_embedding, text_embedding)
            similarities.append({
                "id": i,
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "similarity_score": similarity
            })
        sorted_similarities = sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)
        return sorted_similarities[:5]


def verify_image_embedding(img):
    mms = MultiModalSearch()
    embedding = mms.embed_image(img)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(img):
    documents = load_movies()
    mms = MultiModalSearch(documents)
    results = mms.search_with_image(img)
    return results