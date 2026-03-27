import os
import time
import json

from .invertedindex import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def enhance_query(self, client, query, method):
        prompts = {
            "spell": f"""Fix any spelling errors in the user-provided movie search query below.
                         Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                         Preserve punctuation and capitalization unless a change is required for a typo fix.
                         If there are no spelling errors, or if you're unsure, output the original query unchanged.
                         Output only the final query text, nothing else.
                         User query: "{query}"
                         """,
            "rewrite": f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                           Keep the rewritten query concise (under 10 words).
                           Output only the rewritten query text, nothing else.
                           User query: "{query}"
                           """,
            "expand": f"""Expand the user-provided movie search query below with related terms.
                          Add synonyms and related concepts that might appear in movie descriptions.
                          Output only the additional terms; they will be appended to the original query.
                          User query: "{query}"
                          """
        }
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompts[method]
        )
        return response.text.strip()

    def rerank_individual(self, client, results, query, limit):
        for result in results:
            rerank_result = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=f"""Rate how well this movie matches the search query.
                            Query: "{query}"
                            Movie: {result["doc"]["title"]} - {result["doc"]["description"]}
                            Consider:
                            - Direct relevance to query
                            - User intent (what they're looking for)
                            - Content appropriateness
                            Rate 0-10 (10 = perfect match).
                            Output ONLY the number in your response, no other text or explanation.
                            Score:"""
            )
            try:
                result["rerank_score"] = float(rerank_result.text.strip())
            except ValueError:
                result["rerank_score"] = 0.0
            time.sleep(3)
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:limit]
    
    def rerank_batch(self, client, results, query, limit):
        doc_list_str = "\n".join([f"{i+1}. ID: {result['doc']['id']} - {result['doc']['title']}" for i, result in enumerate(results)])
        rerank_results = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"""Rank the movies listed below by relevance to the following search query.

                        Query: "{query}"

                        Movies:
                        {doc_list_str}

                        Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

                        For example:
                        [75, 12, 34, 2, 1]

                        Ranking:"""
        )
        text = rerank_results.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        ranked_ids = json.loads(text)
        results_by_id = {r["doc"]["id"]: r for r in results}
        sorted_results = []
        for rank, doc_id in enumerate(ranked_ids):
            if doc_id in results_by_id:
                results_by_id[doc_id]["rerank_rank"] = rank + 1
                sorted_results.append(results_by_id[doc_id])
        return sorted_results[:limit]

    def cross_encoder(self, results, query, limit):
        pairs = []
        for result in results:
            doc = result["doc"]
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)
        for i, score in enumerate(scores):
            results[i]["cross_encoder_score"] = score
        sorted_results = sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)
        return sorted_results[:limit]

    def weighted_search(self, query, alpha, limit=5):
        bm25 = self._bm25_search(query, limit * 500)
        bm25_doc_id, bm25_scores = zip(*bm25)

        semantic = self.semantic_search.search_chunks(query, limit * 500)
        semantic_scores = [r["score"] for r in semantic]

        bm25_scores_norm = normalize(bm25_scores)
        semantic_scores_norm = normalize(semantic_scores)

        combined = {}
        for doc_id, bm25_score in zip(bm25_doc_id, bm25_scores_norm):
            combined[doc_id] = {
                "doc": self.idx.docmap[doc_id],
                "bm25_score": bm25_score,
                "semantic_score": 0.0
            }
        for i, result in enumerate(semantic):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": self.idx.docmap[doc_id],
                    "bm25_score": 0.0,
                    "semantic_score": semantic_scores_norm[i]
                }
            else:
                combined[doc_id]["semantic_score"] = semantic_scores_norm[i]

        for doc_id in combined:
            combined[doc_id]["hybrid_score"] = hybrid_score(
                combined[doc_id]["bm25_score"], combined[doc_id]["semantic_score"], alpha
            )

        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25 = self._bm25_search(query, limit * 500)
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        combined = {}
        for i, (doc_id, bm25_score) in enumerate(bm25):
            combined[doc_id] = {
                "doc": self.idx.docmap[doc_id],
                "bm25_rank": i + 1,
                "semantic_rank": None
            }
        for i, result in enumerate(semantic):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": self.idx.docmap[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": i + 1
                }
            else:
                combined[doc_id]["semantic_rank"] = i + 1

        for doc_id in combined:
            bm25_rrf = rrf_score(combined[doc_id]["bm25_rank"], k) if combined[doc_id]["bm25_rank"] else 0
            semantic_rrf = rrf_score(combined[doc_id]["semantic_rank"], k) if combined[doc_id]["semantic_rank"] else 0
            combined[doc_id]["rrf_score"] = bm25_rrf + semantic_rrf

        return sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)[:limit]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize(scores):
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(float(s) - min_score) / (max_score - min_score) for s in scores]


def rrf_score(rank, k=60):
    return 1 / (k + rank)

def evaluate(query, sorted_results, client):
    formatted_results = [
    f"{i+1}. {result['doc']['title']}: {result['doc']['description'][:200]}"
    for i, result in enumerate(sorted_results)
    ]
    evaluation_results = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents= f"""Rate how relevant each result is to this query on a 0-3 scale:

                    Query: "{query}"

                    Results:
                    {chr(10).join(formatted_results)}

                    Scale:
                    - 3: Highly relevant
                    - 2: Relevant
                    - 1: Marginally relevant
                    - 0: Not relevant

                    Do NOT give any numbers other than 0, 1, 2, or 3.

                    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                    [2, 0, 3, 2, 0, 1]"""
    )
    return json.loads(evaluation_results.text)