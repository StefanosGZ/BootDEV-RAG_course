import argparse
import time

from lib.hybrid_search import *
from lib.search_utils import load_movies
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize BM25 and semantic scores to be 0-1")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="List of scores to be normalized")

    weighted_search_hybrid = subparsers.add_parser("weighted-search", help="Do weighted search")
    weighted_search_hybrid.add_argument("query", type=str, help="Query to be performed")
    weighted_search_hybrid.add_argument("--alpha", type=float, default=0.5, help="Alpha value")
    weighted_search_hybrid.add_argument("--limit", type=int, default=5, help="The limit to be retrieved")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Do reciprocal rank fusion")
    rrf_search_parser.add_argument("query", type=str, help="Query to be asked")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="K hyperparameter for rrf search")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit the results to be fetched")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank the query")
    rrf_search_parser.add_argument("--evaluate", action='store_true', help="Evaluate the output")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            hybrid_results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(hybrid_results):
                title = result["doc"]["title"]
                print(f"{i+1}. {title}")
                hybrid_score = result["hybrid_score"]
                bm25_score = result["bm25_score"]
                semantic_score = result["semantic_score"]
                print(f"   Hybrid Score: {hybrid_score:.3f}")
                print(f"   BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f}")
                print(f"   {result["doc"]["description"][:100]}...")

        case "rrf-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            query = args.query
            #print(f"[DEBUG] Original query: {query}")
            limit = args.limit
            client = genai.Client(api_key=api_key)

            if args.enhance:
                enhanced = hs.enhance_query(client, query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'")
                #print(f"[DEBUG] After enhancement: {enhanced}")
                query = enhanced

            if args.rerank_method == "individual":
                rrf_results = hs.rrf_search(query, args.k, limit * 5)
                #print(f"[DEBUG] RRF results: {[r['doc']['title'] for r in rrf_results]}")
                print(f"Re-ranking top {limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                sorted_results = hs.rerank_individual(client, rrf_results, query, limit)
                #print(f"[DEBUG] After reranking: {[r['doc']['title'] for r in sorted_results]}")
                for i, result in enumerate(sorted_results):
                    print(f"{i+1}. {result['doc']['title']}")
                    print(f"   Re-rank Score: {result['rerank_score']:.3f}/10")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                    print(f"   {result['doc']['description'][:100]}")
            
            elif args.rerank_method == "batch":
                rrf_results = hs.rrf_search(query, args.k, limit * 5)
                #print(f"[DEBUG] RRF results: {[r['doc']['title'] for r in rrf_results]}")
                print(f"Re-ranking top {limit} results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k})")
                sorted_results = hs.rerank_batch(client, rrf_results, query, limit)
                #print(f"[DEBUG] After reranking: {[r['doc']['title'] for r in sorted_results]}")
                for i, result in enumerate(sorted_results):
                    print(f"{i+1}. {result['doc']['title']}")
                    print(f"   Re-rank Rank: {result['rerank_rank']}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                    print(f"   {result['doc']['description'][:100]}")
            
            elif args.rerank_method == "cross_encoder":
                rrf_results = hs.rrf_search(query, args.k, limit * 5)
                #print(f"[DEBUG] RRF results: {[r['doc']['title'] for r in rrf_results]}")
                print(f"Re-ranking top {limit * 5} results using cross_encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k})")
                sorted_results = hs.cross_encoder(rrf_results, query, limit)
                #print(f"[DEBUG] After reranking: {[r['doc']['title'] for r in sorted_results]}")
                for i, result in enumerate(sorted_results):
                    print(f"{i+1}. {result['doc']['title']}")
                    print(f"   Cross Encoder Score: {result["cross_encoder_score"]:.3f}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                    print(f"   {result['doc']['description'][:100]}")

            else:
                rrf_results = hs.rrf_search(query, args.k, limit)
                for i, result in enumerate(rrf_results):
                    print(f"{i+1}. {result['doc']['title']}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
                    print(f"   {result['doc']['description'][:100]}")
                sorted_results = rrf_results
            
            if args.evaluate:
                evaluated_results = evaluate(query, sorted_results, client)
                print("\n\n")
                for i, (result, evaluated_result) in enumerate(zip(sorted_results, evaluated_results)):
                    print(f"{i+1}. {result['doc']['title']}: {evaluated_result}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()