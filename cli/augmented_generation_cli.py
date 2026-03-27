import argparse

from lib.search_utils import load_movies
from lib.hybrid_search import *
from lib.augmented_generation import *


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform summarized RAG"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarizing RAG")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit of retrieved documents")

    citations_parser = subparsers.add_parser(
        "citations", help="Citation aware RAG"
    )
    citations_parser.add_argument("query", type=str, help="Search query for summarizing RAG")
    citations_parser.add_argument("--limit", type=int, default=5, help="Limit of retrieved documents")

    question_parser = subparsers.add_parser(
    "question", help="Citation aware RAG"
    )
    question_parser.add_argument("query", type=str, help="Search query for summarizing RAG")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit of retrieved documents")


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            documents = load_movies()
            hs = HybridSearch(documents)
            rrf_results = hs.rrf_search(query, limit = 5, k=60)
            docs = [result["doc"] for result in rrf_results]

            rag_response = basic_rag(query, docs)

            print("Search Results:")
            for doc in docs:
                print(f"- {doc["title"]}")
            print("\n\nRAG Response:")
            print(rag_response)

        case "summarize":
            query = args.query
            limit = args.limit

            documents = load_movies()
            hs = HybridSearch(documents)
            rrf_results = hs.rrf_search(query, k=60, limit=limit)

            results = [f"{result["doc"]["title"]}: {result["doc"]["description"]}" for result in rrf_results]
            results_str = "\n".join(results)

            rag_response = summarize_rag(query, results_str)
            print("Search Results:")
            for result in rrf_results:
                print(f"- {result["doc"]["title"]}")
            print(f"\n\nLLM Summary\n{rag_response}")

        case "citations":
            query = args.query
            limit = args.limit

            documents = load_movies()
            hs = HybridSearch(documents)
            rrf_results = hs.rrf_search(query, k=60, limit=limit)

            documents = "\n".join([f"[{i+1}] {result['doc']['title']}: {result['doc']['description']}" for i, result in enumerate(rrf_results)])            
            rag_response = citation_rag(query, documents)
            print("Search Results:")
            for result in rrf_results:
                print(f"- {result["doc"]["title"]}")
            print(f"\n\nLLM Summary\n{rag_response}")

        case "question":
            query = args.query
            limit = args.limit

            documents = load_movies()
            hs = HybridSearch(documents)
            rrf_results = hs.rrf_search(query, k=60, limit=limit)

            context = "\n".join([f"{result["doc"]["title"]}: {result["doc"]["description"]}" for result in rrf_results])
            rag_response = question_rag(query, context)

            print("Search Results:")
            for result in rrf_results:
                print(f"- {result["doc"]["title"]}")
            print(f"\n\nLLM Summary\n{rag_response}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()