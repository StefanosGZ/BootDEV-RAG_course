#!/usr/bin/env python3
import argparse

from lib.semantic_search import *
from lib.search_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    subparsers.add_parser("verify_embeddings", help="Verfy that the loading or creating of embeddings work")
    subparsers.add_parser("embed_chunks", help="Embed the scemantically parsed chunks")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed the query")
    embedquery_parser.add_argument("embed_query_text", type=str, help="Text to be embedded")

    search_parser = subparsers.add_parser("search", help="Search a movie from the semantic search database using cosine similarity")
    search_parser.add_argument("query", type=str, help="Query to the database")
    search_parser.add_argument("--limit", type=int, default=5, help="The limit of how many solutions are retrieved")
    
    chunk_parser = subparsers.add_parser("chunk", help="Simple chunk")
    chunk_parser.add_argument("text", type=str, help="The text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200 ,help="The chunking size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="The amount that chunks overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Scemantically chunk")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="The maximum chunking size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="The amount that chunks overlap")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for a movie using a query")
    search_chunked_parser.add_argument("query", type=str, help="Query to be searched")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="The limit of fetched movies")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.text)
        
        case "verify_embeddings":
            verify_embeddings()
        
        case "embedquery":
            embed_query_text(args.embed_query_text)

        case "search":
            ss = SemanticSearch()
            documents = load_movies()
            embedded_documents = ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result.get("title")} (score: {result.get("score"):.4f})")
        
        case "chunk":
            chunks = simple_chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case "embed_chunks":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        
        case "search_chunked":
            documents = load_movies()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result["document"]}...")
       
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()