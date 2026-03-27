#!/usr/bin/env python3
import argparse
import json
import math

from lib.invertedindex import InvertedIndex
from lib.keyword_search import search_command, bm25_idf_command, bm25_tf_command
from lib.search_utils import load_movies, BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get IDF score for a term")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF of a term in a database")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to loog up")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default = 5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)
            results = search_command(idx, args.query)
            for result in results:
                print(f"{result["id"]}: {result["title"]}")
            

        case "build":
            data = load_movies()
            idx = InvertedIndex()
            idx.build(data)
            idx.save()

        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)
            idx_freq = idx.get_tf(args.doc_id, args.term)
            print(idx_freq)
        
        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)
            total_doc_count = len(idx.docmap)
            term_match_doc_count = len(idx.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
        case "tfidf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)

            tf = idx.get_tf(args.doc_id, args.term)

            total_doc_count = len(idx.docmap)
            term_match_doc_count = len(idx.get_documents(args.term))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)            
            bm25idf = bm25_idf_command(idx, args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)       
            bm25tf = bm25_tf_command(idx, args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                exit(1)  
            results = idx.bm25_search(args.query, args.limit)
            for result in results:
                doc_id = result[0]
                bm25 = result[1]
                title = idx.docmap[doc_id]["title"]
                print(f"({doc_id}) {title} - Score: {bm25:.2f}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()
