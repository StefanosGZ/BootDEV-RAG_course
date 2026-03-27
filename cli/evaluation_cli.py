import argparse
import json
import os

from lib.hybrid_search import *
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/golden_dataset.json")
    # run evaluation logic here
    with open(DATA_PATH) as f:
        golden_dataset = json.load(f)
    documents = load_movies()
    hs = HybridSearch(documents)
    k = 60
    print(f"k={limit}\n")
    for test in golden_dataset["test_cases"]:
        rrf_results = hs.rrf_search(test["query"], k, limit)

        retrieved_titles = [r["doc"]["title"] for r in rrf_results]
        found_titles = len(set(test["relevant_docs"]) & set(retrieved_titles))

        precision = found_titles/len(retrieved_titles)
        recall = found_titles/len(test["relevant_docs"])
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {test["query"]}")
        print(f"  - Percision@{limit}: {precision :.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 score: {f1:.4f}")
        print(f"  - Retrieved: {", ".join(retrieved_titles)}")
        print(f"  - Relevant:  {", ".join(test["relevant_docs"])}\n")

if __name__ == "__main__":
    main()