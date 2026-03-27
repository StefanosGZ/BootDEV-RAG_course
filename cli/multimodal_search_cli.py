import argparse
from lib.multimodal_search import *

def main():
    parser = argparse.ArgumentParser(description="Image embedding")
    subparsers = parser.add_subparsers(dest="command")
    
    image_search_parser = subparsers.add_parser("image_search")
    image_search_parser.add_argument("image", type=str)

    verify_parser = subparsers.add_parser("verify_image_embedding")
    verify_parser.add_argument("image", type=str)

    args = parser.parse_args()

    match args.command:
        case "image_search":
            results = image_search_command(args.image)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (similarity: {result['similarity_score']:.3f})")
                print(f"   {result['description'][:100]}...")
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()