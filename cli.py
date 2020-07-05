import argparse
import json
import sys

from scipy.sparse import load_npz
from recommender import RECOMMENDER_ALGORITHMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recommender CLI")
    parser.add_argument("matrix_file", help="Sparse user item matrix in npz format")
    parser.add_argument("items_file", help="File of JSON array where each item contains at least the keys title, index")
    parser.add_argument("-a", "--algorithm", choices=RECOMMENDER_ALGORITHMS.keys(), default="cooccurrence", help="Algorithm")

    args = parser.parse_args()

    U = load_npz(args.matrix_file)
    with open(args.items_file) as f:
        items = json.load(f)

    recommender = RECOMMENDER_ALGORITHMS[args.algorithm](U, items)

    try:
        while True:
            q_string = input("Enter a query as JSON (type 'example' for help):\n")
            if q_string.lower().strip() == "example":
                print("Example: {\"authors\": \"Paul Bowles\", \"title\": \"Sky\"}")
            elif len(q_string.strip()) == 0:
                pass
            else:
                try:
                    query = json.loads(q_string)
                    for hit in recommender.recommend(number=5, **query):
                        print(json.dumps(hit, indent=2))
                except json.decoder.JSONDecodeError:
                    print("Query is not valid JSON!")
    except KeyboardInterrupt:
        sys.exit(0)