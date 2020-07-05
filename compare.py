import argparse
import json
import csv

from scipy.sparse import load_npz
from recommender import RECOMMENDER_ALGORITHMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recommender Comparison Utility")
    parser.add_argument("matrix_file", help="Sparse user item matrix in npz format")
    parser.add_argument("items_file", help="File of JSON array where each item contains at least the keys title, index")
    parser.add_argument("eval_file", help="File of JSON array queries to compare, each item having a key 'query'")
    parser.add_argument("output_file", help="Output file")

    args = parser.parse_args()

    U = load_npz(args.matrix_file)
    with open(args.items_file) as f:
        items = json.load(f)

    recommenders = {name: recommender(U, items) for name, recommender in RECOMMENDER_ALGORITHMS.items()}

    with open(args.eval_file) as f:
        queries = json.load(f)

    columns = ("description", "query", "rank", "cooccurrence result", "cooccurrence score", "probability result", "probability score")
    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for query in queries:
            description = query["description"]
            q = query["query"]
            query = q["title"]
            if "authors" in q:
                query = query + " by " + q["authors"]

            c_recs, p_recs = recommenders["cooccurrence"].recommend(5, **q), recommenders["probability"].recommend(5, **q)

            for i,(c_rec,p_rec) in enumerate(zip(c_recs, p_recs)):
                c_score = c_rec["score"]
                p_score = p_rec["score"]
                c_item = c_rec["item"]
                p_item = p_rec["item"]
                c = c_item["title"] + " by " + c_item["authors"][0]
                p = p_item["title"] + " by " + p_item["authors"][0]

                writer.writerow((description, query, i+1, c, int(c_score), p, round(p_score, 4)))