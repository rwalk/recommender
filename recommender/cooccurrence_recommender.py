import argparse
import json
import sys
import numpy as np

from scipy.sparse import load_npz, coo_matrix
from .base_recommender import BaseRecommender

class CooccurrenceRecommender(BaseRecommender):
    
    def __init__(self, U, items):
        '''
        A Co-occurence based recommendation engine
        U: sparse CSR matrix n users by m items
        items: list(dict) of item metadata where each dict contains at least the keys title, index
        '''
        super().__init__(items)
        self._U = U

    def _build_query_vector(self, indexes):
        # build the query vector
        data, I, J = [], [], []
        for idx in indexes:
            if idx:
                I.append(idx)
                J.append(0)
                data.append(1)
        q = coo_matrix((data, (I, J)), shape=(self._U.shape[-1], 1), dtype=np.float64).tocsr()
        return q

    def _score(self, q, number):
        y = self._U.transpose().dot(self._U.dot(q))
        recs = [{
                "item": self._items[i],
                "score": float(score),
             } for i, score in zip(y.indices, y.data)
        ]
        return self._top(recs, lambda x: x["score"], n=number)

    def recommend(self, number=10, **kwargs):
        indexes = self._item_index_lookup(**kwargs)
        q = self._build_query_vector(indexes)
        return self._score(q, number)
