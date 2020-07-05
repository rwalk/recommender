
import numpy as np

from scipy.sparse import coo_matrix, dia_matrix
from .base_recommender import BaseRecommender

class ProbabilityRecommender(BaseRecommender):
    
    def __init__(self, U, items):
        '''
        A Co-occurence based recommendation engine
        U: sparse CSR matrix n users by m items
        items: list(dict) of item metadata where each dict contains at least the keys title, index
        '''
        super().__init__(items)
        self._items = items
        self._U = U
        n, m = U.shape

        # D matrix
        d = U.multiply(U).sum(axis=0)
        self._D = dia_matrix((d,0), shape=(m,m)).tocsr()

        #S matrix
        s = U.transpose().dot(U.dot(np.ones((m, 1)))) - self._D.dot(np.ones((m, 1)))
        s[np.isclose(s, 0)] = 1
        self._S = dia_matrix(((1/s).flatten(),0), shape=(m,m)).tocsr()

    def _build_query_vector(self, indexes):
        # build the query vector
        data, I, J = [], [], []
        for idx in indexes:
            if idx:
                I.append(idx)
                J.append(0)
                data.append(1)
        q = coo_matrix((data, (I, J)), shape=(self._U.shape[-1], 1), dtype=np.float64).tocsr()
        return q/q.data.sum()

    def _score(self, q, number):
        y = self._S.dot(q)
        r = self._U.transpose().dot(self._U.dot(y)) - self._D.dot(y)
        recs = [{
                "item": self._items[i],
                "score": float(score),
             } for i, score in zip(r.indices, r.data)
        ]
        return self._top(recs, lambda x: x["score"], n=number)

    def recommend(self, number=10, **kwargs):
        indexes = self._item_index_lookup(**kwargs)
        q = self._build_query_vector(indexes)
        return self._score(q, number)
