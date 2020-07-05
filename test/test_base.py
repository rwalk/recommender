import random
from unittest import TestCase
from recommender.base_recommender import BaseRecommender

class TestBaseClass(TestCase):

    def test_top(self):
        random.seed(2)
        items = [{ "score": random.random()} for _ in range(10000)]
        top_sort = sorted(items, key=lambda x: x["score"], reverse=True)[:10]
        recommender = BaseRecommender(items)
        self.assertEqual(recommender._top(items, scorer=lambda x: x["score"]), top_sort)