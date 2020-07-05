from .probability_recommender import ProbabilityRecommender
from .cooccurrence_recommender import CooccurrenceRecommender

RECOMMENDER_ALGORITHMS = {
    "probability": ProbabilityRecommender,
    "cooccurrence": CooccurrenceRecommender
}