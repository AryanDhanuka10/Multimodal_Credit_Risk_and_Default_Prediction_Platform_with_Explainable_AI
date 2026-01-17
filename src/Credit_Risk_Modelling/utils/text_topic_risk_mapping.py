import numpy as np


def map_topics_to_risk(topics: np.ndarray):
    """
    Higher topic index ≠ higher risk.
    We rank topics by frequency as a proxy for severity.
    """

    unique, counts = np.unique(topics, return_counts=True)
    freq = dict(zip(unique, counts))

    # More frequent complaint themes = higher systemic risk
    sorted_topics = sorted(freq, key=freq.get, reverse=True)

    risk_map = {
        topic: rank / (len(sorted_topics) - 1)
        for rank, topic in enumerate(sorted_topics)
    }

    return risk_map
