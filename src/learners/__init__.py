from .nq_learner import NQLearner
from .msp_learner import MSPLearner


REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["csp"] = MSPLearner