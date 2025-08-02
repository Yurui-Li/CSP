REGISTRY = {}
from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .msp_agent import MSPAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["csp"] = MSPAgent
