"""
Node Reputation Simulation for Web Application

This module provides the interface for the web application,
using the shared simulator components.
"""

import numpy as np
from shared_simulator import run_interactive_simulation

# Set reproducible seed
seed = "storb"
seed = sum(ord(c) for c in seed) % (2**32 - 1)
np.random.seed(seed)

# The run_interactive_simulation function is imported from shared_simulator
# and can be used directly by the web application
