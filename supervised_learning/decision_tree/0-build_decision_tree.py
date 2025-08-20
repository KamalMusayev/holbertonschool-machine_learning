#!/usr/bin/env python3
"""Decision Tree and Random Forest"""

import numpy as np


class Node:
    """A decision tree node which may have children and a split feature."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        """Initialize a Node with optional children, split feature, and depth."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth


class Decision_Tree:
    """Decision tree object containing the root node and tree parameters."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize
