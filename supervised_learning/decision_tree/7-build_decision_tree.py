#!/usr/bin/env python3
"""
Decision Tree and Random Forest implementation
with Node and Leaf classes.
"""
import numpy as np


class Node:
    """A node class that generalizes everything including root and leaves."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Construct the Node object."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Find the maximum depth."""
        if self.is_leaf:
            return self.depth
        left = self.left_child.max_depth_below() if self.left_child else self.depth
        right = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left, right)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below, only leaves if specified."""
        if self.is_leaf:
            return 1

        if only_leaves:
            left = (self.left_child.count_nodes_below(True)
                    if self.left_child else 0)
            right = (self.right_child.count_nodes_below(True)
                     if self.right_child else 0)
            return left + right
        left = (self.left_child.count_nodes_below(False)
                if self.left_child else 0)
        right = (self.right_child.count_nodes_below(False)
                 if self.right_child else 0)
        return 1 + left + right

    def __str__(self):
        """STR"""
        if self.is_root:
            s = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            s = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            left_str = self.left_child.__str__()
            s += "\n" + self.left_child_add_prefix(left_str).rstrip("\n")

        if self.right_child:
            right_str = self.right_child.__str__()
            s += "\n" + self.right_child_add_prefix(right_str).rstrip("\n")

        return s

    def left_child_add_prefix(self, text):
        """Left Child"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Right Child"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def get_leaves_below(self):
        """Get Leaves"""
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update Bounds"""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if not child:
                continue

            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child is self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """Update Indicator"""

        def is_large_enough(x):
            """Large enough"""
            return np.all([x[:, j] >= bound for j, bound in self.lower.items()],
                          axis=0)

        def is_small_enough(x):
            """Small Enough"""
            return np.all([x[:, j] <= bound for j, bound in self.upper.items()],
                          axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Return the prediction"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Terminal node which is a leaf."""

    def __init__(self, value, depth=None):
        """Construct the leaf object."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the count of 1 leaf."""
        return 1

    def __str__(self):
        """STR"""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Get Leaves"""
        return [self]

    def update_bounds_below(self):
        """Update Bounds"""
        pass

    def pred(self, x):
        """Return the prediction"""
        return self.value


class Decision_Tree():
    """The whole Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Construct the decision tree."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the count of leaves."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Get Leaves"""
        return self.root.get_leaves_below()

    def __str__(self):
        """STR"""
        return self.root.__str__() + "\n"

    def update_bounds(self):
        """Update Bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """Return the prediction"""
        return self.root.pred(x)

    def update_predict(self):
        """Return the prediction 2"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: self._predict_from_leaves(A, leaves)

    def _predict_from_leaves(self, A, leaves):
        """Return the prediction __"""
        res = np.zeros(A.shape[0])
        for leaf in leaves:
            mask = leaf.indicator(A)
            res[mask] = leaf.value
        return res

    def fit(self, explanatory, target, verbose=0):
        self.explanatory = explanatory
        self.target = target
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        self.root.sub_population = np.ones_like(target, dtype=bool)
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Depth                     : {self.depth()}\n"
                  f"    - Number of nodes           : {self.count_nodes()}\n"
                  f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}\n"
                  f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """Recursively fit node and its children."""
        sub_target = self.target[node.sub_population]
        if (node.sub_population.sum() <= self.min_pop or
            node.depth >= self.max_depth or
            np.all(sub_target == sub_target[0])):
            node.value = np.bincount(sub_target).argmax()
            leaf = Leaf(node.value)
            leaf.depth = node.depth
            leaf.sub_population = node.sub_population
            return leaf
        node.feature, node.threshold = self.split_criterion(node)
        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (~(
            self.explanatory[:, node.feature] > node.threshold))
        if (left_population.sum() <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.all(self.target[left_population] ==
                   self.target[left_population][0])):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
        if (right_population.sum() <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.all(self.target[right_population] ==
                   self.target[right_population][0])):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Return a leaf node for a given subpopulation."""
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Return an internal node for a given subpopulation."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy on test data."""
        return np.sum(self.predict(test_explanatory) ==
