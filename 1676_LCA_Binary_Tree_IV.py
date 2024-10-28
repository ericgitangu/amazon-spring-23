from collections import deque
import time
from termcolor import colored

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.val == other.val
        return False

    def __hash__(self):
        return hash(self.val)

class Solution:
    """
    1676. Lowest Common Ancestor of a Binary Tree IV - MEDIUM

    Given the root of a binary tree and an array of TreeNode objects nodes, return the lowest common ancestor (LCA) of all the nodes in nodes. All the nodes will exist in the tree, and all values of the tree's nodes are unique.

    Extending the definition of LCA on Wikipedia: "The lowest common ancestor of n nodes p1, p2, ..., pn in a binary tree T is the lowest node that has every pi as a descendant (where we allow a node to be a descendant of itself) for every valid i". A descendant of a node x is a node y that is on the path from node x to some leaf node.

    Parameters
    ----------
    root : TreeNode
        The root of the binary tree.
    nodes : List[TreeNode]
        An array of TreeNode objects.

    Returns
    -------
    TreeNode
        The lowest common ancestor of all the nodes in the array.

    Examples
    --------
    Example 1:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], nodes = [4,7]
    Output: 2
    Explanation: The lowest common ancestor of nodes 4 and 7 is node 2.

    Example 2:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], nodes = [1]
    Output: 1
    Explanation: The lowest common ancestor of a single node is the node itself.

    Example 3:
    Input: root = [3,5,1,6,2,0,8,null,null,7,4], nodes = [7,6,2,4]
    Output: 5
    Explanation: The lowest common ancestor of the nodes 7, 6, 2, and 4 is node 5.
    """

    def __init__(self, root: TreeNode, nodes: list[TreeNode]):
        """
        Initialize the Solution with the root of the binary tree and the array of nodes.

        Parameters
        ----------
        root : TreeNode
            The root of the binary tree.
        nodes : List[TreeNode]
            An array of TreeNode objects.
        """
        self.root = root
        self.nodes_set = set(nodes)
        
    def __repr__(self):
        return f"Solution(root={self.root}, nodes={self.nodes_set})"
    
    def __str__(self):
        def tree_to_str(node, level=0):
            if not node:
                return ""
            left_str = tree_to_str(node.left, level + 1)
            right_str = tree_to_str(node.right, level + 1)
            return f"{'  ' * level}{node.val}\n{left_str}{right_str}"
        
        root_str = tree_to_str(self.root)
        nodes_str = ', '.join(str(node.val) for node in self.nodes_set)
        return f"Solution(\nroot=\n{root_str}, \nnodes=[{nodes_str}]\n)"
    
    def find_lca(self, root: TreeNode, nodes_set: set[TreeNode]) -> TreeNode:
        if not root or root in nodes_set:
            return root
        left = self.find_lca(root.left, nodes_set)
        right = self.find_lca(root.right, nodes_set)
        if left and right:
            return root
        return left if left else right

    def lowestCommonAncestor(self) -> TreeNode:
        """
        Find the lowest common ancestor (LCA) of all the nodes in the array.

        Returns
        -------
        TreeNode
            The lowest common ancestor of all the nodes in the array.
        """
        return self.find_lca(self.root, self.nodes_set)
    
    def find_lca_iterative(self) -> TreeNode:
        """
        Iterative implementation to find the LCA of the given nodes.
        """
        parent = {self.root: None}
        stack = [self.root]

        # Standard tree traversal to build parent pointers
        while stack:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Collect ancestors for each node
        ancestors = []
        for node in self.nodes_set:
            path = set()
            while node:
                path.add(node)
                node = parent[node]
            ancestors.append(path)

        # Intersect all ancestor paths to find common ancestor
        common_ancestors = set.intersection(*ancestors)
        # The lowest common ancestor is the one deepest in the tree
        lca = max(common_ancestors, key=lambda x: self.depth(x, parent))
        return lca
    
    def depth(self, node, parent) -> int:
        d = 0
        while node:
            node = parent[node]
            d += 1
        return d
    
def build_tree(values):
    if not values:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while i < len(values):
        node = queue.popleft()
        if values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

if __name__ == "__main__":
    # Example 1
    root1 = build_tree([3,5,1,6,2,0,8,None,None,7,4])
    nodes1 = [TreeNode(4), TreeNode(7)]
    sol1 = Solution(root1, nodes1)
    print(str(sol1))
    
    start_time = time.time()
    lca_recursive = sol1.lowestCommonAncestor()
    end_time = time.time()
    print(colored(f"Recursive LCA: {lca_recursive.val}, Time taken: {end_time - start_time:.6f} seconds", 'green'))
    
    start_time = time.time()
    lca_iterative = sol1.find_lca_iterative()
    end_time = time.time()
    print(colored(f"Iterative LCA: {lca_iterative.val}, Time taken: {end_time - start_time:.6f} seconds", 'blue'))
    
    # Example 2
    root2 = build_tree([3,5,1,6,2,0,8,None,None,7,4])
    nodes2 = [TreeNode(1)]
    sol2 = Solution(root2, nodes2)  
    print(str(sol2))
    
    start_time = time.time()
    lca_recursive = sol2.lowestCommonAncestor()
    end_time = time.time()
    print(colored(f"Recursive LCA: {lca_recursive.val}, Time taken: {end_time - start_time:.6f} seconds", 'green'))
    
    start_time = time.time()
    lca_iterative = sol2.find_lca_iterative()
    end_time = time.time()
    print(colored(f"Iterative LCA: {lca_iterative.val}, Time taken: {end_time - start_time:.6f} seconds", 'blue'))
    
    # Example 3
    root3 = build_tree([3,5,1,6,2,0,8,None,None,7,4])
    nodes3 = [TreeNode(7), TreeNode(6), TreeNode(2), TreeNode(4)]
    sol3 = Solution(root3, nodes3)
    print(str(sol3))
    
    start_time = time.time()
    lca_recursive = sol3.lowestCommonAncestor()
    end_time = time.time()
    print(colored(f"Recursive LCA: {lca_recursive.val}, Time taken: {end_time - start_time:.6f} seconds", 'green'))
    
    start_time = time.time()
    lca_iterative = sol3.find_lca_iterative()
    end_time = time.time()
    print(colored(f"Iterative LCA: {lca_iterative.val}, Time taken: {end_time - start_time:.6f} seconds", 'blue'))