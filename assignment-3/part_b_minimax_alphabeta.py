import math

class Node:
    def __init__(self, name, is_max_node=False, value=None):
        self.name = name
        self.is_max_node = is_max_node
        self.value = value
        self.children = []
        self.alpha = -math.inf
        self.beta = math.inf
        self.pruned = False

def build_game_tree():
    root = Node('A', is_max_node=True)
    
    b = Node('B', is_max_node=False)
    c = Node('C', is_max_node=False)
    root.children = [b, c]
    
    d = Node('D', is_max_node=True)
    e = Node('E', is_max_node=True)
    f = Node('F', is_max_node=True)
    g = Node('G', is_max_node=True)
    
    b.children = [d, e]
    c.children = [f, g]
    
    h1 = Node('H1', value=8)
    h2 = Node('H2', value=7)
    i1 = Node('I1', value=9)
    i2 = Node('I2', value=2)
    
    d.children = [h1, h2]
    e.children = [i1, i2]
    
    j1 = Node('J1', value=11)
    j2 = Node('J2', value=8)
    k1 = Node('K1', value=10)
    k2 = Node('K2', value=3)
    
    f.children = [j1, j2]
    g.children = [k1, k2]
    
    l1 = Node('L1', value=12)
    l2 = Node('L2', value=4)
    m1 = Node('M1', value=6)
    m2 = Node('M2', value=9)
    n1 = Node('N1', value=6)
    n2 = Node('N2', value=14)
    o1 = Node('O1', value=12)
    o2 = Node('O2', value=20)
    p1 = Node('P1', value=10)
    p2 = Node('P2', value=2)
    
    h1.children = [l1, l2]
    h2.children = [m1, m2]
    i1.children = [n1, n2]
    i2.children = [o1, o2]
    j1.children = [p1, p2]
    j2.children = []
    k1.children = []
    k2.children = []
    
    return root

def minimax_with_alphabeta(node, depth, alpha, beta, is_maximizing, visited_nodes=None):
    if visited_nodes is None:
        visited_nodes = []
    
    visited_nodes.append(node.name)
    
    if node.value is not None:
        return node.value
    
    if is_maximizing:
        max_eval = -math.inf
        for child in node.children:
            eval_score = minimax_with_alphabeta(child, depth + 1, alpha, beta, False, visited_nodes)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha:
                child.pruned = True
                break
        return max_eval
    else:
        min_eval = math.inf
        for child in node.children:
            eval_score = minimax_with_alphabeta(child, depth + 1, alpha, beta, True, visited_nodes)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            
            if beta <= alpha:
                child.pruned = True
                break
        return min_eval

def minimax_without_alphabeta(node, is_maximizing):
    if node.value is not None:
        return node.value
    
    if is_maximizing:
        max_eval = -math.inf
        for child in node.children:
            eval_score = minimax_without_alphabeta(child, False)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = math.inf
        for child in node.children:
            eval_score = minimax_without_alphabeta(child, True)
            min_eval = min(min_eval, eval_score)
        return min_eval

def print_tree_structure(node, level=0):
    indent = "  " * level
    node_type = "MAX" if node.is_max_node else "MIN" if node.value is None else "LEAF"
    value_info = f" (value={node.value})" if node.value is not None else ""
    pruned_info = " [PRUNED]" if hasattr(node, 'pruned') and node.pruned else ""
    
    print(f"{indent}{node.name} ({node_type}){value_info}{pruned_info}")
    
    for child in node.children:
        print_tree_structure(child, level + 1)

def get_all_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes

def count_pruned_nodes(node):
    pruned_count = 0
    if hasattr(node, 'pruned') and node.pruned:
        pruned_count += 1
    for child in node.children:
        pruned_count += count_pruned_nodes(child)
    return pruned_count

def main():
    print("Minimax Algorithm with Alpha-Beta Pruning")
    print("=" * 50)
    
    root = build_game_tree()
    
    print("Game Tree Structure:")
    print_tree_structure(root)
    print()
    
    print("Leaf Node Values:")
    all_nodes = get_all_nodes(root)
    leaf_nodes = [node for node in all_nodes if node.value is not None]
    for node in sorted(leaf_nodes, key=lambda x: x.name):
        print(f"{node.name}: {node.value}")
    print()
    
    print("Minimax without Alpha-Beta Pruning:")
    result_no_pruning = minimax_without_alphabeta(root, True)
    print(f"Optimal value for root node A: {result_no_pruning}")
    print()
    
    print("Minimax with Alpha-Beta Pruning:")
    visited_nodes = []
    result_with_pruning = minimax_with_alphabeta(root, 0, -math.inf, math.inf, True, visited_nodes)
    print(f"Optimal value for root node A: {result_with_pruning}")
    print()
    
    print("Alpha-Beta Pruning Analysis:")
    pruned_count = count_pruned_nodes(root)
    total_nodes = len(all_nodes)
    print(f"Total nodes in tree: {total_nodes}")
    print(f"Nodes visited: {len(visited_nodes)}")
    print(f"Nodes pruned: {pruned_count}")
    print(f"Efficiency: {((total_nodes - pruned_count) / total_nodes * 100):.1f}%")
    print()
    
    print("Visited nodes order:", " -> ".join(visited_nodes))
    print()
    
    print("Pruned nodes:")
    pruned_nodes = []
    def collect_pruned(node):
        if hasattr(node, 'pruned') and node.pruned:
            pruned_nodes.append(node.name)
        for child in node.children:
            collect_pruned(child)
    
    collect_pruned(root)
    if pruned_nodes:
        print(", ".join(pruned_nodes))
    else:
        print("No nodes were pruned")
    print()
    
    print("Decision Analysis:")
    print("Root node A (MAX) has two children: B and C")
    print("A should choose the action that leads to the maximum value")
    print(f"Optimal move: Choose action that results in value {result_with_pruning}")
    
    print("\nAlgorithm Comparison:")
    print(f"Minimax (no pruning): Evaluated all nodes")
    print(f"Minimax (with pruning): Visited {len(visited_nodes)} nodes")
    print(f"Performance improvement: {((total_nodes - len(visited_nodes)) / total_nodes * 100):.1f}%")

if __name__ == "__main__":
    main()
