from collections import deque

def build_graph():
    graph = {
        'A': ['B', 'H'],
        'B': ['A', 'C', 'I'],
        'C': ['B', 'D', 'G'],
        'D': ['C', 'E', 'F'],
        'E': ['D'],
        'F': ['D'],
        'G': ['C'],
        'H': ['A', 'J', 'M'],
        'I': ['B'],
        'J': ['H', 'K', 'L'],
        'K': ['J'],
        'L': ['J'],
        'M': ['H']
    }
    return graph

def bfs(graph, start, goal):
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        node, path = queue.popleft()
        
        if node in visited:
            continue
            
        visited.add(node)
        
        if node == goal:
            return path
            
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    
    return None

def dfs(graph, start, goal):
    if start == goal:
        return [start]
    
    stack = [(start, [start])]
    visited = set()
    
    while stack:
        node, path = stack.pop()
        
        if node in visited:
            continue
            
        visited.add(node)
        
        if node == goal:
            return path
            
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    
    return None

def main():
    graph = build_graph()
    start = 'A'
    goal = 'L'
    
    print("Tree Traversal: BFS and DFS")
    print("=" * 40)
    print(f"Start Node: {start}")
    print(f"Goal Node: {goal}")
    print()
    
    print("Graph Structure:")
    for node, neighbors in graph.items():
        print(f"{node}: {neighbors}")
    print()
    
    bfs_path = bfs(graph, start, goal)
    dfs_path = dfs(graph, start, goal)
    
    print("BFS (Breadth-First Search) Results:")
    print(f"Path: {' -> '.join(bfs_path) if bfs_path else 'No path found'}")
    print(f"Path Length: {len(bfs_path) - 1 if bfs_path else 'N/A'}")
    print()
    
    print("DFS (Depth-First Search) Results:")
    print(f"Path: {' -> '.join(dfs_path) if dfs_path else 'No path found'}")
    print(f"Path Length: {len(dfs_path) - 1 if dfs_path else 'N/A'}")
    print()
    
    print("Comparison:")
    if bfs_path and dfs_path:
        print(f"BFS finds shortest path: {len(bfs_path) <= len(dfs_path)}")
        print(f"DFS path length: {len(dfs_path)}")
        print(f"BFS path length: {len(bfs_path)}")

if __name__ == "__main__":
    main()
