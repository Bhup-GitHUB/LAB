import heapq

def build_weighted_graph():
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': [('E', 3)],
        'E': []
    }
    return graph

def uniform_cost_search(graph, start, goal):
    if start == goal:
        return [start], 0
    
    frontier = [(0, start, [start])]
    explored = set()
    
    while frontier:
        cost, node, path = heapq.heappop(frontier)
        
        if node in explored:
            continue
            
        explored.add(node)
        
        if node == goal:
            return path, cost
            
        for neighbor, edge_cost in graph[node]:
            if neighbor not in explored:
                new_cost = cost + edge_cost
                new_path = path + [neighbor]
                heapq.heappush(frontier, (new_cost, neighbor, new_path))
    
    return None, float('inf')

def print_search_process(graph, start, goal):
    print("Uniform Cost Search Process:")
    print("=" * 40)
    
    frontier = [(0, start, [start])]
    explored = set()
    step = 1
    
    while frontier:
        print(f"\nStep {step}:")
        print(f"Frontier: {[(cost, node, path) for cost, node, path in frontier]}")
        print(f"Explored: {sorted(explored)}")
        
        cost, node, path = heapq.heappop(frontier)
        
        if node in explored:
            print(f"Node {node} already explored, skipping...")
            continue
            
        explored.add(node)
        print(f"Exploring: {node} (Cost: {cost}, Path: {' -> '.join(path)})")
        
        if node == goal:
            print(f"\nGoal reached! Final path: {' -> '.join(path)}")
            print(f"Total cost: {cost}")
            return path, cost
            
        print(f"Expanding {node}:")
        for neighbor, edge_cost in graph[node]:
            if neighbor not in explored:
                new_cost = cost + edge_cost
                new_path = path + [neighbor]
                heapq.heappush(frontier, (new_cost, neighbor, new_path))
                print(f"  Add {neighbor} to frontier with cost {new_cost}")
        
        step += 1
        
        if step > 10:
            print("Maximum steps reached")
            break
    
    return None, float('inf')

def main():
    graph = build_weighted_graph()
    start = 'A'
    goal = 'E'
    
    print("Uniform Cost Search (UCS) Algorithm")
    print("=" * 50)
    print(f"Start Node: {start}")
    print(f"Goal Node: {goal}")
    print()
    
    print("Graph Structure (Node -> [(Neighbor, Cost), ...]):")
    for node, edges in graph.items():
        print(f"{node}: {edges}")
    print()
    
    print_search_process(graph, start, goal)
    print("\n" + "=" * 50)
    
    path, cost = uniform_cost_search(graph, start, goal)
    
    print("Final UCS Results:")
    if path:
        print(f"Path: {' -> '.join(path)}")
        print(f"Total Cost: {cost}")
        
        print("\nPath Details:")
        total_cost = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            edge_cost = None
            for neighbor, cost_val in graph[current]:
                if neighbor == next_node:
                    edge_cost = cost_val
                    break
            total_cost += edge_cost
            print(f"{current} -> {next_node}: cost {edge_cost}")
        print(f"Total: {total_cost}")
    else:
        print("No path found")

if __name__ == "__main__":
    main()
