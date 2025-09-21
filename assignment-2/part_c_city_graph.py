import heapq

def build_city_graph():
    city_graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 3), ('D', 6)],
        'C': [('D', 1), ('E', 1)],
        'D': [('E', 5)],
        'E': []
    }
    return city_graph

def dijkstra_algorithm(graph, start, goal):
    if start == goal:
        return [start], 0
    
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    previous = {node: None for node in graph}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == goal:
            break
            
        for neighbor, weight in graph[current_node]:
            if neighbor in visited:
                continue
                
            new_distance = current_distance + weight
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
    
    if distances[goal] == float('inf'):
        return None, float('inf')
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return path, distances[goal]

def print_city_search_process(graph, start, goal):
    print("City Graph - Least Cost Path Search")
    print("=" * 50)
    
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    previous = {node: None for node in graph}
    
    pq = [(0, start)]
    visited = set()
    step = 1
    
    while pq:
        print(f"\nStep {step}:")
        print(f"Priority Queue: {pq}")
        print(f"Distances: {distances}")
        print(f"Visited: {sorted(visited)}")
        
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            print(f"Node {current_node} already visited, skipping...")
            continue
            
        visited.add(current_node)
        print(f"Processing: {current_node} (Current distance: {current_distance})")
        
        if current_node == goal:
            print(f"\nDestination {goal} reached!")
            break
            
        print(f"Exploring neighbors of {current_node}:")
        for neighbor, weight in graph[current_node]:
            if neighbor in visited:
                print(f"  {neighbor} already visited, skipping...")
                continue
                
            new_distance = current_distance + weight
            print(f"  {current_node} -> {neighbor}: cost {weight}, total cost {new_distance}")
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
                print(f"    Updated distance to {neighbor}: {new_distance}")
            else:
                print(f"    Distance to {neighbor} not improved")
        
        step += 1
        
        if step > 10:
            print("Maximum steps reached")
            break
    
    if distances[goal] == float('inf'):
        return None, float('inf')
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return path, distances[goal]

def main():
    city_graph = build_city_graph()
    start = 'A'
    goal = 'E'
    
    print("Least Cost Path in a City Graph")
    print("=" * 50)
    print("Scenario: Robot finding least-cost path from A to E")
    print("Intersections are nodes, roads are edges with fuel costs")
    print()
    
    print(f"Start Location: {start}")
    print(f"Destination: {goal}")
    print()
    
    print("City Graph Structure:")
    print("Node -> [(Neighbor, Fuel Cost), ...]")
    for node, edges in city_graph.items():
        print(f"{node}: {edges}")
    print()
    
    path, total_cost = print_city_search_process(city_graph, start, goal)
    
    print("\n" + "=" * 50)
    print("Final Results:")
    
    if path:
        print(f"Least Cost Path: {' -> '.join(path)}")
        print(f"Total Fuel Cost: {total_cost}")
        
        print("\nRoute Details:")
        current_cost = 0
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            edge_cost = None
            for neighbor, cost in city_graph[current_node]:
                if neighbor == next_node:
                    edge_cost = cost
                    break
            
            current_cost += edge_cost
            print(f"  {current_node} -> {next_node}: {edge_cost} fuel units")
        
        print(f"  Total: {current_cost} fuel units")
        
        print("\nAlternative Paths Analysis:")
        print("All possible paths from A to E:")
        all_paths = find_all_paths(city_graph, start, goal)
        for i, p in enumerate(all_paths, 1):
            cost = calculate_path_cost(city_graph, p)
            print(f"  Path {i}: {' -> '.join(p)} (Cost: {cost})")
    
    else:
        print("No path found to destination")

def find_all_paths(graph, start, goal):
    all_paths = []
    
    def dfs_all_paths(current, target, path, visited):
        if current == target:
            all_paths.append(path[:])
            return
        
        for neighbor, _ in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs_all_paths(neighbor, target, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    dfs_all_paths(start, goal, [start], {start})
    return all_paths

def calculate_path_cost(graph, path):
    total_cost = 0
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        for neighbor, cost in graph[current]:
            if neighbor == next_node:
                total_cost += cost
                break
    return total_cost

if __name__ == "__main__":
    main()
