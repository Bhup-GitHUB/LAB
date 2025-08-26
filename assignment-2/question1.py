def bfs():

    tree = {
        'A': ['B'],
        'B': ['C', 'H'], 
        'C': ['D'],
        'D': ['E', 'G'],
        'E': ['F'],
        'F': [],
        'G': [],
        'H': ['I', 'J', 'M'],
        'I': [],
        'J': ['K'],
        'K': ['L'],
        'L': [],
        'M': [],
    }
    
    queue = ['A'] 
    result = []
    
    while queue:
        node = queue.pop(0) 
        result.append(node) 
        
       
        for child in tree[node]:
            queue.append(child)
    
    return result


print(bfs())