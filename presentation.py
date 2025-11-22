# ---------------------------------------------
# Minimum Spanning Tree in Python
# Using Kruskal's Algorithm and Prim's Algorithm
# ---------------------------------------------

# -------------------------
# KRUSKAL’S ALGORITHM
# -------------------------

# Helper: Disjoint Set (Union-Find)
class DisjointSet:
    def _init_(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)

        if xr == yr:
            return False

        # Union by rank
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[yr] < self.rank[xr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1

        return True


def kruskal_mst(nodes, edges):
    """
    nodes: number of vertices
    edges: list of tuples (weight, u, v)
    """
    edges.sort()  # Sort by weight
    ds = DisjointSet(nodes)

    mst = []
    total_cost = 0

    for weight, u, v in edges:
        if ds.union(u, v):   # If adding edge does not form cycle
            mst.append((u, v, weight))
            total_cost += weight

    return mst, total_cost


# -------------------------
# PRIM’S ALGORITHM
# -------------------------
import heapq

def prim_mst(graph, start=0):
    """
    graph: adjacency list representation
    start: starting vertex
    """
    visited = set()
    min_heap = [(0, start, -1)]  # (weight, node, parent)
    mst = []
    total_cost = 0

    while min_heap and len(visited) < len(graph):
        weight, node, parent = heapq.heappop(min_heap)
        if node in visited:
            continue

        visited.add(node)
        if parent != -1:
            mst.append((parent, node, weight))
            total_cost += weight

        for neighbor, w in graph[node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (w, neighbor, node))

    return mst, total_cost


# -------------------------
# SAMPLE INPUT (for testing)
# -------------------------

edges = [
    (4, 0, 1),
    (2, 0, 2),
    (1, 2, 3),
    (5, 1, 3),
    (3, 1, 2)
]

graph = {
    0: [(1, 4), (2, 2)],
    1: [(0, 4), (2, 3), (3, 5)],
    2: [(0, 2), (1, 3), (3, 1)],
    3: [(1, 5), (2, 1)]
}

# Running both algorithms
print("Kruskal MST:")
print(kruskal_mst(4, edges))

print("\nPrim MST:")
print(prim_mst(graph, start=0))