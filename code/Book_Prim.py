import sys
from heapq import heappop, heappush, heapify

class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = {}
        self.distance = sys.maxsize
        self.previous = None
        self.color = None

    def get_neighbor(self, other):
        return self.neighbors.get(other, None)

    def set_neighbor(self, other, weight=0):
        self.neighbors[other] = weight

    def __repr__(self):
        return f"Vertex({self.key})"

    def __str__(self):
        return (
            f"{self.key} connected to: "
            + f"{[x.key for x in self.neighbors]}"
        )

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_key(self):
        return self.key

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.key == other.key
        return False

    def __lt__(self, other):
        if isinstance(other, Vertex):
            return self.distance < other.distance
        return False

    def __hash__(self):
        return hash(self.key)


class Graph:
    def __init__(self):
        self.vertices = {}

    def set_vertex(self, key):
        self.vertices[key] = Vertex(key)

    def get_vertex(self, key):
        return self.vertices.get(key, None)

    def __contains__(self, key):
        return key in self.vertices

    def add_edge(self, from_vert, to_vert, weight=0):
        if from_vert not in self.vertices:
            self.set_vertex(from_vert)
        if to_vert not in self.vertices:
            self.set_vertex(to_vert)
        self.vertices[from_vert].set_neighbor(
            self.vertices[to_vert], weight
        )

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())



"""
这个实现使用了一个字典 visited 来跟踪每个顶点的最短已知距离。
如果在优先队列中发现一个顶点的旧记录，通过检查 visited 中记录的距离来决定是否忽略它。
这样可以确保每个顶点的最新距离总是被正确处理，即便它被多次推入堆中。
"""
def prim(graph, start):
    for vertex in graph:
        vertex.distance = sys.maxsize
        vertex.previous = None
    start.distance = 0
    pq = [(vertex.distance, vertex) for vertex in graph]
    heapify(pq)
    visited = {}

    while pq:
        #print(", ".join(f"{(v[1].key, v[1].distance % 1000)}" for v in pq))
        distance, current_v = heappop(pq)
        if current_v in visited and visited[current_v] < distance:
            continue
        for next_v in current_v.get_neighbors():
            new_distance = current_v.get_neighbor(next_v)
            if new_distance < next_v.distance:
                next_v.previous = current_v
                next_v.distance = new_distance
                heappush(pq, (next_v.distance, next_v))
            #print("".join(f"{v.distance % 1000:<5d}" for v in graph))

print("\n---Prim's---\n")
g = Graph()
vertices = ["A", "B", "C", "D", "E", "F", "G"]
for v in vertices:
    g.set_vertex(v)
g.add_edge("A", "B", 2)
g.add_edge("A", "C", 3)
g.add_edge("B", "A", 2)
g.add_edge("B", "C", 1)
g.add_edge("B", "D", 1)
g.add_edge("B", "E", 4)
g.add_edge("C", "A", 3)
g.add_edge("C", "B", 1)
g.add_edge("C", "F", 5)
g.add_edge("D", "B", 1)
g.add_edge("D", "E", 1)
g.add_edge("E", "B", 4)
g.add_edge("E", "D", 1)
g.add_edge("E", "F", 1)
g.add_edge("F", "C", 5)
g.add_edge("F", "E", 1)
g.add_edge("F", "G", 1)
g.add_edge("G", "F", 1)
print("".join(f"{v:5s}" for v in vertices))
prim(g, g.get_vertex("A"))
print(
    "".join(
        f"{g.get_vertex(v).distance:<5d}"
        for v in vertices
    )
)

"""
---Prim's---

A    B    C    D    E    F    G    
0    1    1    1    1    1    1    
"""