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


# print("\n---Graph---\n")
# g = Graph()
# for i in range(6):
#     g.set_vertex(i)
# print(g.vertices)
# g.add_edge(0, 1, 5)
# g.add_edge(0, 5, 2)
# g.add_edge(1, 2, 4)
# g.add_edge(2, 3, 9)
# g.add_edge(3, 4, 7)
# g.add_edge(3, 5, 3)
# g.add_edge(4, 0, 1)
# g.add_edge(5, 4, 8)
# g.add_edge(5, 2, 1)
# for v in g:
#     for w in v.get_neighbors():
#         print(f"({v.get_key()}, {w.get_key()})")



"""
这个实现使用了一个字典 visited 来跟踪每个顶点的最短已知距离。
如果在优先队列中发现一个顶点的旧记录，通过检查 visited 中记录的距离来决定是否忽略它。
这样可以确保每个顶点的最新距离总是被正确处理，即便它被多次推入堆中。
"""
def dijkstra(graph, start):
    pq = [(v.distance, v) for v in graph]
    start.distance = 0
    heapify(pq)
    visited = {}
    while pq:
        distance, current_v = heappop(pq)
        if current_v in visited and visited[current_v] < distance:
            continue
        for next_v in current_v.get_neighbors():
            new_distance = (
                current_v.distance
                + current_v.get_neighbor(next_v)
            )
            if new_distance < next_v.distance:
                next_v.distance = new_distance
                next_v.previous = current_v
                heappush(pq, (next_v.distance, next_v))
                print("".join(f"{v.distance % 1000:<5d}" for v in graph))

print("\n---Dijkstra's---\n")
g = Graph()
vertices = ["u", "v", "w", "x", "y", "z"]
for v in vertices:
    g.set_vertex(v)
g.add_edge("u", "v", 2)
g.add_edge("u", "w", 5)
g.add_edge("u", "x", 1)
g.add_edge("v", "u", 2)
g.add_edge("v", "w", 3)
g.add_edge("v", "x", 1)
g.add_edge("w", "u", 5)
g.add_edge("w", "v", 3)
g.add_edge("w", "x", 3)
g.add_edge("w", "y", 1)
g.add_edge("w", "z", 5)
g.add_edge("x", "u", 1)
#g.add_edge("x", "v", 2)
g.add_edge("x", "v", 1)
g.add_edge("x", "w", 3)
g.add_edge("x", "y", 1)
g.add_edge("y", "w", 1)
g.add_edge("y", "x", 1)
g.add_edge("y", "z", 1)
g.add_edge("z", "w", 5)
g.add_edge("z", "y", 1)
print("".join(f"{v:5s}" for v in vertices))
dijkstra(g, g.get_vertex("u"))
print(
    "".join(
        f"{g.get_vertex(v).distance:<5d}"
        for v in vertices
    )
)

"""
---Dijkstra's---

u    v    w    x    y    z    
0    2    807  807  807  807  
0    2    5    807  807  807  
0    2    5    1    807  807  
0    2    4    1    807  807  
0    2    4    1    2    807  
0    2    3    1    2    807  
0    2    3    1    2    3    
0    2    3    1    2    3    
"""
