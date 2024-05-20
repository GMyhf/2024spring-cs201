# 01258: Agri-Net
# http://cs101.openjudge.cn/dsapre/01258/
from heapq import heappop, heappush, heapify

def prim(graph, start_node):
    mst = set()
    visited = set([start_node])
    edges = [
        (cost, start_node, to)
        for to, cost in graph[start_node].items()
    ]
    heapify(edges)

    while edges:
        cost, frm, to = heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in visited:
                    heappush(edges, (cost2, to, to_next))

    return mst


while True:
    try:
        N = int(input())
    except EOFError:
        break

    graph = {i: {} for i in range(N)}
    for i in range(N):
        for j, cost in enumerate(map(int, input().split())):
            graph[i][j] = cost

    mst = prim(graph, 0)
    total_cost = sum(cost for frm, to, cost in mst)
    print(total_cost)
