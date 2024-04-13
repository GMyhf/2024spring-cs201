from collections import deque, defaultdict

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()

    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.append(u)

    # 执行拓扑排序
    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None

# 示例调用代码
graph = {
    'cup_milk': ['mix_ingredients'],
    'mix_ingredients': ['pour_batter', 'heat_syrup'],
    'pour_batter': ['turn_pancake'],
    'turn_pancake': ['eat_pancake'],
    'heat_syrup': ['eat_pancake'],
    'heat_griddle': ['pour_batter'],
    'tbl_oil': ['mix_ingredients'],
    'egg': ['mix_ingredients'],
    'eat_pancake': []
}


sorted_vertices = topological_sort(graph)
if sorted_vertices:
    print("Topological sort order:", sorted_vertices)
else:
    print("The graph contains a cycle.")

"""
#Depth First Forest ouput:
#['heat_griddle', 'tbl_oil', 'egg', 'cup_milk', 'mix_ingredients', 'heat_syrup', 'pour_batter', 'turn_pancake', 'eat_pancake']

# Kahn ouput:
Topological sort order: ['cup_milk', 'heat_griddle', 'tbl_oil', 'egg', 'mix_ingredients', 'pour_batter', 'heat_syrup', 'turn_pancake', 'eat_pancake']

"""