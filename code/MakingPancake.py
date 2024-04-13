import sys

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_ertex = Vertex(key)
        self.vertices[key] = new_ertex
        return new_ertex

    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __len__(self):
        return self.num_vertices

    def __contains__(self, n):
        return n in self.vertices

    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        #self.vertices[t].add_neighbor(self.vertices[f], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.discovery = 0
        self.finish = None

    def __lt__(self, o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def setDiscovery(self, dtime):
        self.discovery = dtime

    def setFinish(self, ftime):
        self.finish = ftime

    def getFinish(self):
        return self.finish

    def getDiscovery(self):
        return self.discovery

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.discovery) + ":fin " + str(
            self.finish) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"


class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.topologicalList = []

    def dfs(self):
        for aVertex in self:
            aVertex.color = "white"
            aVertex.predecessor = -1
        for aVertex in self:
            if aVertex.color == "white":
                self.dfsvisit(aVertex)

    def dfsvisit(self, startVertex):
        startVertex.color = "gray"
        self.time += 1
        startVertex.setDiscovery(self.time)
        for nextVertex in startVertex.get_neighbors():
            if nextVertex.color == "white":
                nextVertex.previous = startVertex
                self.dfsvisit(nextVertex)
        startVertex.color = "black"
        self.time += 1
        startVertex.setFinish(self.time)

    def topologicalSort(self):
        self.dfs()
        temp = list(self.vertices.values())
        temp.sort(key = lambda x: x.getFinish(), reverse = True)
        print([(x.key,x.finish) for x in temp])
        self.topologicalList = [x.key for x in temp]
        return self.topologicalList

# Creating the graph
g = DFSGraph()

g.add_vertex('cup_milk')
g.add_vertex('egg')
g.add_vertex('tbl_oil')

g.add_vertex('heat_griddle')
g.add_vertex('cup_mix')
g.add_vertex('pour_cup')
g.add_vertex('turn_pancake')    # turn when bubbly
g.add_vertex('heat_syrup')
g.add_vertex('eat_pancake')

# Adding edges based on dependencies
g.add_edge('cup_milk', 'cup_mix')
g.add_edge('cup_mix', 'pour_cup')
g.add_edge('pour_cup', 'turn_pancake')
g.add_edge('turn_pancake', 'eat_pancake')

g.add_edge('cup_mix', 'heat_syrup')
g.add_edge('heat_syrup', 'eat_pancake')

g.add_edge('heat_griddle', 'pour_cup')
g.add_edge('tbl_oil', 'cup_mix')
g.add_edge('egg', 'cup_mix')



# Getting topological sort of the tasks
topo_order = g.topologicalSort()
print("Topological Sort of the Pancake Making Process:")
print(topo_order)

"""
Output:
函数 topologicalSort 中的调试信息
[('eat_pancake', 18), ('heat_syrup', 16), ('turn_pancake', 14), ('pour_cup', 12), ('cup_mix', 10), ('cup_milk', 8), ('heat_griddle', 6), ('tbl_oil', 4), ('egg', 2)]

Topological Sort of the Pancake Making Process:
['heat_griddle', 'tbl_oil', 'egg', 'cup_milk', 'cup_mix', 'heat_syrup', 'pour_cup', 'turn_pancake', 'eat_pancake']
"""