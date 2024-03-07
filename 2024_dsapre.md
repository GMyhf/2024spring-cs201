# 数据结构与算法pre每日选做

Updated 0802 GMT+8 March 7, 2024

2024 spring, Complied by Hongfei Yan



**说明：**

1）数算课程在春季学期开，前一年计算概论课程结束，正值寒假，同学建议每日推出少许题目练习，因此成为“数算pre每日选做”。

2）“数算pre每日选做”题集会逐渐建成，为避免重复，如果题目出现在  https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md，会给出指引。

3）有同学完成了这些题目，放在自己的 gitbub上面，也可以参考 https://github.com/csxq0605/CS101-spring



# 1-10

## 01094: Sorting It All Out

http://cs101.openjudge.cn/dsapre/01094/



```python
# 23n2300011335
def topo_sort(v):
    global vis,pos,T,topo
    if vis[v] == -1:
        return -1
    if pos[v] != -1:
        return pos[v]
    vis[v] = -1
    p = n
    for i in range(len(T[v])):
        p = min(p,topo_sort(T[v][i]))
        if p == -1:
            return -1
    topo[p-1] = v
    pos[v],vis[v] = p-1,0
    return p-1
while True:
    n,m = map(int,input().split())
    if n == m == 0:
        break
    T = [[] for _ in range(n)]
    E = []
    for _ in range(m):
        s = input()
        E.append([ord(s[0])-ord('A'),ord(s[2])-ord('A')])
    topo = [0 for _ in range(n)]
    for i in range(m):
        p = E[i]
        T[p[0]].append(p[1])
        ans = n
        vis = [0 for _ in range(n)]
        pos = [-1 for _ in range(n)]
        for j in range(n):
            ans = min(ans,topo_sort(j))
        if ans == -1:
            print(f'Inconsistency found after {i+1} relations.')
            break
        elif ans == 0:
            print(f'Sorted sequence determined after {i+1} relations: {"".join([chr(topo[k]+ord("A")) for k in range(n)])}.')
            break
    if ans > 0:
        print("Sorted sequence cannot be determined.")
```





## 01145: Tree Summing

http://cs101.openjudge.cn/dsapre/01145/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-上”





## 01178: Camelot

http://cs101.openjudge.cn/dsapre/01178/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 01258: Agri-Net

http://cs101.openjudge.cn/dsapre/01258/





```python
"""
The problem described is a classic example of finding the Minimum Spanning Tree
 (MST) in a weighted graph. In this scenario, each farm represents a node in
 the graph, and the fiber required to connect each pair of farms represents
 the weight of the edges between the nodes.
One of the most common algorithms to find the MST is Kruskal’s algorithm.
Alternatively, Prim’s algorithm could also be used. Below is a Python
implementation using Kruskal’s algorithm.
First, we need to parse the input, then apply the algorithm to find the MST,
and finally sum the weights of the chosen edges to output the result.
"""
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        elif self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True

def kruskal(n, edges):
    dsu = DisjointSetUnion(n)
    mst_weight = 0
    for weight, u, v in sorted(edges):
        if dsu.union(u, v):
            mst_weight += weight
    return mst_weight

def main():
    while True:
        try:
            n = int(input().strip())
            edges = []
            for i in range(n):
                # Since the input lines may continue onto others, we read them all at once
                row = list(map(int, input().split()))
                for j in range(i + 1, n):
                    if row[j] != 0:  # No need to add edges with 0 weight
                        edges.append((row[j], i, j))
            print(kruskal(n, edges))
        except EOFError:  # Exit the loop when all test cases are processed
            break

if __name__ == "__main__":
    main()
```







## 01321: 棋盘问题

http://cs101.openjudge.cn/dsapre/01321/





```python
# https://www.cnblogs.com/Ayanowww/p/11555193.html
'''
本题知识点：深度优先搜索 + 枚举 + 回溯

题意是要求我们把棋子放在棋盘的'#'上，但不能把两枚棋子放在同一列或者同一行上，问摆好这k枚棋子有多少种情况。

我们可以一行一行地找，当在某一行上找到一个可放入的'#'后，就开始找下一行的'#'，如果下一行没有，就再从下一行找。这样记录哪个'#'已放棋子就更简单了，只需要记录一列上就可以了。
'''
n, k, ans = 0, 0, 0
chess = [['' for _ in range(10)] for _ in range(10)]
take = [False] * 10

def dfs(h, t):
    global ans

    if t == k:
        ans += 1
        return

    if h == n:
        return

    for i in range(h, n):
        for j in range(n):
            if chess[i][j] == '#' and not take[j]:
                take[j] = True
                dfs(i + 1, t + 1)
                take[j] = False

while True:
    n, k = map(int, input().split())
    if n == -1 and k == -1:
        break

    for i in range(n):
        chess[i] = list(input())

    take = [False] * 10
    ans = 0
    dfs(0, 0)
    print(ans)
```







## 01376: Robot

http://cs101.openjudge.cn/dsapre/01376/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目





## 01426: Find The Multiple

http://cs101.openjudge.cn/dsapre/01426/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 01611: The Suspects

http://cs101.openjudge.cn/dsapre/01611/



```python
"""
use a technique called Disjoint-set Union (DSU) or Union-Find, which is a data structure that
provides efficient methods for grouping elements into disjoint (non-overlapping) sets and
for determining whether two elements are in the same set.
"""
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each student initially in their own set
        self.rank = [0] * n  # Rank of each node for path compression

    def find(self, x):
        # Find the representative (root) of the set that x is in
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        # Union the sets that x and y are in
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_y] < self.rank[root_x]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def find_suspects(n, groups):
    uf = UnionFind(n)
    for group in groups:
        for student in group[1:]:
            uf.union(group[0], student)  # Union the first student in the group with all others

    suspect_set = set()
    for i in range(n):
        if uf.find(0) == uf.find(i):  # If student is in the same set as the initial suspect
            suspect_set.add(i)

    return len(suspect_set)

def main():
    while True:
        n, m = map(int, input().split())
        if n == 0 and m == 0:
            break
        groups = [list(map(int, input().split()))[1:] for _ in range(m)]
        print(find_suspects(n, groups))

if __name__ == "__main__":
    main()
```





## 01760: Disk Tree

http://cs101.openjudge.cn/dsapre/01760/

题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-下”





## 01789: Truck History

http://cs101.openjudge.cn/dsapre/01789/





```python
"""
https://www.cnblogs.com/chujian123/p/3375210.html
题意大概是这样的：用一个7位的string代表一个编号，两个编号之间的distance代表这两个编号之间不同字母的个数。
一个编号只能由另一个编号“衍生”出来，代价是这两个编号之间相应的distance，现在要找出一个“衍生”方案，
使得总代价最小，也就是distance之和最小。

题解：问题可以转化为最小代价生成树的问题。因为每两个结点之间都有路径，所以是完全图。 此题的关键是将问题转化
为最小生成树的问题。每一个编号为图的一个顶点，顶点与顶点间的编号差即为这条边的权值，题目所要的就是我们求出
最小生成树来。这里我用prim算法来求最小生成树。
"""
import sys

INF = 100000000


def juli(a, b):
    count = 0
    for i in range(7):
        if a[i] != b[i]:
            count += 1
    return count


def prim(v0, n, g):
    sum = 0
    lowcost = [0] * (n + 1)
    for i in range(1, n + 1):
        lowcost[i] = g[v0][i]
    lowcost[v0] = -1
    for _ in range(1, n):
        min_cost = INF
        v = -1
        for j in range(1, n + 1):
            if lowcost[j] != -1 and lowcost[j] < min_cost:
                v = j
                min_cost = lowcost[j]
        if v != -1:
            sum += lowcost[v]
            lowcost[v] = -1
            for k in range(1, n + 1):
                if lowcost[k] != -1 and g[v][k] < lowcost[k]:
                    lowcost[k] = g[v][k]
    print(f"The highest possible quality is 1/{sum}.")


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        n = int(line)
        if n == 0:
            break
        nodes = [''] * (n + 1)
        for i in range(1, n + 1):
            nodes[i] = sys.stdin.readline().strip()
        g = [[0] * (n + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                g[i][j] = g[j][i] = juli(nodes[i], nodes[j])

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    g[i][j] = 0
                elif g[i][j] == 0:
                    g[i][j] = INF

        prim(1, n, g)


if __name__ == "__main__":
    main()
```









# 11-20



```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```



# 21-30

```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```



# 31-40

```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





# 41-50



```python

```





```python

```





```python

```





```python

```





```python

```





```python

```





```python

```







```python

```







```python

```







```python

```





# 51-60



```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```





# 61-70

```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```





8

```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```





# 71-80

```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```





# 91-102

```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```







```python

```













