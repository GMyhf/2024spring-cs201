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

## 01860: Currency Exchange

http://cs101.openjudge.cn/dsapre/01860/



```python
"""
https://blog.csdn.net/weixin_44226181/article/details/127239846
This problem can be solved using the Bellman-Ford algorithm. The Bellman-Ford algorithm
is used to find the shortest path from a single source vertex to all other vertices
in a weighted graph. In this case, the graph is the currency exchange graph, where
each vertex represents a currency and each edge represents an exchange rate between
two currencies.  The Bellman-Ford algorithm works by iteratively relaxing the edges
of the graph. In each iteration, it checks all edges and updates the shortest path
if a shorter path is found. This process is repeated for N-1 times, where N is the
number of vertices in the graph.
However, in this problem, we are not looking for the shortest path, but rather a
cycle that increases the value. This can be detected by running the Bellman-Ford
algorithm one more time after the N-1 iterations. If the value continues to increase,
then there is a positive cycle.
"""
class Node:
    def __init__(self, a, b, r, c):
        self.a = a
        self.b = b
        self.r = r
        self.c = c

def add(a, b, r, c, e):
    e.append(Node(a, b, r, c))

def bellman_ford(e, dis, n, s, v):
    dis[s] = v
    for _ in range(n-1):
        flag = False
        for edge in e:
            if dis[edge.b] < (dis[edge.a] - edge.c) * edge.r:
                dis[edge.b] = (dis[edge.a] - edge.c) * edge.r
                flag = True
        if not flag:
            return False
    for edge in e:
        if dis[edge.b] < (dis[edge.a] - edge.c) * edge.r:
            return True
    return False

def main():
    n, m, s, v = map(float, input().split())
    n = int(n)
    s = int(s)
    e = []
    dis = [0] * (n + 1)
    for _ in range(int(m)):
        a, b, rab, cab, rba, cba = map(float, input().split())
        add(int(a), int(b), rab, cab, e)
        add(int(b), int(a), rba, cba, e)
    if bellman_ford(e, dis, n, s, v):
        print("YES")
    else:
        print("NO")

if __name__ == "__main__":
    main()
```



## 01944: Fiber Communications

http://cs101.openjudge.cn/dsapre/01944/

```python
# https://www.cnblogs.com/lightspeedsmallson/p/4785834.html
N, P = map(int, input().split())
node_one = []

for i in range(P):
    Q1, Q2 = map(int, input().split())
    node_one.append({'start': min(Q1, Q2), 'end': max(Q1, Q2)})

node_one.sort(key=lambda x: (x['start'], x['end']))

INF = float('inf')
ans = INF

for i in range(1, N + 1):
    to = [0] * (N + 1)

    for j in range(P):
        if node_one[j]['end'] >= i + 1 and node_one[j]['start'] <= i:
            to[1] = max(to[1], node_one[j]['start'])
            to[node_one[j]['end']] = N + 1
        else:
            to[node_one[j]['start']] = max(to[node_one[j]['start']], node_one[j]['end'])

    duandian = 0
    result = 0

    for j in range(1, N + 1):
        if to[j] == 0:
            continue

        if to[j] > duandian:
            if j >= duandian:
                result += (to[j] - j)
            else:
                result += (to[j] - duandian)

            duandian = to[j]

    ans = min(ans, result)

print(ans)
```





## 02039: 反反复复

http://cs101.openjudge.cn/dsapre/02039/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H1: Python入门”



## 02049: Finding Nemo

http://cs101.openjudge.cn/dsapre/02049/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H7: 图应用”



## 02092: Grandpa is Famous

http://cs101.openjudge.cn/dsapre/02092/



```python
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break

    count = [0] * 10001
    for _ in range(n):
        for player in map(int, input().split()):
            count[player] += 1

    max_count = max(count)
    second_max_count = max(x for x in count if x != max_count)

    for player, player_count in enumerate(count):
        if player_count == second_max_count:
            print(player, end=' ')
    print()
```



## 02226: Muddy Fields

http://cs101.openjudge.cn/dsapre/02226/



```python
#23n2300011335
r,c = map(int,input().split())
a,b = 0,0
matrix = [[False for _ in range(1000)] for _ in range(1000)]
match = [0 for _ in range(1000)]
maze = [['' for _ in range(1000)] for _ in range(1000)]
h,s = [[0 for _ in range(1000)] for _ in range(1000)],[[0 for _ in range(1000)] for _ in range(1000)]
for i in range(1,r+1):
    maze[i][1:] = list(input())
    for j in range(1,c+1):
        if maze[i][j] == '*':
            if j == 1 or maze[i][j-1] != '*':
                a += 1
                h[i][j] = a
            else:
                h[i][j] = a
for j in range(1,c+1):
    for i in range(1,r+1):
        if maze[i][j] == '*':
            if i == 1 or maze[i-1][j] != '*':
                b += 1
                s[i][j] = b
            else:
                s[i][j] = b
            matrix[h[i][j]][s[i][j]] = True
visited = []
def dfs(x):
    for i in range(1,b+1):
        if matrix[x][i] and not visited[i]:
            visited[i] = True
            if not match[i] or dfs(match[i]):
                match[i] = x
                return 1
    return 0
ans = 0
for i in range(1,a+1):
    visited = [False for _ in range(1000)]
    ans += dfs(i)
print(ans)
```





## 02253:Frogger

http://cs101.openjudge.cn/dsapre/02253/



```python
"""
定义了一个frog_distance函数，它接受一个石头列表，并计算出青蛙从第一个石头跳到第二个石头的最小跳跃距离。
使用动态规划的思想，通过计算任意两个石头之间的直线距离，并利用最短路径算法（Floyd-Warshall算法）计算出最小跳跃距离。

在主循环中，读取输入并计算每个测试用例的青蛙距离。当输入的石头数量为0时，循环终止。

输出每个测试用例的结果，包括测试用例的序号和青蛙距离，保留3位小数。在每个测试用例后输出一个空行。
"""
import math

def frog_distance(stones):
    n = len(stones)
    distances = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i][j] = 0
            else:
                x1, y1 = stones[i]
                x2, y2 = stones[j]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances[i][j] = distance

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], max(distances[i][k], distances[k][j]))

    return distances[0][1]

# 读取输入
test_case = 1
while True:
    n = int(input())
    if n == 0:
        break

    stones = []
    for _ in range(n):
        x, y = map(int, input().split())
        stones.append((x, y))

    # 计算青蛙距离
    distance = frog_distance(stones)

    # 输出结果
    print("Scenario #{}".format(test_case))
    print("Frog Distance = {:.3f}".format(distance))
    print()
    input()

    test_case += 1
```





## 02255: 重建二叉树

http://cs101.openjudge.cn/dsapre/02255/

题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-上”



## 02299: Ultra-QuickSort

http://cs101.openjudge.cn/dsapre/02299/



```python
"""
这是一个需要你分析特定排序算法的问题。该算法通过交换两个相邻的序列元素来处理n个不同的整数序列，直到序列按升序排序。
对于输入序列，Ultra-QuickSort会产生输出。你的任务是确定Ultra-QuickSort需要执行多少次交换操作才能对给定的输入序列进行排序。  
这个问题可以通过使用归并排序的修改版本来解决，其中我们计算在每次合并步骤中需要的交换次数。在归并排序中，我们将数组分成两半，
对每一半进行排序，然后将它们合并在一起。在合并步骤中，我们可以计算需要交换的次数，因为每当我们从右半部分取出一个元素时，
我们需要交换与左半部分中剩余元素相同数量的次数。
"""
def merge_sort(lst):
    # The list is already sorted if it contains a single element.
    if len(lst) <= 1:
        return lst, 0

    # Divide the input into two halves.
    middle = len(lst) // 2
    left, inv_left = merge_sort(lst[:middle])
    right, inv_right = merge_sort(lst[middle:])

    merged, inv_merge = merge(left, right)

    # The total number of inversions is the sum of inversions in the recursion and the merge process.
    return merged, inv_left + inv_right + inv_merge

def merge(left, right):
    merged = []
    inv_count = 0
    i = j = 0

    # Merge smaller elements first.
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv_count += len(left) - i

    # If there are remaining elements in the left or right half, append them to the result.
    merged += left[i:]
    merged += right[j:]

    return merged, inv_count

while True:
    n = int(input())
    if n == 0:
        break

    lst = []
    for _ in range(n):
        lst.append(int(input()))

    _, inversions = merge_sort(lst)
    print(inversions)
```



## 02337: Catenyms

http://cs101.openjudge.cn/dsapre/02337/



```python
"""
https://blog.51cto.com/u_15684947/5384135
"""
from typing import List, Tuple
import sys

class EulerPath:
    def __init__(self):
        self.maxn = 1005
        self.vis = [0] * self.maxn
        self.in_ = [0] * 128
        self.out = [0] * 128
        self.s = [""] * self.maxn
        self.ans = [""] * self.maxn
        self.len_ = [0] * self.maxn
        self.vv = [[] for _ in range(128)]
        self.tot = 0

    def dfs(self, st: str):
        up = len(self.vv[ord(st)])
        for i in range(up):
            cur = self.vv[ord(st)][i]
            if self.vis[cur[1]]:
                continue
            self.vis[cur[1]] = 1
            self.dfs(cur[0][self.len_[cur[1]] - 1])
            self.ans[self.tot] = cur[0]
            self.tot += 1

    def solve(self):
        t = int(input().strip())
        for _ in range(t):
            self.tot = 0
            n = int(input().strip())
            for i in range(1, 128):
                self.in_[i] = self.out[i] = 0
                self.vv[i].clear()
            for i in range(1, n + 1):
                self.vis[i] = 0
                self.s[i] = input().strip()
                self.len_[i] = len(self.s[i])
            minn = 'z'
            for i in range(1, n + 1):
                st = self.s[i][0]
                ed = self.s[i][self.len_[i] - 1]
                self.vv[ord(st)].append((self.s[i], i))
                self.in_[ord(ed)] += 1
                self.out[ord(st)] += 1
                minn = min(minn, ed, st)
            flag = 1
            ru = 0
            chu = 0
            for i in range(ord('a'), ord('z') + 1):
                self.vv[i] = sorted(self.vv[i])
                if not self.in_[i] and not self.out[i]:
                    continue
                if self.in_[i] == self.out[i]:
                    continue
                elif self.in_[i] - self.out[i] == 1:
                    ru += 1
                elif self.out[i] - self.in_[i] == 1:
                    chu += 1
                    minn = chr(i)
                else:
                    flag = 0
                    break
            if flag == 0 or ru > 1 or chu > 1 or ru != chu:
                print("***")
            else:
                self.dfs(minn)
                if self.tot != n:
                    print("***")
                else:
                    for i in range(n - 1, -1, -1):
                        if i != n - 1:
                            print(".", end='')
                        print(self.ans[i], end='')
                    print()

if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    euler_path = EulerPath()
    euler_path.solve()
```







# 21-30

## 02488: A Knight's Journey

http://cs101.openjudge.cn/dsapre/02488/



```python
move = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def dfs(x, y, step, p, q, visited, ans):
    if step == p * q:
        return True
    for i in range(8):
        dx, dy = x + move[i][0], y + move[i][1]
        if 1 <= dx <= q and 1 <= dy <= p and not visited[dx][dy]:
            visited[dx][dy] = True
            ans[step] = chr(dx + 64) + str(dy)
            if dfs(dx, dy, step + 1, p, q, visited, ans):
                return True
            visited[dx][dy] = False
    return False


n = int(input())
for m in range(1, n + 1):
    p, q = map(int, input().split())
    ans = ["" for _ in range(p * q)]
    visited = [[False] * (p + 1) for _ in range(q + 1)]
    visited[1][1] = True
    ans[0] = "A1"
    if dfs(1, 1, 1, p, q, visited, ans):
        result = "".join(ans)
    else:
        result = "impossible"
    print(f"Scenario #{m}:")
    print(result)
    print()
```





## 02524: 宗教信仰

http://cs101.openjudge.cn/dsapre/02524/



```python
def init_set(n):
    return list(range(n))

def get_father(x, father):
    if father[x] != x:
        father[x] = get_father(father[x], father)
    return father[x]

def join(x, y, father):
    fx = get_father(x, father)
    fy = get_father(y, father)
    if fx == fy:
        return
    father[fx] = fy

def is_same(x, y, father):
    return get_father(x, father) == get_father(y, father)

def main():
    case_num = 0
    while True:
        n, m = map(int, input().split())
        if n == 0 and m == 0:
            break
        count = 0
        father = init_set(n)
        for _ in range(m):
            s1, s2 = map(int, input().split())
            join(s1 - 1, s2 - 1, father)
        for i in range(n):
            if father[i] == i:
                count += 1
        case_num += 1
        print(f"Case {case_num}: {count}")

if __name__ == "__main__":
    main()
```



## 02694: 波兰表达式

http://cs101.openjudge.cn/dsapre/02694/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Basic Exercise 部分相应题目



## 02756: 二叉树（1）

http://cs101.openjudge.cn/dsapre/02756/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-下”



## 02766: 最大子矩阵

http://cs101.openjudge.cn/dsapre/02766/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 02773: 采药

http://cs101.openjudge.cn/dsapre/02773/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 02774: 木材加工

http://cs101.openjudge.cn/dsapre/02774/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “2023期末上机考试（数算B）7题”



## 02788: 二叉树（2）

http://cs101.openjudge.cn/dsapre/02788/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-上”



## 02945: 拦截导弹

http://cs101.openjudge.cn/dsapre/02945/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H3: 递归与动态规划”



## 03720: 文本二叉树

http://cs101.openjudge.cn/dsapre/03720/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “2023期末上机考试（数算B）7题”





# 31-40

## 04079: 二叉搜索树

http://cs101.openjudge.cn/dsapre/04079/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-下”



## 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-上”



## 04082: 树的镜面映射

http://cs101.openjudge.cn/dsapre/04082/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-上”



## 04084: 拓扑排序

http://cs101.openjudge.cn/dsapre/04084/



```python
import heapq

def topological_sort(vertices, edges):
    # Initialize in-degree and connection matrix
    in_edges = [0] * (vertices + 1)
    connect = [[0] * (vertices + 1) for _ in range(vertices + 1)]

    # Populate the in-degree and connection matrix
    for u, v in edges:
        in_edges[v] += 1
        connect[u][v] += 1

    # Priority queue for vertices with in-degree of 0
    queue = []
    for i in range(1, vertices + 1):
        if in_edges[i] == 0:
            heapq.heappush(queue, i)

    # List to store the topological order
    order = []

    # Processing vertices
    while queue:
        u = heapq.heappop(queue)
        order.append(u)
        for v in range(1, vertices + 1):
            if connect[u][v] > 0:
                in_edges[v] -= connect[u][v]
                if in_edges[v] == 0:
                    heapq.heappush(queue, v)

    if len(order) == vertices:
        return order
    else:
        return None

# Read input
vertices, num_edges = map(int, input().split())
edges = []
for _ in range(num_edges):
    u, v = map(int, input().split())
    edges.append((u, v))

# Perform topological sort
order = topological_sort(vertices, edges)

# Output result
if order:
    for i, vertex in enumerate(order):
        if i < len(order) - 1:
            print(f"v{vertex}", end=" ")
        else:
            print(f"v{vertex}")
else:
    print("No topological order exists due to a cycle in the graph.")
```



## 04089: 电话号码

http://cs101.openjudge.cn/dsapre/04089/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 04117: 简单的整数划分问题

http://cs101.openjudge.cn/dsapre/04117/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 04135: 月度开销

http://cs101.openjudge.cn/dsapre/04135/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 04136: 矩形分割

http://cs101.openjudge.cn/dsapre/04136/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 04143: 和为给定数

http://cs101.openjudge.cn/dsapre/04143/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H4: 查找与排序”



## 05344: 最后的最后

http://cs101.openjudge.cn/dsapre/05344/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H2: 线性表”





# 41-50

## 05345: 位查询

http://cs101.openjudge.cn/dsapre/05345/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H2: 线性表”



## 05430: 表达式·表达式树·表达式求值

http://cs101.openjudge.cn/dsapre/05430/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-下”



## 05442: 兔子与星空

http://cs101.openjudge.cn/dsapre/05442/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H7: 图应用”



## 05443: 兔子与樱花

http://cs101.openjudge.cn/dsapre/05443/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H7: 图应用”



## 05467: 多项式加法

http://cs101.openjudge.cn/dsapre/05467/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H2: 线性表”





## 05907: 二叉树的操作

http://cs101.openjudge.cn/dsapre/05907/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “2023期末上机考试（数算B）7题”



## 06250: 字符串最大跨距

http://cs101.openjudge.cn/dsapre/06250/



```python
# 23n2300017735(夏天明BrightSummer)
def find(s, pat):
    nex = [0]
    for i, p in enumerate(pat[1:], 1):
        tmp = nex[i-1]
        while True:
            if p == pat[tmp]:
                nex.append(tmp+1)
                break
            elif tmp:
                tmp = nex[tmp-1]
            else:
                nex.append(0)
                break
    j = 0
    for i, char in enumerate(s):
        while True:
            if char == pat[j]:
                j += 1
                if j == len(pat):
                    return i
                break
            elif j:
                j -= nex[j]
            else:
                break

s, p1, p2 = input().split(',')
try:
    assert((ans := len(s)-find(s, p1)-find(s[::-1], p2[::-1])-2) >= 0)
    print(ans)
except (TypeError, AssertionError):
    print(-1)
```



## 06364: 牛的选举

http://cs101.openjudge.cn/dsapre/06364/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “2023期末上机考试（数算B）7题”



## 06646: 二叉树的深度

http://cs101.openjudge.cn/dsapre/06646/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-下”



## 06648: Sequence

http://cs101.openjudge.cn/dsapre/06648/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 “数算pre每日选做” 中 “H5: 树及算法-下”





# 51-60

## 07161: 森林的带度数层次序列存储

http://cs101.openjudge.cn/dsapre/07161/



```python
# 23n2300011075(才疏学浅)
from collections import deque
class Node:
    def __init__(self):
        self.value=None
        self.degree=0
        self.childs=[]

def build():
    node=Node()
    node.value=l.pop(0)
    node.degree=int(l.pop(0))
    return node

def Tree():
    root=build()
    q=deque([root])
    while q:
        node=q.popleft()
        for i in range(node.degree):
            child=build()
            node.childs.append(child)
            q.append(child)
    return root

def lastorder(tree):
    for child in tree.childs:
        lastorder(child)
    print(tree.value,end=" ")

n=int(input())
for _ in range(n):
    l=list(input().split())
    tree=Tree()
    lastorder(tree)
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













