# 数据结构与算法 上机

[toc]
<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 一 线性结构

### 1. 表达式求值

#### 1.1 前缀表达式求值

```python
def parse(s: str):
    if s in set("+-*/"):
        return s
    return float(s)


def eval(sgn: str, a: float, b: float):
    if sgn == "+":
        return a + b
    if sgn == "-":
        return a - b
    if sgn == "*":
        return a * b
    if sgn == "/":
        return a / b
    raise ValueError("Invalid operator")


def evalable(sgn, a, b):
    return isinstance(sgn, str) and isinstance(a, float) and isinstance(
        b, float)


args = map(parse, input().split())
stack = []

for arg in args:
    stack.append(arg)

    while len(stack) >= 3:
        sgn = stack[-3]
        a = stack[-2]
        b = stack[-1]
        if evalable(sgn, a, b):
            stack.pop()
            stack.pop()
            stack.pop()
            stack.append(eval(sgn, a, b))
        else:
            break

print("{:.6f}".format(stack[0]))
```

### 2. 表达式转换

#### 2.1 中缀表达式转后缀表达式

```python
def parse_expr(s: str):
    expr_list = ["("]
    tmp = ""
    for char in s:
        if char == " ":
            continue
        if char in set("+-*/()"):
            if len(tmp) > 0:
                expr_list.append(tmp)
                tmp = ""
            expr_list.append(char)
        else:
            tmp += char
    if len(tmp) > 0:
        expr_list.append(tmp)
        tmp = ""
    expr_list.append(")")
    return expr_list


def task():
    mid_expr = parse_expr(input())
    sgn_stack = []
    res_stack = []

    for token in mid_expr:
        if token in set("+-"):
            while len(sgn_stack) != 0 and sgn_stack[-1] in set("+-*/"):
                sgn = sgn_stack.pop()
                res_stack.append(sgn)
            sgn_stack.append(token)
        elif token in set("*/"):
            while len(sgn_stack) != 0 and sgn_stack[-1] in set("*/"):
                sgn = sgn_stack.pop()
                res_stack.append(sgn)
            sgn_stack.append(token)
        elif token == "(":
            sgn_stack.append(token)
        elif token == ")":
            while sgn_stack[-1] != "(":
                sgn = sgn_stack.pop()
                res_stack.append(sgn)
            sgn_stack.pop()
        else:
            res_stack.append(token)

    print(" ".join(res_stack))
```



### 3. 单调栈

**例题**

> 给出项数为 n 的整数数列 a_i, i = [0..n]。
>
> 定义函数 f(i) 代表数列中第 i 个元素之后第一个大于 ai 的元素的**下标**。若不存在，则 f(i)=0。
>
> 试求出 f(i), i = [0..n]。

```python
from typing import List
from collections import deque

N = int(input())
arr = [*map(int, input().split())]
mono_stack: List[int] = []
res = deque()

for rev_idx in range(N):
    idx = N - rev_idx - 1
    while mono_stack and arr[mono_stack[-1]] <= arr[idx]:
        mono_stack.pop()
    res.appendleft(mono_stack[-1] + 1 if mono_stack else 0)
    mono_stack.append(idx)

print(*res)
```

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 二 树

### 1. 基本模板

#### 多叉树

```python
class MTreeNode:

    def __init__(self, val: str):
        self.val = val
        self.children: List[MTreeNode] = []
```

#### 二叉树

```python
class BTreeNode:

    def __init__(self, value: str):
        self.value = value
        self.left: Union[None, BTreeNode] = None
        self.right: Union[None, BTreeNode] = None
```

### 2. 前/中/后序遍历

```python
def post_order_traversal(root: TreeNode) -> str:
    
    def post_order_traversal_helper(root: TreeNode, post_order: List[str]) -> None:
        if root is None:
            return
		
        # 更换这里的顺序即可
        post_order_traversal_helper(root.left, post_order)
        post_order_traversal_helper(root.right, post_order)
        post_order.append(root.value)

    post_order = []
    post_order_traversal_helper(root, post_order)
    return ''.join(post_order)
```

### 3. 根据两个遍历结果建树

#### 3.1. 前序遍历 & 中序遍历

```python
def build_tree(pre_order: str, mid_order: str) -> TreeNode:
    if not pre_order:
        return None

    root = TreeNode(pre_order[0])
    root_index = mid_order.index(pre_order[0])

    root.left = None if root_index == 0 else build_tree(
        pre_order[1:root_index + 1], mid_order[:root_index])
    root.right = None if root_index == len(mid_order) - 1 else build_tree(
        pre_order[root_index + 1:], mid_order[root_index + 1:])

    return root
```

#### 3.2. 中序遍历 & 后序遍历

```python
def build_tree(mid_order: str, post_order: str) -> TreeNode:
    root = TreeNode(post_order[-1], None, None)
    root_index = mid_order.index(root.value)

    # calculate the length of left and right sub-tree
    left_length = root_index
    right_length = len(mid_order) - root_index - 1
    assert right_length >= 0

    # build left sub-tree
    if left_length == 0:
        pass
    elif left_length == 1:
        root.left = mid_order[0]
    else:
        left_node = build_tree(mid_order[:left_length],
                               post_order[:left_length])
        root.left = left_node

    # build right sub-tree
    if right_length == 0:
        pass
    elif right_length == 1:
        root.right = mid_order[-1]
    else:
        right_node = build_tree(mid_order[-right_length:],
                                post_order[-right_length - 1:-1])
        root.right = right_node

    return root
```

### 4. 二叉树与多叉树转换

#### 4.1. 二叉树转多叉树

```python
def binary_to_multi(binary_root: BTreeNode) -> MTreeNode:

    def helper(node: Union[BTreeNode, None], parent: MTreeNode):
        if node is None:
            return

        multi_node = MTreeNode(node.val)
        parent.children.append(multi_node)
        helper(node.left, multi_node)
        helper(node.right, parent)

    multi_root = MTreeNode(binary_root.val)
    helper(binary_root.left, multi_root)
    assert binary_root.right is None

    return multi_root
```

#### 4.2. 多叉树转二叉树

```python
def mtree2btree(mroot: MTreeNode) -> BTreeNode:

    def mtree2btree_helper(mnodes: List[MTreeNode]):
        if not mnodes:
            return None

        broot = BTreeNode(mnodes[0].val)
        broot.left = mtree2btree_helper(mnodes[0].children)

        bnode = broot
        for mnode in mnodes[1::]:
            bnode_ = BTreeNode(mnode.val)
            bnode.right = bnode_
            bnode = bnode_
            bnode.left = mtree2btree_helper(mnode.children)
        return broot

    broot = BTreeNode(mroot.val)
    broot.left = mtree2btree_helper(mroot.children)
    return broot
```

### 5. 二叉搜索树

```python
class BST(object):

    def __init__(self):
        self.root: TreeNode = None

    def __put(self, node: TreeNode, val: int) -> Union[TreeNode, None]:

        if node is None:
            return TreeNode(val)

        if val < node.val:
            node.left = self.__put(node.left, val)
            return node
        elif val > node.val:
            node.right = self.__put(node.right, val)
            return node
        else:
            # ignore duplicate
            pass

        return node

    def put(self, val: int) -> None:
        self.root = self.__put(self.root, val)

    def level_order(self) -> List[int]:
        queue: Deque[TreeNode] = deque([self.root])
        res: List[int] = []

        while queue:
            node = queue.popleft()
            res.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        return res
```

### 6. 哈夫曼编码树

```python
class TreeNode(object):

    def __init__(self, val: List[str], weight: int):
        self.val = val
        self.val_set = None
        self.weight = weight
        self.left: Union[None, TreeNode] = None
        self.right: Union[None, TreeNode] = None

    def __eq__(self, other: TreeNode) -> bool:
        if self.weight == other.weight:
            return self.val == other.val
        else:
            return False

    def __gt__(self, other: TreeNode) -> bool:
        if self.weight == other.weight:
            return self.val > other.val
        else:
            return self.weight > other.weight

    def finish(self):
        self.val_set = set(self.val)


def build_tree(n) -> TreeNode:
    node_list: List[TreeNode] = []
    node_heap: List[TreeNode] = []
    for _ in range(n):
        val, weight = input().split()
        node = TreeNode([val], int(weight))
        node_list.append(node)
        heapq.heappush(node_heap, node)

    while len(node_heap) > 1:
        left = heapq.heappop(node_heap)
        right = heapq.heappop(node_heap)

        node = TreeNode(list(heapq.merge(left.val, right.val)),
                        left.weight + right.weight)
        node_list.append(node)

        node.left = left
        node.right = right

        heapq.heappush(node_heap, node)

    for node in node_list:
        node.finish()

    return node_heap[0]


def encode(root: TreeNode, string: str) -> str:
    res = []

    for char in string:
        node = root
        while True:
            if len(node.val) == 1:
                break

            if char in node.left.val_set:
                node = node.left
                res.append('0')
            else:
                node = node.right
                res.append('1')

    return ''.join(res)
```

### 7. 字典树

```python
class TrieNode:

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
```

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 三 并查集

### 1. 基本模板

```python
class UnionFind:

    def __init__(self, N):
        self.N = N
        self.parent = [i for i in range(N)]
        self.size = [1] * N
 
    def find(self, u):
        # 路径压缩
        while u != self.parent[u]:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def connected(self, u, v):
        return self.find(u) == self.find(v)

    def union(self, u, v):
        # 基于 Size 的合并
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return
        if self.size[pu] < self.size[pv]:
            pu, pv = pv, pu
        self.parent[pv] = pu
        self.size[pu] += self.size[pv]
```

**特殊用法**：有时候需要指定合并顺序（冰可乐），此时 union 有固定的父节点。

### 2. 种类并查集

- [0..N] 代表"朋友"，[N..2N] 代表"敌人"
  - 建立"朋友"关系：连接 i 与 j，i + N 与 j + N；
  - 检查"朋友"关系：检查 i 与 j；
  - 建立"敌人"关系：连接 i 与 j + N，i + N 与 j；
  - 检查"敌人"关系：检查 i 与 j + N；
- [0..N] 代表”同类“，[N..2N] 代表”食物“，[2N..3N] 代表”敌人“
  - 建立"同类"关系：连接/检查 i 与 j，i + N 与 j + N，i + 2N 与 j + 2N；
  - 建立 "i 吃 j" 的关系：连接/检查 i 与 j + 2N，i + N 与 j，i + 2N 与 j + N；
    - 即 i 和 j 的敌人是一类，i 的食物和 j 是一类；
    - i 的敌人和 j 的食物是一类（i 吃 j 吃 k，那么 k 必然吃 i）；
  - 检查关系同样只需要检查其中一个元组即可。

**如果考场上出了 Bug，善用 assert 在创建新关系之前检查。**

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 四 无向图

### 1. 不带权图的基本数据结构

```python
class UDGraph(object):

    def __init__(self, N: int, weight: List[int]):
        assert N == len(weight)
        self.N = N
        self.weight = weight
        self.connected: List[Set[int]] = [set() for i in range(N)]

    def add_edge(self, a: int, b: int):
        self.connected[a].add(b)
        self.connected[b].add(a)
```

### 2. 搜索

#### 2.1 广度优先搜索

#### 2.2 深度优先搜索

#### 2.3 一致代价搜索（Dijkstra）

#### 2.4 带约束的一致代价搜索（Constrained Dijkstra）

**边的存储**：用字典，前面是代价，后面是约束

**核心思想**：以前 Visited 只需要维护当前节点到达时候发生的最小代价，而现在需要考虑约束是否满足，即对（当前节点，剩余约束）进行检查。

> 可以考虑把剩余约束更小的一并更新了，但一般不需要这么做（用时不会显著提升）。 

```python
class Graph:

    def __init__(self, V: int):
        self.V = V
        self.edges: List[Dict[int, List[Tuple[T_COST, T_COIN]]]] = []
        self.edges = [{} for _ in range(V)]

    def add_edge(self, u: int, v: int, cost: T_COST, coin: T_COIN) -> None:
        if v in self.edges[u]:
            self.edges[u][v].append((cost, coin))
        else:
            self.edges[u][v] = [(cost, coin)]

    @staticmethod
    def dijkstra_constrained(g: Graph):
        pq: List[Tuple[T_COST, T_COIN, int]] = []
        visited: Dict[Tuple[int, T_COIN], T_COST] = {}
		
        # INIT_STATE = 0
        heapq.heappush(pq, (0, K, 0))
        while pq:
            cost, coin, pos = heapq.heappop(pq)
            if (pos, coin) in visited and visited[(pos, coin)] <= cost:
                continue
            visited[(pos, coin)] = cost
			
            # 一般不需要
            # _coin = coin
            # while _coin > 0:
                # visited[(pos, _coin)] = min(visited.get((pos, _coin), 10000), cost)
                # _coin -= 1
            
            # END_STATE = N - 1
            if pos == N - 1:
                return cost

            for nxt_pos, nxt_list in g.edges[pos].items():
                for (nxt_cost, nxt_coin) in nxt_list:
                    if coin - nxt_coin >= 0:
                        nxt_state = (cost + nxt_cost, coin - nxt_coin, nxt_pos)
                        heapq.heappush(pq, nxt_state)

        return -1
```

### 3. 最小生成树

**边的存储**：Vec\<Edge\>, Edge := Tuple\<Weight, Either, Other\>

>  Either 和 Other 不可区分，但我们设计 API 不需要严格考虑这一点。

```python
class UDGraph:

    def __init__(self, N):
        self.N = N
        self.edges = []

    def add_edge(self, u, v, w):
        self.edges.append((w, u, v))

    @staticmethod
    def kruskal(g: UDGraph) -> Tuple[Int, Int, Number]:
        # g = g.copy()
        
        uf = UnionFind(g.N)
        mst = []

        queue = g.edges
        heapq.heapify(queue)

        while len(mst) < g.N - 1 and queue:
            w, u, v = heapq.heappop(queue)
            if not uf.connected(u, v):
                uf.union(u, v)
                mst.append((u, v, w))
        return mst
```

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 五 有向图

### 1. 拓扑排序

**边的存储**：Vec\<Set\<End\>\> 和 Vec\<Set<Begin\>\>

```python
class DGraph:

    def __init__(self, V: int):
        self.V = V
        self.edges: List[Set[int]] = [set() for _ in range(V)]
        self.rev_edges: List[Set[int]] = [set() for _ in range(V)]

    def add_edge(self, u: int, v: int) -> None:
        self.edges[u].add(v)
        self.rev_edges[v].add(u)

    @staticmethod
    def topological_sort(g: DGraph) -> List[int]:
        # g = g.copy()
        
        stack: List[int] = []
        for i in range(g.V):
            if not g.rev_edges[i]:
                stack.append(i)

        order: List[int] = []
        while stack:
            node = stack.pop()
            order.append(node)
            for neighbor in g.edges[node]:
                g.rev_edges[neighbor].remove(node)
                if not g.rev_edges[neighbor]:
                    stack.append(neighbor)
            g.edges[node].clear()

        if len(order) != g.V:
            return []

        return order
```

### 2. 最短路径

这里给出的是一个不检查环的版本。

```python
class Graph:

    def __init__(self, N: int):
        self.N = N
        self.edges: List[Dict[PlaceNo, Number]] = [{} for _ in range(N)]

    def add_edge(self, src: PlaceNo, dest: PlaceNo, dist: Number):
        self.edges[src][dest] = dist
        self.edges[dest][src] = dist

    def dijkstra(self, src: PlaceNo, dest: PlaceNo) -> List[PlaceNo]:
        queue: List[Tuple[Number, List[PlaceNo]]] = [(0, [src])]
        while queue:
            dist, path = heapq.heappop(queue)
            last = path[-1]
            if last == dest:
                return path
            for to, weight in self.edges[last].items():
                if to not in path:
                    heapq.heappush(queue, (dist + weight, path + [to]))
        raise ValueError("No path found")
```

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 六 其他

### 1. Eratosthenes 素数筛法

输入：数据最大值 $N$

输出：[0..(N+1)] 的布尔序列，若为素数则置为 True

```python
def eratosthenes(n):
    sieve = [True] * (n + 1)
    sieve[0] = False
    sieve[1] = False

    for i in range(2, n + 1):
        if sieve[i]:
            for j in range(i + i, n + 1, i):
                sieve[j] = False
    return sieve
```

### 2. 二分查找模板

```python
left, right = MIN, MAX
while left <= right:
    mid = (left + right) // 2
    if GO_LEFT():
        left = mid + 1
    else:
        right = mid - 1
```

### 3. 归并排序求逆序对个数

```python
def merge(arr, aux, lo, mid, hi):
    for k in range(lo, hi + 1):
        aux[k] = arr[k]

    i = lo
    j = mid + 1
    res = 0
    for k in range(lo, hi + 1):
        if (i > mid):
            arr[k] = aux[j]
            j += 1
        elif (j > hi):
            arr[k] = aux[i]
            i += 1
        elif (aux[j] < aux[i]):
            arr[k] = aux[j]
            j += 1
            res += mid - i + 1
        else:
            arr[k] = aux[i]
            i += 1
    return res


def merge_sort(arr, aux, lo, hi):
    if lo >= hi:
        return 0

    left_rcnt = merge_sort(arr, aux, lo, (lo + hi) // 2)
    right_rcnt = merge_sort(arr, aux, (lo + hi) // 2 + 1, hi)
    merge_rcnt = merge(arr, aux, lo, (lo + hi) // 2, hi)

    return left_rcnt + right_rcnt + merge_rcnt


while True:
    N = int(input())
    if N == 0:
        break

    arr = [int(input()) for _ in range(N)]
    aux = [0 for _ in range(N)]

    res = merge_sort(arr, aux, 0, N - 1)
    print(res)
```

### 4. 最长不增子序列

```python
dp = [1 for _ in range(n)]
for i in range(1, n):
    for j in range(i):
        if arr[i] <= arr[j]:
            dp[i] = max(dp[j] + 1, dp[i])
```

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 典型例题（找灵感/防睿智用）

### 1. 对状态的搜索

#### 02754 八皇后

```python
def dfs(record: List[int]):
    if len(record) == 8:
        results.append(record)
    for i in range(8):
        for j in range(len(record)):
            if record[j] == i or abs(record[j] - i) == len(record) - j:
                break
        else:
            dfs(record + [i])
```

#### 03151 Pots

```python
from collections import deque
from typing import Tuple, Deque, Set, List

A, B, C = map(int, input().split())

# A >= B
IDX = [1, 2] if A >= B else [2, 1]
if A < B:
    A, B = B, A
A_IDX = 0
B_IDX = 1

queue: Deque[Tuple[Tuple[int, int], List[str]]] = deque()
visited: Set[Tuple[int, int]] = set()

queue.append(((0, 0), []))
while queue:
    state, ops = queue.popleft()
    if state[0] == C or state[1] == C:
        print(len(ops))
        print("\n".join(ops))
        exit()

    if state in visited:
        continue
    visited.add(state)

    # FILL
    queue.append(((A, state[1]), ops + [f"FILL({IDX[A_IDX]})"]))
    queue.append(((state[0], B), ops + [f"FILL({IDX[B_IDX]})"]))

    # DROP
    queue.append(((0, state[1]), ops + [f"DROP({IDX[A_IDX]})"]))
    queue.append(((state[0], 0), ops + [f"DROP({IDX[B_IDX]})"]))

    # POUR
    ## A to B
    out = min(B - state[1], state[0])
    queue.append(((state[0] - out, state[1] + out),
                  ops + [f"POUR({IDX[A_IDX]},{IDX[B_IDX]})"]))
    ## B to A
    out = min(A - state[0], state[1])
    queue.append(((state[0] + out, state[1] - out),
                  ops + [f"POUR({IDX[B_IDX]},{IDX[A_IDX]})"]))

print("impossible")
```

#### 01426 Find The Multiple

```python
def task(n: int):
    q: Deque[int] = deque()
    q.append((1 % n, "1"))
    visited: Set[int] = set([1 % n])

    while q:
        mod, s = q.popleft()
        if mod == 0:
            print(s)
            return

        for digit in [0, 1]:
            new_mod = (mod * 10 + digit) % n
            new_s = s + str(digit)

            if new_mod not in visited:
                visited.add(new_mod)
                q.append((new_mod, new_s))
```

#### 28050 骑士周游

```python
from __future__ import annotations
from typing import Tuple, List
import sys

sys.setrecursionlimit(100000)


class KnightTour:

    def __init__(self, N: int, R: int, C: int):
        self.N = N
        self.R = R
        self.C = C

        self.moves = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1),
                      (-2, 1), (-2, -1)]

        self.board = [0] * (N * N)
        self.board[self.pos2idx(self.R, self.C)] = 1

        self.res = False

    def pos2idx(self, x: int, y: int) -> int:
        return x * self.N + y

    def idx2pos(self, idx: int) -> Tuple[int, int]:
        return idx // self.N, idx % self.N

    def is_legal_step(self, x: int, y: int) -> bool:
        return 0 <= x < self.N and 0 <= y < self.N and not self.board[
            self.pos2idx(x, y)]

    def is_finished(self) -> bool:
        return sum(self.board) == self.N * self.N

    def get_weight(self, x: int, y: int) -> int:
        res = 0
        for dx, dy in self.moves:
            curX = x + dx
            curY = y + dy
            if self.is_legal_step(curX, curY):
                res += 1
        return res

    def get_next_moves(self, x: int, y: int) -> List[Tuple[int, int]]:
        res = []
        for dx, dy in self.moves:
            curX = x + dx
            curY = y + dy
            if self.is_legal_step(curX, curY):
                res.append((curX, curY))
        res.sort(key=lambda x: self.get_weight(x[0], x[1]), reverse=False)
        return res

    def knight_tour_helper(self, x: int, y: int) -> None:
        for nx, ny in self.get_next_moves(x, y):
            # Return if the result is found in the previous step
            if self.res:
                return

            # Go to the next step otherwise
            self.board[self.pos2idx(nx, ny)] = 1
            if self.is_finished():
                self.res = True
                return
            self.knight_tour_helper(nx, ny)

            # Backtrack
            self.board[self.pos2idx(nx, ny)] = 0

    def knight_tour(self) -> None:
        self.knight_tour_helper(self.R, self.C)


N = int(input())
R, C = map(int, input().split())

kt = KnightTour(N, R, C)
kt.knight_tour()
print("success" if kt.res else "fail")
```

### 2. 汉诺塔

```python
def iteration(N, t1, t2, t3):
    if N == 1:
        print(f"{N}:{t1}->{t3}")
    else:
        iteration(N - 1, t1, t3, t2)
        print(f"{N}:{t1}->{t3}")
        iteration(N - 1, t2, t1, t3)


N, t1, t2, t3 = input().split()
N = int(N)
iteration(N, t1, t2, t3)
```

### 3. 平衡二叉树

```python
class TreeNode(object):

    def __init__(self, key: int):
        self.key = key
        self.height = 1
        self.left = None
        self.right = None


class AVLTree(object):

    def __init__(self):
        self.root = None

    def insert(self, key: int) -> None:
        self.root = self.__insert(self.root, key)

    def __get_height(self, root: TreeNode) -> int:
        return 0 if root is None else root.height

    def __update_height(self, root: TreeNode) -> None:
        left_height = self.__get_height(root.left)
        right_height = self.__get_height(root.right)
        root.height = max(left_height, right_height) + 1

    def __get_balance_factor(self, root: TreeNode) -> int:
        left_height = self.__get_height(root.left)
        right_height = self.__get_height(root.right)
        return left_height - right_height

    def __right_rotate(self, root: TreeNode) -> TreeNode:
        left = root.left
        root.left = left.right
        left.right = root

        self.__update_height(root)
        self.__update_height(left)

        return left

    def __left_rotate(self, root: TreeNode) -> TreeNode:
        right = root.right
        root.right = right.left
        right.left = root

        self.__update_height(root)
        self.__update_height(right)

        return right

    def __insert(self, root: TreeNode, key: int) -> TreeNode:
        if root is None:
            return TreeNode(key)

        if key < root.key:
            root.left = self.__insert(root.left, key)
        elif key > root.key:
            root.right = self.__insert(root.right, key)
        else:
            # ignore duplicate
            return root

        self.__update_height(root)

        balance_factor = self.__get_balance_factor(root)

        if balance_factor > 1:
            if key < root.left.key:
                # left-left case
                root = self.__right_rotate(root)
            else:
                # left-right case
                root.left = self.__left_rotate(root.left)
                root = self.__right_rotate(root)
        elif balance_factor < -1:
            if key > root.right.key:
                # right-right case
                root = self.__left_rotate(root)
            else:
                # right-left case
                root.right = self.__right_rotate(root.right)
                root = self.__left_rotate(root)

        return root

    def insert(self, key: int) -> None:
        self.root = self.__insert(self.root, key)
```

### 4. 不会的题

**28190 奶牛排队**

> 奶牛在熊大妈的带领下排成了一条直队。显然，不同的奶牛身高不一定相同……
>
> 现在，奶牛们想知道，如果找出一些连续的奶牛，要求最左边的奶牛 A 是最矮的，最右边的 B 是最高的，且 B 高于 A 奶牛。中间如果存在奶牛，则身高不能和 A,B 奶牛相同。问这样的奶牛最多会有多少头？
>
> 从左到右给出奶牛的身高，请告诉它们符合条件的最多的奶牛数（答案可能是 0,2，但不会是 1）。

```python
import bisect

N = int(input())
arr = [int(input()) for _ in range(N)]
assert len(arr) == N

min_stack = []
max_stack = []
result = 0

for j in range(N):
    while len(min_stack) > 0 and arr[min_stack[-1]] >= arr[j]:
        min_stack.pop()
    while len(max_stack) > 0 and arr[max_stack[-1]] < arr[j]:
        max_stack.pop()
    if len(min_stack) > 0:
        k = bisect.bisect(min_stack,
                          max_stack[-1] if len(max_stack) > 0 else -1)
        if (k != len(min_stack)):
            result = max(result, j - min_stack[k] + 1)

    min_stack.append(j)
    max_stack.append(j)

print(result)
```

<!-- 注释语句：导出PDF时会在这里分页 -->

<div style="page-break-after:always"></div>

<!-- 注释语句：导出PDF时会在这里分页 -->

## 常用工具速查

|                             引入                             |                  用法                  |       用途        |
| :----------------------------------------------------------: | :------------------------------------: | :---------------: |
|               from functools import lru_cache                |        @lru_cache(maxsize=1000)        | 递归，记忆化搜索  |
|                           ord chr                            |          ord(char); chr(int)           | ASCII 和 字符转换 |
|              math.ceil, math.floor, math.round               |                                        |       取整        |
|                         bin oct hex                          |                xxx(int)                |     转换进制      |
| bisect.bisect_left, bisect.bisect_right, bisect.insort_left, bisect.insort_right | bisect.xxxxxx_xxxx(List[_Type], _Type) |     二分查找      |
|                         import heapq                         |                                        |        堆         |
|             from collections import defaultdict              |          defaultdict[K_Type]           |                   |
|                    sys.setrecursionlimit                     |                                        |     避免爆栈      |
|                                                              |                                        |                   |
