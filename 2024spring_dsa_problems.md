# 数算（数据结构与算法）题目

*Updated 2025-10-07 10:07 GMT+8*
 *Compiled by Hongfei Yan (2024 Spring)*



> **Logs：**
>
> 2025/01/01: 题目编号前的英文字母大写表示题目的难度等级，分别对应 Easy、Medium 和 Tough。
>
> 2024/4/12: 因为不断汇总进来数算题目，2024_dsapre.md 更名为 2024spring_dsa_problems.md
>
> 2024/2/18:
>
> 1）数算课程在春季学期开，适逢秋季计算概论课程结束，正值寒假，同学建议每日推出少许题目练习，因此创立此题集。
>
> 2）为避免重复，如果题目出现在  https://github.com/GMyhf/2020fall-cs101 计概题集 2020fall_cs101.openjudge.cn_problems.md，会给出指引。如果计概题集中有明显数算题目，也会移过来。
>
> 3）有同学假期时候完成了这些题目，放在 gitbub上面，可以参考
>
> Wangjie Su's GitHub,  https://github.com/csxq0605/CS101-spring
>
> Zixuan Jiang's summary, https://github.com/GMyhf/2024spring-cs201/blob/main/cheatsheet/DataStructuresAndAlgorithms-WinterBreak-20240214-JIANGZixuan.md
>
> Deng Jinwen's summary, https://github.com/JinwenDeng/DSA-Summary
>



# 01094~02299

## T01094: Sorting It All Out

topological sort, http://cs101.openjudge.cn/practice/01094/

An ascending sorted sequence of distinct values is one in which some form of a less-than operator is used to order the elements from smallest to largest. For example, the sorted sequence A, B, C, D implies that A < B, B < C and C < D. in this problem, we will give you a set of relations of the form A < B and ask you to determine whether a sorted order has been specified or not. 

**输入**

Input consists of multiple problem instances. Each instance starts with a line containing two positive integers n and m. the first value indicated the number of objects to sort, where 2 <= n <= 26. The objects to be sorted will be the first n characters of the uppercase alphabet. The second value m indicates the number of relations of the form A < B which will be given in this problem instance. Next will be m lines, each containing one such relation consisting of three characters: an uppercase letter, the character "<" and a second uppercase letter. No letter will be outside the range of the first n letters of the alphabet. Values of n = m = 0 indicate end of input.

**输出**

For each problem instance, output consists of one line. This line should be one of the following three:

Sorted sequence determined after xxx relations: yyy...y.
Sorted sequence cannot be determined.
Inconsistency found after xxx relations.

where xxx is the number of relations processed at the time either a sorted sequence is determined or an inconsistency is found, whichever comes first, and yyy...y is the sorted, ascending sequence.

样例输入

```
4 6
A<B
A<C
B<C
C<D
B<D
A<B
3 2
A<B
B<A
26 1
A<Z
0 0
```

样例输出

```
Sorted sequence determined after 4 relations: ABCD.
Inconsistency found after 2 relations.
Sorted sequence cannot be determined.
```

来源

East Central North America 2001



```python
#23n2310307206胡景博
from collections import deque
def topo_sort(graph):
    in_degree = {u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    q = deque([u for u in in_degree if in_degree[u] == 0])
    topo_order = [];flag = True
    while q:
        if len(q) > 1:
            flag = False#topo_sort不唯一确定
        u = q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(topo_order) != len(graph): return 0
    return topo_order if flag else None
while True:
    n,m = map(int,input().split())
    if n == 0: break
    graph = {chr(x+65):[] for x in range(n)}
    edges = [tuple(input().split('<')) for _ in range(m)]
    for i in range(m):
        a,b = edges[i]
        graph[a].append(b)
        t = topo_sort(graph)
        if t:
            s = ''.join(t)
            print("Sorted sequence determined after {} relations: {}.".format(i+1,s))
            break
        elif t == 0:
            print("Inconsistency found after {} relations.".format(i+1))
            break
    else:
        print("Sorted sequence cannot be determined.")
```



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



Q. 有没有能够随时跟进图的变化的拓扑排序算法。而不是每次图变化都要重新排一遍。

下面是一个可直接 AC，并且真正“增量”维护拓扑序的 Python 实现。它在每次插入一条新边时，只做一次受限的 BFS 和一次局部数组搬移，最坏 O(n+m)，但是常见场景下远快于重排全图。

```python
from collections import deque, defaultdict

def kahn_check(adj, nodes):
    """
    改进的 Kahn 算法，用来判断当前图是否：
      - 有环，返回 0
      - 唯一拓扑序，返回该序列 list
      - 多解，返回 None
    """
    in_deg = {u: 0 for u in nodes}
    for u in adj:
        for v in adj[u]:
            in_deg[v] += 1

    q = deque(u for u in nodes if in_deg[u] == 0)
    topo = []
    unique = True
    while q:
        if len(q) > 1:
            unique = False
        u = q.popleft()
        topo.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)

    if len(topo) < len(nodes):
        return 0      # 有环
    return topo if unique else None

def forward_reachable(adj, start, pos, hi):
    """
    从 start 沿出边做 BFS，只收集那些 pos[w] ≤ hi 的节点。
    """
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen and pos[v] <= hi:
                seen.add(v)
                q.append(v)
    return seen

def solve():
    import sys
    for line in sys.stdin:
        line = line.strip().split()
        if not line:
            continue
        n, m = map(int, line)
        if n == 0 and m == 0:
            break

        # 节点 A, B, ..., chr(ord('A')+n-1)
        nodes = [chr(ord('A') + i) for i in range(n)]
        adj   = defaultdict(list)

        # 初始序列就是 A, B, C, ...
        order = nodes[:]
        pos   = {u:i for i,u in enumerate(order)}

        def insert_edge(u, v):
            """
            插入 u->v：
            1) 如果 pos[u] < pos[v]，无需调整
            2) 否则
               F = 所有能从 v 出发（沿原图）到达且 pos[...] ≤ pos[u] 的节点
               若 u∈F 则成环，返回 False
               否则把 F 从 order 中摘出，插到 u 之后，更新 pos，返回 True
            """
            adj[u].append(v)
            if pos[u] < pos[v]:
                return True

            F = forward_reachable(adj, v, pos, pos[u])
            if u in F:
                return False

            # 从 order 中删掉 F
            new_order = [w for w in order if w not in F]
            # 找 u 在删完 F 后的新位置
            idx = new_order.index(u)
            # 保持 F 在旧 order 中的相对次序
            F_list = [w for w in order if w in F]

            # 重建 order：u 之后插入 F_list
            order.clear()
            order.extend(new_order[:idx+1])
            order.extend(F_list)
            order.extend(new_order[idx+1:])

            # 更新 pos
            for i,w in enumerate(order):
                pos[w] = i
            return True

        # 读入每条关系，在线处理
        rels = [sys.stdin.readline().strip() for _ in range(m)]
        done = False
        for i, rel in enumerate(rels, start=1):
            u, v = rel.split('<')
            ok = insert_edge(u, v)
            if not ok:
                print(f"Inconsistency found after {i} relations.")
                done = True
                break
            chk = kahn_check(adj, nodes)
            if isinstance(chk, list):
                print(f"Sorted sequence determined after {i} relations: {''.join(chk)}.")
                done = True
                break

        if not done:
            print("Sorted sequence cannot be determined.")

if __name__ == "__main__":
    solve()
```

说明

- **数据结构**
  - `order`：当前维护的一个合法拓扑序列。
  - `pos[u]`：节点 `u` 在 `order` 中的下标。
  - `adj`：邻接表，增量插入。
- **增量更新逻辑**
  1. 向 `adj` 添加 `u→v`。
  2. 若 `pos[u] < pos[v]`，说明原序列已满足 `u<v`，无需改动。
  3. 否则，从 `v` 出发做一次受限 BFS，只收集那些 `pos[w] ≤ pos[u]` 的可达节点 `F`。
     - 若 `u∈F`，则新边与原路径 `v→…→u` 构成环，直接报错。
     - 否则，取 `F` 在原 `order` 中的相对次序，整体摘出并插到 `u` 之后，再更新所有 `pos`。
- **唯一性检测**
  每插完一条边后，调用一次改进的 Kahn 算法（`kahn_check`），它在处理队列时监测“可选入度为 0 的节点”是否超过 1，以此判定序列是否**唯一**。

这样，就能做到真正的“在线”增量维护——只有受影响部分被重排，避免了全图重走一遍拓扑排序。



## T01145: Tree Summing

tree, http://cs101.openjudge.cn/practice/01145/

LISP was one of the earliest high-level programming languages and, with FORTRAN, is one of the oldest languages currently being used. Lists, which are the fundamental data structures in LISP, can easily be adapted to represent other important data structures such as trees. 

This problem deals with determining whether binary trees represented as LISP S-expressions possess a certain property. 
Given a binary tree of integers, you are to write a program that determines whether there exists a root-to-leaf path whose nodes sum to a specified integer. For example, in the tree shown below there are exactly four root-to-leaf paths. The sums of the paths are 27, 22, 26, and 18. 

![img](http://media.openjudge.cn/images/1145/1145_1.gif)

Binary trees are represented in the input file as LISP S-expressions having the following form. 

```
empty tree ::= ()

tree 	   ::= empty tree (integer tree tree)
```

The tree diagrammed above is represented by the expression (5 (4 (11 (7 () ()) (2 () ()) ) ()) (8 (13 () ()) (4 () (1 () ()) ) ) ) 

Note that with this formulation all leaves of a tree are of the form (integer () () ) 

Since an empty tree has no root-to-leaf paths, any query as to whether a path exists whose sum is a specified integer in an empty tree must be answered negatively. 

**输入**

The input consists of a sequence of test cases in the form of integer/tree pairs. Each test case consists of an integer followed by one or more spaces followed by a binary tree formatted as an S-expression as described above. All binary tree S-expressions will be valid, but expressions may be spread over several lines and may contain spaces. There will be one or more test cases in an input file, and input is terminated by end-of-file. 

**输出**

There should be one line of output for each test case (integer/tree pair) in the input file. For each pair I,T (I represents the integer, T represents the tree) the output is the string yes if there is a root-to-leaf path in T whose sum is I and no if there is no path in T whose sum is I. 

样例输入

```
22 (5(4(11(7()())(2()()))()) (8(13()())(4()(1()()))))
20 (5(4(11(7()())(2()()))()) (8(13()())(4()(1()()))))
10 (3 
     (2 (4 () () )
        (8 () () ) )
     (1 (6 () () )
        (4 () () ) ) )
5 ()
```

样例输出

```
yes
no
yes
no
```

来源

Duke Internet Programming Contest 1992,UVA 112



实现树：节点链接法。每个节点保存根节点的数据项，以及指向左右子树的链接。

成员 val 保存根节点数据项，成员 left/rightChild 则保存指向左/右子树的引用（同样是TreeNode 对象）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def has_path_sum(root, target_sum):
    if root is None:
        return False

    if root.left is None and root.right is None:  # The current node is a leaf node
        return root.val == target_sum

    left_exists = has_path_sum(root.left, target_sum - root.val)
    right_exists = has_path_sum(root.right, target_sum - root.val)

    return left_exists or right_exists


# Parse the input string and build a binary tree
def parse_tree(s):
    stack = []
    i = 0

    while i < len(s):
        if s[i].isdigit() or s[i] == '-':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '-'):
                j += 1
            num = int(s[i:j])
            node = TreeNode(num)
            if stack:
                parent = stack[-1]
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node
            stack.append(node)
            i = j
        elif s[i] == '[':
            i += 1
        elif s[i] == ']' and s[i - 1] != '[' and len(stack) > 1:
            stack.pop()
            i += 1
        else:
            i += 1

    return stack[0] if len(stack) > 0 else None


while True:
    try:
        s = input()
    except:
        break

    s = s.split()
    target_sum = int(s[0])
    tree = ("").join(s[1:])
    tree = tree.replace('(', ',[').replace(')', ']')
    while True:
        try:
            tree = eval(tree[1:])
            break
        except SyntaxError:
            s = input().split()
            s = ("").join(s)
            s = s.replace('(', ',[').replace(')', ']')
            tree += s

    tree = str(tree)
    tree = tree.replace(',[', '[')
    if tree == '[]':
        print("no")
        continue

    root = parse_tree(tree)

    if has_path_sum(root, target_sum):
        print("yes")
    else:
        print("no")
```



下面给出一个**基于递归直接解析 S-表达式**的版本，不再依赖 `eval`。

```python
import sys

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val        # 节点的数值
        self.left = left      # 左子树
        self.right = right    # 右子树

def has_path_sum(root, target_sum):
    """
    判断从 root 出发，是否存在一条根->叶路径，其节点值之和等于 target_sum
    """
    if root is None:
        return False
    # 到达叶子节点，直接比较剩余 target
    if root.left is None and root.right is None:
        return root.val == target_sum
    # 递归地在左右子树中寻找，更新剩余 target_sum
    return (
        has_path_sum(root.left, target_sum - root.val)
        or has_path_sum(root.right, target_sum - root.val)
    )

def parse_s_expr(s, i):
    """
    递归解析 S-表达式：
    输入：字符串 s，以及当前读取位置 i（指向一个 '('）
    输出：解析出的 TreeNode 或 None，以及新的读取位置 j（指向该子表达式末尾后一个位置）
    """
    # 跳过所有空白
    while i < len(s) and s[i].isspace():
        i += 1
    # 必须以 '(' 开头
    if i >= len(s) or s[i] != '(':
        raise ValueError("格式错误：期待 '('，但在位置 %d 遇到 '%s'" % (i, s[i] if i < len(s) else 'EOF'))
    i += 1  # 跳过 '('

    # 跳过空格，检查是不是空树 "()"
    while i < len(s) and s[i].isspace():
        i += 1
    if i < len(s) and s[i] == ')':
        # 这是一个空树
        return None, i + 1

    # 读整数（可能为负数）
    neg = False
    if s[i] == '-':
        neg = True
        i += 1
    if i >= len(s) or not s[i].isdigit():
        raise ValueError("格式错误：期待数字，但在位置 %d 遇到 '%s'" % (i, s[i] if i < len(s) else 'EOF'))
    num = 0
    while i < len(s) and s[i].isdigit():
        num = num * 10 + int(s[i])
        i += 1
    if neg:
        num = -num

    # 创建当前节点
    node = TreeNode(num)

    # 解析左子树
    node.left, i = parse_s_expr(s, i)
    # 解析右子树
    node.right, i = parse_s_expr(s, i)

    # 跳过空白，接着应有一个 ')'
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s) or s[i] != ')':
        raise ValueError("格式错误：期待 ')'，但在位置 %d 遇到 '%s'" % (i, s[i] if i < len(s) else 'EOF'))
    return node, i + 1

def read_one_case():
    """
    从标准输入读取一个测试用例：
      - 先读包含 target_sum 的那一行
      - 然后根据括号配对读取完整的 S-表达式
    返回 (target_sum, expr_string)，若 EOF 则返回 (None, None)
    """
    line = ''
    # 跳过空行
    while True:
        line = sys.stdin.readline()
        if not line:
            return None, None
        if line.strip():
            break

    parts = line.strip().split(None, 1)
    target_sum = int(parts[0])
    expr = parts[1] if len(parts) > 1 else ''
    # 统计当前括号平衡情况
    balance = expr.count('(') - expr.count(')')
    # 如果还没配对完，就继续读
    while balance > 0:
        nxt = sys.stdin.readline()
        if not nxt:
            break
        expr += nxt.strip()
        balance = expr.count('(') - expr.count(')')
    return target_sum, expr

def main():
    while True:
        target_sum, expr = read_one_case()
        if target_sum is None:
            break  # EOF 退出

        # 解析整棵树
        try:
            root, _ = parse_s_expr(expr, 0)
        except ValueError as e:
            # 格式有问题，当作空树处理
            root = None

        # 空树直接 no
        if root is None:
            print("no")
            continue

        # 判断并输出
        print("yes" if has_path_sum(root, target_sum) else "no")

if __name__ == "__main__":
    main()
```

> **核心思路**
>
> 1. **直接递归解析 S-表达式**
>    - `parse_s_expr` 从一个 `(` 开始，跳过空格
>    - 如果下一个字符立刻是 `)`，就是空树 `()`
>    - 否则先读一个整数值，构造节点
>    - 递归解析左子树，再递归解析右子树
>    - 最后跳过空格、匹配右括号 `)`
> 2. **准确控制读取位置**
>    不再拼接、`eval`、再转回字符串，避免中间出错；而是一次性读入完整字符串并用索引控制解析。
> 3. **输入驱动**
>    `read_one_case` 保证：
>    - 先读含目标值的一行
>    - 根据括号数量动态读剩余行，直到配对完整
>
> 这样凡是合法的 LISP 树，都能被准确地转成二叉树、再去判断根—叶路径和。



实现二叉树：嵌套列表法。用 Python List 来实现二叉树树数据结构；递归的嵌套列表实现二叉树，由具有 3 个
元素的列表实现：第 1 个元素为根节点的值；第 2 个元素是左子树（用列表表示）；第 3 个元素是右子树。

嵌套列表法的优点子树的结构与树相同，是一种递归结构可以很容易扩展到多叉树，仅需要增加列表元素即可。

定义一系列函数来辅助操作嵌套列表
BinaryTree 创建仅有根节点的二叉树，insertLeft/insertRight 将新节点插入树中作为 root 直接的左/右子节点，
原来的左/右子节点变为新节点的左/右子节点。为什么？不为什么，一种实现方式而已。get/setRootVal 则取得或返回根节点，getLeft/RightChild 返回左/右子树。

嵌套列表示例

```python
def BinaryTree(r, left=[], right=[]):
    return([r, left, right])


def getLeftChild(root):
    return(root[1])


def getRightChild(root):
    return(root[2])


def insertLeft(root, newBranch):
    root[1] = BinaryTree(newBranch, left=getLeftChild(root))
    return(root)


def insertRight(root, newBranch):
    root[2] = BinaryTree(newBranch, right=getRightChild(root))
    return(root)


def getRootVal(root):
    return(root[0])


def setRootVal(root, newVal):
    root[0] = newVal


if __name__ == "__main__":
    r = BinaryTree(3)
    insertLeft(r, 4)
    insertLeft(r, 5)
    insertRight(r, 6)
    insertRight(r, 7)
    l = getLeftChild(r)
    print(l)

    setRootVal(l, 9)
    print(r)
    insertLeft(l, 11)
    print(r)
    print(getRightChild(getRightChild(r)))
```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240122171343441.png" alt="image-20240122171343441" style="zoom: 50%;" />



01145:Tree Summing用嵌套列表法AC代码。

```python
def BinaryTree(r,left=[],right=[]):
    return([r,left,right])
def getLeftChild(root):
    return(root[1])
def getRightChild(root):
    return(root[2])
def insertLeft(root,newBranch):
    root[1]=BinaryTree(newBranch,left=getLeftChild(root))
    return(root)
def insertRight(root,newBranch):
    root[2]=BinaryTree(newBranch,right=getRightChild(root))
    return(root)
def getRootVal(root):
    return(root[0])
def setRootVal(root,newVal):
    root[0]=newVal

while True:
    try:
        left=right=0
        astr=input().replace(' ','')
        for i in range(len(astr)):
            if astr[i]=='(' or astr==')':
                bound=i
                break
        num=int(astr[:bound])
        astr=astr[bound:]
        for i in astr:
            if i=='(':
                left+=1
            elif i==')':
                right+=1
        while left!=right:
            bstr=input().replace(' ','')
            for i in bstr:
                if i=='(':
                    left+=1
                elif i==')':
                    right+=1
            astr+=bstr
        if astr=='()':
            print('no')
            continue
        atree=BinaryTree('')
        cur=atree
        aStack=[]
        astr=astr[1:len(astr)-1]
        j=0
        while j<len(astr):
            if astr[j]=='(':
                aStack.append(cur)
                if getLeftChild(cur)==[]:
                    insertLeft(cur,None)
                    cur=getLeftChild(cur)
                else:
                    insertRight(cur,None)
                    cur=getRightChild(cur)
                j+=1
            elif astr[j]==')':
                cur=aStack.pop()
                j+=1
            else:
                anum=''
                while astr[j]!='(' and astr[j]!=')':
                    anum+=astr[j]
                    j+=1
                setRootVal(cur,int(anum))
        #print(num,atree)
        def compare(btree,bnum):
            if getRootVal(btree)==None:
                return(False)
            elif getRootVal(btree)==bnum and (getRootVal(getLeftChild(btree))==None and getRootVal(getRightChild(btree))==None):
                return(True)
            else:
                if compare(getLeftChild(btree),bnum-getRootVal(btree)) or compare(getRightChild(btree),bnum-getRootVal(btree)):
                    return(True)
            return(False)
        if compare(atree,num):
            print('yes')
        else:
            print('no')
    except EOFError:
        break
```



## 01178: Camelot

Centuries ago, King Arthur and the Knights of the Round Table used to meet every year on New Year's Day to celebrate their fellowship. In remembrance of these events, we consider a board game for one player, on which one king and several knight pieces are placed at random on distinct squares.
The Board is an 8x8 array of squares. The King can move to any adjacent square, as shown in Figure 2, as long as it does not fall off the board. A Knight can jump as shown in Figure 3, as long as it does not fall off the board.
![img](http://media.openjudge.cn/images/g180/1178_1.jpg)
During the play, the player can place more than one piece in the same square. The board squares are assumed big enough so that a piece is never an obstacle for other piece to move freely.
The player?s goal is to move the pieces so as to gather them all in the same square, in the smallest possible number of moves. To achieve this, he must move the pieces as prescribed above. Additionally, whenever the king and one or more knights are placed in the same square, the player may choose to move the king and one of the knights together henceforth, as a single knight, up to the final gathering point. Moving the knight together with the king counts as a single move.

Write a program to compute the minimum number of moves the player must perform to produce the gathering.

**输入**

Your program is to read from standard input. The input contains the initial board configuration, encoded as a character string. The string contains a sequence of up to 64 distinct board positions, being the first one the position of the king and the remaining ones those of the knights. Each position is a letter-digit pair. The letter indicates the horizontal board coordinate, the digit indicates the vertical board coordinate.

0 <= number of knights <= 63

**输出**

Your program is to write to standard output. The output must contain a single line with an integer indicating the minimum number of moves the player must perform to produce the gathering.

样例输入

```
D4A3A8H1H8
```

样例输出

```
10
```

来源

IOI 1998



```python
import sys

inf = float('infinity')
kmove = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
knmove = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
kmap = [[inf]*64 for _ in range(64)]
knmap = [[inf]*64 for _ in range(64)]

def ok(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def getxy(p):
    return p % 8, p // 8

def getPosition(x, y):
    return x + y * 8

def init():
    for i in range(64):
        kmap[i][i] = 0
        knmap[i][i] = 0
        x, y = getxy(i)
        for j in range(8):
            tx, ty = kmove[j][0] + x, kmove[j][1] + y
            if ok(tx, ty):
                next = getPosition(tx, ty)
                kmap[i][next] = 1
            tx, ty = knmove[j][0] + x, knmove[j][1] + y
            if ok(tx, ty):
                next = getPosition(tx, ty)
                knmap[i][next] = 1

def floyd():
    for k in range(64):
        for i in range(64):
            for j in range(64):
                kmap[i][j] = min(kmap[i][j], kmap[i][k] + kmap[k][j])
                knmap[i][j] = min(knmap[i][j], knmap[i][k] + knmap[k][j])

init()
floyd()

s = input().strip()
size = len(s)
num = 0
position = [0]*64

for i in range(0, size, 2):
    position[num] = ord(s[i]) - ord('A') + (ord(s[i+1]) - ord('1')) * 8
    num += 1

minmove = inf
total = 0  # Renamed 'sum' to 'total'
for ds in range(64):
    for m in range(64):
        for k in range(1, num):
            total = sum(knmap[position[i]][ds] for i in range(1, num))
            total += kmap[position[0]][m]
            total += knmap[position[k]][m] + knmap[m][ds]
            total -= knmap[position[k]][ds]
            minmove = min(minmove, total)

print(minmove)
```



## 01182: 食物链

并查集, http://cs101.openjudge.cn/practice/01182

动物王国中有三类动物A,B,C，这三类动物的食物链构成了有趣的环形。A吃B， B吃C，C吃A。
现有N个动物，以1－N编号。每个动物都是A,B,C中的一种，但是我们并不知道它到底是哪一种。
有人用两种说法对这N个动物所构成的食物链关系进行描述：
第一种说法是"1 X Y"，表示X和Y是同类。
第二种说法是"2 X Y"，表示X吃Y。
此人对N个动物，用上述两种说法，一句接一句地说出K句话，这K句话有的是真的，有的是假的。当一句话满足下列三条之一时，这句话就是假话，否则就是真话。
1） 当前的话与前面的某些真的话冲突，就是假话；
2） 当前的话中X或Y比N大，就是假话；
3） 当前的话表示X吃X，就是假话。
你的任务是根据给定的N（1 <= N <= 50,000）和K句话（0 <= K <= 100,000），输出假话的总数。

**输入**

第一行是两个整数N和K，以一个空格分隔。
以下K行每行是三个正整数 D，X，Y，两数之间用一个空格隔开，其中D表示说法的种类。
若D=1，则表示X和Y是同类。
若D=2，则表示X吃Y。

**输出**

只有一个整数，表示假话的数目。

样例输入

```
100 7
1 101 1 
2 1 2
2 2 3 
2 3 3 
1 1 3 
2 3 1 
1 5 5
```

样例输出

```
3
```

来源

Noi 01



动物王国中有三类动物A,B,C，这三类动物的食物链构成了有趣的环形。A吃B， B吃C，C吃A。
意思就是所有的种类只有A,B,C三种，只有三个关系A吃B， B吃C，C吃A。
思路：创建3个分组i-A,i-B,i-C。
如果x和y是同类，正确则合并x-A和y-A、x-B和y-B、x-C和y-C。
　　　　　　　　当存在x吃y或者y吃x时不正确。
如果x吃y，正确则合并x-A和y-B、x-B和y-C、x-C和y-A。
　　　　　　当存在x和y是同类或者y吃x时不正确。

这题还有一个特殊之处，在于当两个的吃与被吃关系定下来，由于环形结构，另一个的吃与被吃关系就确定了，因此需要合并三次。

```python
class DisjointSet:
    def __init__(self, n):
        #设[1,n] 区间表示同类，[n+1,2*n]表示x吃的动物，[2*n+1,3*n]表示吃x的动物。
        self.parent = [i for i in range(3 * n + 1)] # 每个动物有三种可能的类型，用 3 * n 来表示每种类型的并查集
        self.rank = [0] * (3 * n + 1)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        elif self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True


def is_valid(n, k, statements):
    dsu = DisjointSet(n)

    def find_disjoint_set(x):
        if x > n:
            return False
        return True

    false_count = 0
    for d, x, y in statements:
        if not find_disjoint_set(x) or not find_disjoint_set(y):
            false_count += 1
            continue
        if d == 1:  # X and Y are of the same type
            if dsu.find(x) == dsu.find(y + n) or dsu.find(x) == dsu.find(y + 2 * n):
                false_count += 1
            else:
                dsu.union(x, y)
                dsu.union(x + n, y + n)
                dsu.union(x + 2 * n, y + 2 * n)
        else:  # X eats Y
            if dsu.find(x) == dsu.find(y) or dsu.find(x + 2*n) == dsu.find(y):
                false_count += 1
            else: #[1,n] 区间表示同类，[n+1,2*n]表示x吃的动物，[2*n+1,3*n]表示吃x的动物
                dsu.union(x + n, y)
                dsu.union(x, y + 2 * n)
                dsu.union(x + 2 * n, y + n)

    return false_count


if __name__ == "__main__":
    N, K = map(int, input().split())
    statements = []
    for _ in range(K):
        D, X, Y = map(int, input().split())
        statements.append((D, X, Y))
    result = is_valid(N, K, statements)
    print(result)

```



《挑战程序设计竞赛（第2版）》的2.4.4并查集，也有讲到。

```python
# 并查集，https://zhuanlan.zhihu.com/p/93647900/
'''
我们设[0,n)区间表示同类，[n,2*n)区间表示x吃的动物，[2*n,3*n)表示吃x的动物。

如果是关系1：
　　将y和x合并。将y吃的与x吃的合并。将吃y的和吃x的合并。
如果是关系2：
　　将y和x吃的合并。将吃y的与x合并。将y吃的与吃x的合并。
原文链接：https://blog.csdn.net/qq_34594236/article/details/72587829
'''
# p = [0]*150001

def find(x):	# 并查集查询
    if p[x] == x:
        return x
    else:
        p[x] = find(p[x])	# 父节点设为根节点。目的是路径压缩。
        return p[x]

n,k = map(int, input().split())

p = [0]*(3*n + 1)
for i in range(3*n+1):	#并查集初始化
    p[i] = i

ans = 0
for _ in range(k):
    a,x,y = map(int, input().split())
    if x>n or y>n:
        ans += 1; continue
    
    if a==1:
        if find(x+n)==find(y) or find(y+n)==find(x):
            ans += 1; continue
        
        # 合并
        p[find(x)] = find(y)				
        p[find(x+n)] = find(y+n)
        p[find(x+2*n)] = find(y+2*n)
    else:
        if find(x)==find(y) or find(y+n)==find(x):
            ans += 1; continue
        p[find(x+n)] = find(y)
        p[find(y+2*n)] = find(x)
        p[find(x+2*n)] = find(y+n)

print(ans)
```





思路：本题用时较长，经过多次思绪打结重写后，找到的一种思路为：为每一个动物分配一个环形食物链`Circle`类，每个动物为自己的食物环的A种动物，并且该食物环的根为该动物，若`D=1`，两个动物所在食物环的根不同，那么经过适当旋转后，合并两个食物环；两个动物所在的食物环相同，但两个动物种类不同，则为假话。若`D=2`，两个动物所在食物环的根不同，那么经过适当旋转后，合并两个食物环；两个动物所在的食物环相同，但两个动物种类不满足A吃B，B吃C，C吃A，则为假话。为什么没有为每一个动物分配一个动物类？因为处理很多未知种类的动物之间的吃与被吃关系让人头大。

```python
# 宋昕杰 物理学院
class Circle:
    def __init__(self, root):
        self.kinds = {root: 0}
        self.root = root

    def join(self, circle, rotate):
        for idx, kind in circle.kinds.items():
            self.kinds[idx] = (kind - rotate) % 3
            ls[idx] = self.root


n, k = map(int, input().split())
ls = [i for i in range(n)]
circles = [Circle(i) for i in range(n)]
ans = 0

for _ in range(k):
    d, x, y = map(int, input().split())
    if x > n or y > n or (d == 2 and x == y):
        ans += 1
        continue
    x -= 1
    y -= 1

    idx_x = circles[ls[x]].kinds[x]
    idx_y = circles[ls[y]].kinds[y]
    if d == 1:
        if ls[x] != ls[y]:
            circles[ls[x]].join(circles[ls[y]], (idx_y - idx_x) % 3)
        else:
            if idx_x != idx_y:
                ans += 1
        continue

    if ls[x] != ls[y]:
        circles[ls[x]].join(circles[ls[y]], (idx_y - idx_x - 1) % 3)
    elif (idx_y - idx_x - 1) % 3 != 0:
        ans += 1

print(ans)
```





## M01258: Agri-Net

prim, http://cs101.openjudge.cn/dsapre/01258/

Farmer John has been elected mayor of his town! One of his campaign promises was to bring internet connectivity to all farms in the area. He needs your help, of course. 
Farmer John ordered a high speed connection for his farm and is going to share his connectivity with the other farmers. To minimize cost, he wants to lay the minimum amount of optical fiber to connect his farm to all the other farms. 
Given a list of how much fiber it takes to connect each pair of farms, you must find the minimum amount of fiber needed to connect them all together. Each farm must connect to some other farm such that a packet can flow from any one farm to any other farm. 
The distance between any two farms will not exceed 100,000. 

**输入**

The input includes several cases. For each case, the first line contains the number of farms, N (3 <= N <= 100). The following lines contain the N x N conectivity matrix, where each element shows the distance from on farm to another. Logically, they are N lines of N space-separated integers. Physically, they are limited in length to 80 characters, so some lines continue onto others. Of course, the diagonal will be 0, since the distance from farm i to itself is not interesting for this problem.

**输出**

For each case, output a single integer length that is the sum of the minimum length of fiber required to connect the entire set of farms.

样例输入

```
4
0 4 9 21
4 0 8 17
9 8 0 16
21 17 16 0
```

样例输出

```
28
```

来源

USACO 102



经典Prim算法

```python
#王昊 光华管理学院
from heapq import heappop, heappush


while True:
    try:
        n = int(input())
    except:
        break
    mat, cur = [], 0
    for i in range(n):
        mat.append(list(map(int, input().split())))
    d, v, q, cnt = [100000 for i in range(n)], set(), [], 0
    d[0] = 0
    heappush(q, (d[0], 0))
    while q:
        x, y = heappop(q)
        if y in v:
            continue
        v.add(y)
        cnt += d[y]
        for i in range(n):
            if d[i] > mat[y][i]:
                d[i] = mat[y][i]
                heappush(q, (d[i], i))
    print(cnt)
```





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



```python
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


```



思路：最小生成树，把边按照权重从小到大排序，每次用并查集判断一条边是否有必要添加

```python
# 钟明衡 物理学院
p = []


def P(x):
    if p[x] != x:
        p[x] = P(p[x])
    return p[x]


while True:
    try:
        n = int(input())
    except EOFError:
        break
    ans = 0
    M = [list(map(int, input().split())) for _ in range(n)]
    p = [i for i in range(n)]
    l = []
    for i in range(n):
        for j in range(n):
            if i != j:
                l.append((i, j, M[i][j]))
    l.sort(key=lambda x: x[2])
    for i, j, k in l:
        pi, pj = P(i), P(j)
        if pi != pj:
            p[pi] = pj
            ans += k
    print(ans)

```





## M01321: 棋盘问题

backtracking, http://cs101.openjudge.cn/pctbook/M01321/

在一个给定形状的棋盘（形状可能是不规则的）上面摆放棋子，棋子没有区别。要求摆放时任意的两个棋子不能放在棋盘中的同一行或者同一列，请编程求解对于给定形状和大小的棋盘，摆放k个棋子的所有可行的摆放方案C。

**输入**

输入含有多组测试数据。
每组数据的第一行是两个正整数，n k，用一个空格隔开，表示了将在一个n*n的矩阵内描述棋盘，以及摆放棋子的数目。 n <= 8 , k <= n
当为-1 -1时表示输入结束。
随后的n行描述了棋盘的形状：每行有n个字符，其中 # 表示棋盘区域， . 表示空白区域（数据保证不出现多余的空白行或者空白列）。

**输出**

对于每一组数据，给出一行输出，输出摆放的方案数目C （数据保证C<2^31）。

样例输入

```
2 1
#.
.#
4 4
...#
..#.
.#..
#...
-1 -1
```

样例输出

```
2
1
```

来源：蔡错@pku



这个题是一个典型的 **回溯搜索（DFS + 剪枝）** 问题，类似于不规则棋盘上的「n 皇后」变形版：

- 棋盘是 n×n，但有些位置不可用（用 `.` 表示）。
- 要摆放 `k` 个棋子。
- 限制条件：任意两个棋子不能在同一行或同一列。
- 要输出所有可行方案的个数。

------

**思路解析**

1. **状态表示**
   - 从第 `row=0` 行开始，一行一行往下考虑是否放棋子。
   - 用一个布尔数组 `used_col[j]` 标记某一列是否已经放过棋子。
2. **搜索分支**
   - 在第 `row` 行：
     - 可以选择不放棋子，继续搜索下一行。
     - 可以在这一行的某一列放棋子（前提是：该格是 `#`，并且该列还没用过），然后继续搜索下一行。
3. **边界条件**
   - 如果已经放了 `k` 个棋子 → 计数 +1。
   - 如果行数到头，还没放够 `k` → 剪枝返回。
4. **复杂度**
   - n ≤ 8，搜索规模不大，DFS完全可行。

------

Python 实现

```python
def solve():
    import sys
    sys.setrecursionlimit(10000)
    input_data = sys.stdin.read().splitlines()

    idx = 0
    while True:
        n, k = map(int, input_data[idx].split())
        idx += 1
        if n == -1 and k == -1:
            break

        board = [list(input_data[idx + i]) for i in range(n)]
        idx += n

        used_col = [False] * n
        res = 0

        def dfs(row, placed):
            nonlocal res
            if placed == k:
                res += 1
                return
            if row == n:
                return

            # 选择不在这一行放棋子
            dfs(row + 1, placed)

            # 在这一行某个可行位置放棋子
            for col in range(n):
                if board[row][col] == '#' and not used_col[col]:
                    used_col[col] = True
                    dfs(row + 1, placed + 1)
                    used_col[col] = False

        dfs(0, 0)
        print(res)

solve()
```



**位运算优化版**（把列状态用一个整数 bitmask 表示）
 核心思路是：用一个整数的二进制位来表示列是否被占用。这样就不需要维护 `used_col[]` 数组，速度和内存都会更快。

------

关键优化点

- `n ≤ 8`，所以最多 8 列，可以用一个 **int 的低 8 位** 表示列状态。
  - 第 j 位 = 1 → 第 j 列已放棋子。
  - 第 j 位 = 0 → 第 j 列还可用。
- 搜索时，如果能放棋子，就把对应列位置 1，再递归。
- 回溯时直接恢复，不需要 `used_col` 数组。

------

Python 位运算版本

```python
def solve():
    import sys
    sys.setrecursionlimit(10000)
    input_data = sys.stdin.read().splitlines()
    
    idx = 0
    while True:
        n, k = map(int, input_data[idx].split())
        idx += 1
        if n == -1 and k == -1:
            break

        board = [list(input_data[idx + i]) for i in range(n)]
        idx += n
        
        res = 0
        
        def dfs(row, placed, colmask):
            nonlocal res
            if placed == k:
                res += 1
                return
            if row == n:
                return
            
            # 这一行不放棋子
            dfs(row + 1, placed, colmask)
            
            # 尝试在这一行放一个棋子
            for col in range(n):
                if board[row][col] == '#' and not (colmask & (1 << col)):
                    dfs(row + 1, placed + 1, colmask | (1 << col))
        
        dfs(0, 0, 0)
        print(res)

solve()
```





```python
# 石贤泽2300012407
def count_ways(board, n, k):
    def backtrack(row, columns):
        if len(columns) == k:
            return 1
        if row >= n:
            return 0
        count = 0
        for col in range(n):
            if board[row][col] == '#' and col not in columns :
                columns.add(col)
                count += backtrack(row + 1, columns)
                columns.remove(col)
        count += backtrack(row + 1, columns)    # 考虑不放置棋子的情况
        return count
    return backtrack(0, set())

while True:
    n, k = map(int, input().split())
    if n == -1 and k == -1:
        break
    board = [input() for j in range(n)]
    print(count_ways(board, n, k))
```



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

bfs, http://cs101.openjudge.cn/practice/01376/

洛谷上有中文题面。机器人搬重物，https://www.luogu.com.cn/problem/P1126

The Robot Moving Institute is using a robot in their local store to transport different items. Of course the robot should spend only the minimum time necessary when travelling from one place in the store to another. The robot can move only along a straight line (track). All tracks form a rectangular grid. Neighbouring tracks are one meter apart. The store is a rectangle N x M meters and it is entirely covered by this grid. The distance of the track closest to the side of the store is exactly one meter. The robot has a circular shape with diameter equal to 1.6 meter. The track goes through the center of the robot. The robot always faces north, south, west or east. The tracks are in the south-north and in the west-east directions. The robot can move only in the direction it faces. The direction in which it faces can be changed at each track crossing. Initially the robot stands at a track crossing. The obstacles in the store are formed from pieces occupying 1m x 1m on the ground. Each obstacle is within a 1 x 1 square formed by the tracks. The movement of the robot is controlled by two commands. These commands are GO and TURN. 
The GO command has one integer parameter n in {1,2,3}. After receiving this command the robot moves n meters in the direction it faces. 

The TURN command has one parameter which is either left or right. After receiving this command the robot changes its orientation by 90o in the direction indicated by the parameter. 

The execution of each command lasts one second. 

Help researchers of RMI to write a program which will determine the minimal time in which the robot can move from a given starting point to a given destination.

**输入**

The input consists of blocks of lines. The first line of each block contains two integers M <= 50 and N <= 50 separated by one space. In each of the next M lines there are N numbers one or zero separated by one space. One represents obstacles and zero represents empty squares. (The tracks are between the squares.) The block is terminated by a line containing four positive integers B1 B2 E1 E2 each followed by one space and the word indicating the orientation of the robot at the starting point. B1, B2 are the coordinates of the square in the north-west corner of which the robot is placed (starting point). E1, E2 are the coordinates of square to the north-west corner of which the robot should move (destination point). The orientation of the robot when it has reached the destination point is not prescribed. We use (row, column)-type coordinates, i.e. the coordinates of the upper left (the most north-west) square in the store are 0,0 and the lower right (the most south-east) square are M - 1, N - 1. The orientation is given by the words north or west or south or east. The last block contains only one line with N = 0 and M = 0.

**输出**

The output contains one line for each block except the last block in the input. The lines are in the order corresponding to the blocks in the input. The line contains minimal number of seconds in which the robot can reach the destination point from the starting point. If there does not exist any path from the starting point to the destination point the line will contain -1. 
![img](http://media.openjudge.cn/images/g378/1376_1.jpg)

样例输入

```
9 10
0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 0 1 0
0 0 0 1 0 0 0 0 0 0
0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 1 0 0 0 0
0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 1 0
7 2 2 7 south
0 0
```

样例输出

```
12
```

来源

Central Europe 1996



```python
from collections import deque


def bfs_min_time(grid, start, end, direction):
    N, M = len(grid), len(grid[0])
    # 定义朝向：0-东, 1-南, 2-西, 3-北
    dir_map = {'E': 0, 'S': 1, 'W': 2, 'N': 3}
    start_dir = dir_map[direction]
    sr, sc, tr, tc = start[0], start[1], end[0], end[1]

    # 机器人中心只能位于网格交点，合法交点要求其周围四个相邻的格子都不能有障碍。
    # 对于交点 (i, j) (i,j均从1开始计数，i∈[1,N-1], j∈[1,M-1])，对应的格子为
    # (i-1,j-1), (i-1,j), (i,j-1), (i,j)
    valid = [[False] * (M) for _ in range(N)]
    for i in range(1, N):
        for j in range(1, M):
            if grid[i - 1][j - 1] == 0 and grid[i - 1][j] == 0 and grid[i][j - 1] == 0 and grid[i][j] == 0:
                valid[i][j] = True

    # 检查起始点和目标点是否合法
    if not valid[sr][sc] or not valid[tr][tc]:
        return -1

    # 定义方向移动，顺序：东, 南, 西, 北
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]

    # BFS: 状态 (r, c, d)
    visited = [[[False] * 4 for _ in range(M)] for _ in range(N)]
    q = deque()
    q.append((sr, sc, start_dir, 0))
    visited[sr][sc][start_dir] = True

    while q:
        r, c, d, steps = q.popleft()
        # 判断是否到达目标位置（朝向不要求匹配）
        if r == tr and c == tc:
            return steps

        # 转向操作
        # Left: d_new = (d+3)%4, Right: d_new = (d+1)%4
        for nd in [(d + 3) % 4, (d + 1) % 4]:
            if not visited[r][c][nd]:
                visited[r][c][nd] = True
                q.append((r, c, nd, steps + 1))

        # 前进1,2,3步，每一步中间都必须合法
        for k in range(1, 4):
            nr = r + dr[d] * k
            nc = c + dc[d] * k
            # 判断越界
            if nr < 1 or nr >= N or nc < 1 or nc >= M:
                break
            # 如果当前位置不合法，则不能继续向前走
            if not valid[nr][nc]:
                break
            if not visited[nr][nc][d]:
                visited[nr][nc][d] = True
                q.append((nr, nc, d, steps + 1))
    return -1


# 读取输入数据
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    grid = [list(map(int, input().split())) for _ in range(n)]
    sx, sy, ex, ey, direction = input().split()
    sx, sy, ex, ey = map(int, [sx, sy, ex, ey])

    direction = direction.upper()  # 确保方向是大写

    # 计算最短时间
    result = bfs_min_time(grid, (sx, sy), (ex, ey), direction[0])
    print(result)
```

> 处理 **机器人转向** 的操作，它的目的是计算左转（`Left`）和右转（`Right`）后对应的新朝向。
>
> **转向的数学公式**
>
> 方向 `d` 是用 **0、1、2、3** 来表示的：
>
> - `0` → **东 (E)**
> - `1` → **南 (S)**
> - `2` → **西 (W)**
> - `3` → **北 (N)**
>
> **左转 (Left)** 和 **右转 (Right)** 的计算方式：
>
> - **左转（逆时针 90°）**：
>   - `d_new = (d - 1) % 4`
>   - 但由于 Python 取模可能会导致负数，为了保持非负数，等价于 `d_new = (d + 3) % 4`
> - **右转（顺时针 90°）**：
>   - `d_new = (d + 1) % 4`
>
> **具体转向示例：**
>
> | 当前方向 | `d` 值 | 左转 `(d+3) % 4` | 右转 `(d+1) % 4` |
> | -------- | ------ | ---------------- | ---------------- |
> | 东 (E)   | 0      | 3 (北 N)         | 1 (南 S)         |
> | 南 (S)   | 1      | 0 (东 E)         | 2 (西 W)         |
> | 西 (W)   | 2      | 1 (南 S)         | 3 (北 N)         |
> | 北 (N)   | 3      | 2 (西 W)         | 0 (东 E)         |
>
> 所以：
>
> ```python
> for nd in [(d + 3) % 4, (d + 1) % 4]:  # 遍历左转和右转的情况
> ```
>
> 这个 `for` 循环的作用是：
>
> - `nd = (d + 3) % 4` **模拟左转**
> - `nd = (d + 1) % 4` **模拟右转**
>
> 





```python
from collections import deque

# Directions: north(0), east(1), south(2), west(3)
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

def bfs(sx, sy, ex, ey, sdir):
    queue = deque([(sx, sy, 0, sdir)])
    visited = [[[0]*4 for _ in range(m+1)] for _ in range(n+1)]
    visited[sx][sy][sdir] = 1

    while queue:
        x, y, time, dir = queue.popleft()
        for i in range(1, 4):  # 1, 2, 3 steps
            nx, ny = x + dx[dir]*i, y + dy[dir]*i
            if nx < 1 or nx >= n or ny < 1 or ny >= m or grid[nx][ny] or grid[nx+1][ny] or grid[nx][ny+1] or grid[nx+1][ny+1]:
                break
            if not visited[nx][ny][dir]:
                visited[nx][ny][dir] = 1
                if nx == ex and ny == ey:
                    return time + 1
                queue.append((nx, ny, time + 1, dir))
        for i in range(4):
            if abs(dir - i) == 2:  # Don't go back
                continue
            if not visited[x][y][i]:  # Turn in place, no need to check boundaries
                visited[x][y][i] = 1
                queue.append((x, y, time + 1, i))
    return -1

while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break

    grid = [[0]*(m+2) for _ in range(n+2)]
    for i in range(1, n+1):
        grid[i] = [0] + list(map(int, input().split())) + [0]

    sx, sy, ex, ey, sdir = input().split()
    sx, sy, ex, ey = map(int, [sx, sy, ex, ey])
    sdir = {'n': 0, 'e': 1, 's': 2, 'w': 3}[sdir[0]]

    if sx == ex and sy == ey:
        print(0)
        continue

    print(bfs(sx, sy, ex, ey, sdir))
```





## 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

Given a positive integer n, write a program to find out a nonzero multiple m of n whose decimal representation contains only the digits 0 and 1. You may assume that n is not greater than 200 and there is a corresponding m containing no more than 100 decimal digits.

**输入**

The input file may contain multiple test cases. Each line contains a value of n (1 <= n <= 200). A line containing a zero terminates the input.

**输出**

For each value of n in the input print a line containing the corresponding value of m. The decimal representation of m must not contain more than 100 digits. If there are multiple solutions for a given value of n, any one of them is acceptable.

样例输入

```
2
6
19
0
```

样例输出

```
10
100100100100100100
111111111111111111
```

来源

Dhaka 2002



考虑到我们的结果至多是100位数，我们需要找到一个能被n整除的数字，并且全是0、1，实际上所有位上都是0、1的数字是可以构成一颗二叉树的，我们从1作为根节点，左右儿子分别进行\*10和*10+1操作——并且我们不关心数字实际的大小，而关心这个数mod n的结果，由于mod运算的结合、分配律，所以我们可以进行vis去重，这大大降低了搜索的成本，于是我们直接用bfs就可以了（虽然题目不要求最短这个性质，但即便如此相比之下bfs也比dfs更高效）



对余数进行BFS（广度优先搜索）。思路是从最小的满足条件的数开始搜索，即1，然后通过不断添加0或1在数的末尾来生成新的数字，直到找到一个数字既是n的倍数又只包含数字0和1。

由于直接操作数字可能会很快超出整数范围，特别是当n很大时，我们可以在BFS过程中仅保存数字的模n值以及该数字的十进制表示。每次从队列中取出一个元素，计算加0或加1后的模n值，如果新模n值为0，则找到了解；否则，如果这个模n值是首次访问，则将其加入队列继续搜索。

下面是Python代码实现：

```python
from collections import deque

def find_multiple(n):
    # 使用队列实现BFS
    q = deque()
    # 初始化队列，存储的是(模n值, 对应的数字字符串)
    q.append((1 % n, "1"))
    visited = set([1 % n])  # 用于记录访问过的模n值，避免重复搜索

    while q:
        mod, num_str = q.popleft()

        # 检查当前模n值是否为0，是则找到答案
        if mod == 0:
            return num_str

        # 尝试在当前数字后加0或加1，生成新的数字，并计算模n值
        for digit in ["0", "1"]:
            new_num_str = num_str + digit
            new_mod = (mod * 10 + int(digit)) % n

            # 如果新模n值未访问过，则加入队列继续搜索
            if new_mod not in visited:
                q.append((new_mod, new_num_str))
                visited.add(new_mod)

def main():
    while True:
        n = int(input())
        if n == 0:
            break
        print(find_multiple(n))

if __name__ == "__main__":
    main()
```

这段代码首先读取输入的n值，然后调用`find_multiple`函数来找到满足条件的最小的由0和1组成的n的倍数。`find_multiple`函数通过广度优先搜索实现，搜索过程中仅记录和处理模n值，这样可以有效避免处理过大的数字。当找到一个模n值为0的数字时，即找到了一个满足条件的倍数，函数返回该数字的字符串表示。



先把输入变成奇数，除掉的2的数量，在最后用'0'补全，然后用bfs从0开始逐位添加0或者1，查找可以整除的数

```python
# 钟明衡 物理学院
def bfs(n):
    l = [0]
    s, e = 0, 1
    while s != e:
        for i in range(s, e):
            for j in (0, 1):
                x = l[i]*10+j
                if x:
                    if x % n:
                        l.append(x)
                    else:
                        return str(x)
        s, e = e, len(l)
    return ''


while (n := int(input())):
    c = 0
    while (n+1) % 2:
        n //= 2
        c += 1
    print(bfs(n)+'0'*c)

```



思路：比较偏bfs的想法。从位数小起步，如果找不到的话，在后面加0或者1。一个简化算法的方式是，如果有mod相同的可以剔掉

```python
#2200015507 王一粟
from collections import deque
def find(n):
    if n == 1:
        return 1
    queue = deque([10,11])
    mylist = [1]
    while queue:
        element = queue.popleft()
        t = element % n
        if t == 0:
            return str(element)
        else:
            if t not in mylist:
                mylist.append(t)
                queue.append(element*10+1)
                queue.append(element*10)
while True:
    n = int(input())
    if n == 0:
        break
    else:
        print(find(n))
```



## 01577: Falling Leaves

http://cs101.openjudge.cn/practice/01577/



![img](http://media.openjudge.cn/images/g579/1577_1.jpg)
Figure 1

Figure 1 shows a graphical representation of a binary tree of letters. People familiar with binary trees can skip over the definitions of a binary tree of letters, leaves of a binary tree, and a binary search tree of letters, and go right to The problem.

A binary tree of letters may be one of two things:

1. It may be empty.
2. It may have a root node. A node has a letter as data and refers to a left and a right subtree. The left and right subtrees are also binary trees of letters.

In the graphical representation of a binary tree of letters:

1. Empty trees are omitted completely.
2. Each node is indicated by
   - Its letter data,
   - A line segment down to the left to the left subtree, if the left subtree is nonempty,
   - A line segment down to the right to the right subtree, if the right subtree is nonempty.

A leaf in a binary tree is a node whose subtrees are both empty. In the example in Figure 1, this would be the five nodes with data B, D, H, P, and Y.

The preorder traversal of a tree of letters satisfies the defining properties:

1. If the tree is empty, then the preorder traversal is empty. 
2. If the tree is not empty, then the preorder traversal consists of the following, in order
   - The data from the root node,
   - The preorder traversal of the root's left subtree,
   - The preorder traversal of the root's right subtree.

The preorder traversal of the tree in Figure 1 is KGCBDHQMPY.

A tree like the one in Figure 1 is also a binary search tree of letters. A binary search tree of letters is a binary tree of letters in which each node satisfies:

The root's data comes later in the alphabet than all the data in the nodes in the left subtree.

The root's data comes earlier in the alphabet than all the data in the nodes in the right subtree.

The problem:

Consider the following sequence of operations on a binary search tree of letters

Remove the leaves and list the data removed
Repeat this procedure until the tree is empty
Starting from the tree below on the left, we produce the sequence of trees shown, and then the empty tree 

![img](http://media.openjudge.cn/images/g579/1577_2.jpg)

by removing the leaves with data

BDHPY
CM
GQ
K

Your problem is to start with such a sequence of lines of leaves from a binary search tree of letters and output the preorder traversal of the tree.

**输入**

The input will contain one or more data sets. Each data set is a sequence of one or more lines of capital letters.

The lines contain the leaves removed from a binary search tree in the stages described above. The letters on a line will be listed in increasing alphabetical order. Data sets are separated by a line containing only an asterisk ('*').

The last data set is followed by a line containing only a dollar sign ('$'). There are no blanks or empty lines in the input.

**输出**

For each input data set, there is a unique binary search tree that would produce the sequence of leaves. The output is a line containing only the preorder traversal of that tree, with no blanks.

样例输入

```
BDHPY
CM
GQ
K
*
AC
B
$
```

样例输出

```
KGCBDHQMPY
BAC
```

来源

Mid-Central USA 2000



```python
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def build_bst(leaves):
    if not leaves:
        return None

    root = TreeNode(leaves[0])
    for leaf in leaves[1:]:
        insert_node(root, leaf)

    return root


def insert_node(root, leaf):
    if leaf < root.data:
        if root.left is None:
            root.left = TreeNode(leaf)
        else:
            insert_node(root.left, leaf)
    else:
        if root.right is None:
            root.right = TreeNode(leaf)
        else:
            insert_node(root.right, leaf)


def preorder_traversal(root):
    if root is None:
        return []
    traversal = [root.data]
    traversal.extend(preorder_traversal(root.left))
    traversal.extend(preorder_traversal(root.right))
    return traversal


# 读取输入数据
flag = 0
while True:
    leaves = []
    while True:
        line = input().strip()
        if line == '*':
            break
        elif line == '$':
            flag = 1
            break
        else:
            leaves.extend(line)

    # 构建二叉搜索树
    root = build_bst(leaves[::-1])

    # 输出前序遍历结果
    traversal_result = preorder_traversal(root)
    print(''.join(traversal_result))
    
    if flag:
        break
```





## 01611: The Suspects

http://cs101.openjudge.cn/practice/01611/

Severe acute respiratory syndrome (SARS), an atypical pneumonia of unknown aetiology, was recognized as a global threat in mid-March 2003. To minimize transmission to others, the best strategy is to separate the suspects from others.
In the Not-Spreading-Your-Sickness University (NSYSU), there are many student groups. Students in the same group intercommunicate with each other frequently, and a student may join several groups. To prevent the possible transmissions of SARS, the NSYSU collects the member lists of all student groups, and makes the following rule in their standard operation procedure (SOP).
Once a member in a group is a suspect, all members in the group are suspects.
However, they find that it is not easy to identify all the suspects when a student is recognized as a suspect. Your job is to write a program which finds all the suspects.

**输入**

The input file contains several cases. Each test case begins with two integers n and m in a line, where n is the number of students, and m is the number of groups. You may assume that 0 < n <= 30000 and 0 <= m <= 500. Every student is numbered by a unique integer between 0 and n−1, and initially student 0 is recognized as a suspect in all the cases. This line is followed by m member lists of the groups, one line per group. Each line begins with an integer k by itself representing the number of members in the group. Following the number of members, there are k integers representing the students in this group. All the integers in a line are separated by at least one space.
A case with n = 0 and m = 0 indicates the end of the input, and need not be processed.

**输出**

For each case, output the number of suspects in one line.

样例输入

```
100 4
2 1 2
5 10 13 11 12 14
2 0 1
2 99 2
200 2
1 5
5 1 2 3 4 5
1 0
0 0
```

样例输出

```
4
1
1
```

来源

Asia Kaohsiung 2003



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



```python
#2200015507 王一粟
def find(x):
    if parent[x] == x:
        return parent[x]
    else:
        parent[x] = find(parent[x])
        return parent[x]
def disjoint(x,y):
    rep_x,rep_y = find(x),find(y)
    if rep_x != rep_y:
        if rank[rep_x] < rank[rep_y]:
            parent[rep_x] = rep_y
        elif rank[rep_x] > rank[rep_y]:
            parent[rep_y] = rep_x
        else:
            parent[rep_y] = rep_x
            rank[rep_x] += 1
def joint(mylist):
    node = mylist[0]
    for element in mylist[1:]:
        disjoint(node,element)
while True:
    n,m = [int(i) for i in input().split()]
    if n == 0 and m == 0:
        break
    parent = [i for i in range(n)]
    rank = [0 for i in range(n)]
    for i in range(m):
        s = [int(i) for i in input().split()]
        joint(s[1:])
    rep_0 = find(0)
    print(len([i for i in parent if find(i) == rep_0]))
```



## 01703: 发现它，抓住它

http://cs101.openjudge.cn/practice/01703/

一个城市中有两个犯罪团伙A和B，你需要帮助警察判断任意两起案件是否是同一个犯罪团伙所为，警察所获得的信息是有限的。假设现在有N起案件（N<=100000），编号为1到N，每起案件由团伙A或团伙B所为。你将按时间顺序获得M条信息（M<=100000），这些信息分为两类：

1. D [a] [b]

其中[a]和[b]表示两起案件的编号，这条信息表明它们属于不同的团伙所为

2. A [a] [b]

其中[a]和[b]表示两起案件的编号，这条信息需要你回答[a]和[b]是否是同一个团伙所为

注意你获得信息的时间是有先后顺序的，在回答的时候只能根据已经接收到的信息做出判断。



**输入**

第一行是测试数据的数量T（1<=T<=20）。
每组测试数据的第一行包括两个数N和M，分别表示案件的数量和信息的数量，其后M行表示按时间顺序收到的M条信息。

**输出**

对于每条需要回答的信息，你需要输出一行答案。如果是同一个团伙所为，回答"In the same gang."，如果不是，回答"In different gangs."，如果不确定，回答”Not sure yet."。

样例输入

```
1
5 5
A 1 2
D 1 2
A 1 2
D 2 4
A 1 4
```

样例输出

```
Not sure yet.
In different gangs.
In the same gang.
```



这个问题可以通过并查集（Union-Find）数据结构来有效解决。并查集是一种非常适合处理集合合并以及查询两个元素是否在同一个集合中的数据结构。

对于这个问题，我们需要稍微扩展并查集的基本操作以适应犯罪团伙的判断。由于信息中只提到两个案件是否属于不同的团伙，我们可以通过将每个案件关联到两个不同的代表元素来表示这种关系：一个代表与它在同一个团伙的案件的代表元素，另一个代表与它在不同团伙的案件的代表元素。

具体来说，对于每个案件，我们实际上在并查集中维护两个节点：一个代表案件本身，另一个代表其对立案件。当我们得知两起案件属于不同的团伙时，我们将一起案件的案件节点与另一起案件的对立案件节点合并，反之亦然。

实现步骤如下：

1. 初始化并查集，对于N起案件，我们需要2N个节点，因为每起案件有两个节点（本身和对立案件）。

2. 对于每条信息：
   - 如果是"D [a] [b]"类型，表示案件a和案件b属于不同团伙，我们将案件a与案件b的对立案件合并，案件b与案件a的对立案件合并。
   - 如果是"A [a] [b]"类型，我们需要判断案件a和案件b是否属于同一个团伙：
     - 如果案件a和案件b属于同一个集合，或者案件a的对立案件和案件b的对立案件属于同一个集合，输出"In the same gang."。
     - 如果案件a和案件b的对立案件属于同一个集合，或者案件b和案件a的对立案件属于同一个集合，输出"In different gangs."。
     - 否则，输出"Not sure yet."。

以下是代码实现的核心部分：

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def solve():
    n, m = map(int, input().split())
    uf = UnionFind(2 * n)  # 初始化并查集，每个案件对应两个节点
    for _ in range(m):
        operation, a, b = input().split()
        a, b = int(a) - 1, int(b) - 1
        if operation == "D":
            uf.union(a, b + n)  # a与b的对立案件合并
            uf.union(a + n, b)  # a的对立案件与b合并
        else:  # "A"
            if uf.find(a) == uf.find(b) or uf.find(a + n) == uf.find(b + n):
                print("In the same gang.")
            elif uf.find(a) == uf.find(b + n) or uf.find(a + n) == uf.find(b):
                print("In different gangs.")
            else:
                print("Not sure yet.")

T = int(input())
for _ in range(T):
    solve()
```

这段代码首先读取测试用例数量 `T`，然后对每个测试用例执行 `solve` 函数，该函数首先创建一个并查集实例，然后根据输入的操作和案件编号更新并查集，并根据并查集的当前状态回答查询。



## 01724: ROADS

Dijkstra, dfs with pruning, http://cs101.openjudge.cn/practice/01724/

同 07735: 道路，http://cs101.openjudge.cn/practice/07735/

N cities named with numbers 1 ... N are connected with one-way roads. Each road has two parameters associated with it : the road length and the toll that needs to be paid for the road (expressed in the number of coins). 
Bob and Alice used to live in the city 1. After noticing that Alice was cheating in the card game they liked to play, Bob broke up with her and decided to move away - to the city N. He wants to get there as quickly as possible, but he is short on cash. 

We want to help Bob to find **the shortest path** from the city 1 to the city N **that he can afford**with the amount of money he has. 

**输入**

The first line of the input contains the integer K, 0 <= K <= 10000, maximum number of coins that Bob can spend on his way. 
The second line contains the integer N, 2 <= N <= 100, the total number of cities. 

The third line contains the integer R, 1 <= R <= 10000, the total number of roads. 

Each of the following R lines describes one road by specifying integers S, D, L and T separated by single blank characters : 

- S is the source city, 1 <= S <= N 
- D is the destination city, 1 <= D <= N 
- L is the road length, 1 <= L <= 100 
- T is the toll (expressed in the number of coins), 0 <= T <=100

Notice that different roads may have the same source and destination cities. 

**输出**

The first and the only line of the output should contain the total length of the shortest path from the city 1 to the city N whose total toll is less than or equal K coins. 
If such path does not exist, only number -1 should be written to the output. 

样例输入

```
5
6
7
1 2 2 3
2 4 3 3
3 4 2 4
1 3 4 1
4 6 2 1
3 5 2 0
5 4 3 2
```

样例输出

```
11
```

来源

CEOI 1998



```python
#何秉儒 物理学院
import heapq

def dijkstra(g):
    while pq:
        dist,node,fee = heapq.heappop(pq)
        if node == n-1 :
            return dist
        for nei,w,f in g[node]:
            n_dist = dist + w
            n_fee = fee + f
            if n_fee <= k:
                dists[nei] = n_dist
                heapq.heappush(pq,(n_dist,nei,n_fee))
    return -1

k,n,r = int(input()),int(input()),int(input())
g = [[] for _ in range(n)]
for i in range(r):
    s,d,l,t = map(int,input().split())
    g[s-1].append((d-1,l,t)) #node,dist,fee

pq = [(0,0,0)] #dist,node,fee
dists = [float('inf')] * n
dists[0] = 0
spend = 0

result = dijkstra(g)
print(result)
```



```python
import heapq
from collections import defaultdict

MAX_COINS = int(input())  # 最大金币数
CITY_COUNT = int(input())  # 城市数目
ROAD_COUNT = int(input())

# 存储道路信息的字典，使用 defaultdict 初始化
roads = defaultdict(list)

for _ in range(ROAD_COUNT):
    start, end, length, money = map(int, input().split())
    start, end = start - 1, end - 1
    roads[start].append((end, length, money))


def bfs(start, end, max_coins):
    queue = [(0, max_coins, start)]  # (距离, 剩余金币, 当前城市)
    visited = set()

    while queue:
        distance, coins, city = heapq.heappop(queue)

        if city == end:
            return distance

        visited.add((city, coins))

        for next_city, road_length, road_money in roads[city]:
            if coins >= road_money:
                new_distance = distance + road_length
                if (next_city, coins - road_money) not in visited:
                    heapq.heappush(queue, (new_distance, coins - road_money, next_city))

    return -1


print(bfs(0, CITY_COUNT - 1, MAX_COINS))

```



Dijkstra

```python
#23n23000111119(武)
from heapq import heappop, heappush
from collections import defaultdict

K, N, R = int(input()), int(input()), int(input())
graph = defaultdict(list)
for i in range(R):
    S, D, L, T = map(int, input().split())
    graph[S].append((D, L, T))


def Dijkstra(graph):
    global K, N, R
    q, ans = [], []
    heappush(q, (0, 0, 1, 0))  # (length,cost,cur,step)
    while q:
        l, cost, cur, step = heappop(q)
        if cur == N: return l
        for next, nl, nc in graph[cur]:
            if cost + nc <= K and step + 1 < N:
                heappush(q, (l + nl, cost + nc, next, step + 1))
    return -1


print(Dijkstra(graph))

```



visited记录上次访问时所花费的money，由于我们利用优先队列实现了短的路径优先弹出，如果再次遇到了之前访问过的点，那路径一定更长，因此只有在花费的money更少并且不超出K的情况下才考虑入队，循环操作直至第一次遇到目的地

```python
# 王昊 光华管理学院
import heapq


K, N, R = map(int, [input() for _ in range(3)])
graph = {i: [] for i in range(1, N+1)}
visited = {i: float('inf') for i in range(1, N+1)}
for _ in range(R):
    S, D, L, T = map(int, input().split())
    graph[S].append((D, L, T))

queue, ans = [(0, 0, 1)], -1
while queue:
    l, t, s = heapq.heappop(queue)
    if s == N:
        ans = l
        break
    visited[s] = t
    for d, z, w in graph[s]:
        if t+w < visited[d] and t+w <= K:
            heapq.heappush(queue, (l+z, t+w, d))
print(ans)
```





```python
# 23n2310307206
import heapq
class edge:
    def __init__(self,start,end,length,money):
        self.start = start
        self.end = end
        self.money = money
        self.length = length
k = int(input())
n = int(input())
r = int(input())
graph = {i:[] for i in range(1,n+1)}
for i in range(r):
    s,d,l,t = map(int,input().split())
    graph[s].append(edge(s,d,l,t))
def dijskra():
    visited=[0]*(n+1)
    ans=-1
    priorQueue=[]
    heapq.heappush(priorQueue,(0,0,1))#length,money,pos
    while priorQueue:
        length,money,pos = heapq.heappop(priorQueue)
        visited[pos] = 1
        if pos == n and money<=k:
            ans=length
            break
        if money > k:

            continue
        for road in graph[pos]:
            pos1 = road.end
            m1 = road.money+money
            l1 = road.length+length
            if m1<=k and visited[pos1] != 1:
                heapq.heappush(priorQueue,(l1,m1,pos1))
        visited[pos] = 0

    print(ans)
dijskra()
```



Dijkstra

```python
# 23n2000012515(heol)
from heapq import heappush as hu, heappop as hp

k, n, r = [int(input()) for _ in range(3)]
edge, vis = [[] for _ in range(n + 1)], [100000] * (n + 1)

for _ in range(r):
    x, y, z, w = map(int, input().split())
    edge[x].append((y, z, w))

q, ans = [], -1
hu(q, (0, 0, 1))

while q:
    l, c, x = hp(q)
    if x == n:
        ans = l
        break
    vis[x] = c
    for y, z, w in edge[x]:
        if c + w < vis[y] and c + w <= k:
            hu(q, (l + z, c + w, y))

print(ans)

```



从城市 1开始深度优先遍历整个图，找到所有能过到达 N 的走法，选一个最优的。《算法基础与在线实践》有讲到。

1）可行性剪枝：

提前预判出一条路走下去不可能走到终点，于是就不走了。在本题中，可行性剪枝十分直观，即发现如果往前走一步到达城市i，花掉的路费就会超过总钱数K那么就不去城市i。

2）最优性剪枝：

在用深搜的方法寻找最优路径（代价最小路径）时，“最优性剪枝”是最常用的方法。思想是，如果提前预判出一条路走下去即便能到终点也不可能是最优的，那就不走了。具体实现的办法是，记下已经找到的起点到终点的目前最优路径的代价C，那么在后续搜索其他路径的过程中，如果走到某个结点k时，发现从起点走到k所花的代价已经大于或等于C，那么就应该立即回退，而不是从k出发继续往前走——因为即便再走下去能够到达终点，所花的代价也一定大于或等于C，相当于徒劳地走。

3）处处最优剪枝：

有时仅仅用上述的可行性剪枝和最优性剪枝还是不够高效。还有一种更强的最优性剪枝方案，是记下从起点到每一个结点V的当前最优路径的代价C。下次探索新路径走到V时如果发现所花的代价已经大于或等于C，则这条新路径就没必要从V再走下去了，应立即回退。这又是用空间换时间的技巧。不妨称这种最优性剪枝为“处处最优剪枝”。

在本题中，仅用可行性剪枝和简单的最优性剪枝是不够的，需要用“处处最优剪枝”。本题中路径的“代价”，实际上就是路径的长度。最优路径就是长度最短的路径。但是由于还有过路费的问题，所以不能直接使用“处处最优剪枝”。因为，同是从起点到达结点V的两条路，长的路不一定就比短的路差。有可能长的那条路花费少，往下走可以走到终点，而短的那条路花费太多，往下走走不到终点。
将过路费考虑进去，可以用 min_lengths\[i][j]记录，表示走到城市i时，在已花掉的过路费为j的条件下，从城市1到i的最优路径的长度。若在后续的搜索中再次走到i时，已花掉的过路费恰好为 j，且此时的路径长度已经超过 min_lengths\[i][j]，则不必再走下去了。



```python
class Road:
    def __init__(self,d,L,t):
       self.d,self.L,self.t = d,L,t


def dfs(s, total_cost, total_length, visited, city_map, min_lengths, k):
    global min_length
    if s == n:
        min_length = min(min_length, total_length)
        return
    for i in range(len(city_map[s])):
        d, L, t = city_map[s][i].d, city_map[s][i].L, city_map[s][i].t
        if visited[d]:
            continue
        cost = t + total_cost
        length = L + total_length
        if cost > k :   # 可行性剪枝：超过预算
            continue
        if (length >= min_length or # 最优性剪枝：超过当前最优解
                length >= min_lengths[d][cost]): # 处处最优性剪枝：超过已经搜索到的最优解
            continue
        min_lengths[d][cost] = length
        visited[d] = True
        dfs(d, cost, length, visited, city_map, min_lengths, k)
        visited[d] = False


k,n,r = int(input()),int(input()),int(input())
city_map = [[] for i in range(n+1)] #邻接表。city_map[i]是从点i有路连到的城市集合
for _ in range(r):
    r = Road(0, 0, 0)
    s, r.d, r.L, r.t = map(int, input().split())
    if s != r.d:
        city_map[s].append(r)
INF = float('inf')
min_length = INF

#min_lengths[i][j]表示从1到i点，花销为j的最短路径的长度
min_lengths = [[INF] * (k + 1) for _ in range(n + 1)]
visited = [False] * (n + 1)
visited[1] = True
dfs(1, 0, 0, visited, city_map, min_lengths, k)
if min_length < INF:
    print(min_length)
else:
    print(-1)


```





## 01760: Disk Tree

前缀树（Trie），http://cs101.openjudge.cn/practice/01760/

Hacker Bill has accidentally lost all the information from his workstation's hard drive and he has no backup copies of its contents. He does not regret for the loss of the files themselves, but for the very nice and convenient directory structure that he had created and cherished during years of work. Fortunately, Bill has several copies of directory listings from his hard drive. Using those listings he was able to recover full paths (like "WINNT\SYSTEM32\CERTSRV\CERTCO~1\X86") for some directories. He put all of them in a file by writing each path he has found on a separate line. Your task is to write a program that will help Bill to restore his state of the art directory structure by providing nicely formatted directory tree.

**输入**

The first line of the input file contains single integer number N (1 <= N <= 500) that denotes a total number of distinct directory paths. Then N lines with directory paths follow. Each directory path occupies a single line and does not contain any spaces, including leading or trailing ones. No path exceeds 80 characters. Each path is listed once and consists of a number of directory names separated by a back slash `\`. 

Each directory name consists of 1 to 8 uppercase letters, numbers, or the special characters from the following list: exclamation mark, number sign, dollar sign, percent sign, ampersand, apostrophe, opening and closing parenthesis, hyphen sign, commercial at, circumflex accent, underscore, grave accent, opening and closing curly bracket, and tilde ("!#$%&'()-@^_`{}~").

**输出**

Write to the output file the formatted directory tree. Each directory name shall be listed on its own line preceded by a number of spaces that indicate its depth in the directory hierarchy. The subdirectories shall be listed in lexicographic order immediately after their parent directories preceded by one more space than their parent directory. Top level directories shall have no spaces printed before their names and shall be listed in lexicographic order. See sample below for clarification of the output format.

样例输入

```
7
WINNT\SYSTEM32\CONFIG
GAMES
WINNT\DRIVERS
HOME
WIN\SOFT
GAMES\DRIVERS
WINNT\SYSTEM32\CERTSRV\CERTCO~1\X86
```

样例输出

```
GAMES
 DRIVERS
HOME
WIN
 SOFT
WINNT
 DRIVERS
 SYSTEM32
  CERTSRV
   CERTCO~1
    X86
  CONFIG
```

来源

Northeastern Europe 2000





这道题本质上是 **前缀树（Trie）** 的应用场景！我们可以使用 **Trie 结构** 来存储和组织目录路径，并通过 **递归遍历** 来打印出格式化的目录树。  

为什么适合使用前缀树（Trie）？

1. 前缀共享：不同路径可能有公共前缀，使用 Trie 可以高效存储和查询这些前缀，而无需重复存储相同的部分。
2. 层次结构：Trie 本身是一个 **树结构**，天然适合表示 **文件系统** 这种 **层次化目录结构**。
3. 字典序排序：Trie 的子节点本质上是 **字典**，可以 按键排序，方便按照要求 字典序输出。

```python
from collections import defaultdict
import sys

class TrieNode:
    """Trie 结点类"""
    def __init__(self):
        self.children = defaultdict(TrieNode)  # 存储子目录
        self.is_end = False  # 该标志在本题中可省略

class Trie:
    """Trie 前缀树"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, path: str):
        """插入目录路径"""
        node = self.root
        for folder in path.split("\\"):  # 以 "\" 分割路径
            node = node.children[folder]  # 如果不存在则自动创建

    def print_tree(self, node=None, depth=0):
        """递归打印目录树"""
        if node is None:
            node = self.root
        for folder in sorted(node.children):  # 按字典序排序
            print(" " * depth + folder)  # 根据深度打印
            self.print_tree(node.children[folder], depth + 1)  # 递归打印子目录

def main():
    # 读取输入
    n = int(sys.stdin.readline().strip())
    trie = Trie()

    for _ in range(n):
        path = sys.stdin.readline().strip()
        trie.insert(path)

    # 输出目录树
    trie.print_tree()

if __name__ == "__main__":
    main()
```

> `Trie` 结构
>
> - `insert(path)`：将路径拆分成目录层级，并插入 Trie。
> - `print_tree(node, depth)`：递归遍历 Trie，并 **按字典序** 打印，使用 **深度控制缩进**。
>
> 时间复杂 度分析
>
> - 插入路径：每个路径最多 80 个字符，总体复杂度 **O(N * M)**（`M` 为路径最大深度）。
> - 打印 Trie：遍历整个树，O(N log N)（因排序）。
> - 整体复杂度：`O(N log N)`，适用于 `N ≤ 500`。





```python
# 23n2300011031
class Node:
    def __init__(self):
        self.children={}
class Trie:
    def __init__(self):
        self.root=Node()
    def insert(self,w):
        cur=self.root
        for u in w.split('\\'):
            if u not in cur.children:
               cur.children[u]=Node()
            cur=cur.children[u]
    def dfs(self,a,layer):
        for c in sorted(a.children):
            print(' '*layer+c)
            self.dfs(a.children[c], layer+1)
s=Trie()
for _ in range(int(input())):
    x=input()
    s.insert(x)
s.dfs(s.root, 0)
```



```python
# 23n2300011072(X)
class Node:
    def __init__(self,name):
        self.name=name
        self.children={}
    def insert(self,path):
        if len(path)==0:
            return
        head,*tail=path
        if head not in self.children:
            self.children[head]=Node(head)
        self.children[head].insert(tail)
    def print_tree(self,depth=0):
        for name in sorted(self.children.keys()):
            print(' '*depth+name)
            self.children[name].print_tree(depth+1)
def build_tree(paths):
    root=Node('')
    for path in paths:
        path=path.split('\\')
        root.insert(path)
    return root
paths=[input() for _ in range(int(input()))]
tree=build_tree(paths)
tree.print_tree()
```



```python
#23n2300017735(夏天明BrightSummer)
def printDir(d, h):
    if not d:
        return
    else:
        for sub in sorted(d.keys()):
            print(' '*h + sub)
            printDir(d[sub], h+1)

n = int(input())
computer = {}
for o in range(n):
    path = input().split('\\')
    curr = computer
    for p in path:
        if p not in curr:
            curr[p] = {}
        curr = curr[p]
printDir(computer, 0)
```





## 01789: Truck History

http://cs101.openjudge.cn/dsapre/01789/

Advanced Cargo Movement, Ltd. uses trucks of different types. Some trucks are used for vegetable delivery, other for furniture, or for bricks. The company has its own code describing each type of a truck. The code is simply a string of exactly seven lowercase letters (each letter on each position has a very special meaning but that is unimportant for this task). At the beginning of company's history, just a single truck type was used but later other types were derived from it, then from the new types another types were derived, and so on. 

Today, ACM is rich enough to pay historians to study its history. One thing historians tried to find out is so called derivation plan -- i.e. how the truck types were derived. They defined the distance of truck types as the number of positions with different letters in truck type codes. They also assumed that each truck type was derived from exactly one other truck type (except for the first truck type which was not derived from any other type). The quality of a derivation plan was then defined as 

$ \frac{1}{\sum_{{t_o,t_d}} d({t_o,t_d})} $

where the sum goes over all pairs of types in the derivation plan such that to is the original type and td the type derived from it and d(to,td) is the distance of the types. 
Since historians failed, you are to write a program to help them. Given the codes of truck types, your program should find the highest possible quality of a derivation plan. 

**输入**

The input consists of several test cases. Each test case begins with a line containing the number of truck types, N, 2 <= N <= 2 000. Each of the following N lines of input contains one truck type code (a string of seven lowercase letters). You may assume that the codes uniquely describe the trucks, i.e., no two of these N lines are the same. The input is terminated with zero at the place of number of truck types. 

**输出**

For each test case, your program should output the text "The highest possible quality is 1/Q.", where 1/Q is the quality of the best derivation plan. 

样例输入

```
4
aaaaaaa
baaaaaa
abaaaaa
aabaaaa
0
```

样例输出

```
The highest possible quality is 1/3.
```

来源

CTU Open 2003



```python
import heapq

def truck_history():
    while True:
        n = int(input())
        if n == 0:
            break

        trucks = [input() for _ in range(n)]
        trucks.sort()

        graph = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                graph[i][j] = graph[j][i] = sum(a!=b for a, b in zip(trucks[i], trucks[j]))

        visited = [False]*n
        min_edge = [float('inf')]*n
        min_edge[0] = 0
        total_distance = 0

        min_heap = [(0, 0)]
        while min_heap:
            d, v = heapq.heappop(min_heap)
            if visited[v]:
                continue
            visited[v] = True
            total_distance += d
            for u in range(n):
                if not visited[u] and graph[v][u] < min_edge[u]:
                    min_edge[u] = graph[v][u]
                    heapq.heappush(min_heap, (graph[v][u], u))

        print(f"The highest possible quality is 1/{total_distance}.")

truck_history()
```



```python
"""
https://www.cnblogs.com/chujian123/p/3375210.html
题意大概是这样的：
用一个7位的string代表一个编号，两个编号之间的distance代表这两个编号之间不同字母的个数。
一个编号只能由另一个编号“衍生”出来，代价是这两个编号之间相应的distance，现在要找出一个“衍生”方案，
使得总代价最小，也就是distance之和最小。

题解：问题可以转化为最小代价生成树的问题。因为每两个结点之间都有路径，所以是完全图。 此题的关键是将问题转化为最小生成树的问题。每一个编号为图的一个顶点，顶点与顶点间的编号差即为这条边的权值，题目所要的就是求出最小生成树来。用prim算法来求最小生成树。
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





## 01860: Currency Exchange

Bellman-Ford, http://cs101.openjudge.cn/practice/01860/

我们城市中有多个货币兑换点。假设每个兑换点专门处理两种特定的货币，并且仅在这两种货币之间进行兑换操作。相同的货币对可能会有多个兑换点同时处理。每个兑换点都有自己的汇率，其中 A 到 B 的汇率表示的是用 1 单位 A 可以兑换到多少单位 B。此外，每个兑换点都收取一定的手续费，该费用是你在进行兑换时需要支付的金额，且手续费总是以原始货币（即兑换前的货币）来计算。

例如，如果你想在某个兑换点将 100 美元兑换成俄罗斯卢布，该兑换点的汇率为 29.75，手续费为 0.39，那么你最终会得到 (100 - 0.39) * 29.75 = 2963.3975 卢布。

当然你知道，在我们城市中你可以使用的货币共有 N 种。让我们将每种货币编号为从 1 到 N 的唯一整数。那么每一个兑换点可以用以下六个数字来描述：整数 A 和 B 表示它兑换的两种货币；实数 RAB、CAB、RBA 和 CBA 分别表示从 A 兑换到 B 以及从 B 兑换到 A 的汇率和手续费。

Nick 手中目前拥有某种货币 S 的一定数量的资金。他想知道是否可以通过一系列兑换操作使得自己的资金增加。当然，最后他希望手中仍然持有货币 S。请帮助他回答这个难题。在整个兑换过程中，Nick 必须始终保持其账户中的资金为非负值。

> Several currency exchange points are working in our city. Let us suppose that each point specializes in two particular currencies and performs exchange operations only with these currencies. There can be several points specializing in the same pair of currencies. Each point has its own exchange rates, exchange rate of A to B is the quantity of B you get for 1A. Also each exchange point has some commission, the sum you have to pay for your exchange operation. Commission is always collected in source currency.
> For example, if you want to exchange 100 US Dollars into Russian Rubles at the exchange point, where the exchange rate is 29.75, and the commission is 0.39 you will get (100 - 0.39) * 29.75 = 2963.3975RUR. 
> You surely know that there are N different currencies you can deal with in our city. Let us assign unique integer number from 1 to N to each currency. Then each exchange point can be described with 6 numbers: integer A and B - numbers of currencies it exchanges, and real RAB, CAB, RBA and CBA - exchange rates and commissions when exchanging A to B and B to A respectively.
> Nick has some money in currency S and wonders if he can somehow, after some exchange operations, increase his capital. Of course, he wants to have his money in currency S in the end. Help him to answer this difficult question. Nick must always have non-negative sum of money while making his operations.

**输入**

输入的第一行包含四个数字：N — 货币种类的数量，M — 兑换点的数量，S — Nick 当前拥有的货币种类编号，V — 他拥有的该货币的数量。接下来的 M 行，每行给出一个兑换点的信息，格式如上所述。数据之间由一个或多个空格分隔。1 ≤ S ≤ N ≤ 100，1 ≤ M ≤ 100，V 是一个实数，范围是 0 ≤ V ≤ 10³。每个兑换点的汇率和手续费都是实数，保留两位小数，其中汇率范围为 10⁻² ≤ 汇率 ≤ 10²，手续费范围为 0 ≤ 手续费 ≤ 10²。

我们将一组兑换操作称为“简单”的，如果这组操作中没有一个兑换点被使用超过一次。你可以假设对于任何一组“简单”的兑换操作，其最终金额与初始金额之比小于 10⁴。

> The first line of the input contains four numbers: N - the number of currencies, M - the number of exchange points, S - the number of currency Nick has and V - the quantity of currency units he has. The following M lines contain 6 numbers each - the description of the corresponding exchange point - in specified above order. Numbers are separated by one or more spaces. 1<=S<=N<=100, 1<=M<=100, V is real number, 0<=V<=10^3.
> For each point exchange rates and commissions are real, given with at most two digits after the decimal point, 10-2<=rate<=102, 0<=commission<=10^2.
> Let us call some sequence of the exchange operations simple if no exchange point is used more than once in this sequence. You may assume that ratio of the numeric values of the sums at the end and at the beginning of any simple sequence of the exchange operations will be less than 10^4.

**输出**

如果 Nick 能通过某些兑换操作增加他的财富，请输出 YES，否则输出 NO。

> If Nick can increase his wealth, output YES, in other case output NO to the output file.



样例输入

```
3 2 1 20.0
1 2 1.00 1.00 1.00 1.00
2 3 1.10 1.00 1.10 1.00
```

样例输出

```
YES
```

来源

Northeastern Europe 2001, Northern Subregion



要判断 Nick 是否能通过一系列兑换操作把初始货币 S 的数量从 V 增加到更大，可以把每种货币看作图中的一个节点，把每个兑换点的两个方向兑换看作两条有向边，然后在这张图上做「最大可达金额」的松弛（relax），等价于在带「非线性」边权下做一次 Bellman–Ford 检测正环。

------

**算法思路**

1. **节点**：货币种类编号 1…N。

2. **边**：每个兑换点 i（描述为 A,B,RAB,CAB,RBA,CBA）对应两条有向边：

   - 从 A 到 B，如果当前在 A 手上有 x 单位，可以兑换到

     $x′=(x−CAB)×RAB$

     但仅当 x>CAB 时才可能兑换，不然兑换后金额为负。

   - 从 B 到 A，同理：

     $x′=(x−CBA)×RBA$

3. **状态**：用数组 best[1..N] 记录在每个货币上能够达到的“最大金额”。初始化：

   $best[S]=V,best[i≠S]=0$.

4. **松弛操作**：对每条有向边 (u→v) 重复下面操作：

   ```text
   if best[u] > fee(u→v)  then
       best[v] = max(best[v], (best[u] - fee(u→v)) * rate(u→v))
   ```

5. **检测“增益回路”**：

   - 纯粹为了把“最终回到 S 的金额 > V”这一目标化为「检测可达正权环」：
     - 在做了 N−1 次松弛之后，若还能在第 N 次松弛中让任意节点的 best 值发生增大，就说明图中存在能够无限增大的“套利回路”；
     - 或者在任何一次松弛中，best[S] 超过了初始值 V，就可以立即判定为 “YES”。

6. **复杂度**：节点数 N≤100，边数 2M≤200。Bellman–Ford 最坏 $O(N ⁣× ⁣M)=10^4$ 级别，完全可以接受。

```python
import sys

def main():
    # 读取第一行：N, M, S, V
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = int(data[2])       # 起始货币编号
    V = float(data[3])     # 起始金额

    # 解析后续每个兑换点的信息
    edges = []
    idx = 4
    for _ in range(M):
        A = int(data[idx]);     B = int(data[idx+1])
        R_ab = float(data[idx+2]);  C_ab = float(data[idx+3])
        R_ba = float(data[idx+4]);  C_ba = float(data[idx+5])
        idx += 6

        # 从 A->B 的边
        edges.append((A, B, R_ab, C_ab))
        # 从 B->A 的边
        edges.append((B, A, R_ba, C_ba))

    # best[i] 表示最终能在货币 i 上得到的最大金额
    best = [0.0] * (N + 1)
    best[S] = V

    # Bellman–Ford 核心：最多做 N 轮松弛
    for iteration in range(1, N + 1):
        updated = False
        for u, v, rate, fee in edges:
            if best[u] > fee:
                x = (best[u] - fee) * rate
                if x > best[v] + 1e-12:  # 加一点 eps 防止浮点误差
                    best[v] = x
                    updated = True

                    # 如果再次更新回了起始货币 S，且金额大于初始值，立刻输出 YES
                    if v == S and best[S] > V:
                        print("YES")
                        return

        # 如果第 N 轮仍有更新，说明存在正增益回路
        if iteration == N and updated:
            print("YES")
            return

        # 没有任何更新，可提前结束
        if not updated:
            break

    # 没发现任何增益方案
    print("NO")


if __name__ == "__main__":
    main()

```

------

用这段代码，可以在 $O(NM)$ 时间内判断是否存在一个“简单序列”（至多用每个兑换点一次）能让 Nick 最终的 S 货币金额超过他最初的 V。





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







## 01941: The Sierpinski Fractal

http://cs101.openjudge.cn/practice/01941/

Consider a regular triangular area, divide it into four equal triangles of half height and remove the one in the middle. Apply the same operation recursively to each of the three remaining triangles. If we repeated this procedure infinite times, we'd obtain something with an area of zero. The fractal that evolves this way is called the Sierpinski Triangle. Although its topological dimension is 2, its Hausdorff-Besicovitch dimension is log(3)/log(2)~1.58, a fractional value (that's why it is called a fractal). By the way, the Hausdorff-Besicovitch dimension of the Norwegian coast is approximately 1.52, its topological dimension being 1. 

For this problem, you are to outline the Sierpinski Triangle up to a certain recursion depth, using just ASCII characters. Since the drawing resolution is thus fixed, you'll need to grow the picture appropriately. Draw the smallest triangle (that is not divided any further) with two slashes, to backslashes and two underscores like this: 

```
 /\
/__\
```

To see how to draw larger triangles, take a look at the sample output.

输入

The input contains several testcases. Each is specified by an integer n. Input is terminated by n=0. Otherwise 1<=n<=10 indicates the recursion depth.

输出

For each test case draw an outline of the Sierpinski Triangle with a side's total length of 2ncharacters. Align your output to the left, that is, print the bottom leftmost slash into the first column. The output must not contain any trailing blanks. Print an empty line after each test case.

样例输入

```
3
2
1
0
```

样例输出

```
       /\
      /__\
     /\  /\
    /__\/__\
   /\      /\
  /__\    /__\
 /\  /\  /\  /\
/__\/__\/__\/__\

   /\
  /__\
 /\  /\
/__\/__\

 /\
/__\
```

来源

Ulm Local 2002



Sierpinski三角形是一种著名的分形图案，它是通过不断地在等边三角形中剔除中心的小三角形而形成的。  

函数f(n)是一个递归函数，用于生成深度为n的Sierpinski三角形。当n为1时，它返回一个包含两个字符串的列表，这两个字符串组成了一个最小的Sierpinski三角形。当n大于1时，它首先调用自身生成深度为n-1的Sierpinski三角形，然后在这个基础上构造深度为n的Sierpinski三角形。  

构造过程分为两步：首先，对于深度为n-1的Sierpinski三角形的每一行，它在左右两侧添加了x个空格，其中x等于2的n-1次方；然后，它将深度为n-1的Sierpinski三角形复制一份，并将两份拼接在一起，形成深度为n的Sierpinski三角形的下半部分。这两步操作完成了深度为n的Sierpinski三角形的构造。  

```python
def f(n):
    if n == 1:
        return [' /\\ ', '/__\\']
    t = f(n - 1)
    x = 2 ** (n - 1)
    res = [' ' * x + u + ' ' * x for u in t]
    res.extend([u + u for u in t])
    return res


al = [f(i) for i in range(1, 11)]
while True:
    n = int(input())
    if n == 0:
        break
    for u in al[n - 1]:
        print(u)
    print()
```



## 01944: Fiber Communications

http://cs101.openjudge.cn/practice/01944/

Farmer John wants to connect his N (1 <= N <= 1,000) barns (numbered 1..N) with a new fiber-optic network. However, the barns are located in a circle around the edge of a large pond, so he can only connect pairs of adjacent barns. The circular configuration means that barn N is adjacent to barn 1.

FJ doesn't need to connect all the barns, though, since only certain pairs of cows wish to communicate with each other. He wants to construct as few 
connections as possible while still enabling all of these pairs to communicate through the network. Given the list of barns that wish to communicate with each other, determine the minimum number of lines that must be laid. To communicate from barn 1 to barn 3, lines must be laid from barn 1 to barn 2 and also from barn 2 to barn 3(or just from barn 3 to 1,if n=3).

**输入**

\* Line 1: Two integers, N and P (the number of communication pairs, 1 <= P <= 10,000)

\* Lines 2..P+1: two integers describing a pair of barns between which communication is desired. No pair is duplicated in the list.

**输出**

One line with a single integer which is the minimum number of direct connections FJ needs to make.

样例输入

```
5 2
1 3
4 5
```

样例输出

```
3
```

提示

[Which connect barn pairs 1-2, 2-3, and 4-5.]

来源

USACO 2002 February



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

http://cs101.openjudge.cn/practice/02039/

Mo和Larry发明了一种信息加密方法。他们首先决定好列数，然后将信息（只包含字母）从上往下依次填入各列，并在末尾补充一些随机字母使其成为一个完整的字母矩阵。例如，若信息是“There's no place like home on a snowy night”并且有5列，Mo会写成：

```
t o i o y
h p k n n
e l e a i
r a h s g
e c o n h
s e m o t
n l e w x
```

注意Mo只会填入字母，且全部是小写形式。在这个例子中，Mo用字母“x”填充了信息使之成为一个完整的矩阵，当然他使用任何字母都是可以的。

Mo根据这个矩阵重写信息：首先从左到右写下第一行，然后从右到左写下第二行，再从左到右写下第三行……以此左右交替地从上到下写下各行字母，形成新的字符串。这样，例子中的信息就被加密为：toioynnkpheleaigshareconhtomesnlewx。

你的工作是帮助Larry从加密后的信息中还原出原始信息（包括填充的字母）。

**输入**

第一行包含一个整数（范围2到20），表示使用的列数。
第二行是一个长度不超过200的字符串。

**输出**

一行，即原始信息。

样例输入

```
5
toioynnkpheleaigshareconhtomesnlewx
```

样例输出

```
theresnoplacelikehomeonasnowynightx
```

来源

East Central North America 2004



```python
# 23n2300011072(X)
cols = int(input())
encrypted = input()
# 计算行数
rows = len(encrypted) // cols
# 创建矩阵
matrix = [['' for _ in range(cols)] for _ in range(rows)]
# 填充矩阵
index = 0
for row in range(rows):
    if row % 2 == 0:  # 从左到右填充
        for col in range(cols):
            matrix[row][col] = encrypted[index]
            index += 1
    else:  # 从右到左填充
        for col in range(cols - 1, -1, -1):
            matrix[row][col] = encrypted[index]
            index += 1
# 从矩阵中提取原始信息
original = ''
for col in range(cols):
    for row in range(rows):
        original += matrix[row][col]
print(original)
```





## 02049: Finding Nemo

http://cs101.openjudge.cn/practice/02049/

Nemo is a naughty boy. One day he went into the deep sea all by himself. Unfortunately, he became lost and couldn't find his way home. Therefore, he sent a signal to his father, Marlin, to ask for help.
After checking the map, Marlin found that the sea is like a labyrinth with walls and doors. All the walls are parallel to the X-axis or to the Y-axis. The thickness of the walls are assumed to be zero.
All the doors are opened on the walls and have a length of 1. Marlin cannot go through a wall unless there is a door on the wall. Because going through a door is dangerous (there may be some virulent medusas near the doors), Marlin wants to go through as few doors as he could to find Nemo.
Figure-1 shows an example of the labyrinth and the path Marlin went through to find Nemo.
![img](http://media.openjudge.cn/images/2049_1.jpg)
We assume Marlin's initial position is at (0, 0). Given the position of Nemo and the configuration of walls and doors, please write a program to calculate the minimum number of doors Marlin has to go through in order to reach Nemo.

**输入**

The input consists of several test cases. Each test case is started by two non-negative integers M and N. M represents the number of walls in the labyrinth and N represents the number of doors. 
Then follow M lines, each containing four integers that describe a wall in the following format: 
x y d t 
(x, y) indicates the lower-left point of the wall, d is the direction of the wall -- 0 means it's parallel to the X-axis and 1 means that it's parallel to the Y-axis, and t gives the length of the wall. 
The coordinates of two ends of any wall will be in the range of [1,199]. 
Then there are N lines that give the description of the doors: 
x y d 
x, y, d have the same meaning as the walls. As the doors have fixed length of 1, t is omitted. 
The last line of each case contains two positive float numbers: 
f1 f2 
(f1, f2) gives the position of Nemo. And it will not lie within any wall or door. 
A test case of M = -1 and N = -1 indicates the end of input, and should not be processed.

**输出**

For each test case, in a separate line, please output the minimum number of doors Marlin has to go through in order to rescue his son. If he can't reach Nemo, output -1.

样例输入

```
8 9
1 1 1 3
2 1 1 3
3 1 1 3
4 1 1 3
1 1 0 3
1 2 0 3
1 3 0 3
1 4 0 3
2 1 1
2 2 1
2 3 1
3 1 1
3 2 1
3 3 1
1 2 0
3 3 0
4 3 1
1.5 1.5
4 0
1 1 0 1
1 1 1 1
2 1 1 1
1 2 0 1
1.5 1.7
-1 -1
```

样例输出

```
5
-1
```

来源

Beijing 2004



```python
from collections import deque

N = 210
Size = 999999
INF = 1<<20
mv = [(1,0),(0,-1),(0,1),(-1,0)]
mapp = [[[0]*2 for _ in range(N)] for _ in range(N)]
vis = [[0]*N for _ in range(N)]

def init():
    global result
    result = 0
    for i in range(N):
        for j in range(N):
            mapp[i][j] = [0, 0]
            vis[i][j] = 0

def BFS(x, y):
    global result
    q = deque()
    q.append((x, y, 0))
    vis[x][y] = 1
    result = INF
    while q:
        t = q.popleft()
        if t[0] == 0 or t[1] == 0 or t[0] > 198 or t[1] > 198:
            result = min(result, t[2])
            continue
        for i in range(4):
            f = [t[0] + mv[i][0], t[1] + mv[i][1]]
            if i == 0 and not vis[f[0]][f[1]] and mapp[t[0]][t[1]][1] != 3:
                f.append(t[2] + 1 if mapp[t[0]][t[1]][1] == 4 else t[2])
                vis[f[0]][f[1]] = 1
                q.append(tuple(f))
            elif i == 1 and not vis[f[0]][f[1]] and mapp[f[0]][f[1]][0] != 3:
                f.append(t[2] + 1 if mapp[f[0]][f[1]][0] == 4 else t[2])
                vis[f[0]][f[1]] = 1
                q.append(tuple(f))
            elif i == 2 and not vis[f[0]][f[1]] and mapp[t[0]][t[1]][0] != 3:
                f.append(t[2] + 1 if mapp[t[0]][t[1]][0] == 4 else t[2])
                vis[f[0]][f[1]] = 1
                q.append(tuple(f))
            elif i == 3 and not vis[f[0]][f[1]] and mapp[f[0]][f[1]][1] != 3:
                f.append(t[2] + 1 if mapp[f[0]][f[1]][1] == 4 else t[2])
                vis[f[0]][f[1]] = 1
                q.append(tuple(f))

while True:
    m, n = map(int, input().split())
    if m == -1 and n == -1:
        break
    init()
    for _ in range(m):
        x, y, d, t = map(int, input().split())
        if d:
            for num in range(t):
                mapp[x-1][y+num][1] = 3
        else:
            for num in range(t):
                mapp[x+num][y-1][0] = 3
    for _ in range(n):
        x, y, d = map(int, input().split())
        if d:
            mapp[x-1][y][1] = 4
        else:
            mapp[x][y-1][0] = 4
    Nemo_x, Nemo_y = map(float, input().split())
    xx, yy = int(Nemo_x + 0.0001), int(Nemo_y + 0.0001)
    if n == 0 and m == 0:
        print(0)
        continue
    if xx <= 0 or yy <= 0 or xx >= 199 or yy >= 199:
        print(0)
    else:
        BFS(xx, yy)
        print(result if result != INF else -1)
```





## 02092: Grandpa is Famous

http://cs101.openjudge.cn/practice/02092/

The whole family was excited by the news. Everyone knew grandpa had been an extremely good bridge player for decades, but when it was announced he would be in the Guinness Book of World Records as the most successful bridge player ever, whow, that was astonishing!
The International Bridge Association (IBA) has maintained, for several years, a weekly ranking of the best players in the world. Considering that each appearance in a weekly ranking constitutes a point for the player, grandpa was nominated the best player ever because he got the highest number of points.
Having many friends who were also competing against him, grandpa is extremely curious to know which player(s) took the second place. Since the IBA rankings are now available in the internet he turned to you for help. He needs a program which, when given a list of weekly rankings, finds out which player(s) got the second place according to the number of points.

**输入**

The input contains several test cases. Players are identified by integers from 1 to 10000. The first line of a test case contains two integers N and M indicating respectively the number of rankings available (2 <= N <= 500) and the number of players in each ranking (2 <= M <= 500). Each of the next N lines contains the description of one weekly ranking. Each description is composed by a sequence of M integers, separated by a blank space, identifying the players who figured in that weekly ranking. You can assume that:

- in each test case there is exactly one best player and at least one second best player,
- each weekly ranking consists of M distinct player identifiers.

The end of input is indicated by N = M = 0.

**输出**

For each test case in the input your program must produce one line of output, containing the identification number of the player who is second best in number of appearances in the rankings. If there is a tie for second best, print the identification numbers of all second best players in increasing order. Each identification number produced must be followed by a blank space.

样例输入

```
4 5
20 33 25 32 99
32 86 99 25 10
20 99 10 33 86
19 33 74 99 32
3 6
2 34 67 36 79 93
100 38 21 76 91 85
32 23 85 31 88 1
0 0
```

样例输出

```
32 33
1 2 21 23 31 32 34 36 38 67 76 79 88 91 93 100
```

来源

South America 2004



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



## 02192: Zipper

http://cs101.openjudge.cn/practice/02192/

Given three strings, you are to determine whether the third string can be formed by combining the characters in the first two strings. The first two strings can be mixed arbitrarily, but each must stay in its original order.

For example, consider forming "tcraete" from "cat" and "tree":

String A: cat
String B: tree
String C: tcraete

As you can see, we can form the third string by alternating characters from the two strings. As a second example, consider forming "catrtee" from "cat" and "tree":

String A: cat
String B: tree
String C: catrtee

Finally, notice that it is impossible to form "cttaree" from "cat" and "tree".

**输入**

The first line of input contains a single positive integer from 1 through 1000. It represents the number of data sets to follow. The processing for each data set is identical. The data sets appear on the following lines, one data set per line.

For each data set, the line of input consists of three strings, separated by a single space. All strings are composed of upper and lower case letters only. The length of the third string is always the sum of the lengths of the first two strings. The first two strings will have lengths between 1 and 200 characters, inclusive.

**输出**

For each data set, print:

Data set n: yes

if the third string can be formed from the first two, or

Data set n: no

if it cannot. Of course n should be replaced by the data set number. See the sample output below for an example.

样例输入

```
3
cat tree tcraete
cat tree catrtee
cat tree cttaree
```

样例输出

```
Data set 1: yes
Data set 2: yes
Data set 3: no
```

来源

Pacific Northwest 2004



```python
# 袁籁2300010728
from functools import lru_cache


@lru_cache
def f(a, b, c):
    if len(c) == 0:
        return True
    else:
        if len(a) and c[0] == a[0] and f(a[1:], b, c[1:]):
            return True
        elif len(b) and c[0] == b[0] and f(a, b[1:], c[1:]):
            return True
        else:
            return False


n = int(input())
for _ in range(n):
    a, b, c = input().split()
    x = len(c)
    if f(a, b, c):
        print('Data set %d: yes' % (_ + 1))
    else:
        print('Data set %d: no' % (_ + 1))

```





## 02226: Muddy Fields

http://cs101.openjudge.cn/practice/02226/

Rain has pummeled the cows' field, a rectangular grid of R rows and C columns (1 <= R <= 50, 1 <= C <= 50). While good for the grass, the rain makes some patches of bare earth quite muddy. The cows, being meticulous grazers, don't want to get their hooves dirty while they eat.

To prevent those muddy hooves, Farmer John will place a number of wooden boards over the muddy parts of the cows' field. Each of the boards is 1 unit wide, and can be any length long. Each board must be aligned parallel to one of the sides of the field.

Farmer John wishes to minimize the number of boards needed to cover the muddy spots, some of which might require more than one board to cover. The boards may not cover any grass and deprive the cows of grazing area but they can overlap each other.

Compute the minimum number of boards FJ requires to cover all the mud in the field.

**输入**

\* Line 1: Two space-separated integers: R and C

\* Lines 2..R+1: Each line contains a string of C characters, with '*' representing a muddy patch, and '.' representing a grassy patch. No spaces are present.

**输出**

\* Line 1: A single integer representing the number of boards FJ needs.

样例输入

```
4 4
*.*.
.***
***.
..*.
```

样例输出

```
4
```

提示

OUTPUT DETAILS:

Boards 1, 2, 3 and 4 are placed as follows:
1.2.
.333
444.
..2.
Board 2 overlaps boards 3 and 4.

来源

USACO 2005 January Gold



A well‐known approach is to first label each maximal horizontal contiguous segment and each maximal vertical contiguous segment, build a bipartite graph between these segments (with an edge if a muddy cell belongs to both), and then compute the maximum matching (which, by Kőnig’s theorem, equals the minimum vertex cover – i.e. the minimum boards needed).

```python
def min_boards(R, C, field):
    # Label horizontal segments.
    hor = [[0] * C for _ in range(R)]
    hor_id = 0
    for r in range(R):
        c = 0
        while c < C:
            if field[r][c] == '*':
                hor_id += 1
                # label contiguous '*' segment in row r
                while c < C and field[r][c] == '*':
                    hor[r][c] = hor_id
                    c += 1
            else:
                c += 1

    # Label vertical segments.
    ver = [[0] * C for _ in range(R)]
    ver_id = 0
    for c in range(C):
        r = 0
        while r < R:
            if field[r][c] == '*':
                ver_id += 1
                # label contiguous '*' segment in column c
                while r < R and field[r][c] == '*':
                    ver[r][c] = ver_id
                    r += 1
            else:
                r += 1

    # Build bipartite graph: for each horizontal segment, list all vertical segments that intersect it.
    graph = {i: set() for i in range(1, hor_id + 1)}
    for r in range(R):
        for c in range(C):
            if field[r][c] == '*':
                h = hor[r][c]
                v = ver[r][c]
                graph[h].add(v)

    # Use DFS to find an augmenting path in the bipartite graph.
    match = {}  # maps vertical segment -> horizontal segment

    def dfs(u, seen):
        for v in graph[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match or dfs(match[v], seen):
                match[v] = u
                return True
        return False

    result = 0
    for u in range(1, hor_id + 1):
        if dfs(u, set()):
            result += 1
    return result

if __name__ == "__main__":
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        exit(0)
    R = int(data[0])
    C = int(data[1])
    field = data[2:]
    print(min_boards(R, C, field))

```

> Explanation
>
> 1. **Segment Labeling:**
>    - **Horizontal Segments:** For each row, we scan left-to-right. When we hit a `'*'`, we label all consecutive `'*'`cells as one horizontal segment.
>    - **Vertical Segments:** For each column, we scan top-to-bottom. When we hit a `'*'`, we label all consecutive `'*'` cells as one vertical segment.
> 2. **Graph Construction:**
>    - For every cell with mud (`'*'`), we know its horizontal segment id (from `hor`) and its vertical segment id (from `ver`). We then add an edge from the horizontal segment to the vertical segment.
> 3. **Maximum Bipartite Matching:**
>    - We use a DFS-based augmenting path algorithm to compute the maximum matching. By Kőnig’s theorem, this matching size is equal to the minimum number of boards required.





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





## 02253: Frogger

http://cs101.openjudge.cn/dsapre/02253/

Freddy Frog is sitting on a stone in the middle of a lake. Suddenly he notices Fiona Frog who is sitting on another stone. He plans to visit her, but since the water is dirty and full of tourists' sunscreen, he wants to avoid swimming and instead reach her by jumping. 
Unfortunately Fiona's stone is out of his jump range. Therefore Freddy considers to use other stones as intermediate stops and reach her by a sequence of several small jumps. 
To execute a given sequence of jumps, a frog's jump range obviously must be at least as long as the longest jump occuring in the sequence. 
The frog distance (humans also call it minimax distance) between two stones therefore is defined as the minimum necessary jump range over all possible paths between the two stones. 

You are given the coordinates of Freddy's stone, Fiona's stone and all other stones in the lake. Your job is to compute the frog distance between Freddy's and Fiona's stone.

**输入**

The input will contain one or more test cases. The first line of each test case will contain the number of stones n (2<=n<=200). The next n lines each contain two integers xi,yi (0 <= xi,yi <= 1000) representing the coordinates of stone #i. Stone #1 is Freddy's stone, stone #2 is Fiona's stone, the other n-2 stones are unoccupied. There's a blank line following each test case. Input is terminated by a value of zero (0) for n. 

**输出**

For each test case, print a line saying "Scenario #x" and a line saying "Frog Distance = y" where x is replaced by the test case number (they are numbered from 1) and y is replaced by the appropriate real number, printed to three decimals. Put a blank line after each test case, even after the last one. 

样例输入

```
2
0 0
3 4

3
17 4
19 4
18 5

0
```

样例输出

```
Scenario #1
Frog Distance = 5.000

Scenario #2
Frog Distance = 1.414
```

来源

Ulm Local 1997



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

http://cs101.openjudge.cn/prctice/02255/

**输入**

输入可能有多组，以EOF结束。 每组输入包含两个字符串，分别为树的前序遍历和中序遍历。每个字符串中只包含大写字母且互不重复。

**输出**

对于每组输入，用一行来输出它后序遍历结果。

样例输入

```
DBACEGF ABCDEFG
BCAD CBAD
```

样例输出

```
ACBFGED
CDAB
```

提示

以英文题面为准



通过递归地划分左右子树并重建二叉树，然后再按照左子树、右子树、根节点的顺序进行后序遍历。

```python
def build_tree(preorder, inorder):
    if not preorder:
        return ''
    
    root = preorder[0]
    root_index = inorder.index(root)
    
    left_preorder = preorder[1:1 + root_index]
    right_preorder = preorder[1 + root_index:]
    
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    
    left_tree = build_tree(left_preorder, left_inorder)
    right_tree = build_tree(right_preorder, right_inorder)
    
    return left_tree + right_tree + root

while True:
    try:
        preorder, inorder = input().split()
        postorder = build_tree(preorder, inorder)
        print(postorder)
    except EOFError:
        break

```





## 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/

In this problem, you have to analyze a particular sorting algorithm. The algorithm processes a sequence of n distinct integers by swapping two adjacent sequence elements until the sequence is sorted in ascending order. For the input sequence 
9 1 0 5 4 , 
Ultra-QuickSort produces the output 
0 1 4 5 9 . 
Your task is to determine how many swap operations Ultra-QuickSort needs to perform in order to sort a given input sequence. 

**输入**

The input contains several test cases. Every test case begins with a line that contains a single integer n < 500,000 -- the length of the input sequence. Each of the the following n lines contains a single integer 0 ≤ a[i] ≤ 999,999,999, the i-th input sequence element. Input is terminated by a sequence of length n = 0. This sequence must not be processed.

**输出**

For every input sequence, your program prints a single line containing an integer number op, the minimum number of swap operations necessary to sort the given input sequence.

样例输入

```
5
9
1
0
5
4
3
1
2
3
0
```

样例输出

```
6
0
```

来源

Waterloo local 2005.02.05



```python
"""
问题：分析特定排序算法，通过交换两个相邻的序列元素来处理n个不同的整数序列，直到序列按升序排序。
任务是确定需要执行多少次交换操作才能对给定的输入序列进行排序。  

可以通过使用归并排序的修改版本来解决，计算在每次合并步骤中需要的交换次数。
在归并排序中，将数组分成两半，对每一半进行排序，然后将它们合并在一起。

在合并步骤中，计算需要交换的次数，因为每当从右半部分取出一个元素时，
需要交换与左半部分中剩余元素相同数量的次数。
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
            inv_count += len(left) - i #left[i~mid)都比right[j]要大，他们都会与right[j]构成逆序对，将他们加入答案

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



卢卓然-23-生命科学学院，思路：

Ultra-QuickSort题目使用的排序方法是归并排序（本题解答是基于归并排序的解法，用其他解法如冒泡排序时间复杂度较高）。在归并排序中，将序列递归地分为左右两半，分别排序后再合并到一起。在归并排序中，**合并函数的书写**是重点，也是本题关注的点。在两个有序序列进行合并的过程中，若通过**交换两个相邻数字的位置**来实现合并，一共需要交换多少次，就是本题需要解决的问题。

以下是归并排序（mergesort）的代码。`merge(arr,l,m,r)`函数是合并两个有序序列的函数。函数的主体是三个while。第一个while运用双指针法，是对两个序列的合并，涉及到了元素位置的改变。第二、三个while是简单地将L1或L2中的剩余有序元素复制到arr队尾，并不涉及到元素位置的改变。所以只需关注第一个while的内容。



现在来分析用双指针法合并两个有序序列时，与其（复杂度上）等效的交换相邻数字的方法是怎样实现的。在双指针法中，是将 `L1[i]`和`L2[j]`中更小的一个放在arr的k处，在两个指针逐渐变大的过程中，将合并后的递增序列覆盖在`arr`的一段上。这种方法相对交换相邻数字，无疑很节省时间复杂度。交换相邻数字，关心的只是数字之间的相对位置，而不是绝对位置。所以可以假定这样的规则：当`L1[i]<=L2[j]`时，不改变数字的相对位置；当`L1[i]>L2[j]`时，将L2[j]通过不断换位向前移动，从而“插入”到L1[i]的前面一个位置。

那么L2[j]需要换多少次才能达到L1[i]前面呢？想象用交换位置法得到新序列的过程，那么在L2[j]动身之前，**L2中所有L2[j]之前的元素已经全部跑到了L1[i]的左边，并且与L1中L1[i]左边的元素组成了递增序列**。L2[j]的“目的地”就是这个已组成的递增序列和L1[i]之间的位置。L2和目的地之间相隔的元素，也就是**L1中L1[i]及其之后的元素**，其数量为`n1-i`（n1为原L1的长度）。所以在这一步中，交换的次数为`d+=(n1-i)`。

随着递归的进行，每一次合并中d不断累加，最终就可以得到总交换次数。

```python
import sys
sys.setrecursionlimit(100000)
d=0
def merge(arr,l,m,r):
    '''对l到m和m到r两段进行合并'''
    global d
    n1=m-l+1#L1长
    n2=r-m#L2长
    L1=arr[l:m+1]
    L2=arr[m+1:r+1]
    ''' L1和L2均为有序序列'''
    i,j,k=0,0,l#i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i<n1 and j<n2:
        if L1[i]<=L2[j]:
            arr[k]=L1[i]
            i+=1
        else:
            arr[k]=L2[j]
            d+=(n1-i)#精髓所在
            j+=1
        k+=1
    while i<n1:
        arr[k]=L1[i]
        i+=1
        k+=1
    while j<n2:
        arr[k]=L2[j]
        j+=1
        k+=1
def mergesort(arr,l,r):
    '''对arr的l到r一段进行排序'''
    if l<r:#递归结束条件，很重要
        m=(l+r)//2
        mergesort(arr,l,m)
        mergesort(arr,m+1,r)
        merge(arr,l,m,r)
results=[]
while True:
    n=int(input())#序列长
    if n==0:
        break
    array=[]
    for b in range(n):
        array.append(int(input()))
    d=0
    mergesort(array,0,n-1)
    results.append(d)
for r in results:
    print(r)
```





# 02337~05344

## 02337: Catenyms

http://cs101.openjudge.cn/practice/02337/

A catenym is a pair of words separated by a period such that the last letter of the first word is the same as the last letter of the second. For example, the following are catenyms: 

```
dog.gopher
gopher.rat
rat.tiger
aloha.aloha
arachnid.dog
```

A compound catenym is a sequence of three or more words separated by periods such that each adjacent pair of words forms a catenym. For example, 

aloha.aloha.arachnid.dog.gopher.rat.tiger

Given a dictionary of lower case words, you are to find a compound catenym that contains each of the words exactly once.

**输入**

The first line of standard input contains t, the number of test cases. Each test case begins with 3 <= n <= 1000 - the number of words in the dictionary. n distinct dictionary words follow; each word is a string of between 1 and 20 lowercase letters on a line by itself.

**输出**

For each test case, output a line giving the lexicographically least compound catenym that contains each dictionary word exactly once. Output "***" if there is no solution. 

样例输入

```
2
6
aloha
arachnid
dog
gopher
rat
tiger
3
oak
maple
elm
```

样例输出

```
aloha.arachnid.dog.gopher.rat.tiger
***
```

来源

Waterloo local 2003.01.25







问题建模

1. **图的构造**
   - 顶点：26 个字母 `'a'–'z'`。
   - 有向边：每个单词 `w` 视为一条从 `w[0]` 指向 `w[-1]` 的边，边上存储整个单词。
2. **欧拉路径的必要条件**
   - 最多一个顶点满足 `出度 = 入度 + 1`（起点）。
   - 最多一个顶点满足 `入度 = 出度 + 1`（终点）。
   - 其余顶点都满足 `入度 = 出度`。
   - 涉及到的所有顶点在“无向”意义上必须连通。
3. **字典序最小**
   - 对每个出发字母，用一个 **最小堆**（`heapq`）按完整单词排序，取边时总是弹出堆顶，保证当前能选的最小单词优先使用。
4. **Hierholzer 算法**
   - 从合法的起点（如果不存在“`出度 = 入度 + 1`”则从最小字母出发）出发，DFS 弹栈构造路径，最后逆序输出。

------

代码实现

```python
import sys
import heapq
from collections import defaultdict, deque


def find_eulerian_path(words):
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    adj = defaultdict(list)
    used_letters = set()

    # 1. 构造图：入度、出度，并在 adj[u] 中维护 (word, v) 的最小堆
    for w in words:
        u, v = w[0], w[-1]
        outdeg[u] += 1
        indeg[v] += 1
        used_letters |= {u, v}
        heapq.heappush(adj[u], (w, v))

    # 2. 检查度数条件，找可能的起点
    start, plus1, minus1 = None, 0, 0
    for ch in used_letters:
        o, i = outdeg[ch], indeg[ch]
        if o == i + 1:
            plus1 += 1
            start = ch
        elif i == o + 1:
            minus1 += 1
        elif i != o:
            return None
    if not ((plus1 == 1 and minus1 == 1) or (plus1 == 0 and minus1 == 0)):
        return None

    # 3. 如果没有唯一起点，就从最小的有出度的字母开始
    if start is None:
        start = min(ch for ch in used_letters if outdeg[ch] > 0)

    # 4. 连通性检查（无向图）
    seen = {start}
    q = deque([start])
    undirected = defaultdict(list)
    for u in adj:
        for _, v in adj[u]:
            undirected[u].append(v)
            undirected[v].append(u)
    while q:
        u = q.popleft()
        for v in undirected[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    if seen != used_letters:
        return None

    # 5. Hierholzer：DFS 弹栈
    path = deque()

    def dfs(u):
        heap = adj[u]
        while heap:
            w, v = heapq.heappop(heap)
            dfs(v)
            path.appendleft(w)

    dfs(start)

    # 6. 检查是否用了所有单词
    if len(path) != len(words):
        return None
    return '.'.join(path)


def solve():
    input = sys.stdin.readline
    t = int(input())
    for _ in range(t):
        n = int(input())
        words = [input().strip() for _ in range(n)]
        ans = find_eulerian_path(words)
        print(ans if ans is not None else "***")


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    solve()
```

- **复杂度**：
  - 建图和入度/出度统计：O(N log N)（每个单词入堆）。
  - DFS（Hierholzer）：O(N log N)。
- 能正确处理最大 N=1000 的用例，并保证字典序最小。



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



## 02442: Sequence

heap, merge, http://cs101.openjudge.cn/practice/02442/

中文版是 http://cs101.openjudge.cn/practice/06648/



Given m sequences, each contains n non-negative integer. Now we may select one number from each sequence to form a sequence with m integers. It's clear that we may get n ^ m this kind of sequences. Then we can calculate the sum of numbers in each sequence, and get n ^ m values. What we need is the smallest n sums. Could you help us?

输入

The first line is an integer T, which shows the number of test cases, and then T test cases follow. The first line of each case contains two integers m, n (0 < m <= 100, 0 < n <= 2000). The following m lines indicate the m sequence respectively. No integer in the sequence is greater than 10000.

输出

For each test case, print a line with the smallest n sums in increasing order, which is separated by a space.

样例输入

```
1
2 3
1 2 3
2 2 3
```

样例输出

```
3 3 4
```

来源

POJ Monthly,Guang Lin



利用堆合并的方法依次求解两序列的最小 n 个和，从而逐步合并 m 个序列，避免枚举所有 n^m 种组合。

```python
import sys
import heapq

def merge(arr1, arr2, n):
    """
    将两个有序数组 arr1 和 arr2 合并，求出所有组合中最小的 n 个和
    使用堆来进行合并搜索
    """
    heap = []
    visited = set()
    # 初始候选项：(arr1[0]+arr2[0], 0, 0)
    heapq.heappush(heap, (arr1[0] + arr2[0], 0, 0))
    visited.add((0, 0))
    result = []
    while len(result) < n:
        s, i, j = heapq.heappop(heap)
        result.append(s)
        # 如果 arr1 中的下一个数存在，尝试加入候选项
        if i + 1 < n and (i + 1, j) not in visited:
            heapq.heappush(heap, (arr1[i + 1] + arr2[j], i + 1, j))
            visited.add((i + 1, j))
        # 如果 arr2 中的下一个数存在，尝试加入候选项
        if j + 1 < n and (i, j + 1) not in visited:
            heapq.heappush(heap, (arr1[i] + arr2[j + 1], i, j + 1))
            visited.add((i, j + 1))
    return result

def main():
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    T = int(next(it))
    results = []
    for _ in range(T):
        m = int(next(it))
        n = int(next(it))
        # 读取第一个序列，并排序
        current = sorted(int(next(it)) for _ in range(n))
        # 依次与后续的 m-1 个序列合并
        for _ in range(m - 1):
            seq = sorted(int(next(it)) for _ in range(n))
            current = merge(current, seq, n)
        results.append(" ".join(map(str, current)))
    sys.stdout.write("\n".join(results))

if __name__ == "__main__":
    main()
```

代码说明

- **merge 函数**  
  该函数接受两个有序数组 `arr1` 与 `arr2`，利用最小堆依次寻找组合中最小的 n 个和。我们用 `visited` 集合避免重复放入堆中。
- **主函数**  
  先读取测试用例数 T，再依次处理每个测试用例。每个测试用例中，首先将第一个序列排序作为初始的结果，再依次将后续序列与当前结果进行合并。最终输出最小的 n 个和。

该算法利用堆优化，每次合并时间复杂度约为 O(n log n)，适合 m 与 n 的题目范围。







参考链接：https://blog.csdn.net/liuwei_nefu/article/details/5645528
题意是  给出  m组数，每组 n个数  然后从m组中 每组选出一个进行求和 ，然后取其中前n小的数输出。 选择的总数自然是 n的m次方，暴力法自然是超时的。

一个简单的思路是，从第一组到第m组依次处理。
首先第一组的n个数自然是最小的n个数，  然后这n个数和第二组的n个组进行组合，形成n×n个数，保留其前n个数，再处理第三组，依次类推直到第m组。

为什么保留前n个数就可以了呢？   我们以第一组和第二组例所得的n×n个数为例，假设保留n+1个数，且这第n+1个数加上第三组的某个数x的和  在下一步中需要保留（即在下一步操作中属于前n小的数之一），然而前n个数中的任意一个数+x  < 第n+1个数+x  ，此时得出矛盾， 由此可知，每次处理后的n×n个数中只需保留前n个数即可

```python
import heapq

t = int(input())
for _ in range(t):
    m, n = map(int, input().split())
    seq1 = sorted(map(int, input().split()))
    for _ in range(m - 1):
        seq2 = sorted(map(int, input().split()))

        # 使用最小堆存储可能的最小和以及对应的索引
        min_heap = [(seq1[i] + seq2[0], i, 0) for i in range(n)]
        heapq.heapify(min_heap)
        result = []
        for _ in range(n):
            current_sum, i, j = heapq.heappop(min_heap)
            result.append(current_sum)
            if j + 1 < len(seq2):
                heapq.heappush(min_heap, (seq1[i] + seq2[j + 1], i, j + 1))
        seq1 = result
    print(*seq1)
```



## 02488: A Knight's Journey

http://cs101.openjudge.cn/practice/02488/

**Background**
The knight is getting bored of seeing the same black and white squares again and again and has decided to make a journey
around the world. Whenever a knight moves, it is two squares in one direction and one square perpendicular to this. The world of a knight is the chessboard he is living on. Our knight lives on a chessboard that has a smaller area than a regular 8 * 8 board, but it is still rectangular. Can you help this adventurous knight to make travel plans?

![img](http://media.openjudge.cn/images/g1490/2488_1.jpg)

**Problem**
Find a path such that the knight visits every square once. The knight can start and end on any square of the board.

**输入**

The input begins with a positive integer n in the first line. The following lines contain n test cases. Each test case consists of a single line with two positive integers p and q, such that 1 <= p * q <= 26. This represents a p * q chessboard, where p describes how many different square numbers 1, . . . , p exist, q describes how many different square letters exist. These are the first q letters of the Latin alphabet: A, . . .

**输出**

The output for every scenario begins with a line containing "Scenario #i:", where i is the number of the scenario starting at 1. Then print a single line containing the lexicographically first path that visits all squares of the chessboard with knight moves followed by an empty line. The path should be given on a single line by concatenating the names of the visited squares. Each square name consists of a capital letter followed by a number.
If no such path exist, you should output impossible on a single line.

样例输入

```
3
1 1
2 3
4 3
```

样例输出

```
Scenario #1:
A1

Scenario #2:
impossible

Scenario #3:
A1B3C1A2B4C2A3B1C3A4B2C4
```

来源

TUD Programming Contest 2005, Darmstadt, Germany



思路：通过回溯法枚举所有可能路径，并优化提高搜索效率。

回溯法：从起点开始，尝试所有合法移动方向。标记已访问方格，递归探索下一个位置。若访问完所有方格，记录路径；若无路可走，回溯并尝试其他方向。

优化：按列字母（A, B, ...）和行号（1, 2, ...）升序尝试移动，确保找到的第一条路径即为字典序最小。并且，若当前路径无法覆盖所有方格，提前终止该分支。

```python
def knight_tour(p, q):
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    
    total = p * q
    path = []
    visited = [[False for _ in range(q)] for _ in range(p)]
    
    def backtrack(row, col):
        path.append(f"{chr(ord('A') + col)}{row + 1}")
        visited[row][col] = True
        
        if len(path) == total:
            return True
        
        next_steps = []
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < p and 0 <= nc < q and not visited[nr][nc]:
                next_steps.append((nc, nr))
        
        for nc, nr in sorted(next_steps):
            if backtrack(nr, nc):
                return True
        
        path.pop()
        visited[row][col] = False
        return False
    
    for start_row in range(p):
        for start_col in range(q):
            if backtrack(start_row, start_col):
                return ''.join(path)
    return "impossible"

n = int(input())
for i in range(n):
    p, q = map(int, input().split())
    result = knight_tour(p, q)
    print(f"Scenario #{i+1}:")
    print(result)
    print()
```





【陈宣之 23生科】思路：Dijstra，把路径放在第一位，用heapq取字典序最小

```python
# 陈宣之 23生科
import heapq
def dfs(x,y,r,c):
    global table,directions
    q=[[table[x][y],(x,y)]]
    while q:
        way,(x,y)=heapq.heappop(q)
        if len(way)==r*c*2:
            return way
        for dx,dy in directions:
            nx,ny=x+dx,y+dy
            if 0<=nx<r and 0<=ny<c and table[nx][ny] not in way:
                heapq.heappush(q,[way+table[nx][ny],(nx,ny)])
    return 0


n=int(input())
directions=[(-2,-1),(-2,1),(-1,2),(-1,-2),(1,-2),(1,2),(2,-1),(2,1)]
for _ in range(n):
    p,q=map(int,input().split())
    table=[]
    for i in range(p):
        temp=[]
        for j in range(q):
            temp.append(chr(ord("A")+j)+str(i+1))
        table.append(temp)
    judge=False
    for j in range(q):
        for i in range(p):
            if dfs(i,j,p,q):
                judge=True
                print("Scenario #",_+1,":",sep="")
                print(dfs(i,j,p,q))
                break
        if judge:
            break
    if not judge:
        print("Scenario #",_+1,":",sep="")
        print("impossible")
    if _<n-1:
        print()
```





回溯即可，重点是需要排列好自己的方向使得第一个弄出来的就是字典序第一个。

```python
# 曹以楷
"""
@File        :   knights_journey_02488.py
@Time        :   2025/03/07 18:59:38
@Author      :   Usercyk
@Description :   Get the possible paths for a knight to pass every squares in pxq board.
"""


class Solution:
    """
    The solution class
    """
    MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
             (1, -2), (1, 2), (2, -1), (2, 1)]

    def __init__(self) -> None:
        self.path = []
        self.p = -1
        self.q = -1
        self.board = []
        self.flag = False

    def travel(self, step: int = 1, x: int = 0, y: int = 0) -> bool:
        """
        Travel the pxq board

        Arguments:
            step -- the current step
            x -- current pos x
            y -- current pos y

        Returns:
            Can the knight travel through all the board
        """
        if step == self.p*self.q:
            self.flag = True
            return True

        for dy, dx in self.MOVES:
            nx, ny = x+dx, y+dy

            if all((not self.flag, 0 <= nx < self.p, 0 <= ny < self.q)):
                if self.board[nx][ny] != 1:
                    self.board[nx][ny] = 1
                    self.path[step] = (nx, ny)
                    self.travel(step+1, nx, ny)
                    self.board[nx][ny] = 0

        return self.flag

    def re_init(self, p: int, q: int):
        """
        Init the board and paths

        Arguments:
            p -- the numbers
            q -- the alphabets
        """
        self.p, self.q = p, q
        self.path = [(0, 0) for _ in range(p*q)]

        self.board = [[0]*(q+1) for _ in range(p+1)]
        self.board[0][0] = 1

        self.flag = False

    def solve(self):
        """
        Solve the problem
        """
        for i in range(int(input())):
            self.re_init(*map(int, input().split()))

            print(f"Scenario #{i+1}:")
            if self.travel():
                ans = (chr(c[1]+ord("A"))+str(c[0]+1) for c in self.path)
                print("".join(ans))
            else:
                print("impossible")
            print("")


if __name__ == "__main__":
    Solution().solve()

```





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



## 02499: Binary Tree

http://cs101.openjudge.cn/practice/02499/

Binary trees are a common data structure in computer science. In this problem we will look at an infinite binary tree where the nodes contain a pair of integers. The tree is constructed like this:

- The root contains the pair (1, 1).
- If a node contains (a, b) then its left child contains (a + b, b) and its right child (a, a + b)

**Problem**
Given the contents (a, b) of some node of the binary tree described above, suppose you are walking from the root of the tree to the given node along the shortest possible path. Can you find out how often you have to go to a left child and how often to a right child?

**输入**

The first line contains the number of scenarios.
Every scenario consists of a single line containing two integers i and j (1 <= i, j <= 2*109) that represent
a node (i, j). You can assume that this is a valid node in the binary tree described above.

**输出**

The output for every scenario begins with a line containing "Scenario #i:", where i is the number of the scenario starting at 1. Then print a single line containing two numbers l and r separated by a single space, where l is how often you have to go left and r is how often you have to go right when traversing the tree from the root to the node given in the input. Print an empty line after every scenario.

样例输入

```
3
42 1
3 4
17 73
```

样例输出

```
Scenario #1:
41 0

Scenario #2:
2 1

Scenario #3:
4 6
```

来源

TUD Programming Contest 2005 (Training Session), Darmstadt, Germany



这个问题可以用 **递归或迭代方法** 解决，我们需要从 `(1,1)` 根节点出发，找到从根到 `(i,j)` 这个节点的路径，并统计 **左移和右移的次数**。

**分析题目**

**树的结构：**

- 根节点 `(1,1)`
- 左子树： `(a + b, b)`
- 右子树： `(a, a + b)`

对于 **给定的 `(i, j)`**，我们要找到它在二叉树中的位置，并计算 **左转和右转的次数**：

- **如果 `i > j`**，意味着 `(i, j)` 来自 **左子树**，所以 **左转**。
- **如果 `i < j`**，意味着 `(i, j)` 来自 **右子树**，所以 **右转**。
- **如果 `i == j`**，那么 `i, j` 必须是 `(1,1)`，但由于题目保证 `(i,j)` 是合法的，我们不需要考虑这种情况。

因此，我们可以不断递归：

- 如果 `(i > j)`，它的 **父节点** 是 `(i - j, j)`，左转计数 `+1`。
- 如果 `(i < j)`，它的 **父节点** 是 `(i, j - i)`，右转计数 `+1`。

最终 `(1,1)` 会是终点。

**Python 代码**

```python
def count_moves(i, j):
    left_moves = 0
    right_moves = 0

    while i != 1 and j != 1:  # 终止条件: (1,1)
        if i > j:
            left_moves += i // j  # 计算可以跳跃多少次
            i %= j  # 直接更新 i，减少迭代次数
            if i == 0:  # 避免 ZeroDivisionError
                i = 1
        else:
            right_moves += j // i  # 计算可以跳跃多少次
            j %= i  # 直接更新 j，减少迭代次数
            if j == 0:  # 避免 ZeroDivisionError
                j = 1

    # 可能 i != 1 或 j != 1，需要再补一次
    if i > 1:
        left_moves += i - 1
    elif j > 1:
        right_moves += j - 1

    return left_moves, right_moves


n = int(input())  # 读取测试用例数量
for case_num in range(1, n + 1):
    i, j = map(int, input().split())  # 读取 i, j
    left, right = count_moves(i, j)

    # 输出格式
    print(f"Scenario #{case_num}:")
    print(left, right)
    if case_num != n:
        print()  # 题目要求每个案例后面空行

```

> **优化点**
>
> 1. **使用 `i // j` 和 `i % j` 来优化计算**：
>    - 由于 `(i, j)` 总是其父节点 `(i - j, j)` 或 `(i, j - i)`，我们可以直接 **跳跃** `i // j` 或 `j // i` 步，而不是一层层递归，减少递归调用。
>    - 例如 `(42, 1)`，它的 **父节点** 是 `(41,1)`，然后 `(40,1)`，一直到 `(1,1)`，所以左转 `41` 次，而不是递归 41 次。
>
> 2. **用 `while` 迭代代替递归**：
>    - 避免递归的栈溢出问题（因为 `i, j` 可以达到 `2 * 10^9`）。
>    - 迭代方式更加 **高效**，可以 **O(log(max(i, j)))** 解决问题。
>
> **复杂度分析**
>
> 每次迭代都将 `i, j` 变成 `i % j` 或 `j % i`，类似 **欧几里得算法（求 GCD）**，因此：
>
> - **时间复杂度：** `O(log(max(i, j)))`
> - **空间复杂度：** `O(1)`
>
> 这个方法在大数据情况下也能高效运行！





ee-张坤思路：减法变成除法，就可以大大提高效率。可以假设一个极端情况 1和10000 减法要用9999次 除法只用1次。武昱达：辗转相除。

```python
n = int(input())
for i in range(n):
    a, b = map(int, input().split())
    l, r = 0, 0
    while a != 1 or b != 1:
        if a == 1:
            r += b - a
            b = 1
        elif b == 1:
            l += a - 1
            a = 1
        elif a > b:
            l += a // b
            a -= b * (a // b)
        elif a < b:
            r += b // a
            b -= a * (b // a)
    print("Scenario #{}:".format(i + 1))
    print(l, r)
    print()
```



只要结点不包含1值，左右整除不会余0。

```python
# 23n2300011329 洪亮
def binarytree(l, r, x, y):
    if l == 1:
        return [x, y+r-l]
    elif r == 1:
        return [x+l-r, y]
    elif l > r:
        n = l // r
        # if l == r * n:
        #     n -= 1
        ans = binarytree(l - r * n, r, x + n, y)
    else:
        n = r // l
        # if r == l * n:
        #     n -= 1
        ans = binarytree(l, r - l * n, x, y + n)
    return ans


for _ in range(int(input())):
    l, r = map(int, input().split())
    ans = binarytree(l, r, 0, 0)
    print(f'Scenario #{_ + 1}:')
    print(ans[0], ans[1])
    print()
```





## 02502: Subway

dijkstra, http://cs101.openjudge.cn/practice/02502/

You have just moved from a quiet Waterloo neighbourhood to a big, noisy city. Instead of getting to ride your bike to school every day, you now get to walk and take the subway. Because you don't want to be late for class, you want to know how long it will take you to get to school. 
You walk at a speed of 10 km/h. The subway travels at 40 km/h. Assume that you are lucky, and whenever you arrive at a subway station, a train is there that you can board immediately. You may get on and off the subway any number of times, and you may switch between different subway lines if you wish. All subway lines go in both directions.

**输入**

Input consists of the x,y coordinates of your home and your school, followed by specifications of several subway lines. Each subway line consists of the non-negative integer x,y coordinates of each stop on the line, in order. You may assume the subway runs in a straight line between adjacent stops, and the coordinates represent an integral number of metres. Each line has at least two stops. The end of each subway line is followed by the dummy coordinate pair -1,-1. In total there are at most 200 subway stops in the city. 

**输出**

Output is the number of minutes it will take you to get to school, rounded to the nearest minute, taking the fastest route.

样例输入

```
0 0 10000 1000
0 200 5000 200 7000 200 -1 -1 
2000 600 5000 600 10000 600 -1 -1
```

样例输出

```
21
```

来源

Waterloo local 2001.09.22



✅ 带注释的 Dijkstra 最短路径算法（支持步行与地铁）：

```python
import math
import heapq

# 计算两点之间的欧几里得距离
def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 读取起点（家）和终点（学校）坐标
sx, sy, ex, ey = map(int, input().split())

# min_time: 记录从起点到每个地铁站/终点的最短时间（单位：小时）
min_time = {}

# rails: 记录所有地铁连接（双向）
rails = set()

# 读取所有地铁线路
while True:
    try:
        rail = list(map(int, input().split()))
        if rail == [-1, -1]:
            break
        # 解析当前地铁线路的所有站点
        stations = [(rail[2 * i], rail[2 * i + 1]) for i in range(len(rail) // 2 - 1)]

        for j, station in enumerate(stations):
            # 初始化所有地铁站点的最短时间为无穷大
            min_time[station] = float('inf')
            # 添加地铁线路中相邻站点的双向连接
            if j != len(stations) - 1:
                rails.add((station, stations[j + 1]))
                rails.add((stations[j + 1], station))
    except EOFError:
        break  # 输入结束

# 把起点和终点加入时间表中
min_time[(sx, sy)] = 0  # 起点时间为 0
min_time[(ex, ey)] = float('inf')  # 终点初始化为无穷大

# 使用小根堆实现 Dijkstra 算法，按时间升序处理节点
min_heap = [(0, sx, sy)]  # (当前耗时, 当前x, 当前y)

while min_heap:
    curr_time, x, y = heapq.heappop(min_heap)

    # 如果当前耗时不是最短路径中记录的值，说明已经被更新，跳过
    if curr_time > min_time[(x, y)]:
        continue

    # 如果已经到达终点，提前结束
    if (x, y) == (ex, ey):
        break

    # 遍历所有可达点（隐式图）
    for position in min_time.keys():
        if position == (x, y):
            continue  # 自己跳过
        nx, ny = position

        # 计算当前位置到下一个点的距离
        dis = get_distance(x, y, nx, ny)

        # 判断是否为地铁连接：地铁速度是步行的4倍
        rail_factor = 4 if ((position, (x, y)) in rails or ((x, y), position) in rails) else 1

        # 计算到该点的所需时间（单位：小时）
        new_time = curr_time + dis / (10000 * rail_factor)

        # 如果时间更短，则更新并加入堆中
        if new_time < min_time[position]:
            min_time[position] = new_time
            heapq.heappush(min_heap, (new_time, nx, ny))

# 输出从起点到终点的最短时间，转换为分钟并四舍五入
print(round(min_time[(ex, ey)] * 60))
```

------

✅ 小结

- **地铁速度**是步行的 4 倍 → 用 `rail_factor = 4` 简化处理。
- **图是隐式图**：所有站点间的连边不是预先建好，而是在 Dijkstra 中动态判断。
- 只对包含的点建图（避免不必要计算，提升效率）。

这个代码不仅**简洁清晰**，还**容易维护和扩展**，比如日后加入不同速度的公交车或地铁线路都很方便。







```python
#2300012739 汤子瑜
import math
import heapq

def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

sx, sy, ex, ey = map(int, input().split())
min_time = {}
rails = set()

while True:
    try:
        rail = list(map(int, input().split()))
        if rail == [-1, -1]:
            break
        stations = [(rail[2 * i], rail[2 * i + 1]) for i in range(len(rail) // 2 - 1)]
        for j, station in enumerate(stations):
            min_time[station] = float('inf')
            if j != len(stations) - 1:
                rails.add((station, stations[j + 1]))
                rails.add((stations[j + 1], station))
    except EOFError:
        break

min_time[(sx, sy)], min_time[(ex, ey)] = 0, float('inf')
min_heap = [(0, sx, sy)]

while min_heap:
    curr_time, x, y = heapq.heappop(min_heap)
    if curr_time > min_time[(x, y)]:
        continue

    if (x, y) == (ex, ey):
        break

    for position in min_time.keys():
        if position == (x, y):
            continue
        nx, ny = position
        dis = get_distance(x, y, nx, ny)
        rail_factor = 4 if ((position, (x, y)) in rails or ((x, y), position) in rails) else 1
        new_time = curr_time + dis / (10000 * rail_factor)
        if new_time < min_time[position]:
            min_time[position] = new_time
            heapq.heappush(min_heap, (new_time, nx, ny))

print(round(min_time[(ex, ey)] * 60))

```



```python
import heapq
import math

def distance(x1, y1, x2, y2, speed):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / speed

a, b, c, d = map(int, input().split())
home = (a, b)
school = (c, d)

# Subway stations and their coordinates
stations = [home, school]
subway_lines = []

# Reading the subway lines
while True:
    try:
        line = list(map(int, input().split()))
        if line == [-1, -1]:
            break
        subway_line = []
        for i in range(0, len(line) - 2, 2):
            subway_line.append((line[i], line[i + 1]))
        subway_lines.append(subway_line)
        stations.extend(subway_line)
    except EOFError:
        break

# Number of stations
n = len(stations)

# Distance matrix
dis = [[float('inf')] * n for _ in range(n)]

# Walking distance
for i in range(n):
    for j in range(i + 1, n):
        dis[i][j] = dis[j][i] = distance(stations[i][0], stations[i][1], stations[j][0], stations[j][1], 10 / 3.6)

# Subway distance
for line in subway_lines:
    for i in range(len(line) - 1):
        u = stations.index(line[i])
        v = stations.index(line[i + 1])
        dis[u][v] = dis[v][u] = distance(line[i][0], line[i][1], line[i + 1][0], line[i + 1][1], 40 / 3.6)

# Dijkstra's algorithm
def dijkstra(start):
    min_heap = [(0, start)]
    min_time = [float('inf')] * n
    min_time[start] = 0
    while min_heap:
        current_time, u = heapq.heappop(min_heap)
        if current_time > min_time[u]:
            continue
        for v in range(n):
            if u != v and current_time + dis[u][v] < min_time[v]:
                min_time[v] = current_time + dis[u][v]
                heapq.heappush(min_heap, (min_time[v], v))
    return min_time

min_time = dijkstra(0)
print(round(min_time[1] / 60))  # Convert seconds to minutes and round to nearest minute
```





## 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Basic Exercise 部分相应题目



## 02756: 二叉树（1）

http://cs101.openjudge.cn/practice/02756/



![img](http://media.openjudge.cn/images/2756_1.jpg)
如上图所示，由正整数1, 2, 3, ...组成了一棵无限大的二叉树。从某一个结点到根结点（编号是1的结点）都有一条唯一的路径，比如从10到根结点的路径是(10, 5, 2, 1)，从4到根结点的路径是(4, 2, 1)，从根结点1到根结点的路径上只包含一个结点1，因此路径就是(1)。对于两个结点x和y，假设他们到根结点的路径分别是(x1, x2, ... ,1)和(y1, y2, ... ,1)（这里显然有x = x1，y = y1），那么必然存在两个正整数i和j，使得从xi和 yj开始，有$x_i = y_j , x_{i + 1} = y_{j + 1}, x_{i + 2} = y_{j + 2},...$ 现在的问题就是，给定x和y，要求xi（也就是yj)。

**输入**

输入只有一行，包括两个正整数x和y，这两个正整数都不大于1000。

**输出**

输出只有一个正整数xi。

样例输入

```
10 4
```

样例输出

```
2
```



这个问题涉及到二叉树中两个节点的最近公共祖先问题。这里的二叉树是一个特殊的完全二叉树，其中节点编号的方式是根节点编号为1，对于任意节点`N`，其左子节点的编号为`2 * N`，右子节点的编号为`2 * N + 1`。

要找到两个节点`x`和`y`的公共祖先，我们可以回溯其到根节点的路径，并找到路径上的最后一个公共节点。在这个完全二叉树中，我们可以简单地通过整除2来获得父节点的编号，即节点`N`的父节点是`N // 2`。

下面是Python代码示例，用于找到任意两个节点的最近公共祖先：

```python
def find_common_ancestor(x, y):
    # 创建两个集合用于存储x和y的所有祖先节点
    ancestors_x = set()
    ancestors_y = set()
  
    # 回溯x到根节点的路径并保存
    while x > 0:
        ancestors_x.add(x)
        x //= 2

    # 回溯y到根节点的路径
    # 并在每一步检查当前节点是否也是x的祖先节点
    while y > 0:
        if y in ancestors_x:
            return y  # 找到了公共祖先
        y //= 2

    return 1  # 如果没有找到公共祖先，默认返回根节点1

# 读取输入
x, y = map(int, input().split())

# 查找并输出x和y的最近公共祖先
print(find_common_ancestor(x, y))

```



```python
def common(x, y):
    if x == y:
        return x
    if x < y:
        return common(x, y//2)
    else:
        return common(x//2, y)


m, n = map(int, input().split())

print(common(m, n))
```





## 02766: 最大子矩阵

http://cs101.openjudge.cn/practice/02766/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 02773: 采药

http://cs101.openjudge.cn/practice/02773/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 02774: 木材加工

http://cs101.openjudge.cn/practice/02774/

木材厂有一些原木，现在想把这些木头切割成一些长度相同的小段木头，需要得到的小段的数目是给定了。当然，我们希望得到的小段越长越好，你的任务是计算能够得到的小段木头的最大长度。

木头长度的单位是厘米。原木的长度都是正整数，我们要求切割得到的小段木头的长度也要求是正整数。

**输入**

第一行是两个正整数*N*和*K*(1 ≤ *N* ≤ 10000, 1 ≤ *K* ≤ 10000)，*N*是原木的数目，*K*是需要得到的小段的数目。
接下来的*N*行，每行有一个1到10000之间的正整数，表示一根原木的长度。
　

**输出**

输出能够切割得到的小段的最大长度。如果连1厘米长的小段都切不出来，输出"0"。

样例输入

```
3 7
232
124
456
```

样例输出

```
114
```

来源

NOIP 2004



可以参考04135: 月度开销，08210: 河中跳房子

```python
n, k = map(int, input().split())
expenditure = []
for _ in range(n):
    expenditure.append(int(input()))


def check(x):
    num = 0
    for i in range(n):
        num += expenditure[i] // x

    return num >= k

lo = 1
hi = max(expenditure) + 1

if sum(expenditure) < k:
    print(0)
    exit()

ans = 1
while lo < hi:
    mid = (lo + hi) // 2
    if check(mid):
        ans = mid
        lo = mid + 1
    else:
        hi = mid

print(ans)
```



## 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/

在计算机上看到文件系统的结构通常很有用。Microsoft Windows上面的"explorer"程序就是这样的一个例子。但是在有图形界面之前，没有图形化的表示方法的，那时候最好的方式是把目录和文件的结构显示成一个"图"的样子，而且使用缩排的形式来表示目录的结构。比如：

```
ROOT
|     dir1
|     file1
|     file2
|     file3
|     dir2
|     dir3
|     file1
file1
file2
```

这个图说明：ROOT目录包括三个子目录和两个文件。第一个子目录包含3个文件，第二个子目录是空的，第三个子目录包含一个文件。

**输入**

你的任务是写一个程序读取一些测试数据。每组测试数据表示一个计算机的文件结构。每组测试数据以'*'结尾，而所有合理的输入数据以'#'结尾。一组测试数据包括一些文件和目录的名字（虽然在输入中我们没有给出，但是我们总假设ROOT目录是最外层的目录）。在输入中,以']'表示一个目录的内容的结束。目录名字的第一个字母是'd'，文件名字的第一个字母是'f'。文件名可能有扩展名也可能没有（比如fmyfile.dat和fmyfile）。文件和目录的名字中都不包括空格,长度都不超过30。一个目录下的子目录个数和文件个数之和不超过30。

**输出**

在显示一个目录中内容的时候，先显示其中的子目录（如果有的话），然后再显示文件（如果有的话）。文件要求按照名字的字母表的顺序显示（目录不用按照名字的字母表顺序显示，只需要按照目录出现的先后显示）。对每一组测试数据，我们要先输出"DATA SET x:"，这里x是测试数据的编号（从1开始）。在两组测试数据之间要输出一个空行来隔开。

你需要注意的是，我们使用一个'|'和5个空格来表示出缩排的层次。

样例输入

```
file1
file2
dir3
dir2
file1
file2
]
]
file4
dir1
]
file3
*
file2
file1
*
#
```

样例输出

```
DATA SET 1:
ROOT
|     dir3
|     |     dir2
|     |     file1
|     |     file2
|     dir1
file1
file2
file3
file4

DATA SET 2:
ROOT
file1
file2
```

提示

一个目录和它的子目录处于不同的层次
一个目录和它的里面的文件处于同一层次

来源: 翻译自 Pacific Northwest 1998 的试题



洪亮 物理学院

思路：把目录看成节点，每个目录节点设置子目录节点列表和子文件列表。输出格式：当前目录，遍历子目录，遍历子文件。

```python
class Node:
    def __init__(self,name):
        self.name=name
        self.dirs=[]
        self.files=[]

def print_(root,m):
    pre='|     '*m
    print(pre+root.name)
    for Dir in root.dirs:
        print_(Dir,m+1)
    for file in sorted(root.files):
        print(pre+file)
        
tests,test=[],[]
while True:
    s=input()
    if s=='#':
        break
    elif s=='*':
        tests.append(test)
        test=[]
    else:
        test.append(s)
for n,test in enumerate(tests,1):
    root=Node('ROOT')
    stack=[root]
    print(f'DATA SET {n}:')
    for i in test:
        if i[0]=='d':
            Dir=Node(i)
            stack[-1].dirs.append(Dir)
            stack.append(Dir)
        elif i[0]=='f':
            stack[-1].files.append(i)
        else:
            stack.pop()
    print_(root,0)
    print()
```



考虑递归，对于输入文件 f 开头则存起来，d 开头则多记一个缩进递归，] 结尾则将存贮的files排序输出并结束该字典。

```python
# 夏天明，元培学院
from sys import exit

class dir:
    def __init__(self, dname):
        self.name = dname
        self.dirs = []
        self.files = []
    
    def getGraph(self):
        g = [self.name]
        for d in self.dirs:
            subg = d.getGraph()
            g.extend(["|     " + s for s in subg])
        for f in sorted(self.files):
            g.append(f)
        return g

n = 0
while True:
    n += 1
    stack = [dir("ROOT")]
    while (s := input()) != "*":
        if s == "#": exit(0)
        if s[0] == 'f':
            stack[-1].files.append(s)
        elif s[0] == 'd':
            stack.append(dir(s))
            stack[-2].dirs.append(stack[-1])
        else:
            stack.pop()
    print(f"DATA SET {n}:")
    print(*stack[0].getGraph(), sep='\n')
    print()
```



钟明衡-23-物理学院：写了一个File类，默认名称为'ROOT'，当出现file就存储，dir就建一个新的类接收，直到']'结束

输出时，先输出原顺序的dir，要在每一行以前加上'|     '，这个可以自动嵌套，然后输出排序了的file。

OJ上大部分树题目其实不用类也能做，用defaultdict存索引还是很方便的。但是，递归思想在很多方面都可以使用，是一种省事而且优美的方法。我对递归的理解是，规定一个base case，然后告诉计算机大概要做什么，让它自己用同一套逻辑去算就好了。包括写类的时候，指针指向同样是这个类的元素，也是一种递归想法。举个例子，建树的时候，把左右子节点都看成一棵新的树，大概就是这么个想法。这种思路在各种场景下都有不错的表现。

```python
class File:
    def __init__(self):
        self.name = 'ROOT'
        self.files = []
        self.dirs = []

    def __str__(self):
        return '\n'.join([self.name]+['|     '+s for d in self.dirs for s in str(d).split('\n')]+sorted(self.files))

    def build(self, parent, s):
        if s[0] == 'f':
            parent.files.append(s)
        else:
            dir = File()
            dir.name = s
            parent.dirs.append(dir)
            while True:
                s = input()
                if s == ']':
                    break
                dir.build(dir, s)


x = 0
while True:
    s = input()
    if s == '#':
        break
    x += 1
    root = File()
    while s != '*':
        root.build(root, s)
        s = input()
    print('DATA SET %d:' % x)
    print(root, end='\n\n')

```





```python
# 蒋子轩23工学院
def print_structure(node,indent=0):
	#indent为缩进个数
    prefix='|     '*indent
    print(prefix+node['name'])
    for dir in node['dirs']:
    	#若为目录继续递归
        print_structure(dir,indent+1)
    for file in sorted(node['files']):
    	#若为文件直接打印
        print(prefix+file)
dataset=1
datas=[]
temp=[]
#读取输入
while True:
    line=input()
    if line=='#':
        break
    if line=='*':
        datas.append(temp)
        temp=[]
    else:
        temp.append(line)
for data in datas:
    print(f'DATA SET {dataset}:')
    root={'name':'ROOT','dirs':[],'files':[]}
    stack=[root]
    #用栈实现后进先出
    for line in data:
        if line[0]=='d':
            dir={'name':line,'dirs':[],'files':[]}
            stack[-1]['dirs'].append(dir)
            stack.append(dir)
        elif line[0]=='f':
            stack[-1]['files'].append(line)
        else:  #某种结束符
            stack.pop()
    print_structure(root)
    if dataset<len(datas):
        print()
    dataset+=1
```



罗熙佑思路：根据树的递归定义操作。这题之前拿 C++ 做过，当时几乎不会树，靠递归手搓，代码又臭又长，写起来也容易错，当时de了很久bug;现在系统学了树就感觉蛮简单了，可以用`map`+`lambda`函数来处理了缩进，其他地方模版性比较强；代码写了详细注释，可供参考。

```Python
# 罗熙佑
from typing import List
class Node:
    def __init__(self, v) -> None:
        self.v = v # value
        self.c = [] # children

def build_tree(s: List[str]) -> Node:
    '''
    从s中读取数据并建树，返回root；
    s是一组输入数据的列表，原输入的每一行是一个列表一个元素
    s: sequence
    '''
    # base case 1: s读完，说明本组数据结束
    if not s:
        return None
    
    t = s.pop() # token

    # base case 2: 读到]代表目录结束，读到f代表叶节点
    if t == ']':
        return None
    if t[0] == 'f':
        return Node(t)

    # step1: build root
    root = Node(t)

    # step2: build subtree and connect
    while nd := build_tree(s): # 如果建出None就不加，用海牛让代码更简洁
        root.c.append(nd)

    # step3: sort
    dire = [n for n in root.c if n.v[0] == 'd']
    file = [n for n in root.c if n.v[0] == 'f']
    file.sort(key=lambda u: u.v) # 把文件按字典序排序
    root.c = dire + file

    #step 4: return
    return root

def traversal(root: Node) -> List[str]:
    '''把以root为根的树的待打印内容放到sq列表，每一行内容作为一个元素；并返回sq'''
    sq = [] # sequence

    # step1: add root
    sq.append(root.v)

    # step2: add subtree, from left to right
    for n in root.c: # node
        # 2.1: if node is leaf, i.e. file
        if n.v[0] == 'f':
            sq.append(n.v)
        # 2.2: 
        else: # if node is dir, add indentation for every node of the subtree whose root is the 'dir'
            sq.extend(map(lambda x: "|     " + x, traversal(n)))

    # step3: return
    return sq
            
def main() -> None:
    # read input data
    s = ['ROOT']
    idx = []
    cnt = 0
    while True:
        t = input()
        if t == '#':
            break
        s.append(t)
        cnt += 1
        if t == '*':
            idx.append(cnt)
            s[-1] = 'ROOT'

    for i in range(1, len(idx) + 1):
        print(f"DATA SET {i}:")
        root = build_tree(s[idx[i - 1] - 1: idx[i - 2] - 1 if i > 1 else None: -1])
        print('\n'.join(traversal(root)), end='\n\n')

if __name__ == "__main__":
    main()
```





## 02788: 二叉树（2）

http://cs101.openjudge.cn/practice/02788/



![img](http://media.openjudge.cn/images/2756_1.jpg)

如上图所示，由正整数1，2，3……组成了一颗二叉树。我们已知这个二叉树的最后一个结点是n。现在的问题是，结点m所在的子树中一共包括多少个结点。

比如，n = 12，m = 3那么上图中的结点13，14，15以及后面的结点都是不存在的，结点m所在子树中包括的结点有3，6，7，12，因此结点m的所在子树中共有4个结点。

**输入**

输入数据包括多行，每行给出一组测试数据，包括两个整数m，n (1 <= m <= n <= 1000000000)。最后一组测试数据中包括两个0，表示输入的结束，这组数据不用处理。

**输出**

对于每一组测试数据，输出一行，该行包含一个整数，给出结点m所在子树中包括的结点的数目。

样例输入

```
3 12
0 0
```

样例输出

```
4
```



利用每层节点的编号区间来计算子树中存在的节点数量。这样既能保证效率，也能避免超时问题：

```python
import sys

def count_subtree_nodes(m, n):
    count = 0
    left = m
    right = m
    # 每层的节点编号范围为 [left, right]
    while left <= n:
        count += min(n, right) - left + 1
        left *= 2
        right = right * 2 + 1
    return count

def main():
    input_stream = sys.stdin
    for line in input_stream:
        m, n = map(int, line.split())
        if m == 0 and n == 0:
            break
        print(count_subtree_nodes(m, n))

if __name__ == '__main__':
    main()
```

说明

1. **每层计算**：
   - 对于子树中每一层，节点编号的区间为 `[left, right]`，其中初始 `left = m` 和 `right = m`。
   - 下一层的区间为 `[left*2, right*2+1]`。
2. **节点数计算**：
   - 对于每一层，实际存在的节点数为 `min(n, right) - left + 1`，确保不超过 `n`。

这种方法的时间复杂度约为 O(log n)，能高效解决问题，并且避免了内存和时间上的超限问题。





> 完全二叉树
>
> ​           每个结点
>
> 左孩子 2\*i          右孩子 2\*i+1
>

 

设置left right 按照完全二叉树每一行去遍历算就行，算最左结点、最右结点.

```python
while True:
    m, n = map(int, input().split())
    if m == 0 and n == 0:
        break

    ans = 1
    num = 1  # Number of nodes in the first level of the complete binary tree
    left = 2 * m
    right = 2 * m + 1
    while right <= n:
        num = num * 2
        ans += num
        left = left * 2
        right = right * 2 + 1

    if left <= n:
        ans += n - left + 1

    print(ans)
```





```python
import sys

def count_subtree_nodes(m, n):
    left_bound, right_bound, depth = 2 * m, 2 * m + 1, 1

    while left_bound <= n:
        if right_bound < n:
            depth += 1
            left_bound *= 2
            right_bound = 2 * right_bound + 1
        else:
            return (1 << depth) + (n - left_bound)

    return (1 << depth) - 1

def main():
    input_data = sys.stdin.read().strip().split("\n")
    for line in input_data:
        m, n = map(int, line.split())
        if m == 0 and n == 0:
            break
        print(count_subtree_nodes(m, n))

if __name__ == "__main__":
    main()
```





```python
#23n2300010763
def compute(m,n):
    cnt = 1
    while m*cnt<=n:
        cnt *= 2
    return min(n-(m-1)*(cnt//2),cnt-1)


while True:
    m,n = map(int,input().split())
    if not m and not n:
        break
    print(compute(m,n))
```



```c++
#include<iostream>
using namespace std;

int main()
{
    int m,n,sum;
    while(scanf("%d%d",&m,&n)==2 && (m||n)) {
        sum=0;
        int d=1;
        while(1) {
            if(m<=n) {
                sum+=d;
                m=2*m+1;
                d=d*2;  
            }
            else{ 
                if(m-n<d)
                    sum = sum+d-(m-n);
                break;
            }
        }

        printf("%d\n",sum);
    }
}
```





## 02945: 拦截导弹

http://cs101.openjudge.cn/dsapre/02945/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional problems 部分



数据不大可以用搜索（其实主要是不想写动规）

```python
# 2300012148熊奕凯
ans=0
a=int(input())
l=list(map(int,input().split()))
def dfs(cnt, pos):
    global ans
    cnt+=1
    if cnt>ans:
        ans=cnt
    if pos == a-1:
        return
    else:
        for i in range(pos+1,a):
            if l[i]<=l[pos]:
                dfs(cnt,i)
        return
for i in range(0,a):
    dfs(0,i)
print(ans)
```



## 03151: Pots

bfs, http://cs101.openjudge.cn/practice/03151/

You are given two pots, having the volume of **A** and **B** liters respectively. The following operations can be performed:

1. FILL(i)    fill the pot **i** (1 ≤ **i** ≤ 2) from the tap;

2. DROP(i)   empty the pot **i** to the drain;

3. POUR(i,j)  pour from pot **i** to pot **j**; after this operation either the pot **j** is full (and there may be some water left in the pot **i**), or the pot **i** is empty (and all its contents have been moved to the pot **j**).

   

Write a program to find the shortest possible sequence of these operations that will yield exactly **C** liters of water in one of the pots.

**输入**

On the first and only line are the numbers **A**, **B**, and **C**. These are all integers in the range from 1 to 100 and **C**≤max(**A**,**B**).

**输出**

The first line of the output must contain the length of the sequence of operations **K**. The following **K** lines must each describe one operation. If there are several sequences of minimal length, output any one of them. If the desired result can’t be achieved, the first and only line of the file must contain the word ‘**impossible**’.

样例输入

```
3 5 4
```

样例输出

```
6
FILL(2)
POUR(2,1)
DROP(1)
POUR(2,1)
FILL(2)
POUR(2,1)
```



```python
# 23生科崔灏梵
def bfs(A, B, C):
    start = (0, 0)
    visited = set()
    visited.add(start)
    queue = [(start, [])]

    while queue:
        (a, b), actions = queue.pop(0)

        if a == C or b == C:
            return actions

        next_states = [(A, b), (a, B), (0, b), (a, 0), (min(a + b, A),\
                max(0, a + b - A)), (max(0, a + b - B), min(a + b, B))]

        for i in next_states:
            if i not in visited:
                visited.add(i)
                new_actions = actions + [get_action(a, b, i)]
                queue.append((i, new_actions))

    return ["impossible"]


def get_action(a, b, next_state):
    if next_state == (A, b):
        return "FILL(1)"
    elif next_state == (a, B):
        return "FILL(2)"
    elif next_state == (0, b):
        return "DROP(1)"
    elif next_state == (a, 0):
        return "DROP(2)"
    elif next_state == (min(a + b, A), max(0, a + b - A)):
        return "POUR(2,1)"
    else:
        return "POUR(1,2)"


A, B, C = map(int, input().split())
solution = bfs(A, B, C)

if solution == ["impossible"]:
    print(solution[0])
else:
    print(len(solution))
    for i in solution:
        print(i)
```



## 03447: 银河贸易问题

bfs, http://cs101.openjudge.cn/practice/03447/

随着一种称为“￥”的超时空宇宙飞船的发明，一种叫做“￥￥”的地球与遥远的银河之间的商品进出口活动应运而生。￥￥希望从PluralZ星团中的一些银河进口商品，这些银河中的行星盛产昂贵的商品和原材料。初步的报告显示：
（1） 每个银河都包含至少一个和最多26个行星，在一个银河的每个行星用A~Z中的一个字母给以唯一的标识。
（2） 每个行星都专门生产和出口一种商品，在同一银河的不同行星出口不同的商品。
（3） 一些行星之间有超时空货运航线连接。如果行星A和B相连，则它们可以自由贸易；如果行星C与B相连而不与A相连，则A和C之间仍可通过B进行贸易，不过B要扣留5%的货物作为通行费。一般来说，只要两个行星之间可以通过一组货运航线连接，他们就可以进行贸易，不过每个中转站都要扣留5%的货物作为通行费。
（4） 在每个银河至少有一个行星开放一条通往地球的￥航线。对商业来说，￥航线和其他星际航线一样。
￥￥已经对每个行星的主要出口商品定价（不超过10的正实数），数值越高，商品的价值越高。在本地市场，越值钱的商品获利也越高。问题是要确定如果要考虑通行费是，哪个行星的商品价值最高。

**输入**

输入包含若干银河的描述。每个银河的描述开始的第1行是一个整数n，表示银河的行星数。接下来的n行每行包括一个行星的描述，即：
（1） 一行用以代表行星的字母；
（2） 一个空格；
（3） 以d.dd的形式给出该行星的出口商品的价值；
（4） 一个空格；
（5） 一个包含字母和（或）字符“*”的字符串；字母表示一条通往该行星的货运航线；“*”表示该行星向地球开放￥货运航线。

**输出**

对每个银河的描述，输出一个字母P表示在考虑通行费的前提下，行星P具有最高价值的出口商品。如果用有最高价值的出口商品的行星多于一个，只需输出字母序最小的那个行星。

样例输入

```
5
E 0.01 *A
D 0.01 A*
C 0.01 *A
A 1.00 EDCB
B 0.01 A*
```

样例输出

```
A
```



注意不要字典覆盖，多条路从S到D，价格不一样。

```python
# 肖添天
from collections import defaultdict, deque

n = int(input())
graph = defaultdict(set)
to_earth = set()
price = {}
for i in range(n):
    a, b, c = input().split()
    b = float(b)
    price[a] = b if a not in price else max(price[a], b)
    for x in c:
        if x == "*":
            to_earth.add(a)
        else:
            graph[a].add(x)
            graph[x].add(a)

def bfs(start):
    Q = deque([start])
    visited = set()
    visited.add(start)
    cnt = 0
    while Q:
        l = len(Q)
        for _ in range(l):
            f = Q.popleft()
            if f in to_earth:
                return price[start] * (0.95 ** cnt)
            for x in graph[f]:
                if x not in visited:
                    Q.append(x)
                    visited.add(x)
        cnt += 1
    return 0


ans = []
for planet in price.keys():
    ans.append((bfs(planet), planet))

ans.sort(key=lambda x: [-x[0], x[1]])
print(ans[0][1])
```





## M03424: Candies

dijkstra, http://cs101.openjudge.cn/practice/03424/

During the kindergarten days, flymouse was the monitor of his class. Occasionally the head-teacher brought the kids of flymouse’s class a large bag of candies and had flymouse distribute them. All the kids loved candies very much and often compared the numbers of candies they got with others. A kid A could had the idea that though it might be the case that another kid B was better than him in some aspect and therefore had a reason for deserving more candies than he did, he should never get a certain number of candies fewer than B did no matter how many candies he actually got, otherwise he would feel dissatisfied and go to the head-teacher to complain about flymouse’s biased distribution.

snoopy shared class with flymouse at that time. flymouse always compared the number of his candies with that of snoopy’s. He wanted to make the difference between the numbers as large as possible while keeping every kid satisfied. Now he had just got another bag of candies from the head-teacher, what was the largest difference he could make out of it?

**输入**

The input contains a single test cases. The test cases starts with a line with two integers *N* and *M*not exceeding 30 000 and 150 000 respectively. *N* is the number of kids in the class and the kids were numbered 1 through *N*. snoopy and flymouse were always numbered 1 and *N*. Then follow *M*lines each holding three integers *A*, *B* and *c* in order, meaning that kid *A* believed that kid *B*should never get over *c* candies more than he did.

**输出**

Output one line with only the largest difference desired. The difference is guaranteed to be finite.

样例输入

```
2 2
1 2 5
2 1 4
```

样例输出

```
5
```

提示

32-bit signed integer type is capable of doing all arithmetic.

来源

POJ Monthly--2006.12.31, Sempr



你的理解是正确的：

- 将每个不等式

  $x_B−x_A≤c$

  看作有向边 A→B，权重为 c。

- 对任意一条从 1 到 N 的路径 $1=v_0→v_1→⋯→v_k=N$，都有

  $x_N−x_1≤ \sum_{i=0}^{k−1}c_{v_i v_{i+1}}$.

- 要使 $x_N−x_1$ 尽可能大，就要取所有路径权重之中的最小值──也就是在这张有向图上求从 1 到 N 的最短路。

因为所有 $c≥0$，可以直接用 Dijkstra 算法。下面给出一个 $O(Mlog⁡N)$ 的实现：

```python
import sys
import threading
import heapq

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        A, B, c = map(int, input().split())
        graph[A].append((B, c))
    INF = 10**30
    dist = [INF] * (N+1)
    dist[1] = 0
    pq = [(0, 1)]  # (当前距离, 节点)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == N:
            break    # 提前退出
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    # 输出从 1 到 N 的最短路距离，即为最大可实现的 x_N - x_1
    print(dist[N])

if __name__ == "__main__":
    threading.Thread(target=main).start()
```

**说明：**

1. 我们把每条约束 xB≤xA+c 转化为图中的一条边 A→B，权重为 c。
2. 用 Dijkstra 从节点 1 出发，算出到节点 N 的最短距离 `dist[N]`。
3. 这个最短距离恰好就是在所有满足约束的分配方案中，xN−x1 能达到的最大值。



参考：https://blog.csdn.net/Maxwei_wzj/article/details/60464314

题目大意：幼儿园一个班里有N个小朋友（标号为1~N），一个小朋友flymouse（为N号）被校长指定去发糖，有M个条件，每个条件三个参数A,B,c，表示小朋友A不希望小朋友B有比他多超过c个的糖，班里还有另一个小朋友snoopy（为1号），flymouse希望自己得到的糖果比snoopy的尽量多，求最大的差值。

做法：这里引入一个叫<mark>差分约束系统</mark>的东西，大概就是给定一系列这样形式的不等式：`xi-xj<=bk`，然后求某两个xa和xb的差的最大值，即`max(xa-xb)`。正确的方法是，如果存在一个`xi-xj<=bk`这样的不等式，就从j引一条指向i的边权为bk的有向边，这样就可以构成一个有向图，然后求`max(xa-xb)`就是求从b到a的最短路径。为什么呢？因为我们看任意一条简单路径：b,s1,s2,...,sn,a，其中相邻两点间边权依次为b0,b1,...,bn，所以`xs1-xb<=b0,xs2-xs1<=b1,...,xa-xsn<=bn`,所以`xa-xb=(xa-xsn)+...+(xs2-xs1)+(xs1-xb)<=b0+b1+...+bn`，所以我们可以得到`xa-xb`必定不超过任意从b到a的简单路径上边权的和，也就是说任何一条路径都是一个上界，所以要求最大值也就是求最小的上界，也就是求最短路了。

> 差分约束系统是一个整体系统，我们需要找到所有约束都满足的前提下，目标变量差值的最大值。

而这一题模型比较简单，构图很容易，对于每个条件直接连A->B，边权为c即可，然后求从1到N的最短路。



标准的Dijkstra就可以。

```python
import heapq

def dijkstra(N, G, start):
    INF = float('inf')
    dist = [INF] * (N + 1)  # 存储源点到各个节点的最短距离
    dist[start] = 0  # 源点到自身的距离为0
    pq = [(0, start)]  # 使用优先队列，存储节点的最短距离
    while pq:
        d, node = heapq.heappop(pq)  # 弹出当前最短距离的节点
        if d > dist[node]:  # 如果该节点已经被更新过了，则跳过
            continue
        for neighbor, weight in G[node]:  # 遍历当前节点的所有邻居节点
            new_dist = dist[node] + weight  # 计算经当前节点到达邻居节点的距离
            if new_dist < dist[neighbor]:  # 如果新距离小于已知最短距离，则更新最短距离
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))  # 将邻居节点加入优先队列
    return dist



N, M = map(int, input().split())
G = [[] for _ in range(N + 1)]  # 图的邻接表表示
for _ in range(M):
    s, e, w = map(int, input().split())
    G[s].append((e, w))


start_node = 1  # 源点
shortest_distances = dijkstra(N, G, start_node)  # 计算源点到各个节点的最短距离
print(shortest_distances[-1])  # 输出结果
```



```python
import heapq

class Edge:
    def __init__(self, k=0, w=0):
        self.k, self.w = k, w  # 有向边的终点和边权值，或当前k到源点的距离

    def __lt__(self, other):
        return self.w < other.w


def dijkstra(N, G):
    bUsed = [False] * (N + 1)  # bUsed[i]为True表示源到i的最短路已经求出
    INF = float('inf')
    pq = []
    heapq.heappush(pq, Edge(1, 0))  # 源点是1号点,1号点到自己的距离是0
    while pq:
        p = heapq.heappop(pq)
        if bUsed[p.k]:  # 已经求出了最短路
            continue
        bUsed[p.k] = True
        if p.k == N:  # 因只要求1-N的最短路，所以要break
            break
        for edge in G[p.k]:
            if not bUsed[edge.k]:
                heapq.heappush(pq, Edge(edge.k, p.w + edge.w))
    return p.w


N, M = map(int, input().split())
G = [[] for _ in range(N + 1)]
for _ in range(M):
    s, e, w = map(int, input().split())
    G[s].append(Edge(e, w))

shortest_distance = dijkstra(N, G)
print(shortest_distance)

```





## 03720: 文本二叉树

http://cs101.openjudge.cn/practice/03720/



![img](https://raw.githubusercontent.com/GMyhf/img/main/img/202401302229085.jpg)
如上图，一棵每个节点都是一个字母，且字母互不相同的二叉树，可以用以下若干行文本表示:



```
A
-B
--*
--C
-D
--E
---*
---F
```



在这若干行文本中：

1) 每个字母代表一个节点。该字母在文本中是第几行，就称该节点的行号是几。根在第1行
2) 每个字母左边的'-'字符的个数代表该结点在树中的层次（树根位于第0层）
3) 若某第 i 层的非根节点在文本中位于第n行，则其父节点必然是第 i-1 层的节点中，行号小于n,且行号与n的差最小的那个
4) 若某文本中位于第n行的节点(层次是i) 有两个子节点，则第n+1行就是其左子节点，右子节点是n+1行以下第一个层次为i+1的节点
5) 若某第 i 层的节点在文本中位于第n行，且其没有左子节点而有右子节点，那么它的下一行就是 i+1个'-' 字符再加上一个 '*' 



给出一棵树的文本表示法，要求输出该数的前序、后序、中序遍历结果

**输入**

第一行是树的数目 n

接下来是n棵树，每棵树以'0'结尾。'0'不是树的一部分
每棵树不超过100个节点

**输出**

对每棵树，分三行先后输出其前序、后序、中序遍历结果
两棵树之间以空行分隔

样例输入

```
2
A
-B
--*
--C
-D
--E
---*
---F
0
A
-B
-C
0
```

样例输出

```
ABCDEF
CBFEDA
BCAEFD

ABC
BCA
BAC
```

来源: Guo Wei





```python
class Node:
    def __init__(self, x, depth):
        self.x = x
        self.depth = depth
        self.lchild = None
        self.rchild = None

    def preorder_traversal(self):
        nodes = [self.x]
        if self.lchild and self.lchild.x != '*':
            nodes += self.lchild.preorder_traversal()
        if self.rchild and self.rchild.x != '*':
            nodes += self.rchild.preorder_traversal()
        return nodes

    def inorder_traversal(self):
        nodes = []
        if self.lchild and self.lchild.x != '*':
            nodes += self.lchild.inorder_traversal()
        nodes.append(self.x)
        if self.rchild and self.rchild.x != '*':
            nodes += self.rchild.inorder_traversal()
        return nodes

    def postorder_traversal(self):
        nodes = []
        if self.lchild and self.lchild.x != '*':
            nodes += self.lchild.postorder_traversal()
        if self.rchild and self.rchild.x != '*':
            nodes += self.rchild.postorder_traversal()
        nodes.append(self.x)
        return nodes


def build_tree():
    n = int(input())
    for _ in range(n):
        tree = []
        stack = []
        while True:
            s = input()
            if s == '0':
                break
            depth = len(s) - 1
            node = Node(s[-1], depth)
            tree.append(node)

            # Finding the parent for the current node
            while stack and tree[stack[-1]].depth >= depth:
                stack.pop()
            if stack:  # There is a parent
                parent = tree[stack[-1]]
                if not parent.lchild:
                    parent.lchild = node
                else:
                    parent.rchild = node
            stack.append(len(tree) - 1)

        # Now tree[0] is the root of the tree
        yield tree[0]


# Read each tree and perform traversals
for root in build_tree():
    print("".join(root.preorder_traversal()))
    print("".join(root.postorder_traversal()))
    print("".join(root.inorder_traversal()))
    print()

```



```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def parse_tree(lines):
    # last_node_at_level[i] 记录最近一个深度为 i 的真实节点
    last_node_at_level = {}
    # expect_right[i] 标记深度 i 的节点下一个真实节点应该挂在它的 right
    expect_right = {}

    root = None

    for line in lines:
        # 计算深度
        level = 0
        while line[level] == '-':
            level += 1
        val = line[level]

        if val == '*':
            # 标记：depth = level-1 的节点左子为空，下一个真实节点挂到它的 right
            expect_right[level-1] = True
            # 不创建实际节点
            continue

        node = Node(val)
        # 第一个节点当作 root
        if level == 0:
            root = node

        # 如果不是根，就找父节点
        if level > 0:
            parent = last_node_at_level[level-1]
            # 如果父节点标记了 expect_right，则挂到 right
            if expect_right.get(level-1, False):
                parent.right = node
                expect_right[level-1] = False  # 重置标记
            else:
                # 否则，先挂左，再挂右
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node

        # 更新同层最新节点
        last_node_at_level[level] = node

    return root

def preorder(root):
    return '' if not root else root.val + preorder(root.left) + preorder(root.right)

def inorder(root):
    return '' if not root else inorder(root.left) + root.val + inorder(root.right)

def postorder(root):
    return '' if not root else postorder(root.left) + postorder(root.right) + root.val

if __name__ == '__main__':
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    idx = 1

    for ti in range(n):
        # 读取一棵树的所有行
        lines = []
        while data[idx] != '0':
            lines.append(data[idx])
            idx += 1
        idx += 1  # 跳过 '0'

        root = parse_tree(lines)
        # 输出三种遍历
        print(preorder(root))
        print(postorder(root))
        print(inorder(root))
        if ti != n-1:
            print()

```



## 04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

栈是常用的一种数据结构，有n个元素在栈顶端一侧等待进栈，栈顶端另一侧是出栈序列。你已经知道栈的操作有两种：push和pop，前者是将一个元素进栈，后者是将栈顶元素弹出。现在要使用这两种操作，由一个操作序列可以得到一系列的输出序列。请你编程求出对于给定的n，计算并输出由操作数序列1，2，…，n，经过一系列操作可能得到的输出序列总数。

**输入**

就一个数n(1≤n≤15)。

**输出**

一个数，即可能输出序列的总数目。

样例输入

```
3
```

样例输出

```
5
```

提示

先了解栈的两种基本操作，进栈push就是将元素放入栈顶，栈顶指针上移一位，等待进栈队列也上移一位，出栈pop是将栈顶元素弹出，同时栈顶指针下移一位。
　　 用一个过程采模拟进出栈的过程，可以通过循环加递归来实现回溯：重复这样的过程，如果可以进栈则进一个元素，如果可以出栈则出一个元素。就这样一个一个地试探下去，当出栈元素个数达到n时就计数一次(这也是递归调用结束的条件)。



这是一个经典的栈操作模拟问题。我们需要统计对于从 `1` 到 `n` 的数字，所有可能的出栈序列数量。这个问题的本质就是 **栈的合法出栈序列个数问题**。

这个数量恰好是 **第 n 个卡特兰数（Catalan Number）**，其定义为：
$$
C_n = \frac{1}{n+1} \binom{2n}{n}
$$


​				
​					

或者通过递归模拟也可以得到。下面我们给出递归+回溯的写法，逐个模拟进栈和出栈的操作。

------

✅ Python 代码如下：

```python
def count_sequences(n):
    def dfs(push_num, stack, popped):
        nonlocal count
        # 如果已经弹出了 n 个数，说明这个出栈序列是合法的
        if popped == n:
            count += 1
            return
        # 尝试进栈：如果还有数字没进栈
        if push_num <= n:
            stack.append(push_num)
            dfs(push_num + 1, stack, popped)
            stack.pop()
        # 尝试出栈：如果栈不空
        if stack:
            top = stack.pop()
            dfs(push_num, stack, popped + 1)
            stack.append(top)

    count = 0
    dfs(1, [], 0)
    return count

# 读取输入
n = int(input())
print(count_sequences(n))
```

✅ 拓展：卡特兰数法（更高效）

你也可以直接用卡特兰数公式来快速计算：

```python
import math

def catalan_number(n):
    return math.comb(2 * n, n) // (n + 1)

n = int(input())
print(catalan_number(n))
```

两种方法都可以，递归方式更直观模拟操作，适合理解；卡特兰数方法更高效，适合大 n（比如 n = 15）时使用。



```python
"""
尝试所有可能的进栈和出栈顺序，并统计满足条件的序列数量。
当进栈次数和出栈次数都达到n时，即得到一个有效的出栈序列，计数器加1
"""
def count_stack_sequences(n):
    def backtrack(open_count, close_count):
        if open_count == n and close_count == n:
            return 1
        total_count = 0
        if open_count < n:
            total_count += backtrack(open_count + 1, close_count)
        if close_count < open_count:
            total_count += backtrack(open_count, close_count + 1)
        return total_count

    return backtrack(0, 0)

if __name__ == "__main__":
    n = int(input())
    result = count_stack_sequences(n)
    print(result)

```



加lru_cache就快了，相当于dp。n=15,也是瞬间出。

```python
"""
尝试所有可能的进栈和出栈顺序，并统计满足条件的序列数量。
当进栈次数和出栈次数都达到n时，即得到一个有效的出栈序列，计数器加1
"""

from functools import lru_cache

def count_stack_sequences(n):
    
    @lru_cache(None)
    def backtrack(open_count, close_count):
        if open_count == n and close_count == n:
            return 1
        total_count = 0
        if open_count < n:
            total_count += backtrack(open_count + 1, close_count)
        if close_count < open_count:
            total_count += backtrack(open_count, close_count + 1)
        return total_count

    return backtrack(0, 0)

if __name__ == "__main__":
    n = int(input())
    result = count_stack_sequences(n)
    print(result)
```





##  04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

定义一个数组，初始化为空。在数组上执行两种操作：

1、增添1个元素，把1个新的元素放入数组。

2、输出并删除数组中最小的数。

使用堆结构实现上述功能的高效算法。

**输入**

第一行输入一个整数n，代表操作的次数。
每次操作首先输入一个整数type。
当type=1，增添操作，接着输入一个整数u，代表要插入的元素。
当type=2，输出删除操作，输出并删除数组中最小的元素。
1<=n<=100000。

**输出**

每次删除操作输出被删除的数字。

样例输入

```
4
1 5
1 1
1 7
2
```

样例输出

```
1
```

提示

每组测试数据的复杂度为O(nlogn)的算法才能通过本次，否则会返回TLE(超时)
需要使用最小堆结构来实现本题的算法



练习自己写个BinaryHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。

```python
class BinaryHeap:
    def __init__(self):
        self._heap = []

    def _perc_up(self, i):
        while (i - 1) // 2 >= 0:
            parent_idx = (i - 1) // 2
            if self._heap[i] < self._heap[parent_idx]:
                self._heap[i], self._heap[parent_idx] = (
                    self._heap[parent_idx],
                    self._heap[i],
                )
            i = parent_idx

    def insert(self, item):
        self._heap.append(item)
        self._perc_up(len(self._heap) - 1)

    def _perc_down(self, i):
        while 2 * i + 1 < len(self._heap):
            sm_child = self._get_min_child(i)
            if self._heap[i] > self._heap[sm_child]:
                self._heap[i], self._heap[sm_child] = (
                    self._heap[sm_child],
                    self._heap[i],
                )
            else:
                break
            i = sm_child

    def _get_min_child(self, i):
        if 2 * i + 2 > len(self._heap) - 1:
            return 2 * i + 1
        if self._heap[2 * i + 1] < self._heap[2 * i + 2]:
            return 2 * i + 1
        return 2 * i + 2

    def delete(self):
        self._heap[0], self._heap[-1] = self._heap[-1], self._heap[0]
        result = self._heap.pop()
        self._perc_down(0)
        return result

    def heapify(self, not_a_heap):
        self._heap = not_a_heap[:]
        i = len(self._heap) // 2 - 1    # 超过中点的节点都是叶子节点
        while i >= 0:
            #print(f'i = {i}, {self._heap}')
            self._perc_down(i)
            i = i - 1



n = int(input().strip())
bh = BinaryHeap()
for _ in range(n):
    inp = input().strip()
    if inp[0] == '1':
        bh.insert(int(inp.split()[1]))
    else:
        print(bh.delete())
```



## 04079: 二叉搜索树

http://cs101.openjudge.cn/dsapre/04079/

二叉搜索树在动态查表中有特别的用处，一个无序序列可以通过构造一棵二叉搜索树变成一个有序序列，构造树的过程即为对无序序列进行排序的过程。每次插入的新的结点都是二叉搜索树上新的叶子结点，在进行插入操作时，不必移动其它结点，只需改动某个结点的指针，由空变为非空即可。

  这里，我们想探究二叉树的建立和序列输出。

**输入**

只有一行，包含若干个数字，中间用空格隔开。（数字可能会有重复）

**输出**

输出一行，对输入数字建立二叉搜索树后进行前序周游的结果。

样例输入

```
41 467 334 500 169 724 478 358 962 464 705 145 281 827 961 491 995 942 827 436 
```

样例输出

```
41 467 334 169 145 281 358 464 436 500 478 491 724 705 962 827 961 942 995 
```



要解决这个问题，首先需要了解二叉搜索树（Binary Search Tree，BST）的基本属性和操作。二叉搜索树是一种特殊的二叉树，满足以下性质：

- 每个节点的键值大于其左子树上任意节点的键值。
- 每个节点的键值小于其右子树上任意节点的键值。
- 左右子树也分别为二叉搜索树。



在BST上进行前序遍历（Preorder Traversal）可以按照以下步骤进行：

1. 访问根节点。
2. 递归地对左子树进行前序遍历。
3. 递归地对右子树进行前序遍历。

如果使用迭代而不是递归，您可以使用栈（Stack）数据结构来实现前序遍历。



```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def insert_into_bst(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    elif val > root.val:
        root.right = insert_into_bst(root.right, val)
    return root

def preorder_traversal(root):
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right) if root else []

def preorderTraversal(root):
    if root is None:
        return []

    stack = []
    result = []
    stack.append(root)

    while stack:
        node = stack.pop()
        result.append(node.val)

        # 先将右子节点入栈，再将左子节点入栈
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result

# 读取输入并转换成整数列表
numbers = list(map(int, input().split()))

# 构造二叉搜索树
bst_root = None
for num in numbers:
    bst_root = insert_into_bst(bst_root, num)

# 前序遍历二叉搜索树并输出
#print(' '.join(map(str, preorder_traversal(bst_root))))
print(' '.join(map(str, preorderTraversal(bst_root))))
```



```c++
#include <iostream>

using namespace std;

struct Node {
	int num;
	Node *left, *right;
	
	Node(int n) : num(n), left(NULL), right(NULL) {}
};

void insert(Node *&root, int num) {
	if (root == NULL) {
		root = new Node(num);
		return;
	}
	
	if (num == root->num) {
		return;
	}
	
	if (num < root->num) {
		insert(root->left, num);
	} else {
		insert(root->right, num);
	}
}

void preorder(Node *root) {
	if (!root)
		return;
	cout << root->num << " ";
	preorder(root->left);
	preorder(root->right);
}

void deleteTree(Node *root) {
	if (!root)
		return;
	deleteTree(root->left);
	deleteTree(root->right);
	delete root;
}

int main() {
	Node *root = NULL;
	int num;
	while (cin >> num) {
		insert(root, num);
	}
	
	preorder(root);
	cout << endl;
	
	deleteTree(root);
	return 0;
}
```



## 04080:Huffman编码树

http://cs101.openjudge.cn/practice/04080/

构造一个具有n个外部节点的扩充二叉树，每个外部节点Ki有一个Wi对应，作为该外部节点的权。使得这个扩充二叉树的叶节点带权外部路径长度总和最小：

​                   Min( W1 * L1 + W2 * L2 + W3 * L3 + … + Wn * Ln)

Wi:每个节点的权值。

Li:根节点到第i个外部叶子节点的距离。

编程计算最小外部路径长度总和。

输入

第一行输入一个整数n，外部节点的个数。第二行输入n个整数，代表各个外部节点的权值。
2<=N<=100

输出

输出最小外部路径长度总和。

样例输入

```
4
1 1 3 5
```

样例输出

```
17
```





这个问题其实是一个经典的 **最优二叉树** 构造问题，也叫做 **霍夫曼编码（Huffman Coding）**。我们要构造一棵 **扩充二叉树**（即每个非叶子节点都有两个子节点），使得所有 **叶子节点的带权路径长度和最小**。

---

✅ 思路简述（霍夫曼算法）：

我们通过以下贪心策略构造一棵最优二叉树：

1. 将所有权值作为初始节点，放入一个最小堆中。
2. 重复执行以下操作直到只剩一个节点：
   - 从堆中取出两个最小权值节点 `a` 和 `b`
   - 合并为一个新节点，权值为 `a + b`
   - 把这个新节点的权值加入堆
   - 这次合并会产生一个代价：`a + b`，将其加入总路径代价中
3. 最终累加的合并代价即为 **最小外部带权路径长度总和**。

---

✅ 示例解释

输入：

```
4
1 1 3 5
```

构造过程：

- 1 + 1 → 新节点权值 2，总成本 += 2
- 2 + 3 → 新节点权值 5，总成本 += 5
- 5 + 5 → 新节点权值 10，总成本 += 10
- 最终成本 = 2 + 5 + 10 = **17**

---

✅ Python 实现

```python
import heapq

def min_weighted_path_length(n, weights):
    heapq.heapify(weights)
    total = 0
    while len(weights) > 1:
        a = heapq.heappop(weights)
        b = heapq.heappop(weights)
        combined = a + b
        total += combined
        heapq.heappush(weights, combined)
    return total

# 读取输入
n = int(input())
weights = list(map(int, input().split()))
print(min_weighted_path_length(n, weights))
```





```python
from heapq import heapify,heappop,heappush

class Node:
    def __init__(self,val,fre):
        self.val=val
        self.fre=fre
        self.left=None
        self.right=None
    def __lt__(self,other):
        return self.fre<other.fre
    
def HuffmanTree(val_fre):
    while len(val_fre)>1:
        left=heappop(val_fre)
        right=heappop(val_fre)
        merge=Node(None,left.fre+right.fre)
        merge.left=left
        merge.right=right
        heappush(val_fre,merge)
    return val_fre[0]

def calculate(node,depth):
    if node.left==node.right==None:
        return node.fre*depth
    return calculate(node.left,depth+1)+calculate(node.right,depth+1)

n=int(input())
val_fre=[]
fres=list(map(int,input().split()))
for i in range(n):
    val_fre.append(Node(i,fres[i]))
heapify(val_fre)
root=HuffmanTree(val_fre)
ans=calculate(root,0)
print(ans)
```



## 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



我们都知道用“左儿子右兄弟”的方法可以将一棵一般的树转换为二叉树，如：

```
    0                             0
  / | \                          /
 1  2  3       ===>             1
   / \                           \
  4   5                           2
                                 / \
                                4   3
                                 \
                                  5
```

现在请你将一些一般的树用这种方法转换为二叉树，并输出转换前和转换后树的高度。

**输入**

输入是一个由“u”和“d”组成的字符串，表示一棵树的深度优先搜索信息。比如，dudduduudu可以用来表示上文中的左树，因为搜索过程为：0 Down to 1 Up to 0 Down to 2 Down to 4 Up to 2 Down to 5 Up to 2 Up to 0 Down to 3 Up to 0。
你可以认为每棵树的结点数至少为2，并且不超过10000。

**输出**

按如下格式输出转换前和转换后树的高度：
h1 => h2
其中，h1是转换前树的高度，h2是转换后树的高度。

样例输入

```
dudduduudu
```

样例输出

```
2 => 4
```



思路：感觉很久没有建树了，练习了建树。对于输入序列，‘ud’表示兄弟节点，单独‘d’表示子节点，‘u’表示回归到父节点。据此建立二叉树

```python
# 赵思懿，生科
class BinaryTreeNode:
    def __init__(self):
        self.parent = None
        self.left = None
        self.right = None

def tree_height(root):  # 计算二叉树高度
    if not root:
        return -1
    else:
        return max(tree_height(root.left), tree_height(root.right)) + 1

def original_tree_height(arr):  # 原树高度
    height, max_height = 0, 0
    for action in arr:
        if action == 'd':
            height += 1
        elif action == 'u':
            height -= 1
        max_height = max(max_height, height)
    return max_height

def build_binary_tree(arr):  # 根据输入序列建立二叉树
    root = BinaryTreeNode()
    current_node = root
    for action in arr:
        if action == 'd':
            current_node.left = BinaryTreeNode()
            current_node.left.parent = current_node
            current_node = current_node.left
        elif action == 'x':
            current_node.right = BinaryTreeNode()
            current_node.right.parent = current_node.parent
            current_node = current_node.right
        elif action == 'u':
            current_node = current_node.parent
    return root

input_sequence = input().replace('ud', 'x')
binary_tree_root = build_binary_tree(input_sequence)
print(original_tree_height(input_sequence), '=>', tree_height(binary_tree_root))

```





```python
# 23n2300011072(X)
class TreeNode:
    def __init__(self):
        self.children = []
        self.first_child = None
        self.next_sib = None


def build(seq):
    root = TreeNode()
    stack = [root]
    depth = 0
    for act in seq:
        cur_node = stack[-1]
        if act == 'd':
            new_node = TreeNode()
            if not cur_node.children:
                cur_node.first_child = new_node
            else:
                cur_node.children[-1].next_sib = new_node
            cur_node.children.append(new_node)
            stack.append(new_node)
            depth = max(depth, len(stack) - 1)
        else:
            stack.pop()
    return root, depth


def cal_h_bin(node):
    if not node:
         return -1
    return max(cal_h_bin(node.first_child), cal_h_bin(node.next_sib)) + 1


seq = input()
root, h_orig = build(seq)
h_bin = cal_h_bin(root)
print(f'{h_orig} => {h_bin}')

```



一般的树因为是深搜，所以d和u的数量是匹配的，只需要找到down最多的次数就可以；转换成二叉树后的新
树，因为同一层除左孩外其余均是左孩的子节点，所以每次有d时把新高度+1，存储在栈中，每次在原来的基础上增加高度。

```python
"""
calculates the height of a general tree and its corresponding binary tree using the 
"left child right sibling" method. The input is a string composed of "u" and "d", 
representing the depth-first search information of a tree. 

uses a stack to keep track of the heights of the nodes in the binary tree. 
When it encounters a "d", it increases the heights and updates the maximum heights if necessary. 
When it encounters a "u", it decreases the old height and sets the new height to the top of the stack. 
Finally, it returns the maximum heights of the tree and the binary tree in the required format.
"""
def tree_heights(s):
    old_height = 0
    max_old = 0
    new_height = 0
    max_new = 0
    stack = []
    for c in s:
        if c == 'd':
            old_height += 1
            max_old = max(max_old, old_height)

            new_height += 1
            stack.append(new_height)
            max_new = max(max_new, new_height)
        else:
            old_height -= 1

            new_height = stack[-1]
            stack.pop()
    return f"{max_old} => {max_new}"

s = input().strip()
print(tree_heights(s))
```



思路：根据dfs序列递归建树，但并不需要把树转换为实体二叉树，可以递归地求高度(Height)和转换后高度(NewH)。

```python
# 卢卓然 生命科学学院
class Node:
    def __init__(self):
        self.child = []
    
    def getHeight(self):
        return 1 + max([nd.getHeight() for nd in self.child], default=-1)
    
    def getNewH(self):
        return 1 + max([nd.getNewH() + i for i, nd in enumerate(self.child)], default=-1)

def call():
    res = Node()
    while s and s.pop() == 'd':
        res.child.append(call())
    return res
    
s = list(input())[::-1]
root = call()
print(f"{root.getHeight()} => {root.getNewH()}")
```





思路：我写的方法还是分别建起来两个树，其中左儿子右兄弟方法建树还挺麻烦的，但是后来一想，每一个节点都是左儿子右兄弟方法建树，那不就是递归吗。

```python
# 蔡嘉华 物理学院
class TreeNode:
    def __init__(self):
        self.children = []
        self.left = None
        self.right = None

def height1(root):
    if not root:
        return -1
    elif not root.children:
        return 0
    h = 0
    for child in root.children:
        h = max(h, height1(child))
    return h + 1

def height2(root):
    if not root:
        return -1
    elif not root.left and not root.right:
        return 0
    return 1 + max(height2(root.left), height2(root.right))

root = TreeNode()
nodes = [root]
steps = list(input())
for step in steps:
    if step == 'd':
        node = TreeNode()
        nodes[-1].children.append(node)
        nodes.append(node)
    else:
        nodes.pop()

def prase_tree(root: TreeNode):
    if root.children:
        root.left = prase_tree(root.children.pop(0))
        cur = root.left
        while root.children:
            cur.right = prase_tree(root.children.pop(0))
            cur = cur.right
    return root

h1 = height1(root)
root0 = prase_tree(root)
h2 = height2(root0)
print(f'{h1} => {h2}')
```





```c++
#include <iostream>
#include <stack>
#include <string>
using namespace std;

int main()
{
    string str;
    cin >> str;

    int oldheight = 0;
    int maxold = 0;
    int newheight = 0;
    int maxnew = 0;
    stack<int> s;
    for (int i=0; i<str.size(); i++) {
        if (str[i] == 'd') {
            oldheight++;
            if (oldheight > maxold)
                maxold = oldheight;

            newheight++;
            s.push(newheight);
            if (newheight > maxnew)
                maxnew = newheight;
        }
        else {
            oldheight--;

            newheight = s.top();
            s.pop();
        }
    }

    cout << maxold << " => " << maxnew << endl;
}
```



## 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/

一棵树的镜面映射指的是对于树中的每个结点，都将其子结点反序。例如，对左边的树，镜面映射后变成右边这棵树。 

```
    a                             a
  / | \                         / | \
 b  c  f       ===>            f  c  b
   / \                           / \
  d   e                         e   d
```

我们在输入输出一棵树的时候，常常会把树转换成对应的二叉树，而且对该二叉树中只有单个子结点的分支结点补充一个虚子结点“$”，形成“伪满二叉树”。

例如，对下图左边的树，得到下图右边的伪满二叉树 

```
    a                             a
  / | \                          / \
 b  c  f       ===>             b   $
   / \                         / \
  d   e                       $   c                          
                                 / \
                                d   f
                               / \
                              $   e
```

然后对这棵二叉树进行前序遍历，如果是内部结点则标记为0，如果是叶结点则标记为1，而且虚结点也输出。

现在我们将一棵树以“伪满二叉树”的形式输入，要求输出这棵树的镜面映射的宽度优先遍历序列。

**输入**

输入包含一棵树所形成的“伪满二叉树”的前序遍历。
第一行包含一个整数，表示结点的数目。
第二行包含所有结点。每个结点用两个字符表示，第一个字符表示结点的编号，第二个字符表示该结点为内部结点还是外部结点，内部结点为0，外部结点为1。结点之间用一个空格隔开。
数据保证所有结点的编号都为一个小写字母。

**输出**

输出包含这棵树的镜面映射的宽度优先遍历序列，只需要输出每个结点的编号，编号之间用一个空格隔开。

样例输入

```
9
a0 b0 $1 c0 d0 $1 e1 f1 $1
```

样例输出

```
a f c b e d
```

提示

样例输入输出对应着题目描述中的那棵树。



根据给定的伪满二叉树前序遍历，输出其镜面映射的宽度优先遍历序列。镜面映射是指将每个节点的子节点顺序反转，而宽度优先遍历则需要按层级遍历所有节点。

思路

1. **输入处理**：读取输入的节点数目和前序遍历序列。
2. **层级计算**：通过遍历前序序列，维护当前层级信息。内部节点（标记为0）会增加层级，而外部节点（标记为1）会减少层级。
3. **层级分组**：将每个非虚节点按计算的层级分组。
4. **逆序输出**：对每个层级的节点逆序，并按层级顺序合并所有节点。

```python
from collections import defaultdict

n = int(input())
if n == 0:
    print()
    exit()

preorder = input().split()

# 初始化根节点
root = preorder[0][0]
root_type = preorder[0][1]

tier = defaultdict(list)
tier[0].append(root)

nodes = [root]
level = 0
types = {root: root_type}

for i in range(1, n):
    current = preorder[i]
    name = current[0]
    typ = current[1]
    types[name] = typ

    prev_node = nodes[-1]
    prev_type = types[prev_node]

    # 计算层级变化
    if prev_type == '1':
        level -= 1
    else:
        level += 1

    nodes.append(name)

    # 只添加非虚节点到对应层级
    if name != '$':
        tier[level].append(name)

# 按层级顺序排序并逆序每层节点
sorted_levels = sorted(tier.items(), key=lambda x: x[0])
result = []
for level, chars in sorted_levels:
    result.extend(reversed(chars))

print(' '.join(result))
```

代码解释

1. **输入处理**：读取输入的节点数和前序序列，处理根节点。
2. **层级计算**：遍历前序序列中的每个节点，根据前一个节点的类型（内部或外部）调整当前层级。
3. **层级分组**：将每个非虚节点添加到对应的层级列表中。
4. **逆序输出**：对每个层级的节点进行逆序处理，并按层级顺序合并所有结果，生成最终的宽度优先遍历序列。

这种方法避免了显式构建树结构，直接通过前序序列的遍历和层级计算，高效地得到镜面映射后的遍历结果。





```python
import sys
from collections import deque

sys.setrecursionlimit(10000)

# --- 第一步：读取输入并分割成 token 列表 ---
n = int(sys.stdin.readline().strip())          # 结点总数（包括虚节点）
tokens = sys.stdin.readline().split()           # 每个 token 形如 'a0'、'b1'、'$1'

# --- 第二步：将前序序列解析成“伪满二叉树” ---
idx = 0  # 全局索引，用于在 tokens 列表中遍历

class BinNode:
    __slots__ = ('label', 'is_leaf', 'left', 'right')
    def __init__(self, label, is_leaf):
        self.label = label      # 结点字符（小写字母或 '$'）
        self.is_leaf = is_leaf  # True=叶子或虚节点，False=内部节点
        self.left = None        # 左子指针（在左-子/右-兄弟表示中存第一孩子）
        self.right = None       # 右子指针（同一层的下一个兄弟）

def parse_binary():
    """
    递归按前序解析伪满二叉树：
    - 如果当前是内部节点（flag=='0'），则继续读两个子节点
    - 如果是叶子或虚节点（flag=='1'），则不再递归
    """
    global idx
    label = tokens[idx][0]
    flag  = tokens[idx][1]
    idx += 1
    node = BinNode(label, flag == '1')
    if not node.is_leaf:
        # 内部节点必然有左右两个孩子
        node.left = parse_binary()
        node.right = parse_binary()
    return node

root_bin = parse_binary()  # 根节点

# --- 第三步：将“左-子/右-兄弟”二叉树转换回 N 叉树，忽略 '$' 虚节点 ---
class NaryNode:
    __slots__ = ('label', 'children')
    def __init__(self, label):
        self.label = label    # 真实结点字符
        self.children = []    # 孩子列表

def build_nary(bin_node):
    """
    将二叉表示转换成 N 叉表示：
    - 跳过 bin_node 为 None 或 label=='$'
    - bin_node.left 指向第一个孩子；通过 .right 链接拿到所有兄弟
    """
    if bin_node is None or bin_node.label == '$':
        return None

    nnode = NaryNode(bin_node.label)
    child = bin_node.left
    while child:
        nch = build_nary(child)
        if nch:
            nnode.children.append(nch)
        child = child.right
    return nnode

root = build_nary(root_bin)

# --- 第四步：对 N 叉树做镜像（对每个节点的孩子列表反序） ---
def mirror_nary(node):
    if node is None:
        return
    node.children.reverse()
    for ch in node.children:
        mirror_nary(ch)

mirror_nary(root)

# --- 第五步：宽度优先遍历并输出结果 ---
q = deque([root])
res = []
while q:
    u = q.popleft()
    res.append(u.label)
    for ch in u.children:
        q.append(ch)

# 最终只输出结点编号，空格分隔
print(' '.join(res))
```

1. **读取与分割**  
   - 读入结点数 `n`（包含真实结点和虚节点 `$`）。  
   - 按空格分割出形如 `a0`、`$1` 的 token 列表。

2. **解析伪满二叉树**  
   - 利用前序遍历特点，递归构建二叉树结构（左-子/右-兄弟编码）。  
   - `'0'` 标记内部节点，需继续读左右两棵子树；`'1'` 标记叶子或虚节点。

3. **还原 N 叉树**  
   - 跳过所有 label 为 `'$'` 的节点。  
   - 通过左指针找第一个孩子，通过右指针串联剩余兄弟。

4. **镜像操作**  
   - 递归对每个节点的子节点列表做 `reverse()`，即完成“左右子树互换”效果。

5. **宽度优先遍历**  
   - 使用队列按层次输出，最终打印各真实结点的标签，空格分隔。

该方案时间复杂度约 O(n)，空间复杂度 O(n)。





前序遍历（Preorder Traversal）是二叉树遍历的一种方式，它的遍历顺序是先访问根节点，然后按照左子树和右子树的顺序递归地遍历子树。

`print_tree`函数不仅仅是简单地进行宽度优先遍历，而是通过一系列的栈和队列操作来间接实现了镜像映射的效果。

`print_tree`函数：实现了宽度优先遍历的变种。首先，它遍历树的“右侧视图”，将遇到的非虚节点按遍历顺序存入栈`s`中。然后，通过逆序将栈`s`的内容移入队列`Q`中，以此实现宽度优先遍历的某种形式。对于树的每一层，都先遍历右子树，再遍历左子树，最后将栈`s`中的元素逆序放回队列`Q`以继续遍历。这个过程通过间接的方式实现了镜面映射的效果。

​	发现“镜面”这一要求看着吓人，实际上没什么，它不会改变层与层之间的关系，只要每一层逆序输出就好。

另外发现：04082 树的镜面映射，同样程序用了两遍，lin28~35, line45~52. 在KMP中，也是有同样程序用了两遍。



在这个二叉树中，右节点是和自己“同级”的，左节点是自己的“下级”。这样建立二叉树之后，就能直接写出广度优先遍历，而不需要再把二叉树变为原来的树了。

```python
from collections import deque

class TreeNode:
    def __init__(self, x):
        self.x = x
        self.children = []

def create_node():
    return TreeNode('')

def build_tree(tempList, index):
    node = create_node()
    node.x = tempList[index][0]
    if tempList[index][1] == '0':
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
    return node, index

def print_tree(p):
    Q = deque()
    s = deque()

    # 遍历右子节点并将非虚节点加入栈s
    while p is not None:
        if p.x != '$':
            s.append(p)
        p = p.children[1] if len(p.children) > 1 else None

    # 将栈s中的节点逆序放入队列Q
    while s:
        Q.append(s.pop())

    # 宽度优先遍历队列Q并打印节点值
    while Q:
        p = Q.popleft()
        print(p.x, end=' ')

        # 如果节点有左子节点，将左子节点及其右子节点加入栈s
        if p.children:
            p = p.children[0]
            while p is not None:
                if p.x != '$':
                    s.append(p)
                p = p.children[1] if len(p.children) > 1 else None

            # 将栈s中的节点逆序放入队列Q
            while s:
                Q.append(s.pop())


n = int(input())
tempList = input().split()

# 构建多叉树
root, _ = build_tree(tempList, 0)

# 执行宽度优先遍历并打印镜像映射序列
print_tree(root)
```



按左儿子右兄弟的规则慢慢把树搓出来

```python
# 童溯 数学科学学院
class binarynode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.children = []
        self.parent = None

n = int(input())
lst = input().split()
stack = []
nodes = []
for x in lst:
    temp = binarynode(x[0])
    nodes.append(temp)
    if stack:
        if stack[-1].left:
            stack[-1].right = temp
            stack.pop()
        else:
            stack[-1].left = temp
    if x[1] == "0":
        stack.append(temp)

for x in nodes:
    if x.left and x.left.value != "$":
        x.children.append(x.left)
        x.left.parent = x
    if x.right and x.right.value != "$":
        x.parent.children.append(x.right)
        x.right.parent = x.parent

for x in nodes:
    x.children = x.children[::-1]

lst1 = [nodes[0]]
for x in lst1:
    if x.children:
        lst1 += x.children
print(" ".join([x.value for x in lst1]))
```





一步一步来。先把前序输入+内外节点的信息转成二叉树，第二步是把二叉树转成原来的树，第三步是bfs镜像输出。

```python
# 叶晨熙 化学与分子工程
class Noden:
    def __init__(self, value):
        self.child = []
        self.value = value
        self.parent = None

class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        self.parent = None

def bfs(noden):
    queue.pop(0)
    out.append(noden.value)
    if noden.child:
        for k in reversed(noden.child):
            queue.append(k)
    if queue:
        bfs(queue[0])
        

def ex(node):
    ans.append(node.value)
    if node.left:
        ex(node.left)
    if node.right:
        ex(node.right)

def reverse(node):
    if node.right == None:
        return node
    else:
        return reverse(node.parent)

def build(s, node, state):
    if not s:
        return
    if state == '0':
        new = Node(s[0][0])
        node.left = new
        new.parent = node
    else:
        pos = reverse(node.parent)
        new = Node(s[0][0])
        pos.right = new
        new.parent = pos
    build(s[1:], new, s[0][1])

def bi_to_n(node):
    if node.left:
        if node.left.value != '$':
            newn = Noden(node.left.value)
            dic[node.left] = newn
            dic[node].child.append(newn)
            newn.parent = dic[node]
            bi_to_n(node.left)
    if node.right:
        if node.right.value != '$':
            newn = Noden(node.right.value)
            dic[node.right] = newn
            dic[node].parent.child.append(newn)
            newn.parent = dic[node].parent
            bi_to_n(node.right)

n = int(input())
k = input().split()
root = Node(k[0][0])
k.pop(0)
if k:
    build(k, root, k[0][1])
ans = []
ex(root)
#print(ans)
dic = {}
dic[root] = Noden(root.value)
bi_to_n(root)
rootn = dic[root]
#print(rootn)
queue = [rootn]
out =[]
bfs(rootn)
print(' '.join(out))

```



就是按照伪满二叉树构造特点分层。遇到1标记的叶子节点就说明下一个是兄弟节点上移一层输出。

```python
# 赵策 数学科学学院
from collections import defaultdict
n=int(input())
preorder=input().split()
root,type=list(preorder[0])
dic={}
nodes=[root]
dic[root]=type
tier=defaultdict(list)
tier[0].append(root)
level=0
for i in range(1,n):
    name,type=list(preorder[i])
    dic[name]=type
    if dic[nodes[-1]]=='1':
        level-=1
    else:
        level+=1
    nodes.append(name)
    if name!='$':
        tier[level].append(name)
res=''
for i in sorted(tier.items()):
    res+=''.join(i[1])[::-1]
print(' '.join(res))
```





镜面映射简单，每层倒着输出就行；关键是这个树怎么构建，其实是斜着来表示树的层级，具体实现有点类似dfs，遇到顶就回溯

构建的数据结构很有意思，其实就是一个双层列表，我给他起了个名字叫梯子

```python
# 赵新坤 物理学院
class ladder:
    def __init__(self):
        self.height=-1
        self.matrix=[]

    def add_level(self):
        self.height+=1
        self.matrix.append([])

    def insert_i_level(self,node,level):
        if self.height<level:
            self.add_level()
        self.matrix[level].append(node)

def traversal(Ladder):
    ans=[]
    for i in range(Ladder.height+1):
        stack=[]
        for j in Ladder.matrix[i]:
            stack.insert(0,j)
        ans.extend(stack)
    print(' '.join(ans))

n=int(input())
data=[str(x) for x in input().split()]
L=ladder()
level=0
while data:
#    print(f'current level is {level}')
    p=data.pop(0)
    if p =='$1':
        level-=1
#        print('turn right')
    else:
        L.insert_i_level(p[0],level)
#        print(f'add {p[0]} into level {level}')
        if p[-1]!='1':
            level+=1
        else:
            level-=1

#for i in range(L.height+1):
#    print(L.matrix[i])
traversal(L)
```





## 04084: 拓扑排序

http://cs101.openjudge.cn/dsapre/04084/

给出一个图的结构，输出其拓扑排序序列，要求在同等条件下，编号小的顶点在前。

**输入**

若干行整数，第一行有2个数，分别为顶点数v和弧数a，接下来有a行，每一行有2个数，分别是该条弧所关联的两个顶点编号。
v<=100, a<=500

**输出**

若干个空格隔开的顶点构成的序列(用小写字母)。

样例输入

```
6 8
1 2
1 3
1 4
3 2
3 5
4 5
6 4
6 5
```

样例输出

```
v1 v3 v2 v6 v4 v5
```



可能有孤立点。可能两个点之间可以有多条弧，用dic构造的图，同学入度-1能AC。

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



```python
#23n2300011075(才疏学浅)
v,a=map(int,input().split())
node=["v"+str(i) for i in range(v+1)]
dic1={i:0 for i in node}
dic2={i:[] for i in node}
for _ in range(a):
    f,t=map(int,input().split())
    dic1[node[t]]+=1
    dic2[node[f]].append(node[t])
vis=set()
cnt=0
ans=[]
while cnt<v:
    for i in range(1,v+1):
        if dic1[node[i]]==0 and node[i] not in vis:
            vis.add(node[i])
            ans.append(node[i])
            cnt+=1
            for nodes in dic2[node[i]]:
                dic1[nodes]-=1
            break
print(*ans)
```



## 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

给你一些电话号码，请判断它们是否是一致的，即是否有某个电话是另一个电话的前缀。比如：

Emergency 911
Alice 97 625 999
Bob 91 12 54 26

在这个例子中，我们不可能拨通Bob的电话，因为Emergency的电话是它的前缀，当拨打Bob的电话时会先接通Emergency，所以这些电话号码不是一致的。

**输入**

第一行是一个整数t，1 ≤ t ≤ 40，表示测试数据的数目。
每个测试样例的第一行是一个整数n，1 ≤ n ≤ 10000，其后n行每行是一个不超过10位的电话号码。

**输出**

对于每个测试数据，如果是一致的输出“YES”，如果不是输出“NO”。

样例输入

```
2
3
911
97625999
91125426
5
113
12340
123440
12345
98346
```

样例输出

```
NO
YES
```



> 04089:电话号码。使用方法二（直接比较）作为首选，因为实现简单且在Python中对于题目给出的数据规模（n≤10000）效率足够。当处理更大规模数据时，则应选择Trie实现（方法一）。方法一：Trie（字典树）实现，方法二：排序后直接比较。即更简单的方法是对号码排序后逐个检查是否为前缀。方法三：递归字典实现。另一种Trie的递归实现方式。



方法一：

要解决这个问题，需要判断是否存在某个电话号码是另一个电话号码的前缀。这可以通过构建一个 **字典树（Trie）** 来高效实现。

**解题思路**

1. **字典树（Trie）**：
   - 字典树是一种专门用来处理字符串前缀问题的数据结构。
   - 每个节点存储一个字符，路径表示字符串的前缀。
   - 如果某个节点已经是某个电话号码的结尾（即完整电话号码），那么后续插入的任何电话号码都会以它为前缀。

2. **算法步骤**：
   - 对于每个测试样例：
     1. 构建一个空的字典树。
     2. 将所有电话号码按长度从短到长排序（因为短的号码更可能是长号码的前缀）。
     3. 遍历每个电话号码：
        - 在字典树中查找该号码是否已经存在完整的前缀。
        - 如果存在，则直接输出 "NO"。
        - 否则，将该号码插入字典树。
     4. 如果所有号码都成功插入且没有冲突，输出 "YES"。

3. **复杂度分析**：
   - 插入和查找操作在字典树中的时间复杂度为 $O(L)$，其中 L 是电话号码的长度。
   - 因此，总的时间复杂度为 $O(t \cdot n \cdot L)$，其中 t 是测试样例数，n 是电话号码的数量，L 是电话号码的最大长度。

---

实现代码

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_number = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, number):
        node = self.root
        for digit in number:
            if digit not in node.children:
                node.children[digit] = TrieNode()
            node = node.children[digit]
            # 如果当前节点已经是某个电话号码的结尾，则说明存在前缀冲突
            if node.is_end_of_number:
                return False
        # 插入完成后，标记为完整电话号码
        node.is_end_of_number = True
        # 如果当前节点还有子节点，说明有其他号码以它为前缀
        return len(node.children) == 0
    
    def is_consistent(self, numbers):
        # 按长度从短到长排序，确保短号码先被检查
        numbers.sort(key=len)
        for number in numbers:
            if not self.insert(number):
                return False
        return True

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    t = int(data[0])  # 测试样例数量
    index = 1
    results = []
    
    for _ in range(t):
        n = int(data[index])  # 当前测试样例的电话号码数量
        index += 1
        numbers = data[index:index + n]
        index += n
        
        trie = Trie()
        if trie.is_consistent(numbers):
            results.append("YES")
        else:
            results.append("NO")
    
    print("\n".join(results))

# 调用主函数
if __name__ == "__main__":
    main()
```

代码解释

1. **TrieNode 类**：
   - `children`：存储子节点，键是数字字符，值是子节点对象。
   - `is_end_of_number`：标记当前节点是否是一个完整电话号码的结尾。

2. **Trie 类**：
   - `insert` 方法：在字典树中插入一个电话号码，同时检查是否存在前缀冲突。
   - `is_consistent` 方法：遍历所有电话号码，依次插入字典树，返回是否一致。

3. **主函数**：
   - 读取输入数据，解析测试样例。
   - 对每个测试样例调用 `Trie` 的 `is_consistent` 方法，输出结果。

---

注意事项

- 输入电话号码可能包含前导零，因此不能将其转换为整数。
- 排序电话号码时按照长度从小到大排序，可以减少不必要的冲突检查。
- 使用字典树能够高效地解决前缀问题，避免暴力比较的高时间复杂度。



https://www.geeksforgeeks.org/trie-insert-and-search/

**Definition:** A trie (prefix tree, derived from retrieval) is a multiway tree data structure used for storing strings over an alphabet. It is used to store a large amount of strings. The pattern matching can be done efficiently using tries.

使用字典实现的字典树（Trie）。它的主要功能是插入和搜索字符串。

这里特意把 insert 和 search 分开写，也是保留了Trie本身的完整性。

```python
class TrieNode:
    def __init__(self):
        self.child={}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1


t = int(input())
p = []
for _ in range(t):
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(str(input()))
    nums.sort(reverse=True)
    s = 0
    trie = Trie()
    for num in nums:
        s += trie.search(num)
        trie.insert(num)
    if s > 0:
        print('NO')
    else:
        print('YES')
```





方法二：排序后直接比较。即更简单的方法是对号码排序后逐个检查是否为前缀。

```python
# 雷逸鸣 物理学院
class Solution:
    def is_consistent(self, phone_numbers):
        phone_numbers.sort()  # 对电话号码列表进行排序
        for i in range(len(phone_numbers) - 1):
            if phone_numbers[i + 1].startswith(phone_numbers[i]):
                return False
        return True

def main():
    t = int(input().strip())
    for _ in range(t):
        n = int(input().strip())
        phone_numbers = [input().strip() for _ in range(n)]
        solution = Solution()
        if solution.is_consistent(phone_numbers):
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()

```



方法三：递归字典实现。是另一种Trie的递归实现方式。

用字典嵌套建树，按照每一位数字，如果是子节点就在子节点继续添加，如果不是就新建子节点，在此添加，最后记录end；遍历的时候按深度优先，如果end和其他节点同时存在，就是NO

```python
# 赵新坤 物理学院
def trav(adct,i):
    ans=[]
    kys=[k for k in adct.keys()]
    if 'end' in kys and len(kys)>1:
        ans.append('NO')
    for key in adct.keys():
#        print('     '*i+key)
        if adct[key]:
            ans.extend(trav(adct[key],i+1))
    return ans
def insert(alst,adct):
    p=alst.pop(0)
    if p not in adct.keys():
        adct[p]={}
    if alst:    
        adct[p]=insert(alst,adct[p])
    else :
        adct[p].update({'end':None })
    return adct
t=int(input())
anslst=[]
for _ in range(t):
    n=int(input())
    tree={}
    ans='YES'
    num_set=set()
    for i in range(n):
        ipt=str(input())
        if ipt in num_set:
            ans='NO'
        num_set.add(ipt)
        ipt_num_lst=[x for x in ipt]
        tree=insert(ipt_num_lst,tree)
    if 'NO' in trav(tree,0):
        ans='NO'
    anslst.append(ans)
for i in anslst:
    print(i)

```



```python
def insert(phone, node):
    for i, digit in enumerate(phone):
        if 'end' in node:
            # 当前路径已经是某个号码的结尾，但我们还在往下走，说明前缀冲突
            return False
        if digit not in node:
            node[digit] = {}
        node = node[digit]
    # 插入结束，如果这个节点还有子节点，说明有其他号码以它为前缀
    if node:
        return False
    node['end'] = None
    return True

t = int(input())
for _ in range(t):
    n = int(input())
    numbers = [input().strip() for _ in range(n)]
    tree = {}
    consistent = True
    for num in sorted(numbers):  # 先排序，前缀更容易被检测出来
        if not insert(num, tree):
            consistent = False
            break
    print("YES" if consistent else "NO")
```



【蔡东辰、24工学院】只记得字典树是字典套字典，但课件里的代码只是粗略扫了一眼，没记住具体怎么写，于是自己想了一个。对于这题而言，通过递归计算总叶子节点数量，如果与输入的数据数量不一致就输出NO。但是一开始写的时候一直wa，和ai共同探索后才发现number里的数据一开始以int形式存了，导致有先导0时会出错，去了int之后果然ac了，感觉是值得进我错题本的题

```python
class Trie:
    def __init__(self,key = None):
        self.dic = {}
        self.key = key
    def add(self,s):
        if s[0] not in self.dic:
            self.dic[s[0]] = Trie(s[0])
        a = self.dic[s[0]]
        if len(s) > 1:
            a.add(s[1:])
    def leaf_nums(self):
        if len(self.dic) == 0:
            return 1
        num = 0
        for i in self.dic:
            num += self.dic[i].leaf_nums()
        return num
t = int(input())
for i in range(t):
    n = int(input())
    trie = Trie()
    number = []
    for j in range(n):
        number.append(input())
    number.sort(key = lambda x:-len(x))
    for k in number:
        trie.add(str(k))
    if trie.leaf_nums() == n:
        print('YES')
    else:
        print('NO')
```





## 04093: 倒排索引查询

data structures, http://cs101.openjudge.cn/practice/04093/

现在已经对一些文档求出了倒排索引，对于一些词得出了这些词在哪些文档中出现的列表。

要求对于倒排索引实现一些简单的查询，即查询某些词同时出现，或者有些词出现有些词不出现的文档有哪些。

**输入**

第一行包含一个数N，1 <= N <= 100，表示倒排索引表的数目。
接下来N行，每行第一个数ci，表示这个词出现在了多少个文档中。接下来跟着ci个数，表示出现在的文档编号，编号不一定有序。1 <= ci <= 1000，文档编号为32位整数。
接下来一行包含一个数M，1 <= M <= 100，表示查询的数目。
接下来M行每行N个数，每个数表示这个词要不要出现，1表示出现，-1表示不出现，0表示无所谓。数据保证每行至少出现一个1。

**输出**

共M行，每行对应一个查询。输出查询到的文档编号，按照编号升序输出。
如果查不到任何文档，输出"NOT FOUND"。

样例输入

```
3
3 1 2 3
1 2
1 3
3
1 1 1
1 -1 0
1 -1 -1
```

样例输出

```
NOT FOUND
1 3
1
```



在实际搜索引擎在处理基于倒排索引的查询时，搜索引擎确实会优先关注各个查询词的倒排表的合并和交集处理，而不是直接准备未出现文档的集合。这种方法更有效，特别是在处理大规模数据集时，因为它允许系统动态地调整和优化查询过程，特别是在有复杂查询逻辑（如多个词的组合、词的排除等）时。详细解释一下搜索引擎如何使用倒排索引来处理查询：

倒排索引查询的核心概念
1. 倒排索引结构：
   - 对于每个词（token），都有一个关联的文档列表，这个列表通常是按文档编号排序的。
   - 每个文档在列表中可能还会有附加信息，如词频、位置信息等。
2. 处理查询：
   - 单词查询：对于单个词的查询，搜索引擎直接返回该词的倒排列表。
   - 多词交集查询：对于包含多个词的查询，搜索引擎找到每个词的倒排列表，然后计算这些列表的交集。
   这个交集代表了所有查询词都出现的文档集合。
   - 复杂逻辑处理：对于包含逻辑运算（AND, OR, NOT）的查询，搜索引擎会结合使用集合的
   交集（AND）、并集（OR）和差集（NOT）操作来处理查询。特别是在处理 NOT 逻辑时，
   它并不是去查找那些未出现词的文档集合，而是从已经确定的结果集中排除含有这个词的文档。

更贴近实际搜索引擎的处理实现，如下：

```python
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

word_documents = []

# 读取每个词的倒排索引
for _ in range(N):
    ci = int(data[index])
    index += 1
    documents = sorted(map(int, data[index:index + ci]))
    index += ci
    word_documents.append(documents)

M = int(data[index])
index += 1

results = []

# 处理每个查询
for _ in range(M):
    query = list(map(int, data[index:index + N]))
    index += N

    # 集合存储各词的文档集合（使用交集获取所有词都出现的文档）
    included_docs = []
    excluded_docs = set()

    # 解析查询条件
    for i in range(N):
        if query[i] == 1:
            included_docs.append(word_documents[i])
        elif query[i] == -1:
            excluded_docs.update(word_documents[i])

    # 仅在有包含词时计算交集
    if included_docs:
        result_set = set(included_docs[0])
        for docs in included_docs[1:]:
            result_set.intersection_update(docs)
        result_set.difference_update(excluded_docs)
        final_docs = sorted(result_set)
        results.append(" ".join(map(str, final_docs)) if final_docs else "NOT FOUND")
    else:
        results.append("NOT FOUND")

# 输出所有查询结果
for result in results:
    print(result)
```



利用集合运算来处理倒排索引查询。思路如下：

1. **建立倒排索引**  
   对于每个词，读入出现该词的文档编号，将其存入一个集合中。

2. **处理查询**  
   对于每个查询，按照查询向量（每个位置取值 1、-1、0）：
   - 对于标记为 1 的词（必须出现），求这些词对应集合的交集。
   - 对于标记为 -1 的词（必须不出现），从交集中减去这些词出现的文档编号。
   - 0 的词不作限制。
   
3. **输出结果**  
   将最终得到的候选文档集合按升序输出；如果为空，则输出 "NOT FOUND"。

下面是完整代码：

```python
#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    
    # 读入倒排索引的词数
    N = int(next(it))
    inverted = []
    for _ in range(N):
        # 每个词的出现文档数
        count = int(next(it))
        docs = set()
        for _ in range(count):
            docs.add(int(next(it)))
        inverted.append(docs)
    
    # 读入查询数目
    M = int(next(it))
    output_lines = []
    for _ in range(M):
        # 每个查询包含 N 个数字
        query = [int(next(it)) for _ in range(N)]
        candidate = None
        # 处理必须出现的词（值为 1）：取交集
        for j in range(N):
            if query[j] == 1:
                if candidate is None:
                    candidate = inverted[j].copy()
                else:
                    candidate &= inverted[j]
        # 处理必须不出现的词（值为 -1）：从候选集合中剔除
        for j in range(N):
            if query[j] == -1:
                candidate -= inverted[j]
        # 输出结果
        if candidate:
            result_line = " ".join(map(str, sorted(candidate)))
            output_lines.append(result_line)
        else:
            output_lines.append("NOT FOUND")
    
    sys.stdout.write("\n".join(output_lines))
    
if __name__ == '__main__':
    main()
```

**代码说明**

- **倒排索引构建**  
  从输入中读入每个词出现的文档编号，并用 `set` 保存，方便后续交并集操作。

- **查询处理**  
  - 先对所有标记为 1 的词取交集，即得到文档中同时包含所有这些词的候选集合。
  - 再对所有标记为 -1 的词，从候选集合中减去这些文档编号。
  - 0 表示不关心，不做任何操作。

- **输出**  
  如果候选集合非空，则将文档编号排序后输出，否则输出 "NOT FOUND"。



```python
n = int(input())
files = [] # 储存所有单词的文档归属数据
for i in range(n):
    data = list(map(int,input().split()))
    files.append(set(data[1:])) # 将每个单词对应的文档数据转化为集合，方便后续的操作

m = int(input())
for i in range(m):

    data = list(map(int,input().split()))
    t = data.index(1) # 第一个1要特殊处理，作为初始值
    a = set(files[t]) # 所有有可能合题的文档（同样要储存在集合内）
    for j in range(t + 1,n):
        if data[j] == 1:
            a = a & files[j] # 取交集（集合之间的∩、-、∪运算时间复杂度极低！）

    b = set() # 所有一定不合题的文档（同样要储存在集合内）
    for j in range(n):
        if data[j] == -1:
            b = b | files[j] # 取并集
    
    # 注意：一定要先把所有要包含的文档全部放入a，所有不能包含的文档统一放入b，最后再a-b！
    # 否则会损失一些b中的信息，例如如果是“无所谓、不能包含2、要包含12”，
    # 如果一步一步使用交集和差集的话，则1和2都合题，这与第二个条件矛盾！因此不能一步一步来~
    c = a - b # a与b的差集
    finale = list(c)
    if finale:
        finale.sort()
        print(" ".join(map(str,finale)))
    else:print("NOT FOUND")
```





```python

n = int(input())
lis = []
all_documents = set()

# Read each word's document list
for _ in range(n):
    data = list(map(int, input().split()))
    doc_set = set(data[1:])
    lis.append(doc_set)
    all_documents.update(doc_set)

# Prepare the not-present sets 未出现文档集合
lis1 = [all_documents - doc_set for doc_set in lis]

# Read number of queries
m = int(input())

# Process each query
for _ in range(m):
    query = list(map(int, input().split()))
    result_set = None

    # Determine result set based on requirements in query
    for num, requirement in enumerate(query):
        if requirement != 0:
            current_set = lis[num] if requirement == 1 else lis1[num]
            result_set = current_set if result_set is None else result_set.intersection(current_set)

    if not result_set:
        print("NOT FOUND")
    else:
        print(' '.join(map(str, sorted(result_set))))

```





## 04117: 简单的整数划分问题

http://cs101.openjudge.cn/practice/04117/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Tough 部分相应题目



## 04130: Saving Tang Monk

http://cs101.openjudge.cn/practice/04130/

题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Tough 部分相应题目





## 04135: 月度开销

http://cs101.openjudge.cn/dsapre/04135/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 04136: 矩形分割

http://cs101.openjudge.cn/dsapre/04136/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 04137:最小新整数

monotonous-stack, http://cs101.openjudge.cn/practice/04137/

给定一个十进制正整数n(0 < n < 1000000000)，每个数位上数字均不为0。n的位数为m。
现在从m位中删除k位(0<k < m)，求生成的新整数最小为多少？
例如: n = 9128456, k = 2, 则生成的新整数最小为12456

**输入**

第一行t, 表示有t组数据；
接下来t行，每一行表示一组测试数据，每组测试数据包含两个数字n, k。

**输出**

t行，每行一个数字，表示从n中删除k位后得到的最小整数。

样例输入

```
2
9128456 2
1444 3
```

样例输出

```
12456
1
```



何为单调栈？顾名思义，单调栈即满足单调性的栈结构。例如：维护一个整数的单调递增栈。

```python
# 23n2300012276(管骏杰）
def removeKDigits(num, k):
    stack = []
    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    while k:
        stack.pop()
        k -= 1
    return int(''.join(stack))
t = int(input())
results = []
for _ in range(t):
    n, k = input().split()
    results.append(removeKDigits(n, int(k)))
for result in results:
    print(result)
```



## M04140: 方程求解

牛顿迭代法, AI, binary search, http://cs101.openjudge.cn/practice/04140/

求下面方程的根：$ f(x) = x^3 - 5x^2 + 10x - 80 = 0 $。

**输入**

\-

**输出**

精确到小数点后9位。

样例输入

```
-
```

样例输出

```
-
```





可以用 **牛顿迭代法** 或 **二分法**  来求解方程 $ f(x) = x^3 - 5x^2 + 10x - 80 = 0 $。



**方法 1：牛顿迭代法**（Newton-Raphson Method）

牛顿法适用于方程可微的情况，需要计算导数：

$$
f'(x) = 3x^2 - 10x + 10
$$
更新公式：
$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$



Python 实现，输出保留到小数点后 9 位：

```python
def f(x):
    return x**3 - 5*x**2 + 10*x - 80

def f_prime(x):
    return 3*x**2 - 10*x + 10

def newton_method(x0, eps=1e-10, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-12:
            raise ValueError("Derivative too small, method fails.")
        x_new = x - fx / fpx
        if abs(x_new - x) < eps:
            return x_new
        x = x_new
    raise ValueError("Newton method did not converge.")

# 初始猜测（可以图像估计或经验选择）
initial_guess = 4.0
root = newton_method(initial_guess)

# 输出结果，保留9位小数
print(f"{root:.9f}")
```

说明：

- 使用初始猜测 `x0 = 4.0`，该函数在 x=4 附近有根。
- 精度设置为 `1e-10`，确保输出时小数点后 9 位准确。
- 若牛顿法不收敛，会抛出异常。

这个程序没有输入，因此运行时直接输出结果即可，符合题目要求。



> 用<mark>牛顿迭代法</mark>实现。
>
> 因为大语言模型的训练过程中涉及到了梯度下降（或其变种，如SGD、Adam等），用于优化模型参数以最小化损失函数。两种方法都是通过迭代的方式逐步接近最优解。每一次迭代都基于当前点的局部信息调整参数，试图找到一个比当前点更优的新点。理解牛顿迭代法有助于深入理解基于梯度的优化算法的工作原理，特别是它们如何利用导数信息进行决策。
>
> > **牛顿迭代法**
> >
> > - **目的**：主要用于寻找一个函数 $f(x)$ 的根，即找到满足 $f(x)=0$ 的 $x$ 值。不过，通过适当变换目标函数，它也可以用于寻找函数的极值。
> > - **方法基础**：利用泰勒级数的一阶和二阶项来近似目标函数，在每次迭代中使用目标函数及其导数的信息来计算下一步的方向和步长。
> > - **迭代公式**：$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $ 对于求极值问题，这可以转化为$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $，这里 $f'(x)$ 和 $f''(x)$ 分别是目标函数的一阶导数和二阶导数。
> > - **特点**：牛顿法通常具有更快的收敛速度（尤其是对于二次可微函数），但是需要计算目标函数的二阶导数（Hessian矩阵在多维情况下），并且对初始点的选择较为敏感。
> >
> > **梯度下降法**
> >
> > - **目的**：直接用于寻找函数的最小值（也可以通过取负寻找最大值），尤其在机器学习领域应用广泛。
> > - **方法基础**：仅依赖于目标函数的一阶导数信息（即梯度），沿着梯度的反方向移动以达到减少函数值的目的。
> > - **迭代公式**：$ x_{n+1} = x_n - \alpha \cdot \nabla f(x_n) $ 这里 $\alpha$ 是学习率，$\nabla f(x_n)$ 表示目标函数在 $x_n$ 点的梯度。
> > - **特点**：梯度下降不需要计算复杂的二阶导数，因此在高维空间中相对容易实现。然而，它的收敛速度通常较慢，特别是当目标函数的等高线呈现出椭圆而非圆形时（即存在条件数大的情况）。
> >
> > **相同与不同**
> >
> > - **相同点**：两者都可用于优化问题，试图找到函数的极小值点；都需要目标函数至少一阶可导。
> > - **不同点**：
> >   - 牛顿法使用了更多的局部信息（即二阶导数），因此理论上收敛速度更快，但在实际应用中可能会遇到计算成本高、难以处理大规模数据集等问题。
> >   - 梯度下降则更为简单，易于实现，特别是在高维空间中，但由于只使用了一阶导数信息，其收敛速度可能较慢，尤其是在接近极值点时。
>
> 



思路：先算一下迭代的方程$$\phi(x)=x-\frac{x^3-5x^2+10x-80}{3x^2-10x+10}=\frac{2x^3-5x^2+80}{3x^2-10x+10}$$
与此同时，利用一点点数学，可以发现原方程有且仅有一实根，且在5到6之间。

```python
# coding: utf-8
"""
@File        :   newton_04140.py
@Time        :   2025/03/01 16:14:57
@Author      :   Usercyk
@Description :   Using Newton's method to solve the equation f(x) = 0
"""


class Solution:
    """
    The solution
    """

    def phi(self, x: float) -> float:
        """
        The iteration function

        Arguments:
            x -- The nth x value

        Returns:
            The (n+1)th x value
        """
        return (2*x**3-5*x**2+80)/(3*x**2-10*x+10)

    def solve(self, x_init: float = 5.0, eps: float = 1e-15) -> float:
        """
        Solve the equation f(x) = x**3-5*x**2+10*x-80 = 0

        Keyword Arguments:
            x_init -- The initial value of x (default: {5.0})
            eps -- The precision (default: {1e-15})

        Returns:
            The solution of the equation
        """
        x = x_init
        while True:
            x_next = self.phi(x)
            if abs(x_next-x) < eps:
                return x_next
            x = x_next


if __name__ == "__main__":
    print(f"{Solution().solve():.9f}")

```





**方法 2：二分法**

二分法适用于单调区间，我们需要先找到根所在的区间，然后不断缩小范围，直到精度满足要求。

```python
def f(x):
    return x ** 3 - 5 * x ** 2 + 10 * x - 80


def binary_search(a, b, tol=1e-9):
    """在区间 [a, b] 内使用二分法找到方程 f(x) = 0 的根"""
    if f(a) * f(b) > 0:
        raise ValueError("二分法要求 f(a) 和 f(b) 符号相反，确保根在区间内")

    while abs(b - a) > tol:
        mid = (a + b) / 2
        if f(mid) == 0:
            return mid
        elif f(mid) * f(a) < 0: # 说明区间内有根
            b = mid
        else:
            a = mid

    return (a + b) / 2 # 根就在这个很小的区间里


# 选择合适的区间（先观察 f(x) 在不同区间的符号变化）
root1 = binary_search(3, 10)
print(f"{root1:.9f}")
```

**比较**

| 方法       | 适用情况       | 优点     | 缺点                             |
| ---------- | -------------- | -------- | -------------------------------- |
| **二分法** | 确保有根的区间 | 一定收敛 | 需要选区间，收敛较慢             |
| **牛顿法** | 初值合理时     | 收敛快   | 需要计算导数，初值不好可能不收敛 |





## 04143: 和为给定数

http://cs101.openjudge.cn/dsapre/04143/

**输入**

共三行： 第一行是整数n(0 < n <= 100,000)，表示有n个整数。 第二行是n个整数。整数的范围是在0到10^8之间。 第三行是一个整数m（0 <= m <= 2^30)，表示需要得到的和。

**输出**

若存在和为m的数对，输出两个整数，小的在前，大的在后，中间用单个空格隔开。若有多个数对满足条件，选择数对中较小的数更小的。若找不到符合要求的数对，输出一行No。

样例输入

`4 2 5 1 4 6`

样例输出

`1 5`



```python
# 23n2300011760(喜看稻菽千重浪)
n=int(input())-1;m=0
A=sorted(map(int,input().split()))
s=int(input())
while m<n:
    while m<n and A[m]+A[n]>s:n-=1
    while m<n and A[m]+A[n]<s:m+=1
    if m<n and A[m]+A[n]==s:print(A[m],A[n]);break
else:print("No")
```



## 05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

假设这里有36张扑克牌，分别为A1~A9,B1~B9,C1~C9,D1~D9，其中A代表方片，B代表草花，C代表红桃，D代表黑桃，那么，设定如下的排序规则：

1.对于两张卡牌，X1Y1与X2Y2，X1与X2表示A～D，Y1与Y2表示1～9，如果X1与X2不同，那么依照D>C>B>A的方式进行排序

2.假如有X1与X2相同时，那么就比较Y1与Y2的大小。

例如，对于如下的四张牌，有如下的升序排序结果：

D3，C4，A4，C1

升序排序的结果为A4，C1，C4，D3

有人提出了如下的排序策略：

先建立9个队列，用于存放点数的大小，将卡牌依点数存放入各自的队列之中，然后再按队列1到队列9依次出队。

例如，对于上面的结果，依次进队后，结果如下：

队列1：C1；队列3：D3，队列4：C4，A4

将其依次出队后，结果为C1，D3，C4，A4

然后，再建立4个队列，用于存放花色。将卡牌依花色A～D存放入队列1～4中，然后再按队列1到队列4依次出队。

例如，对于上面刚刚出队的序列C1，D3，C4，A4，将其依次进队，结果如下：

队列1：A4；队列3：C1，C4；队列4：D3

将其依次出队后，结果为A4，C1，C4，D3，排序结束。



请根据上面的算法，编写一个用队列对扑克牌排序的程序，要求依照上面的排序规则，根据先花色后点数的方法进行排序。



输入

输入分为两行，第一行为一个整数n，表示一共有n张牌（1<=n<=100）
第二行用XY的形式表示每一张牌，其中X为A～D，Y为1～9

输出

输出三个部分
第一个部分为第一次进队出队的结果，用Queue1:...表示，共9行，结果用空格分隔，下同
第二部分为第二次进队出队的结果，用QueueA:...表示，共4行
第三部分为一行，即将卡牌排序后的结果（升序排序）

样例输入

```
8
D8 A6 C3 B8 C5 A1 B5 D3
```

样例输出

```
Queue1:A1
Queue2:
Queue3:C3 D3
Queue4:
Queue5:C5 B5
Queue6:A6
Queue7:
Queue8:D8 B8
Queue9:
QueueA:A1 A6
QueueB:B5 B8
QueueC:C3 C5
QueueD:D3 D8
A1 A6 B5 B8 C3 C5 D3 D8
```

提示

第二次入队出队时，可以复用第一次时9个队列中的4个。所以其实只需要开辟9个队列即可。



```python
from collections import deque


n = int(input())
queues = [deque() for _ in range(9)]
cards = deque(list(input().split()))

while cards:
    card = cards.popleft()
    queues[int(card[1])-1].append(card)

qs = {'A': deque(), 'B': deque(), 'C': deque(), 'D': deque()}
for i in range(9):
    tmp = []
    while queues[i]:
        card = queues[i].popleft()
        qs[card[0]].append(card)
        tmp.append(card)
    print(f'Queue{i+1}:'+' '.join(tmp))

result = []
for char in qs.keys():
    tmp = []
    while qs[char]:
        card = qs[char].popleft()
        result.append(card)
        tmp.append(card)
    print(f'Queue{char}:' + ' '.join(tmp))
print(*result)
```



## 05344: 最后的最后

http://cs101.openjudge.cn/practice/05344/

 [弗拉维奥·约瑟夫斯](http://http//zh.wikipedia.org/wiki/弗拉維奧·約瑟夫斯)是1世纪的一名犹太历史学家。他在自己的日记中写道，在一次战中，他和他的40个战友被罗马军队包围在洞中。他们讨论是自杀还是被俘，最终决定自杀，并以抽签的方式决定谁杀掉谁。约瑟夫斯和另外一个人是最后两个留下的人。约瑟夫斯说服了那个人，他们将向罗马军队投降，不再自杀。约瑟夫斯把他的存活归因于运气或天意，他不知道是哪一个。

  在计算机科学与数学中，就有一个以此命名的问题：**约瑟夫斯问题**（有时也称为**约瑟夫斯置换**）。在计算机编程的算法中，类似问题又称为**约瑟夫环**。具体描述如下：有![n](http://upload.wikimedia.org/math/7/b/8/7b8b965ad4bca0e41ab51de7b31363a1.png)个囚犯站成一个圆圈，准备处决。首先从一个人开始，越过![k-2](http://upload.wikimedia.org/math/7/2/1/721e20007292e8066d890e8d365d268d.png)个人（因为第一个人已经被越过），并杀掉第*k*个人。接着，再越过![k-1](http://upload.wikimedia.org/math/1/4/4/14464ac1dfe6fa8ad8fda94bb6f01571.png)个人，并杀掉第*k*个人。这个过程沿着圆圈一直进行，直到最终只剩下一个人留下，这个人就可以继续活着。问题是，给定了![n](http://upload.wikimedia.org/math/7/b/8/7b8b965ad4bca0e41ab51de7b31363a1.png)和![k](http://upload.wikimedia.org/math/8/c/e/8ce4b16b22b58894aa86c421e8759df3.png)，一开始要站在什么地方才能避免被处决？

  为了让大家熟悉循环链表的使用，对该题进行模拟。我们要求将之前的所有被kill掉的囚犯的编号输出。

**输入**

题中描述的囚犯数n（即编号为1至n，n不大于1000）和间隔数k（k大于等于2，小于n）

**输出**

顺序输出被kill掉的囚犯的编号，中间以空格隔开

样例输入

```
10 2
```

样例输出

```
2 4 6 8 10 3 7 1 9
```



```python
class Node:
    def __init__(self, number):
        self.number = number
        self.next = None

def josephus_circle(n, k):
    # 创建循环链表
    head = Node(1)
    current = head
    for i in range(2, n + 1):
        new_node = Node(i)
        current.next = new_node
        current = new_node
    current.next = head  # 形成环

    result = []
    current = head
    prev = None

    while current.next != current:
        # 找到第k个节点
        for _ in range(k - 1):
            prev = current
            current = current.next
        # 杀掉第k个节点
        result.append(str(current.number))
        prev.next = current.next
        current = prev.next

    # 最后剩下的一个人
    #result.append(str(current.number))
    #return ' '.join(result[:-1])  # 根据题意，只输出被杀掉的编号
    return ' '.join(result)

# 读取输入
n, k = map(int, input().split())

# 计算并输出结果
print(josephus_circle(n, k))
```

该实现的时间复杂度为 O(n*k)，对于 `n` 和 `k` 较大的情况，可以考虑优化算法，如使用数学方法求解约瑟夫斯问题的位置，但本题要求模拟过程，故采用链表方法。



```python
# 23n2300011119（武）
from collections import deque
n,k=map(int,input().split())
queue=deque(i for i in range(1,n+1))
flag=k
res=[]
# 1 2 3 4 5 6 7 8 9 10
while len(queue)>=2:
    a=queue.popleft()
    queue.append(a)
    if k-2!=0:
        for _ in range(k-2):
            a = queue.popleft()
            queue.append(a)
    b=queue.popleft()
    res.append(b)
res_new=[str(i) for i in res]
print(" ".join(res_new))
```





# 05345~20456

## 05345: 位查询

http://cs101.openjudge.cn/practice/05345/

给出N个范围在[0, 65535]的整数，编程支持以下的操作： 


（1）修改操作：C d，所有的数都增加d。如果超过65535，把结果模65536。 0 <= d <= 65535 
（2）查询操作：Q i，统计在N个正整数中有多少个整数其对应的二进制形式的第i位二进制位为非0。0 <= i <= 15。并且最低位i为0。


　　最后，输出所有查询操作的统计值。

**输入**

输入的第一行为两个正整数N,M,其中N为操作的整数的个数，而M为具体有多少个操作。
输入的第二行为N个正整数，为进行操作的N个正整数。
下面有M行，分别表示M个操作。

N<=100000,M<=200000

**输出**

输出所有查询操作Q的统计值，每一个查询操作统计结果输出为一行。

样例输入

```
3 5
1 2 4
Q 1
Q 2
C 1
Q 1
Q 2
```

样例输出

```
1
1
2
1
```

提示

只输出查询操作Q的统计值。



"最低位i为0"的意思是指编号上个位数是第零位。比如说 $4 = (100)_2$，这里面1是第二位，而不是第三位的意思。

```python
n,m  = map(int, input().split())
a = list(map(int, input().split()))
for _ in range(m):
    op, i = input().split()
    i = int(i)
    if op == 'C':
        for j in range(n):
            a[j] += i
            a[j] %= 65535
    elif op == 'Q':
        cnt = 0
        for j in a:
            s = bin(j)[2:][::-1]
            if i < len(s) and s[i] == '1':
                cnt += 1
        print(cnt)
```



```python
def modify_nums(nums, d):
    for i in range(len(nums)):
        nums[i] = (nums[i] + d) % 65536

def count_bits(nums, i):
    count = 0
    for num in nums:
        if (num >> i) & 1:
            count += 1
    return count

N, M = map(int, input().split())
nums = list(map(int, input().split()))

for _ in range(M):
    operation, value = input().split()
    if operation == 'Q':
        i = int(value)
        result = count_bits(nums, i)
        print(result)
    elif operation == 'C':
        d = int(value)
        modify_nums(nums, d)
```



## 05430: 表达式·表达式树·表达式求值

http://cs101.openjudge.cn/dsapre/05430/

众所周知，任何一个表达式，都可以用一棵表达式树来表示。例如，表达式a+b*c，可以表示为如下的表达式树：

  +
 / \
a  *
  / \
  b c

现在，给你一个中缀表达式，这个中缀表达式用变量来表示（不含数字），请你将这个中缀表达式用表达式二叉树的形式输出出来。

**输入**

输入分为三个部分。
第一部分为一行，即中缀表达式(长度不大于50)。中缀表达式可能含有小写字母代表变量（a-z），也可能含有运算符（+、-、*、/、小括号），不含有数字，也不含有空格。
第二部分为一个整数n(n < 10)，表示中缀表达式的变量数。
第三部分有n行，每行格式为C　x，C为变量的字符，x为该变量的值。

**输出**

输出分为三个部分，第一个部分为该表达式的逆波兰式，即该表达式树的后根遍历结果。占一行。
第二部分为表达式树的显示，如样例输出所示。如果该二叉树是一棵满二叉树，则最底部的叶子结点，分别占据横坐标的第1、3、5、7……个位置（最左边的坐标是1），然后它们的父结点的横坐标，在两个子结点的中间。如果不是满二叉树，则没有结点的地方，用空格填充（但请略去所有的行末空格）。每一行父结点与子结点中隔开一行，用斜杠（/）与反斜杠（\）来表示树的关系。/出现的横坐标位置为父结点的横坐标偏左一格，\出现的横坐标位置为父结点的横坐标偏右一格。也就是说，如果树高为m，则输出就有2m-1行。
第三部分为一个整数，表示将值代入变量之后，该中缀表达式的值。需要注意的一点是，除法代表整除运算，即舍弃小数点后的部分。同时，测试数据保证不会出现除以0的现象。

样例输入

```
a+b*c
3
a 2
b 7
c 5
```

样例输出

```
abc*+
   +
  / \
 a   *
    / \
    b c
37
```





```python
'''
表达式树是一种特殊的二叉树。对于你的问题，需要先将中缀表达式转换为后缀表达式
（逆波兰式），然后根据后缀表达式建立表达式树，最后进行计算。

首先使用stack进行中缀到后缀的转换，然后根据后缀表达式建立表达式二叉树，
再通过递归和映射获取表达式的值。
最后，打印出整棵树（取自 23n2300017735，夏天明BrightSummer）

中缀表达式转后缀表达式 https://zq99299.github.io/dsalg-tutorial/dsalg-java-hsp/05/05.html
'''
#from collections import deque as q
import operator as op
#import os


class Node:
    def __init__(self, x):
        self.value = x
        self.left = None
        self.right = None


def priority(x):
    if x == '*' or x == '/':
        return 2
    if x == '+' or x == '-':
        return 1
    return 0


def infix_trans(infix):
    postfix = []
    op_stack = []
    for char in infix:
        if char.isalpha():
            postfix.append(char)
        else:
            if char == '(':
                op_stack.append(char)
            elif char == ')':
                while op_stack and op_stack[-1] != '(':
                    postfix.append(op_stack.pop())
                op_stack.pop()
            else:
                while op_stack and priority(op_stack[-1]) >= priority(char) and op_stack[-1] != '(':
                    postfix.append(op_stack.pop())
                op_stack.append(char)
    while op_stack:
        postfix.append(op_stack.pop())
    return postfix


def build_tree(postfix):
    stack = []
    for item in postfix:
        if item in '+-*/':
            node = Node(item)
            node.right = stack.pop()
            node.left = stack.pop()
        else:
            node = Node(item)
        stack.append(node)
    return stack[0]


def get_val(expr_tree, var_vals):
    if expr_tree.value in '+-*/':
        operator = {'+': op.add, '-': op.sub, '*': op.mul, '/': op.floordiv}
        return operator[expr_tree.value](get_val(expr_tree.left, var_vals), get_val(expr_tree.right, var_vals))
    else:
        return var_vals[expr_tree.value]

# 计算表达式树的深度。它通过递归地计算左右子树的深度，并取两者中的最大值再加1，得到整个表达式树的深度。


def getDepth(tree_root):
    #return max([self.child[i].getDepth() if self.child[i] else 0 for i in range(2)]) + 1
    left_depth = getDepth(tree_root.left) if tree_root.left else 0
    right_depth = getDepth(tree_root.right) if tree_root.right else 0
    return max(left_depth, right_depth) + 1

    '''
    首先，根据表达式树的值和深度信息构建第一行，然后构建第二行，该行包含斜线和反斜线，
    用于表示子树的链接关系。接下来，如果当前深度为0，表示已经遍历到叶子节点，直接返回该节点的值。
    否则，递减深度并分别获取左子树和右子树的打印结果。最后，将左子树和右子树的每一行拼接在一起，
    形成完整的树形打印图。
    
打印表达式树的函数。表达式树是一种抽象数据结构，它通过树的形式来表示数学表达式。在这段程序中，
函数printExpressionTree接受两个参数：tree_root表示树的根节点，d表示树的总深度。
首先，函数会创建一个列表graph，列表中的每个元素代表树的一行。第一行包含根节点的值，
并使用空格填充左右两边以保持树的形状。第二行显示左右子树的链接情况，使用斜杠/表示有左子树，
反斜杠\表示有右子树，空格表示没有子树。

接下来，函数会判断深度d是否为0，若为0则表示已经达到树的最底层，直接返回根节点的值。否则，
将深度减1，然后递归调用printExpressionTree函数打印左子树和右子树，
并将结果分别存储在left和right中。

最后，函数通过循环遍历2倍深度加1次，将左子树和右子树的每一行连接起来，存储在graph中。
最后返回graph，即可得到打印好的表达式树。
    '''


def printExpressionTree(tree_root, d):  # d means total depth

    graph = [" "*(2**d-1) + tree_root.value + " "*(2**d-1)]
    graph.append(" "*(2**d-2) + ("/" if tree_root.left else " ")
                 + " " + ("\\" if tree_root.right else " ") + " "*(2**d-2))

    if d == 0:
        return tree_root.value
    d -= 1
    '''
    应该是因为深度每增加一层，打印宽度就增加一倍，打印行数增加两行
    '''
    #left = printExpressionTree(tree_root.left, d) if tree_root.left else [
    #    " "*(2**(d+1)-1)]*(2*d+1)
    if tree_root.left:
        left = printExpressionTree(tree_root.left, d)
    else:
        #print("left_d",d)
        left = [" "*(2**(d+1)-1)]*(2*d+1)
        #print("left_left",left)

    right = printExpressionTree(tree_root.right, d) if tree_root.right else [
        " "*(2**(d+1)-1)]*(2*d+1)

    for i in range(2*d+1):
        graph.append(left[i] + " " + right[i])
        #print('graph=',graph)
    return graph



infix = input().strip()
n = int(input())
vars_vals = {}
for i in range(n):
    line = input().split()
    vars_vals[line[0]] = int(line[1])
    
'''
infix = "a+(b-c*d*e)"
#infix = "a+b*c"
n = 5
vars_vals = {'a': 2, 'b': 7, 'c': 5, 'd':1, 'e':1}
'''

postfix = infix_trans(infix)
tree_root = build_tree(postfix)
print(''.join(str(x) for x in postfix))
expression_value = get_val(tree_root, vars_vals)


for line in printExpressionTree(tree_root, getDepth(tree_root)-1):
    print(line.rstrip())


print(expression_value)
```



## 05442: 兔子与星空

prim, kruskal, http://cs101.openjudge.cn/practice/05442/

很久很久以前，森林里住着一群兔子。兔子们无聊的时候就喜欢研究星座。如图所示，天空中已经有了n颗星星，其中有些星星有边相连。兔子们希望删除掉一些边，然后使得保留下的边仍能是n颗星星连通。他们希望计算，保留的边的权值之和最小是多少？





![img](http://media.openjudge.cn/images/upload/1353513346.jpg)

**输入**

第一行只包含一个表示星星个数的数n，n不大于26，并且这n个星星是由大写字母表里的前n个字母表示。接下来的n-1行是由字母表的前n-1个字母开头。最后一个星星表示的字母不用输入。对于每一行，以每个星星表示的字母开头，然后后面跟着一个数字，表示有多少条边可以从这个星星到后面字母表中的星星。如果k是大于0，表示该行后面会表示k条边的k个数据。每条边的数据是由表示连接到另一端星星的字母和该边的权值组成。权值是正整数的并且小于100。该行的所有数据字段分隔单一空白。该星星网络将始终连接所有的星星。该星星网络将永远不会超过75条边。没有任何一个星星会有超过15条的边连接到其他星星（之前或之后的字母）。在下面的示例输入，数据是与上面的图相一致的。

**输出**

输出是一个整数，表示最小的权值和

样例输入

```
9
A 2 B 12 I 25
B 3 C 10 H 40 I 8
C 2 D 18 G 55
D 1 E 44
E 2 F 60 G 38
F 0
G 1 H 35
H 1 I 35
```

样例输出

```
216
```

提示

考虑看成最小生成树问题，注意输入表示。



The problem you're describing is a classic Minimum Spanning Tree (MST) problem. The MST problem is a common problem in graph theory that asks for a spanning tree of a graph such that the sum of its edge weights is as small as possible.  In this case, the stars represent the nodes of the graph, and the edges between them represent the connections between the stars. The weight of each edge is given in the problem statement. The goal is to find a subset of these edges such that all stars are connected and the sum of the weights of these edges is minimized.

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))

solve()
```



思路：用的kruskal，这个算法比那些用图的算法看起来明白多了

kruskal适配这种vertex.key连续变化的情况，因为并查集建立还有find方法中都要用到列表的有序性。

```python
# 蔡嘉华 物理学院
class DisjSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0]*n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xset, yset = self.find(x), self.find(y)
        if self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
        else:
            self.parent[xset] = yset
            if self.rank[xset] == self.rank[yset]:
                self.rank[yset] += 1

def kruskal(n, edges):
    dset = DisjSet(n)
    edges.sort(key = lambda x:x[2])
    sol = 0
    for u, v, w in edges:
        u, v = ord(u)-65, ord(v)-65
        if dset.find(u) != dset.find(v):
            dset.union(u, v)
            sol += w
    if len(set(dset.find(i) for i in range(n))) > 1:
        return -1
    return sol

n = int(input())
edges = []
for _ in range(n-1):
    arr = input().split()
    root, m = arr[0], int(arr[1])
    for i in range(m):
        edges.append((root, arr[2+2*i], int(arr[3+2*i])))
print(kruskal(n, edges))
```





## 05443: 兔子与樱花

dijkstra, Floyd-Warshall, http://cs101.openjudge.cn/practice/05443/

很久很久之前，森林里住着一群兔子。有一天，兔子们希望去赏樱花，但当他们到了上野公园门口却忘记了带地图。现在兔子们想求助于你来帮他们找到公园里的最短路。

**输入**

输入分为三个部分。
第一个部分有P+1行（P<30），第一行为一个整数P，之后的P行表示上野公园的地点, 字符串长度不超过20。
第二个部分有Q+1行（Q<50），第一行为一个整数Q，之后的Q行每行分别为两个字符串与一个整数，表示这两点有直线的道路，并显示二者之间的矩离（单位为米）。
第三个部分有R+1行（R<20），第一行为一个整数R，之后的R行每行为两个字符串，表示需要求的路线。

**输出**

输出有R行，分别表示每个路线最短的走法。其中两个点之间，用->(矩离)->相隔。

样例输入

```
6
Ginza
Sensouji
Shinjukugyoen
Uenokouen
Yoyogikouen
Meijishinguu
6
Ginza Sensouji 80
Shinjukugyoen Sensouji 40
Ginza Uenokouen 35
Uenokouen Shinjukugyoen 85
Sensouji Meijishinguu 60
Meijishinguu Yoyogikouen 35
2
Uenokouen Yoyogikouen
Meijishinguu Meijishinguu
```

样例输出

```
Uenokouen->(35)->Ginza->(80)->Sensouji->(60)->Meijishinguu->(35)->Yoyogikouen
Meijishinguu
```







由于图中所有边的权值均为非负数，可以利用经典的 Dijkstra 算法：维护一个全局的距离字典，当发现更短的路径时更新之。

```python
import heapq
from collections import defaultdict

p = int(input())
points = [input().strip() for _ in range(p)]
maps = defaultdict(list)
for _ in range(int(input())):
    a, b, d = input().split()
    d = int(d)
    maps[a].append((b, d))
    maps[b].append((a, d))

def dijkstra(src, dst):
    INF = float('inf')
    dist = {point: INF for point in points}
    path = {point: "" for point in points}
    dist[src] = 0
    path[src] = src
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v, w in maps[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                path[v] = path[u] + f"->({w})->" + v
                heapq.heappush(pq, (nd, v))
    return path[dst]

for _ in range(int(input())):
    s, t = input().split()
    print(dijkstra(s, t))

```



【柯有为】思路：图的最短路径问题，用Dijkstra方法。由于是求最短路径，所以需要额外添加一个储存路径的字典，最后将路径转化为所要求的格式输出即可。

```python
import heapq

def Dijkstra(graph, start, end):
    if start == end:
        return []
    
    distances = {place:float('inf') for place in graph.keys()}
    paths = {place:[] for place in graph.keys()}
    distances[start] = 0
    heap = []
    heapq.heappush(heap, (0, [], start))

    while heap:
        d, path, cur = heapq.heappop(heap)
        for neighbor, nd in graph[cur].items():
            if d + nd < distances[neighbor]:
                distances[neighbor] = d + nd
                paths[neighbor] = path + [neighbor]
                heapq.heappush(heap, (distances[neighbor], paths[neighbor], neighbor))
    
    return paths[end]

graph = dict()
P = int(input())
for _ in range(P):
    graph[input()] = dict()
Q = int(input())
for _ in range(Q):
    u, v, d = map(str,input().split())
    graph[u][v] = graph[v][u] = int(d)

R = int(input())
for _ in range(R):
    start, end = map(str,input().split())
    path = Dijkstra(graph, start, end)
    output = start
    cur = start
    for vertex in path:
        output += f"->({graph[cur][vertex]})->{vertex}"
        cur = vertex
    print(output)
```





思路：多添加一个路径进入pos即可。

```python
# 谭琳诗倩、2200013722
import heapq
import math
def dijkstra(graph,start,end,P):
    if start == end: return []
    dist = {i:(math.inf,[]) for i in graph}
    dist[start] = (0,[start])
    pos = []
    heapq.heappush(pos,(0,start,[]))
    while pos:
        dist1,current,path = heapq.heappop(pos)
        for (next,dist2) in graph[current].items():
            if dist2+dist1 < dist[next][0]:
                dist[next] = (dist2+dist1,path+[next])
                heapq.heappush(pos,(dist1+dist2,next,path+[next]))
    return dist[end][1]

P = int(input())
graph = {input():{} for _ in range(P)}
for _ in range(int(input())):
    place1,place2,dist = input().split()
    graph[place1][place2] = graph[place2][place1] = int(dist)

for _ in range(int(input())):
    start,end = input().split()
    path = dijkstra(graph,start,end,P)
    s = start
    current = start
    for i in path:
        s += f'->({graph[current][i]})->{i}'
        current = i
    print(s)
```



【曹以楷 24物理学院】思路：Floyd-Warshall

```python
from itertools import product
from typing import List

class Solution:
    INF = 1 << 30

    def solve(self) -> None:
        n = int(input())
        locations = [input().strip() for _ in range(n)]

        distances = [[self.INF] * n for _ in range(n)]
        next_node = [[-1] * n for _ in range(n)]

        # 初始化自身到自身的距离为 0
        for i in range(n):
            distances[i][i] = 0

        # 读取边信息并初始化邻接矩阵和路径表
        for _ in range(int(input())):
            a, b, d = input().split()
            u, v, dist = locations.index(a), locations.index(b), int(d)
            if dist < distances[u][v]:
                distances[u][v] = distances[v][u] = dist
                next_node[u][v] = v
                next_node[v][u] = u

        # Floyd-Warshall 算法计算所有点对最短路径
        for k, i, j in product(range(n), repeat=3):
            if distances[i][j] > distances[i][k] + distances[k][j]:
                distances[i][j] = distances[i][k] + distances[k][j]
                next_node[i][j] = next_node[i][k]

        # 查询路径
        for _ in range(int(input())):
            a, b = input().split()
            u, v = locations.index(a), locations.index(b)
            if distances[u][v] == self.INF:
                print("No path")
            else:
                print(self.reconstruct_path(next_node, u, v, locations, distances))

    def reconstruct_path(self, next_node: List[List[int]],
                         u: int, v: int, locations: List[str], distances: List[List[int]]) -> str:
        path_indices = [u]
        while u != v:
            u = next_node[u][v]
            path_indices.append(u)

        # 构造格式化路径字符串
        result = locations[path_indices[0]]
        for i in range(1, len(path_indices)):
            from_idx, to_idx = path_indices[i - 1], path_indices[i]
            result += f"->({distances[from_idx][to_idx]})->{locations[to_idx]}"
        return result

if __name__ == "__main__":
    Solution().solve()

```





【张俊龙 24工学院】思路：定义最小距离以及走最小距离的第一步，之后三重循环维护。

> 代码的功能是：**实现一个带权无向图的最短路径查询系统**，它会先读入点和边的信息，预处理所有点对之间的最短路径（使用 Floyd-Warshall 算法），然后支持多次查询输出从一个点到另一个点的路径和路径长度。

```python
def floyd_warshall(p, length, nxt):
    for k in range(p):
        for i in range(p):
            if length[i][k] == float('inf'):
                continue
            for j in range(p):
                if length[k][j] == float('inf'):
                    continue
                if length[i][k] + length[k][j] < length[i][j]:
                    length[i][j] = length[i][k] + length[k][j]
                    nxt[i][j] = nxt[i][k]

def reconstruct_path(u, v, nxt, name, length):
    if u == v:
        return name[u]
    path = [u]
    while u != v:
        u = nxt[u][v]
        if u is None:
            return "NO PATH"
        path.append(u)
    result = name[path[0]]
    for i in range(1, len(path)):
        result += f'->({length[path[i - 1]][path[i]]})->{name[path[i]]}'
    return result

# -------------------------------
# Main
# -------------------------------
p = int(input())
name = []
di = {}
for i in range(p):
    place = input().strip()
    name.append(place)
    di[place] = i

length = [[float('inf')] * p for _ in range(p)]
nxt = [[None] * p for _ in range(p)]

for i in range(p):
    length[i][i] = 0
    nxt[i][i] = i

q = int(input())
for _ in range(q):
    a, b, c = input().split()
    u, v, d = di[a], di[b], int(c)
    if d < length[u][v]:  # Take shortest if multiple edges
        length[u][v] = length[v][u] = d
        nxt[u][v] = v
        nxt[v][u] = u

# Compute all-pairs shortest paths
floyd_warshall(p, length, nxt)

r = int(input())
for _ in range(r):
    a, b = input().split()
    u, v = di[a], di[b]
    print(reconstruct_path(u, v, nxt, name, length))

```

**`floyd_warshall(...)`：**

- 经典三重循环版本。
- 如果从 `i` 到 `j` 可以通过中间点 `k` 走得更短，则更新路径和 `nxt[i][j]`。

**`reconstruct_path(...)`：**

- 利用 `nxt` 重建路径。
- 输出格式为：`A->(距离)->B->(距离)->C`。





思路：用 $\text{Floyd}$​ 算法很容易求多源汇最短路径长度，本题重点是记录具体路径。此外可以将每个顶点都哈希以便后续的 dp 操作。

简单讲解该算法：$dp[k][x][y]$ 的含义是只允许用节点 $1\sim k$ 作为中间点时节点 $x$ 到节点 $y$ 的最短路长度；很显然 $dp[n][x][y]$ 就是节点 $x$ 到节点 $y$ 的最短路长度。然后根据经过 $k$ 点和不经过 $k$ 点两种情况分类，可以得到转移方程：

$$dp[k][x][y] = min(dp[k-1][x][y], dp[k-1][x][k]+dp[k-1][k][y]).$$​

再做滚动数组优化去掉第一维即可。

```python
# 物理学院 罗熙佑
from functools import lru_cache


@lru_cache(None)
def get_path(i, j):
    if i == j:
        return f'{rhs[i]}'

    return get_path(i, path[i][j]) + f'->({dp[path[i][j]][j]})->{rhs[j]}'


p = int(input())
hs = {input(): i for i in range(p)}
rhs = {i: name for name, i in hs.items()}

dp = [[0 if i == j else float('inf') for j in range(p)] for i in range(p)]
path = [[i for j in range(p)] for i in range(p)]  # 从i到j经过的最后一个中转点, 不中转时为起点
q = int(input())
for _ in range(q):
    a, b, w = input().split()
    a, b, w = hs[a], hs[b], int(w)
    dp[a][b] = w
    dp[b][a] = w

for k in range(p):
    for i in range(p):
        for j in range(p):
            dist = dp[i][k] + dp[k][j]
            if dist < dp[i][j]:
                dp[i][j] = dist
                path[i][j] = k  # 因为k是从小往大迭代的, 所以最后记录到的是最后一个中转点

r = int(input())
for _ in range(r):
    a, b = map(lambda x: hs[x], input().split())
    print(get_path(a, b))
    
```





使用图论中的经典算法，如迪杰斯特拉（Dijkstra）算法，来找到两点之间的最短路径。输出格式的要求,每步都需要显示两个地点和它们之间的距离。

```python
import heapq

def dijkstra(adjacency, start):
    distances = {vertex: float('infinity') for vertex in adjacency}
    previous = {vertex: None for vertex in adjacency}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in adjacency[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous

def shortest_path_to(adjacency, start, end):
    distances, previous = dijkstra(adjacency, start)
    path = []
    current = end
    while previous[current] is not None:
        path.insert(0, current)
        current = previous[current]
    path.insert(0, start)
    return path, distances[end]

# Read the input data
P = int(input())
places = {input().strip() for _ in range(P)}

Q = int(input())
graph = {place: {} for place in places}
for _ in range(Q):
    src, dest, dist = input().split()
    dist = int(dist)
    graph[src][dest] = dist
    graph[dest][src] = dist  # Assuming the graph is bidirectional

R = int(input())
requests = [input().split() for _ in range(R)]

# Process each request
for start, end in requests:
    if start == end:
        print(start)
        continue

    path, total_dist = shortest_path_to(graph, start, end)
    output = ""
    for i in range(len(path) - 1):
        output += f"{path[i]}->({graph[path[i]][path[i+1]]})->"
    output += f"{end}"
    print(output)

```



## 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

二叉搜索树在动态查表中有特别的用处，一个无序序列可以通过构造一棵二叉搜索树变成一个有序序列，

构造树的过程即为对无序序列进行排序的过程。每次插入的新的结点都是二叉搜索树上新的叶子结点，在进行

插入操作时，不必移动其它结点，只需改动某个结点的指针，由空变为非空即可。

这里，我们想探究二叉树的建立和层次输出。

**输入**

只有一行，包含若干个数字，中间用空格隔开。（数字可能会有重复，对于重复的数字，只计入一个）

**输出**

输出一行，对输入数字建立二叉搜索树后进行按层次周游的结果。

样例输入

```
51 45 59 86 45 4 15 76 60 20 61 77 62 30 2 37 13 82 19 74 2 79 79 97 33 90 11 7 29 14 50 1 96 59 91 39 34 6 72 7
```

样例输出

```
51 45 59 4 50 86 2 15 76 97 1 13 20 60 77 90 11 14 19 30 61 82 96 7 29 37 62 79 91 6 33 39 74 34 72
```

提示

输入输出的最后都不带空格和回车换行



按输入顺序遍历数字，用一个 `seen` 集合来跳过重复值，对于每个新值，插入 BST。

```python
from collections import deque
import sys

# 定义二叉树结点
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# 插入结点到 BST 中
def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    return root

# 层次遍历
def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(str(node.val))
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

def main():
    data = sys.stdin.read().strip().split()
    seen = set()
    root = None
    for tok in data:
        num = int(tok)
        if num in seen:
            continue
        seen.add(num)
        root = insert(root, num)
    # 输出层次遍历，最后不带多余空格或换行
    out = ' '.join(level_order(root))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()

```



The problem is asking to construct a binary search tree (BST) from a sequence of numbers and then perform a level order traversal (also known as breadth-first search) on the BST.

Here is a step-by-step plan:

1. Create a TreeNode class to represent each node in the tree.
2. Create a function `insert` that takes a node and a value as input and inserts the value into the BST rooted at the node.
3. Create a function `level_order_traversal` that takes the root of the tree as input and returns the level order traversal of the tree.
   - Use a queue to store the nodes to be visited.
   - While the queue is not empty, dequeue a node, visit it, and enqueue its children.
4. Read the sequence of numbers from the input, construct the BST, perform the level order traversal, and output the result.

Here is the Python code that implements this plan:

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.value:
        node.left = insert(node.left, value)
    elif value > node.value:
        node.right = insert(node.right, value)
    return node

def level_order_traversal(root):
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal

numbers = list(map(int, input().strip().split()))
numbers = list(dict.fromkeys(numbers))  # remove duplicates
root = None
for number in numbers:
    root = insert(root, number)
traversal = level_order_traversal(root)
print(' '.join(map(str, traversal)))
```

This code reads the sequence of numbers from the input, removes duplicates, constructs the BST, performs the level order traversal, and prints the result.





## 05467: 多项式加法

http://cs101.openjudge.cn/dsapre/05467/

我们经常遇到两多项式相加的情况，在这里，我们就需要用程序来模拟实现把两个多项式相加到一起。首先，我们会有两个多项式，每个多项式是独立的一行，每个多项式由系数、幂数这样的多个整数对来表示。

如多项式2x20- x17+ 5x9- 7x7+ 16x5+ 10x4 + 22x2- 15

对应的表达式为：2 20 -1 17 5 9 - 7 7 16 5 10 4 22 2 -15 0。 

**为了标记每行多项式的结束，在表达式后面加上了一个幂数为负数的整数对。**

同时输入表达式的幂数大小顺序是随机的。

我们需要做的就是把所给的两个多项式加起来。



**输入**

输入包括多行。
第一行整数n,表示有多少组的多项式需要求和。(1 < n < 100)
下面为2n行整数，每一行都是一个多项式的表达式。表示n组需要相加的多项式。
每行长度小于300。

**输出**

输出包括n行，每行为1组多项式相加的结果。
在每一行的输出结果中，多项式的每一项用“[x y]”形式的字符串表示，x是该项的系数、y 是该项的幂数。要求按照每一项的幂从高到低排列，即先输出幂数高的项、再输出幂数低的项。
系数为零的项不要输出。

样例输入

```
2
-1 17 2 20 5 9 -7 7 10 4 22 2 -15 0 16 5 0 -1
2 19 7 7 3 17 4 4 15 10 -10 5 13 2 -7 0 8 -8
-1 17 2 23 22 2 6 8 -4 7 -18 0 1 5 21 4 0 -1
12 7 -7 5 3 17 23 4 15 10 -10 5 13 5 2 19 9 -7
```

样例输出

```
[ 2 20 ] [ 2 19 ] [ 2 17 ] [ 15 10 ] [ 5 9 ] [ 6 5 ] [ 14 4 ] [ 35 2 ] [ -22 0 ]
[ 2 23 ] [ 2 19 ] [ 2 17 ] [ 15 10 ] [ 6 8 ] [ 8 7 ] [ -3 5 ] [ 44 4 ] [ 22 2 ] [ -18 0 ]
```

提示

第一组样例数据的第二行末尾的8 -8，因为幂次-8为负数，所以这一行数据结束，8 -8不要参与计算。



```python
#23n2300011072(X)
from collections import defaultdict
def add(a):
    i=0
    while 1:
        m,n=a[i],a[i+1]
        if n<0:
            break
        res[n]+=m
        i+=2
for _ in range(int(input())):
    res=defaultdict(int)
    add(list(map(int,input().split())))
    add(list(map(int,input().split())))
    for i in sorted(res,reverse=True):
        if res[i]!=0:
            print(f'[ {res[i]} {i} ] ',end='')
    print()
```



## 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/

定义一个双端队列，进队操作与普通队列一样，从队尾进入。出队操作既可以从队头，也可以从队尾。编程实现这个数据结构。

**输入**
第一行输入一个整数t，代表测试数据的组数。
每组数据的第一行输入一个整数n，表示操作的次数。
接着输入n行，每行对应一个操作，首先输入一个整数type。
当type=1，进队操作，接着输入一个整数x，表示进入队列的元素。
当type=2，出队操作，接着输入一个整数c，c=0代表从队头出队，c=1代表从队尾出队。
n <= 1000

**输出**
对于每组测试数据，输出执行完所有的操作后队列中剩余的元素,元素之间用空格隔开，按队头到队尾的顺序输出，占一行。如果队列中已经没有任何的元素，输出NULL。

样例输入

```
2
5
1 2
1 3
1 4
2 0
2 1
6
1 1
1 2
1 3
2 0
2 1
2 0
```

样例输出

```
3
NULL
```



计概的话，可以用 from collections import deque。数算课程的话，可以练习自己实现deque。

```python
class Node:
    def __init__(self, value=None):
        self.value = value
        self.next = None
        self.prev = None

class MyDeque:
    def __init__(self):
        self.head = None
        self.tail = None

    def isEmpty(self):
        return self.head is None

    def append(self, value):
        """添加元素到队尾"""
        new_node = Node(value)
        if self.isEmpty():
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def appendleft(self, value):
        """添加元素到队头"""
        new_node = Node(value)
        if self.isEmpty():
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    def pop(self):
        """从队尾移除元素"""
        if self.isEmpty():
            return None
        ret_value = self.tail.value
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        return ret_value

    def popleft(self):
        """从队头移除元素"""
        if self.isEmpty():
            return None
        ret_value = self.head.value
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        return ret_value

    def printDeque(self):
        """打印队列元素"""
        elements = []
        current = self.head
        while current:
            elements.append(current.value)
            current = current.next
        return elements

# 接下来是根据题目要求处理输入输出的部分
t = int(input())  # 测试数据的组数
for _ in range(t):
    n = int(input())  # 操作次数
    my_deque = MyDeque()
    for _ in range(n):
        parts = list(map(int, input().split()))
        if parts[0] == 1:  # 进队操作
            my_deque.append(parts[1])
        elif parts[0] == 2:  # 出队操作
            if parts[1] == 0:
                my_deque.popleft()
            else:
                my_deque.pop()
    if my_deque.isEmpty():
        print("NULL")
    else:
        print(' '.join(map(str, my_deque.printDeque())))

```



## 05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/

给定一棵二叉树，在二叉树上执行两个操作：

1. 节点交换

把二叉树的两个节点交换。
![img](https://raw.githubusercontent.com/GMyhf/img/main/img/1368411159.jpg)

2. 前驱询问

询问二叉树的一个节点对应的子树最左边的节点。
![img](https://raw.githubusercontent.com/GMyhf/img/main/img/1368411165.jpg)

**输入**

第一行输出一个整数t(t <= 100)，代表测试数据的组数。

对于每组测试数据，第一行输入两个整数n m，n代表二叉树节点的个数，m代表操作的次数。

随后输入n行，每行包含3个整数X Y Z，对应二叉树一个节点的信息。X表示节点的标识，Y表示其左孩子的标识，Z表示其右孩子的标识。

再输入m行，每行对应一次操作。每次操作首先输入一个整数type。

当type=1，节点交换操作，后面跟着输入两个整数x y，表示将标识为x的节点与标识为y的节点交换。输入保证对应的节点不是祖先关系。

当type=2，前驱询问操作，后面跟着输入一个整数x，表示询问标识为x的节点对应子树最左的孩子。

1<=n<=100，节点的标识从0到n-1，根节点始终是0.
m<=100

**输出**

对于每次询问操作，输出相应的结果。

样例输入

```
2
5 5
0 1 2
1 -1 -1
2 3 4
3 -1 -1
4 -1 -1
2 0
1 1 2
2 0
1 3 4
2 2
3 2
0 1 2
1 -1 -1
2 -1 -1
1 1 2
2 0
```

样例输出

```
1
3
4
2
```



用字典+列表的确比类方便

```python
# 数学科学学院 王镜廷 2300010724
def find_leftmost_node(son, u):
    while son[u][0] != -1:
        u = son[u][0]
    return u

def main():
    t = int(input())
    for _ in range(t):
        n, m = map(int, input().split())

        son = [-1] * (n + 1)  # 存储每个节点的子节点
        parent = {}  # 存储每个节点的父节点和方向，{节点: (父节点, 方向)}

        for _ in range(n):
            i, u, v = map(int, input().split())
            son[i] = [u, v]
            parent[u] = (i, 0)  # 左子节点
            parent[v] = (i, 1)  # 右子节点

        for _ in range(m):
            s = input().split()
            if s[0] == "1":
                u, v = map(int, s[1:])
                fu, diru = parent[u]
                fv, dirv = parent[v]
                son[fu][diru] = v
                son[fv][dirv] = u
                parent[v] = (fu, diru)
                parent[u] = (fv, dirv)
            elif s[0] == "2":
                u = int(s[1])
                root = find_leftmost_node(son, u)
                print(root)

if __name__ == "__main__":
    main()
```



```python
# 23n2300011072 蒋子轩
def swap(x, y):
    tree[loc[x][0]][loc[x][1]] = y
    tree[loc[y][0]][loc[y][1]] = x
    loc[x], loc[y] = loc[y], loc[x]


for _ in range(int(input())):
    n, m = map(int, input().split())
    tree = {}
    loc = [[] for _ in range(n)]
    for _ in range(n):
        a, b, c = map(int, input().split())
        tree[a] = [b, c]
        loc[b], loc[c] = [a, 0], [a, 1]
    for _ in range(m):
        op = list(map(int, input().split()))
        if op[0] == 1:
            swap(op[1], op[2])
        else:
            cur = op[1]
            while tree[cur][0] != -1:
                cur = tree[cur][0]
            print(cur)
```



OOP方式。同一个parent要特殊考虑。否则同一个 parent，48行相当于又交换。

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

class BinaryTree:
    def __init__(self, n):
        self.root = TreeNode(0)
        self.node_dict = {0: self.root}
        self.build_tree(n)

    def build_tree(self, n):
        for _ in range(n):
            idx, left, right = map(int, input().split())
            if idx not in self.node_dict:
                self.node_dict[idx] = TreeNode(idx)
            node = self.node_dict[idx]
            if left != -1:
                if left not in self.node_dict:
                    self.node_dict[left] = TreeNode(left)
                left_node = self.node_dict[left]
                node.left = left_node
                left_node.parent = node
            if right != -1:
                if right not in self.node_dict:
                    self.node_dict[right] = TreeNode(right)
                right_node = self.node_dict[right]
                node.right = right_node
                right_node.parent = node

    def swap_nodes(self, x, y):
        node_x = self.node_dict[x]
        node_y = self.node_dict[y]
        px, py = node_x.parent, node_y.parent

        if px == py:
            px.left, px.right = px.right, px.left
            return

        # Swap in the parent's children references
        if px.left == node_x:
            px.left = node_y
        else:
            px.right = node_y

        if py.left == node_y:
            py.left = node_x
        else:
            py.right = node_x

        # Swap their parent references
        node_x.parent, node_y.parent = py, px

    def find_leftmost_child(self, x):
        node = self.node_dict[x]
        while node.left:
            node = node.left
        return node.val

def main():
    t = int(input())
    for _ in range(t):
        n, m = map(int, input().split())
        tree = BinaryTree(n)
        for _ in range(m):
            op, *args = map(int, input().split())
            if op == 1:
                x, y = args
                tree.swap_nodes(x, y)
            elif op == 2:
                x, = args
                print(tree.find_leftmost_child(x))

if __name__ == "__main__":
    main()
```







思路：建树，注意交换节点的时候搞清楚哪个变量在引用什么，否则容易出现死循环之类的

```python
# 夏天明 元培学院
class Node:
    def __init__(self, name):
        self.name = name
        self.child = [None, None]
        self.parent = None
    
    def findLef(self):
        curr = self
        while (lef := nodes[curr.child[0]]):
            curr = lef
        return curr.name

for o in range(int(input())):
    n, m = map(int, input().split())
    nodes = [Node(i) for i in range(n)] + [None]
    for o in range(n):
        x, *idx = map(int, input().split())
        nodes[x].child = idx
        for i in [0,1]:
            if idx[i] != -1:
                nodes[idx[i]].parent = (nodes[x],i) 
    for o in range(m):
        token, *idx = map(int, input().split())
        if token == 1:
            p = [nodes[i].parent for i in idx]
            for i in [0,1]:
                p[i][0].child[p[i][1]] = idx[1-i]
                nodes[idx[i]].parent = p[1-i]
        else:
            print(nodes[idx[0]].findLef())
```





思路：没有特别的操作。20min。

```python
# 谭琳诗倩、2200013722
class BinaryTree:
    def __init__(self,root):
        self.root = root
        self.left = None
        self.right = None
        self.father = None

for _ in range(int(input())):
    n,m = map(int,input().split())
    tree_list = list(BinaryTree(i) for i in range(n))

    for __ in range(n):
        root,left,right = map(int,input().split())
        if left != -1:
            tree_list[root].left = tree_list[left]
            tree_list[left].father = tree_list[root]
        if right != -1:
            tree_list[root].right = tree_list[right]
            tree_list[right].father = tree_list[root]

    for __ in range(m):
        type,*tu = map(int,input().split())

        if type == 1: # swap
            x,y = tu
            tree1,tree2 = tree_list[x],tree_list[y]
            father1 = tree1.father
            father2 = tree2.father
            if father2 is father1:
                father2.left,father2.right = father2.right,father2.left

            else:
                if father1.left == tree1:
                    father1.left = tree2
                else: father1.right = tree2

                if father2.left == tree2:
                    father2.left = tree1
                else: father2.right = tree1
                tree1.father,tree2.father = father2,father1

        elif type == 2:
            node = tree_list[tu[0]]
            while node.left:
                node = node.left
            print(node.root)
```





```python
# 23n2300011072(X) 蒋子轩
class TreeNode:
    def __init__(self,val=0):
        self.val=val
        self.left=None
        self.right=None
def build_tree(nodes_info):
    nodes=[TreeNode(i) for i in range(n)]
    for val,left,right in nodes_info:
        if left!=-1:
            nodes[val].left=nodes[left]
        if right!=-1:
            nodes[val].right=nodes[right]
    return nodes
def swap_nodes(nodes,x,y):
    for node in nodes:
        if node.left and node.left.val in[x,y]:
            node.left=nodes[y] if node.left.val==x else nodes[x]
        if node.right and node.right.val in[x,y]:
            node.right=nodes[y] if node.right.val==x else nodes[x]
def find_leftmost(node):
    while node and node.left:
        node=node.left
    return node.val if node else -1
for _ in range(int(input())):
    n,m=map(int,input().split())
    nodes_info=[tuple(map(int,input().split())) for _ in range(n)]
    ops=[tuple(map(int,input().split())) for _ in range(m)]
    nodes=build_tree(nodes_info)
    for op in ops:
        if op[0]==1:
            swap_nodes(nodes,op[1],op[2])
        elif op[0]==2:
            print(find_leftmost(nodes[op[1]]))
```



思路：只记录父子关系即可

```python
# Xinjie Song, Phy
for _ in range(int(input())):
    n, m = map(int, input().split())
    edges = {i: [0, 0, 0] for i in range(n)}
    for i in range(n):
        x, y, z = map(int, input().split())
        edges[x][1], edges[x][2] = y, z
        if y != -1:
            edges[y][0] = x
        if z != -1:
            edges[z][0] = x
    for __ in range(m):
        s = list(map(int, input().split()))
        if s[0] == 1:
            x, y = s[1], s[2]
            x_father, y_father = edges[x][0], edges[y][0]
            x_idx, y_idx = 1 + (edges[x_father][2] == x), 1 + (edges[y_father][2] == y)
            edges[x_father][x_idx], edges[y_father][y_idx], edges[x][0], edges[y][0] = y, x, y_father, x_father
        else:
            x = s[1]
            while edges[x][1] != -1:
                x = edges[x][1]
            print(x)
```







## 06250: 字符串最大跨距

http://cs101.openjudge.cn/dsapre/06250/

有三个字符串S,S1,S2，其中，S长度不超过300，S1和S2的长度不超过10。想检测S1和S2是否同时在S中出现，且S1位于S2的左边，并在S中互不交叉（即，S1的右边界点在S2的左边界点的左侧）。计算满足上述条件的最大跨距（即，最大间隔距离：最右边的S2的起始点与最左边的S1的终止点之间的字符数目）。如果没有满足条件的S1，S2存在，则输出-1。 

例如，S = "abcd123ab888efghij45ef67kl", S1="ab", S2="ef"，其中，S1在S中出现了2次，S2也在S中出现了2次，最大跨距为：18。

**输入**

三个串：S, S1, S2，其间以逗号间隔（注意，S, S1, S2中均不含逗号和空格）；

**输出**

S1和S2在S最大跨距；若在S中没有满足条件的S1和S2，则输出-1。

样例输入

```
abcd123ab888efghij45ef67kl,ab,ef
```

样例输出

```
18
```



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



## 06263: 布尔表达式

http://cs101.openjudge.cn/practice/06263/

输入一个布尔表达式，请你输出它的真假值。
比如：( V | V ) & F & ( F | V )
V表示true，F表示false，&表示与，|表示或，!表示非。
上式的结果是F

**输入**

输入包含多行，每行一个布尔表达式，表达式中可以有空格，总长度不超过1000

**输出**

对每行输入，如果表达式为真，输出"V",否则出来"F"

样例输入

```
( V | V ) & F & ( F| V)
!V | V & V & !F & (F | V ) & (!F | F | !V & V)
(F&F|V|!V&!F&!(F|F&V))
```

样例输出

```
F
V
V
```



```python
#23n2300010834 焦晨航
while True:
    try:
        s=input()
    except EOFError:
        break
    s=s.replace('V','True').replace('F','False')
    s=s.replace('&',' and ').replace('|',' or ').replace('!',' not ')
    if eval(s):
        print('V')
    else:
        print('F')
```





```python
# 23n2300011119(武)
def ShuntingYard(l:list):
    stack,output=[],[]
    for i in l:
        if i==" ":continue
        if i in 'VF':output.append(i)
        elif i=='(':stack.append(i)
        elif i in '&|!':
            while True:
                if i=='!':break
                elif not stack:break
                elif stack[-1]=="(":
                    break
                else:output.append(stack.pop())
            stack.append(i)
        elif i==')':
            while stack[-1]!='(':
                output.append(stack.pop())
            stack.pop()
    if stack:output.extend(reversed(stack))
    return output

def Bool_shift(a):
    if a=='V':return True
    elif a=='F':return False
    elif a==True:return 'V'
    elif a==False:return 'F'

def cal(a,operate,b=None):
    if operate=="&":return Bool_shift(Bool_shift(a) and Bool_shift(b))
    if operate=="|":return Bool_shift(Bool_shift(a) or Bool_shift(b))
    if operate=="!":return Bool_shift(not Bool_shift(a))

def post_cal(l:list):
    stack=[]
    for i in l:
        if i in 'VF':stack.append(i)
        elif i in "&|!":
            if i=="!":
                stack.append(cal(stack.pop(),'!'))
            else:
                a,b=stack.pop(),stack.pop()
                stack.append(cal(a,i,b))
    return stack[0]

while True:
    try:print(post_cal(ShuntingYard(list(input()))))
    except EOFError:break
```





## 06364: 牛的选举

http://cs101.openjudge.cn/dsapre/06364/

现在有N（1<=N<=50000）头牛在选举它们的总统，选举包括两轮：第一轮投票选举出票数最多的K（1<=K<=N）头牛进入第二轮；第二轮对K头牛重新投票，票数最多的牛当选为总统。



现在给出每头牛i在第一轮期望获得的票数Ai（1<=Ai<=1,000,000,000），以及在第二轮中（假设它进入第二轮）期望获得的票数Bi（1<=Bi<=1,000,000,000），请你预测一下哪头牛将当选总统。幸运的是，每轮投票都不会出现票数相同的情况。    



**输入**

第1行：N和K
第2至N+1行：第i+1行包括两个数字：Ai和Bi

**输出**

当选总统的牛的编号（牛的编号从1开始）

样例输入

```
5 3
3 10
9 2
5 6
8 4
6 5
```

样例输出

```
5
```



```python
n, k = map(int, input().split())
cows = []
for i in range(n):
    a, b = map(int, input().split())
    cows.append((a, b, i + 1))
cows.sort(key=lambda x: x[0], reverse=True)
second_round_cows = cows[:k]
second_round_cows.sort(key=lambda x: x[1], reverse=True)
print(second_round_cows[0][2])
```





## M06640: 倒排索引

string, implementation, data structure, http://cs101.openjudge.cn/practice/06640/

给定一些文档，要求求出某些单词的倒排表。

对于一个单词，它的倒排表的内容为出现这个单词的文档编号。

**输入**

第一行包含一个数N，1 <= N <= 1000，表示文档数。
接下来N行，每行第一个数ci，表示第i个文档的单词数。接下来跟着ci个用空格隔开的单词，表示第i个文档包含的单词。文档从1开始编号。1 <= ci <= 100。
接下来一行包含一个数M，1 <= M <= 1000，表示查询数。
接下来M行，每行包含一个单词，表示需要输出倒排表的单词。
每个单词全部由小写字母组成，长度不会超过256个字符，大多数不会超过10个字符。

**输出**

对于每一个进行查询的单词，输出它的倒排表，文档编号按从小到大排序。
如果倒排表为空，输出"NOT FOUND"。

样例输入

```
3
2 hello world
4 the world is great
2 great news
4
hello
world
great
pku
```

样例输出

```
1
1 2
2 3
NOT FOUND
```



要实现一个程序来创建和查询倒排索引，可以使用 字典结构来高效地完成任务。以下是具体的步骤：

1. 首先，解析输入，为每个单词构建倒排索引，即记录每个单词出现在哪些文档中。
2. 使用字典存储倒排索引，其中键为单词，值为一个有序列表，列表中包含出现该单词的文档编号。
3. 对于每个查询，检查字典中是否存在该单词，如果存在，则返回升序文档编号列表；如果不存在，则返回 "NOT FOUND"。

```python
from collections import defaultdict
def main():
    n = int(input())
    index = 1
    inverted_index = defaultdict(set)   # 构建倒排索引
    for i in range(1, n + 1):
        parts = input().split()
        doc_id = i
        num_words = int(parts[0])
        words = parts[1:num_words + 1]
        for word in words:
            inverted_index[word].add(doc_id)

    m = int(input())
    results = []

    # 查询倒排索引
    for _ in range(m):
        query = input()
        if query in inverted_index:
            results.append(" ".join(map(str, sorted(list(inverted_index[query])))))
        else:
            results.append("NOT FOUND")

    # 输出查询结果
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
```



其实每行第一个数字根本没用，读成一个str列表之后取[1:]就行了
其他数据结构也完全不需要

```python
N = int(input())
lst = []
for _ in range(N):
    lst.append(list(input().split())[1:])
M = int(input())
for _ in range(M):
    s = input().strip()
    ans = []
    for i,m in enumerate(lst):
        if s in m:
            ans.append(i+1)
    print(' '.join(map(str,ans)))
```





## 06646: 二叉树的深度

http://cs101.openjudge.cn/dsapre/06646/

给定一棵二叉树，求该二叉树的深度

二叉树深度定义：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的节点个数为树的深度

**输入**

第一行是一个整数n，表示二叉树的结点个数。二叉树结点编号从1到n，根结点为1，n <= 10
接下来有n行，依次对应二叉树的n个节点。
每行有两个整数，分别表示该节点的左儿子和右儿子的节点编号。如果第一个（第二个）数为-1则表示没有左（右）儿子

**输出**

输出一个整型数，表示树的深度

样例输入

```
3
2 3
-1 -1
-1 -1
```

样例输出

```
2
```



```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(nodes):
    if not nodes:
        return None

    tree_nodes = [None] * (len(nodes) + 1)
    for i in range(1, len(nodes) + 1):
        tree_nodes[i] = TreeNode(i)

    for i, (left, right) in enumerate(nodes, start=1):
        if left != -1:
            tree_nodes[i].left = tree_nodes[left]
        if right != -1:
            tree_nodes[i].right = tree_nodes[right]

    return tree_nodes[1]


def tree_depth(root):
    if not root:
        return 0
    left_depth = tree_depth(root.left)
    right_depth = tree_depth(root.right)
    return max(left_depth, right_depth) + 1


def main():
    n = int(input())
    nodes = []
    index = 1
    for _ in range(n):
        left, right = map(int, input().split())
        nodes.append((left, right))

    root = build_tree(nodes)
    depth = tree_depth(root)
    print(depth)


if __name__ == "__main__":
    main()

```



```python
class TreeNode:
    # 二叉树节点定义
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def build_tree(node_list):
    # 根据节点信息构建二叉树
    nodes = {i: TreeNode(i) for i in range(1, len(node_list) + 1)}
    for i, (left, right) in enumerate(node_list, 1):
        if left != -1:
            nodes[i].left = nodes[left]
        if right != -1:
            nodes[i].right = nodes[right]
    return nodes[1]  # 返回树的根节点

def max_depth(root):
    # 计算二叉树的最大深度
    if root is None:
        return 0
    else:
        left_depth = max_depth(root.left)
        right_depth = max_depth(root.right)
        return max(left_depth, right_depth) + 1

# 读取输入并解析
n = int(input())
node_list = []
for _ in range(n):
    left, right = map(int, input().split())
    node_list.append((left, right))

# 构建二叉树并计算深度
root = build_tree(node_list)
depth = max_depth(root)

# 输出结果
print(depth)
```





```python
# 钟明衡 物理学院
# 用两个列表来存储每个节点左右子树的索引，判断深度用dfs进行先序遍历
ans, l, r = 1, [-1], [-1]


def dfs(n, count):
    global ans, l, r
    if l[n] != -1:
        dfs(l[n], count + 1)
    if r[n] != -1:
        dfs(r[n], count + 1)
    ans = max(ans, count)


n = int(input())
for i in range(n):
    a, b = map(int, input().split())
    l.append(a)
    r.append(b)
dfs(1, 1)
print(ans)
```



## 06648: Sequence

http://cs101.openjudge.cn/practice/06648/

英文版，http://cs101.openjudge.cn/practice/02442/



给定m个数字序列，每个序列包含n个非负整数。我们从每一个序列中选取一个数字组成一个新的序列，显然一共可以构造出`n^m`个新序列。接下来我们对每一个新的序列中的数字进行求和，一共会得到`n^m`个和，请找出最小的n个和

**输入**

输入的第一行是一个整数T，表示测试用例的数量，接下来是T个测试用例的输入
每个测试用例输入的第一行是两个正整数m（0 < m <= 100）和n(0 < n <= 2000)，然后有m行，每行有n个数，数字之间用空格分开，表示这m个序列
序列中的数字不会大于10000

**输出**

对每组测试用例，输出一行用空格隔开的数，表示最小的n个和

样例输入

```
1
2 3
1 2 3
2 2 3
```

样例输出

```
3 3 4
```



思路：输入时将各条序列sort，先只考虑两条序列，（0,0）一定最小，用heapq存储，下一步最小一定在（i+1, j）和（i, j+1）之间，以此类推找到最小的n个存为序列seq，再将seq与第三条序列重复操作，以此类推。注意m=1的情况。

```python
import sys
import heapq

def merge(arr1, arr2, n):
    """
    将两个有序数组 arr1 和 arr2 合并，求出所有组合中最小的 n 个和
    使用堆来进行合并搜索
    """
    heap = []
    visited = set()
    # 初始候选项：(arr1[0]+arr2[0], 0, 0)
    heapq.heappush(heap, (arr1[0] + arr2[0], 0, 0))
    visited.add((0, 0))
    result = []
    while len(result) < n:
        s, i, j = heapq.heappop(heap)
        result.append(s)
        # 如果 arr1 中的下一个数存在，尝试加入候选项
        if i + 1 < n and (i + 1, j) not in visited:
            heapq.heappush(heap, (arr1[i + 1] + arr2[j], i + 1, j))
            visited.add((i + 1, j))
        # 如果 arr2 中的下一个数存在，尝试加入候选项
        if j + 1 < n and (i, j + 1) not in visited:
            heapq.heappush(heap, (arr1[i] + arr2[j + 1], i, j + 1))
            visited.add((i, j + 1))
    return result

def main():
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    T = int(next(it))
    results = []
    for _ in range(T):
        m = int(next(it))
        n = int(next(it))
        # 读取第一个序列，并排序
        current = sorted(int(next(it)) for _ in range(n))
        # 依次与后续的 m-1 个序列合并
        for _ in range(m - 1):
            seq = sorted(int(next(it)) for _ in range(n))
            current = merge(current, seq, n)
        results.append(" ".join(map(str, current)))
    sys.stdout.write("\n".join(results))

if __name__ == "__main__":
    main()
```





虑到n^m个和的数量可能非常大，我们不能直接存储它们。因此，我们可以通过逐步合并两个序列来找到最小的n个和，而不是一次性生成所有可能的和。

为了找到最小的n个和，我们可以按照以下步骤操作：

1. 对每个序列进行排序，确保我们可以从最小的元素开始处理。
2. 使用一个最小堆（优先队列）来维护当前可能的最小和。最初，我们只将每个序列的最小元素（即每个序列的第一个元素）的和放入最小堆中。
3. 每次从最小堆中取出当前的最小和，然后探索通过替换这个和中的某个元素来得到下一个可能的最小和。
4. 重复这个过程，直到我们找到了n个最小的和。

以下是一个更为内存效率的Python代码解决方案：



```python
import heapq

t = int(input())
for _ in range(t):
    m, n = map(int, input().split())
    seq1 = sorted(map(int, input().split()))
    for _ in range(m - 1):
        seq2 = sorted(map(int, input().split()))

        # 使用最小堆存储可能的最小和以及对应的索引
        min_heap = [(seq1[i] + seq2[0], i, 0) for i in range(n)]
        heapq.heapify(min_heap)
        result = []
        for _ in range(n):
            current_sum, i, j = heapq.heappop(min_heap)
            result.append(current_sum)
            if j + 1 < len(seq2):
                heapq.heappush(min_heap, (seq1[i] + seq2[j + 1], i, j + 1))
        seq1 = result
    print(*seq1)
```

> **这个解题思路，类似于机器学习中梯度下降减少loss？** 
>
> 你的观察非常准确！这个解题思路在某种程度上确实类似于机器学习中的梯度下降算法，尤其是在减少“损失”（或目标函数）的过程中。分析两者的相似之处：
>
> **梯度下降与当前算法的相似性**
>
> 1. **迭代优化**：
>    - **梯度下降**：通过迭代地调整参数，逐步减小损失函数的值，最终达到最优解。
>    - **当前算法**：通过迭代地合并序列，每次选择当前最小的和，逐步优化结果序列。
>
> 2. **局部最优选择**：
>    - **梯度下降**：在每一步选择使损失函数下降最快的方向（即梯度的反方向）。
>    - **当前算法**：使用最小堆（优先队列）每次选择当前最小的和，确保每一步都是局部最优的选择。
>
> 3. **逐步逼近全局最优**：
>    - **梯度下降**：通过不断迭代，期望最终达到全局最优（在凸函数的情况下）。
>    - **当前算法**：通过逐步合并序列，最终得到一个优化后的结果序列。
>
> **类比梯度下降的关键点**
>
> - **选择最小损失**：在梯度下降中，选择使损失函数下降最快的方向；在这里，选择当前最小的和。
> - **更新参数**：梯度下降中更新模型参数；这里更新结果序列`seq1`。
> - **迭代过程**：两者都是通过多次迭代逐步逼近最优解。
>
> **总结**
>
> 你的解题思路确实与梯度下降算法在概念上有许多相似之处，尤其是在迭代优化和局部最优选择方面。这种贪心策略通过每一步选择当前最优的选项，最终得到一个整体较优的结果，类似于梯度下降通过每一步的微小调整逐步减少损失函数值。
>
> 这种相似性不仅帮助理解算法的工作原理，也展示了不同领域中优化问题的共通性。继续深入理解这些概念，可以帮助你在解决更复杂的问题时更加得心应手！



```python
import heapq

def merge_sequences(seq1, seq2, n):
    # 对两个序列进行排序
    seq1.sort()
    seq2.sort()
    # 使用最小堆存储可能的最小和以及对应的索引
    min_heap = [(seq1[i] + seq2[0], i, 0) for i in range(len(seq1))]
    # 生成最小n个和
    result = []
    while n > 0 and min_heap:
        current_sum, i, j = heapq.heappop(min_heap)
        result.append(current_sum)
        if j + 1 < len(seq2):
            heapq.heappush(min_heap, (seq1[i] + seq2[j + 1], i, j + 1))
        n -= 1
    return result

def min_sequence_sums(m, n, sequences):
    # 对所有序列进行排序
    for seq in sequences:
        seq.sort()
    # 逐步合并序列
    current_min_sums = sequences[0]
    for i in range(1, m):
        current_min_sums = merge_sequences(current_min_sums, sequences[i], n)
    return current_min_sums

# 读取输入数据
T = int(input())  # 读取测试用例的数量
for _ in range(T):
    m, n = map(int, input().split())  # 对于每个测试用例，读取m和n
    sequences = [list(map(int, input().split())) for _ in range(m)]
    results = min_sequence_sums(m, n, sequences)
    print(' '.join(map(str, results[:n])))
```

这段代码定义了两个函数：`merge_sequences` 用于合并两个已排序的序列并找到最小的n个和，而`min_sequence_sums` 用于逐步合并所有序列。注意，由于题目要求输出最小的n个和，所以每次合并操作后我们仅保留n个和。这样可以保证内存使用量不会超过题目要求的限制。





## 07161: 森林的带度数层次序列存储

http://cs101.openjudge.cn/dsapre/07161/

对于树和森林等非线性结构，我们往往需要将其序列化以便存储。有一种树的存储方式称为带度数的层次序列。我们可以通过层次遍历的方式将森林序列转化为多个带度数的层次序列。

例如对于以下森林：

<img src="http://media.openjudge.cn/images/upload/1401904592.png" alt="img" style="zoom:50%;" />

两棵树的层次遍历序列分别为：C E F G K H J / D X I

每个结点对应的度数为：3 3 0 0 0 0 0 / 2 0 0

我们将以上序列存储起来，就可以在以后的应用中恢复这个森林。在存储中，我们可以将第一棵树表示为C 3 E 3 F 0 G 0 K 0 H 0 J 0，第二棵树表示为D 2 X 0 I 0。



现在有一些通过带度数的层次遍历序列存储的森林数据，为了能够对这些数据进行进一步处理，首先需要恢复他们。



**输入**

输入数据的第一行包括一个正整数n，表示森林中非空的树的数目。
随后的 n 行，每行给出一棵树的带度数的层次序列。
树的节点名称为A-Z的单个大写字母。

**输出**

输出包括一行，输出对应森林的后根遍历序列。

样例输入

```
2
C 3 E 3 F 0 G 0 K 0 H 0 J 0
D 2 X 0 I 0
```

样例输出

```
K H J E F G C X I D
```





利用队列根据“带度数的层次序列”构造树，然后进行后根遍历。代码如下：

```python
from collections import deque
import sys

# 定义树的结点
class Node:
    def __init__(self, value, degree):
        self.value = value
        self.degree = degree
        self.children = []

# 根据带度数的层次序列构造树
def build_tree(tokens):
    # tokens 的格式：[字母, 度数, 字母, 度数, ...]
    # 第一个结点为根
    root = Node(tokens[0], int(tokens[1]))
    queue = deque([root])
    index = 2  # 下一个待处理的token索引
    while queue and index < len(tokens):
        current = queue.popleft()
        # current.degree 个孩子依次出现在 tokens 中
        for _ in range(current.degree):
            # 每个孩子由两个元素构成：字母和度数
            child = Node(tokens[index], int(tokens[index+1]))
            current.children.append(child)
            queue.append(child)
            index += 2
    return root

# 后根遍历（后序遍历）：先遍历所有子树，再访问根节点
def postorder(node, output):
    for child in node.children:
        postorder(child, output)
    output.append(node.value)

def main():
    input_lines = sys.stdin.read().splitlines()
    if not input_lines:
        return
    n = int(input_lines[0].strip())
    result = []
    # 对于每一棵树进行构造并后序遍历
    for i in range(1, n+1):
        # 将一行的内容按空格分割
        tokens = input_lines[i].split()
        if not tokens:
            continue
        root = build_tree(tokens)
        temp = []
        postorder(root, temp)
        result.extend(temp)
    # 输出时以空格分隔各个结点
    print(" ".join(result))

if __name__ == "__main__":
    main()
```

代码说明

1. **构造树：**  
   利用队列按照层次顺序为每个结点分配其子结点，注意每个结点的子结点个数由紧随其后读入的数字决定。

2. **后根遍历：**  
   递归地先遍历所有子树，然后将当前结点值加入输出列表。

3. **主函数：**  
   读取输入，依次构造每棵树，并将每棵树的后序遍历结果依次合并，最后按要求格式输出。



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
from collections import deque

def postorder(node, children, result):
    if node in children:
        for child in children[node]:
            postorder(child, children, result)
    result.append(node)

def process_tree(tree_data):
    nodes = deque()
    children = {}
    it = iter(tree_data)
    root = next(it)
    degree = int(next(it))
    nodes.append((root, degree))
    children[root] = []

    while nodes:
        current_node, degree = nodes.popleft()
        for _ in range(degree):
            if not it:
                break
            child = next(it)
            child_degree = int(next(it))
            if current_node not in children:
                children[current_node] = []
            children[current_node].append(child)
            nodes.append((child, child_degree))
            children[child] = []

    # 后根遍历
    result = []
    postorder(root, children, result)
    return result

def main():
    n = int(input().strip())
    forest_data = []
    for _ in range(n):
        forest_data.append(input().strip().split())

    results = []
    for tree_data in forest_data:
        result = process_tree(tree_data)
        results.extend(result)

    print(" ".join(results))

if __name__ == "__main__":
    main()

```



## 07206: 我是最快的马

bfs, http://cs101.openjudge.cn/practice/07206/

我们都知道，在中国象棋中，马是走日字步的。现给定马的起始坐标与终点坐标，求出马最快能到达的路线。如果有多条路线都是步数最少的，则输出路线的数目
注意，此时棋盘上可能会有一些其它的棋子，这些棋子是会憋马脚的，注意！

**输入**

前两行为起点坐标与终点坐标，第三行为一个数字M，之后的M行为棋盘上其它棋子的坐标(M<32,坐标行列值<=10)

**输出**

如果最快路线唯一，则输出路线。否则只输出一个数字，为最快路线的数目N

样例输入

```
0 0
2 4
1
1 1
```

样例输出

```
(0,0)-(1,2)-(2,4)
```



<img src="/Users/hfyan/Library/Application Support/typora-user-images/image-20240525200326139.png" alt="image-20240525200326139" style="zoom: 50%;" />





```python
def solve():
    import sys
    from collections import deque
    start_r, start_c = map(int, input().split())
    end_r, end_c = map(int, input().split())
    M = int(input())
    obstacles = set()
    for _ in range(M):
        r, c = map(int, input().split())
        obstacles.add((r, c))

    # 棋盘坐标范围
    MIN, MAX = 0, 10

    # 马的走法：每个元组为(dx, dy, (block_dx, block_dy))
    moves = [
        (2, 1, (1, 0)), (2, -1, (1, 0)),
        (-2, 1, (-1, 0)), (-2, -1, (-1, 0)),
        (1, 2, (0, 1)), (-1, 2, (0, 1)),
        (1, -2, (0, -1)), (-1, -2, (0, -1))
    ]

    def in_bounds(pos):
        r, c = pos
        return MIN <= r <= MAX and MIN <= c <= MAX

    start = (start_r, start_c)
    end = (end_r, end_c)

    # BFS：dist记录最短步数，ways记录最短路径数，prev在唯一路径时记录前驱便于重构路径
    dist, ways, prev = {}, {}, {}
    dist[start] = 0
    ways[start] = 1
    prev[start] = None
    q = deque([start])

    while q:
        cur = q.popleft()
        r, c = cur
        for dx, dy, (br, bc) in moves:
            # 计算“蹩马腿”所在位置
            block = (r + br, c + bc)
            # 如果该位置在棋盘上且被其它棋子占据，则该方向走法被阻挡
            if in_bounds(block) and block in obstacles:
                continue
            newPos = (r + dx, c + dy)
            if not in_bounds(newPos) or newPos in obstacles:
                continue
            nd = dist[cur] + 1
            if newPos not in dist:
                dist[newPos] = nd
                ways[newPos] = ways[cur]
                # 若当前点只有一条路径，则记录前驱；如果不唯一，则设为None
                prev[newPos] = cur if ways[cur] == 1 else None
                q.append(newPos)
            elif nd == dist[newPos]:
                ways[newPos] += ways[cur]
                # 前驱不再唯一
                prev[newPos] = None

    # 如果目标不可达，输出"无解"
    if end not in dist:
        print("无解")
        return

    # 如果只有唯一一条最短路径，则重构路径输出格式 "(r,c)-(r,c)-..."
    if ways[end] == 1:
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        route_str = "-".join("({},{})".format(r, c) for r, c in path)
        print(route_str)
    else:
        # 多条最短路径则输出路径数目
        print(str(ways[end]))


if __name__ == '__main__':
    solve()

"""
0 0
2 2
0

16
"""
```

另外，题面“坐标行列值<=10” 没有说明是否可以到-10。如果 MIN, MAX = -10, 10范围，上例输入情况下结果是54。



## E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

阿尔吉侬是一只聪明又慵懒的小白鼠，它最擅长的就是走各种各样的迷宫。今天它要挑战一个非常大的迷宫，研究员们为了鼓励阿尔吉侬尽快到达终点，就在终点放了一块阿尔吉侬最喜欢的奶酪。现在研究员们想知道，如果阿尔吉侬足够聪明，它最少需要多少时间就能吃到奶酪。

  迷宫用一个R×C的字符矩阵来表示。字符S表示阿尔吉侬所在的位置，字符E表示奶酪所在的位置，字符#表示墙壁，字符.表示可以通行。阿尔吉侬在1个单位时间内可以从当前的位置走到它上下左右四个方向上的任意一个位置，但不能走出地图边界。

**输入**

第一行是一个正整数T（1 <= T <= 10），表示一共有T组数据。
每一组数据的第一行包含了两个用空格分开的正整数R和C（2 <= R, C <= 200），表示地图是一个R×C的矩阵。
接下来的R行描述了地图的具体内容，每一行包含了C个字符。字符含义如题目描述中所述。保证有且仅有一个S和E。

**输出**

对于每一组数据，输出阿尔吉侬吃到奶酪的最少单位时间。若阿尔吉侬无法吃到奶酪，则输出“oop!”（只输出引号里面的内容，不输出引号）。每组数据的输出结果占一行。

样例输入

```
3
3 4
.S..
###.
..E.
3 4
.S..
.E..
....
3 4
.S..
####
..E.
```

样例输出

```
5
1
oop!
```



这道题的意思很清晰：标准的**迷宫最短路径**问题。

**思路总结**：

- 用**BFS（宽度优先搜索）**，因为BFS可以保证第一次到达终点时就是最短路径。
- 每次从当前位置向上、下、左、右四个方向扩展。
- 遇到墙壁（`#`）或者越界，跳过。
- 记录访问过的位置，避免走回头路。
- 如果遍历完也没有到达终点，输出 `"oop!"`。

完整的代码：

```python
from collections import deque

def solve_maze():
    T = int(input())
    for _ in range(T):
        R, C = map(int, input().split())
        maze = [list(input().strip()) for _ in range(R)]

        # 找起点 S
        for i in range(R):
            for j in range(C):
                if maze[i][j] == 'S':
                    start = (i, j)
                if maze[i][j] == 'E':
                    end = (i, j)

        # BFS
        queue = deque()
        visited = [[False] * C for _ in range(R)]
        queue.append((start[0], start[1], 0))  # (row, col, distance)
        visited[start[0]][start[1]] = True

        found = False

        while queue:
            x, y, dist = queue.popleft()
            if (x, y) == end:
                print(dist)
                found = True
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < R and 0 <= ny < C:
                    if not visited[nx][ny] and maze[nx][ny] != '#':
                        visited[nx][ny] = True
                        queue.append((nx, ny, dist + 1))

        if not found:
            print("oop!")

# 调用主函数
solve_maze()
```

------

**简短解释一下代码逻辑：**

- `queue` 中保存 `(当前x, 当前y, 走到这里用的时间/步数)`。
- 每次扩展上下左右四个方向。
- 走到终点 `E` 时，直接输出当前走了多少步。
- 如果最后 `queue` 为空了还没到达 `E`，说明无法到达，输出 `"oop!"`。





## 07734: 虫子的生活

disjoint set, http://cs101.openjudge.cn/practice/07734/

Hopper 博士正在研究一种罕见种类的虫子的性行为。他假定虫子只表现为两种性别，并且虫子只与异性进行交互。在他的实验中，不同的虫子个体和虫子的交互行为很容易区分开来，因为这些虫子的背上都被标记有一些标号。

现在给定一系列的虫子的交互，现在让你判断实验的结果是否验证了他的关于没有同性恋的虫子的假设或者是否存在一些虫子之间的交互证明假设是错的。

**输入**

输入的第一行包含实验的组数。每组实验数据第一行是虫子的个数（至少1个，最多2000个) 和交互的次数 (最多1000000次) ，以空格间隔. 在下面的几行中,每次交互通过给出交互的两个虫子的标号来表示，标号之间以空格间隔。已知虫子从1开始连续编号。

**输出**

每组测试数据的输出为2行，第一行包含 "Scenario #i:", 其中 i 是实验数据组数的标号，从1开始,第二行为 "No suspicious bugs found!" 如果实验结果和博士的假设相符,或 "Suspicious bugs found!" 如果Hopper的博士的假设是错误的

样例输入

```
2
3 3
1 2
2 3
1 3
4 2
1 2
3 4
```

样例输出

```
Scenario #1:
Suspicious bugs found!

Scenario #2:
No suspicious bugs found!
```



注意每组数据中间有个空行

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


def solve_bug_life(scenarios):
    for i in range(1, scenarios + 1):
        n, m = map(int, input().split())
        uf = UnionFind(2 * n + 1)  # 为每个虫子创建两个节点表示其可能的两种性别
        suspicious = False
        for _ in range(m):
            u, v = map(int, input().split())
            if suspicious:
                continue

            if uf.is_connected(u, v):
                suspicious = True
            uf.union(u, v + n)  # 将u的一种性别与v的另一种性别关联
            uf.union(u + n, v)  # 同理


        print(f'Scenario #{i}:')
        print('Suspicious bugs found!' if suspicious else 'No suspicious bugs found!')
        print()


# 读取场景数量并解决问题
scenarios = int(input())
solve_bug_life(scenarios)
```





## 07735: 道路

http://cs101.openjudge.cn/practice/07735/

同 01724: ROADS，Dijkstra, dfs with pruning, http://cs101.openjudge.cn/practice/01724/

```python
import heapq
from collections import defaultdict

MAX_COINS = int(input())  # 最大金币数
CITY_COUNT = int(input())  # 城市数目
ROAD_COUNT = int(input())

# 存储道路信息的字典，使用 defaultdict 初始化
roads = defaultdict(list)

for _ in range(ROAD_COUNT):
    start, end, length, money = map(int, input().split())
    start, end = start - 1, end - 1
    roads[start].append((end, length, money))

def bfs(start, end, max_coins):
    queue = [(0, max_coins, start)]  # (距离, 剩余金币, 当前城市)
    visited = set()

    while queue:
        distance, coins, city = heapq.heappop(queue)

        if city == end:
            return distance

        visited.add((city, coins))

        for next_city, road_length, road_money in roads[city]:
            if coins >= road_money:
                new_distance = distance + road_length
                if (next_city, coins - road_money) not in visited:
                    heapq.heappush(queue, (new_distance, coins - road_money, next_city))

    return -1

print(bfs(0, CITY_COUNT - 1, MAX_COINS))
```



思路：要在Dijkstra算法的基础上实现剪枝，我们用一个数组记录到当前节点的最小费用（初始为无穷大），若优先队列弹出的元素满足费用大于最小费用，则该节点之前一定被访问过，且存在一种走法比当前走法费用和长度都小。故这个元素一定是无效的，可以舍去，从而实现了剪枝。

题解中直接删去visited检查也可以。但理论上内存应该会更大。

```python
# 应硕丞 数学科学学院
import heapq
k = int(input())
n = int(input())
r = int(input())
graph = {i:[] for i in range(1, n+1)}
for _ in range(r):
    s, d, dl, dt = map(int, input().split())
    graph[s].append((dl,dt,d))
que = [(0,0,1)]
fee = [10000]*101
def dijkstra(g):
    while que:
        l, t, d = heapq.heappop(que)
        if d == n:
            return l
        if t>fee[d]:
            continue
        fee[d] = t
        for dl, dt, next_d in g[d]:
            if t+dt <= k:
                heapq.heappush(que,(l+dl, t+dt, next_d))
    return -1
print(dijkstra(graph))
```



思路：dijkstra变体，要删去visited检查。

```python
#何秉儒 物理学院
import heapq

def dijkstra(g):
    while pq:
        dist,node,fee = heapq.heappop(pq)
        if node == n-1 :
            return dist
        for nei,w,f in g[node]:
            n_dist = dist + w
            n_fee = fee + f
            if n_fee <= k:
                dists[nei] = n_dist
                heapq.heappush(pq,(n_dist,nei,n_fee))
    return -1

k,n,r = int(input()),int(input()),int(input())
g = [[] for _ in range(n)]
for i in range(r):
    s,d,l,t = map(int,input().split())
    g[s-1].append((d-1,l,t)) #node,dist,fee

pq = [(0,0,0)] #dist,node,fee
dists = [float('inf')] * n
dists[0] = 0
spend = 0

result = dijkstra(g)
print(result)

```





## 07745: 整数奇偶排序

http://cs101.openjudge.cn/dsapre/07745/

给定10个整数的序列，要求对其重新排序。排序要求:

1.奇数在前，偶数在后；

2.奇数按从大到小排序；

3.偶数按从小到大排序。



**输入**

输入一行，包含10个整数，彼此以一个空格分开，每个整数的范围是大于等于0，小于等于100。

**输出**

按照要求排序后输出一行，包含排序后的10个整数，数与数之间以一个空格分开。

样例输入

`4 7 3 13 11 12 0 47 34 98`

样例输出

`47 13 11 7 3 0 4 12 34 98`

来源: 1873



```python
# 读取输入的整数序列
numbers = list(map(int, input().split()))

# 将整数序列分为奇数列表和偶数列表
odd_numbers = [num for num in numbers if num % 2 == 1]
even_numbers = [num for num in numbers if num % 2 == 0]

# 对奇数列表按照从大到小的顺序排序
odd_numbers.sort(reverse=True)

# 对偶数列表按照从小到大的顺序排序
even_numbers.sort()

# 合并排序后的奇数列表和偶数列表
sorted_numbers = odd_numbers + even_numbers

# 输出结果
print(' '.join(map(str, sorted_numbers)))
```



## 07576: 败方树

http://cs101.openjudge.cn/practice/07576/

给定一个整数数组，要求对数组中的元素构建败方树（数组相邻元素两两比较，从第一个元素开始）。之后修改数组中的元素，要求输出初始构建以及修改后得到的败方树的所有内部结点代表的整数(从左到右从上到下输出）

**输入**

第一行为数组的元素个数n和修改的次数m。
第二行为n个整数，即数组的元素。
接下来m行代表m次修改操作，每次操作修改数组中的一个元素，每一行包括两个整数，第一个为被修改元素在数组中的标号，第二个为修改之后的元素值。

**输出**

输出m+1行。
第一行为初始构建的败方树的所有内部结点代表的整数(按照树的结点从左到右从上到下的顺序输出）
接下来m行为接下来m次修改后得到的败方树的所有内部结点代表的整数(按照树的结点从左到右从上到下的顺序输出）

样例输入

```
8 1
10 9 20 6 16 12 90 17
3 15
```

样例输出

```
6 12 9 17 10 20 16 90
9 12 15 17 10 20 16 90
```



从Python 3.7开始，提供了一个新的内置装饰器——`@dataclass`

`@dataclass` 自动为 `Product` 类生成了几种基础方法。例如，用于初始化对象的 `__init__` 方法、用于生成对象的字符串表示形式的 `__repr__` 方法，以及用于比较对象相等性的 `__eq__` 方法。

因此，任何时候只要需要定义主要用于存储数据的类时，请不要忘记利用 `@dataclass`装饰器的强大功能。

```python
from collections import deque
from dataclasses import dataclass

@dataclass
class TreeNode:
    value: int
    min_win: int
    left: 'TreeNode' = None
    right: 'TreeNode' = None

def build_tree(values):
    stack = deque(TreeNode(value, value) for value in values)
    while len(stack) > 1:
        left_node = stack.popleft()
        right_node = stack.popleft()
        new_node = TreeNode(max(left_node.min_win, right_node.min_win),
                            min(left_node.min_win, right_node.min_win))
        new_node.left, new_node.right = left_node, right_node
        stack.append(new_node)

    root = TreeNode(stack[0].min_win, stack[0].min_win)
    root.left = stack[0]
    return root

def show(n, root):
    stack = deque([root])
    result = []
    while stack:
        if len(result) == n:
            print(*result)
            return
        current_node = stack.popleft()
        result.append(current_node.value)
        if current_node.left:
            stack.append(current_node.left)
        if current_node.right:
            stack.append(current_node.right)


n, m = map(int, input().split())
initial_values = list(map(int, input().split()))
root = build_tree(initial_values)
show(n, root)
for _ in range(m):
    position, value = map(int, input().split())
    initial_values[position] = value
    root = build_tree(initial_values)
    show(n, root)
```





## 08581: 扩展二叉树

tree, dfs, http://cs101.openjudge.cn/practice/08581/

由于先序、中序和后序序列中的任一个都不能唯一确定一棵二叉树，所以对二叉树做如下处理，将二叉树的空结点用·补齐，如图所示。我们把这样处理后的二叉树称为原二叉树的扩展二叉树，扩展二叉树的先序和后序序列能唯一确定其二叉树。 现给出扩展二叉树的先序序列，要求输出其中序和后序序列。

![img](http://media.openjudge.cn/images/upload/1440300244.png)

**输入**

扩展二叉树的先序序列（全部都由大写字母或者.组成）

**输出**

第一行：中序序列
第二行：后序序列

样例输入

```
ABD..EF..G..C..
```

样例输出

```
DBFEGAC
DFGEBCA
```



通过递归方法解析扩展二叉树的先序序列，并输出其中序和后序序列：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_tree(s, index):
    # 如果当前字符为'.'，表示空结点，返回None，并将索引后移一位
    if s[index] == '.':
        return None, index + 1
    # 否则创建一个结点
    node = Node(s[index])
    index += 1
    # 递归构造左子树
    node.left, index = build_tree(s, index)
    # 递归构造右子树
    node.right, index = build_tree(s, index)
    return node, index

def inorder(node, res):
    if node is None:
        return
    inorder(node.left, res)
    res.append(node.val)
    inorder(node.right, res)

def postorder(node, res):
    if node is None:
        return
    postorder(node.left, res)
    postorder(node.right, res)
    res.append(node.val)

def main():
    s = input().strip()
    root, _ = build_tree(s, 0)
    
    in_res = []
    inorder(root, in_res)
    
    post_res = []
    postorder(root, post_res)
    
    print("".join(in_res))
    print("".join(post_res))

if __name__ == '__main__':
    main()
```

代码说明

- **build_tree 函数**  
  递归地根据扩展二叉树的先序序列构造二叉树：  
  - 当遇到 `'.'` 时表示空结点，直接返回 `None`。  
  - 否则以当前字符创建一个结点，然后递归构造其左子树和右子树。

- **inorder 和 postorder 函数**  
  分别实现中序遍历（左-根-右）和后序遍历（左-右-根）。

- **main 函数**  
  读取输入字符串，构造树后计算中序和后序遍历结果，并输出。  





由于二叉树的空结点都用.补齐了，直接进行递归就可以得到唯一确定的树。

```python
class BinaryTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def build_tree(lst):
    if not lst:
        return None

    value = lst.pop()
    if value == '.':
        return None

    root = BinaryTreeNode(value)
    root.left = build_tree(lst)
    root.right = build_tree(lst)

    return root


def inorder(root):
    if not root:
        return []

    left = inorder(root.left)
    right = inorder(root.right)
    return left + [root.value] + right


def postorder(root):
    if not root:
        return []

    left = postorder(root.left)
    right = postorder(root.right)
    return left + right + [root.value]


lst = list(input())
root = build_tree(lst[::-1])
in_order_result = inorder(root)
post_order_result = postorder(root)
print(''.join(in_order_result))
print(''.join(post_order_result))
```



嵌套括号表示法Nested parentheses representation。直接用元组（root, left, right）来代表一棵树。

ABD..EF..G..C..
('A', ('B', ('D', None, None), ('E', ('F', None, None), ('G', None, None))), ('C', None, None))

```python
def build_tree(preorder):
    if not preorder or preorder[0] == '.':
        return None, preorder[1:]
    root = preorder[0]
    left, preorder = build_tree(preorder[1:])
    right, preorder = build_tree(preorder)
    return (root, left, right), preorder

def inorder(tree):
    if tree is None:
        return ''
    root, left, right = tree
    return inorder(left) + root + inorder(right)

def postorder(tree):
    if tree is None:
        return ''
    root, left, right = tree
    return postorder(left) + postorder(right) + root

# 输入处理
preorder = input().strip()

# 构建扩展二叉树
tree, _ = build_tree(preorder)

# 输出结果
print(inorder(tree))
print(postorder(tree))
```



递归建树，节点满载了就pop出去，最后递归得到答案

```python
#杨天健 信息科学技术学院
class Node:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None

def mid(root):
    return ("" if root.left==None else mid(root.left))+(""if root.data=='.'else root.data)+(""if root.right==None else mid(root.right))

def post(root):
    return ("" if root.left==None else post(root.left))+(""if root.right==None else post(root.right))+(""if root.data=='.'else root.data)

s=input()
root=Node(s[0])
stack=[root]
for i in range(1,len(s)):
    a=Node(s[i])
    if stack[-1].left==None :
        stack[-1].left=a
        if a.data!='.':
            stack.append(a)
    else :
        stack[-1].right=a
        stack.pop()
        if s[i]!='.':
            stack.append(a)

print(mid(root))
print(post(root))

#ABD..EF..G..C..
```



## 08758: 2的幂次方表示

http://cs101.openjudge.cn/practice/08758/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 09201: Freda的越野跑

http://cs101.openjudge.cn/practice/09201/

Freda报名参加了学校的越野跑。越野跑共有N人参加，在一条笔直的道路上进行。这N个人在起点处站成一列，相邻两个人之间保持一定的间距。比赛开始后，这N个人同时沿着道路向相同的方向跑去。换句话说，这N个人可以看作x轴上的N个点，在比赛开始后，它们同时向x轴正方向移动。
假设越野跑的距离足够远，这N个人的速度各不相同且保持匀速运动，那么会有多少对参赛者之间发生“赶超”的事件呢？

输入

第一行1个整数N。
第二行为N 个非负整数，按从前到后的顺序给出每个人的跑步速度。
对于50%的数据，2<=N<=1000。
对于100%的数据，2<=N<=100000。

输出

一个整数，表示有多少对参赛者之间发生赶超事件。

样例输入

```
5
1 3 10 8 5
```

样例输出

```
7
```

提示

我们把这5个人依次编号为A,B,C,D,E，速度分别为1,3,10,8,5。
在跑步过程中：
B,C,D,E均会超过A，因为他们的速度都比A快；
C,D,E都会超过B，因为他们的速度都比B快；
C,D,E之间不会发生赶超，因为速度快的起跑时就在前边。



```python
import sys

def merge_sort(a, temp, left, right):
    if right - left <= 1:
        return 0
    mid = (left + right) // 2
    inv_count = merge_sort(a, temp, left, mid) + merge_sort(a, temp, mid, right)
    i, j, k = left, mid, left
    while i < mid and j < right:
        if a[i] < a[j]:
            temp[k] = a[i]
            i += 1
        else:
            temp[k] = a[j]
            j += 1
            inv_count += mid - 
            
        k += 1
    while i < mid:
        temp[k] = a[i]
        i += 1
        k += 1
    while j < right:
        temp[k] = a[j]
        j += 1
        k += 1
    for i in range(left, right):
        a[i] = temp[i]
    return inv_count

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))
temp = [0] * n
print(n * (n - 1) // 2 - merge_sort(a, temp, 0, n))
```



```python
#蒋子轩
from bisect import *
n=int(input())
a=list(map(int,input().split()))
sorted_list=[]
cnt=0
for num in a:
    pos=bisect_left(sorted_list,num)
    cnt+=pos
    insort_left(sorted_list,num)
print(cnt)
```



```python
# 23物院宋昕杰 树状数组
n = int(input())
tr = [0] * (n + 1)


def lowbit(x):
    return x & -x


def query(x, y):  			#查询[x, y]，索引从1开始
    x -= 1
    ans = 0
    while y > x:
        ans += tr[y]
        y -= lowbit(y)
    while x > y:
        ans -= tr[x]
        x -= lowbit(x)
    return ans


def add(i, k):				#原数组第i个数加上k，更新树状数组
    while i <= n:
        tr[i] += k
        i += lowbit(i)


ls = list(map(int, input().split()))
for i in range(1, n + 1):		#O(nlogn)建树
    add(i, 1)
keys = sorted(ls)
dic = {}
for i in range(n):
    if keys[i] not in dic:
        dic[keys[i]] = i
ans = 0
for i in range(n - 1, -1, -1):
    idx = dic[ls[i]]
    ans += query(1, idx)
    add(idx + 1, -1)
print(ans)
```



## 09202: 舰队、海域出击！

Topological Order, http://cs101.openjudge.cn/practice/09202/

作为一名海军提督，Pachi将指挥一支舰队向既定海域出击！
Pachi已经得到了海域的地图，地图上标识了一些既定目标和它们之间的一些单向航线。如果我们把既定目标看作点、航线看作边，那么海域就是一张有向图。不幸的是，Pachi是一个会迷路的提督QAQ，所以他在包含环(圈)的海域中必须小心谨慎，而在无环的海域中则可以大展身手。
受限于战时的消息传递方式，海域的地图只能以若干整数构成的数据的形式给出。作为舰队的通讯员，在出击之前，请你告诉提督海域中是否包含环。

例如下面这个海域就是无环的：

![img](http://media.openjudge.cn/images/upload/1450686482.png)

而下面这个海域则是有环的（C-E-G-D-C）：

![img](http://media.openjudge.cn/images/upload/1450686544.png)



**输入**

每个测试点包含多组数据，每组数据代表一片海域，各组数据之间无关。
第一行是数据组数T。
每组数据的第一行两个整数N，M，表示海域中既定目标数、航线数。
接下来M行每行2个不相等的整数x,y，表示从既定目标x到y有一条单向航线（所有既定目标使用1~N的整数表示）。
描述中的图片仅供参考，其顶点标记方式与本题数据无关。

1<=N<=100000，1<=M<=500000，1<=T<=5
注意：输入的有向图不一定是连通的。

**输出**

输出包含T行。
对于每组数据，输出Yes表示海域有环，输出No表示无环。

样例输入

```
2
7 6
1 2
1 3
2 4
2 5
3 6
3 7
12 13
1 2
2 3
2 4
3 5
5 6
4 6
6 7
7 8
8 4
7 9
9 10
10 11
10 12
```

样例输出

```
No
Yes
```

提示

输入中的两张图就是描述中给出的示例图片。



拓扑排序检查有向图是否存在环

```python
# 蒋子轩 工院
from collections import deque,defaultdict
def topo_sort(graph):
    in_degree={u:0 for u in range(1,n+1)}
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0])
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return 'Yes'
    return 'No'
for _ in range(int(input())):
    n,m=map(int,input().split())
    graph=defaultdict(list)
    for _ in range(m):
        u,v=map(int,input().split())
        graph[u].append(v)
    print(topo_sort(graph))
```





可以dfs完成对一个节点的所有邻居的访问后，将这个节点标记为已完全访问（color[node] = 2）。这意味着我们已经探索了从这个节点出发可以到达的所有节点，而且没有发现环。

```python
from collections import defaultdict

def dfs(node, color):
    color[node] = 1
    for neighbour in graph[node]:
        if color[neighbour] == 1:
            return True
        if color[neighbour] == 0 and dfs(neighbour, color):
            return True
    color[node] = 2
    return False

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        x, y = map(int, input().split())
        graph[x].append(y)
    color = [0] * (N + 1)
    is_cyclic = False
    for node in range(1, N + 1):
        if color[node] == 0:
            if dfs(node, color):
                is_cyclic = True
                break
    print("Yes" if is_cyclic else "No")
```



拓扑排序检查有向图是否存在环。它的基本思想是通过计算每个顶点的入度，并从入度为 0 的顶点开始进行深度优先搜索（DFS）。在 DFS 过程中，每遍历到一个顶点时，将其标记为已访问，并将其所有邻居的入度减一。如果邻居的入度减为 0，则继续对邻居进行 DFS。如果最终所有顶点都被访问到，则图中不存在环；否则，存在环。

```python
from collections import defaultdict

def dfs(p):
    vis[p] = True
    for q in graph[p]:
        in_degree[q] -= 1
        if in_degree[q] == 0:
            dfs(q)

for _ in range(int(input())):
    n, m = map(int, input().split())
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)
    vis = [False] * (n + 1) 
    for _ in range(m):
        x, y = map(int, input().split())
        graph[x].append(y)
        in_degree[y] += 1
    for k in range(1, n + 1):  
        if in_degree[k] == 0 and not vis[k]:  
            dfs(k)
    flag = any(not vis[i] for i in range(1, n + 1))  
    print('Yes' if flag else 'No')

```







## 14683: 合并果子

http://cs101.openjudge.cn/dsapre/14683/

有n堆果子（n<=10000），多多决定把所有的果子合成一堆。

每一次合并，多多可以把两堆果子合并到一起，消耗的体力等于两堆果子数量之和。可以看出，所有的果子经过n-1次合并之后，就只剩下一堆了。多多在合并果子时总共消耗的体力等于每次合并所耗体力之和。

设计出合并的次序方案，使多多耗费的体力最少，并输出这个最小的体力耗费值。

**输入**

两行，第一行是一个整数n(1<＝n<=10000)，表示果子的种类数。
第二行包含n个整数，用空格分隔，第i个整数ai(1<＝ai<=20000)是第i堆果子的数目。

**输出**

一行，这一行只包含一个整数，也就是最小的体力耗费值。输入数据保证这个值小于2^31。

样例输入

```
3 
1 2 9
```

样例输出

```
15
```

提示：哈夫曼编码



```python
import heapq

n = int(input())
l = list(map(int, input().split()))
heapq.heapify(l)
ans = 0

while len(l) > 1:
    a = heapq.heappop(l)
    b = heapq.heappop(l)
    ans += a + b
    heapq.heappush(l, a + b)

print(ans)
```







## 17968: 整型关键字的散列映射

http://cs101.openjudge.cn/practice/17968/

给定一系列整型关键字和素数P，用除留余数法定义的散列函数H（key)=key%M，将关键字映射到长度为M的散列表中，用线性探查法解决冲突

**输入**

输入第一行首先给出两个正整数N（N<=1000）和M（>=N的最小素数），分别为待插入的关键字总数以及散列表的长度。
第二行给出N个整型的关键字。数字之间以空格分隔。

**输出**

在一行内输出每个整型关键字的在散列表中的位置。数字间以空格分隔。

样例输入

```
4 5
24 13 66 77
```

样例输出

```
4 3 1 2
```



这个题目的输入数据可能不是标准形式，特殊处理，整体读入 sys.stdin.read

```python
def insert_hash_table(keys, M):
    table = [0.5] * M  # 用 0.5 表示空位
    result = []

    for key in keys:
        index = key % M
        i = index

        while True:
            if table[i] == 0.5 or table[i] == key:
                result.append(i)
                table[i] = key
                break
            i = (i + 1) % M

    return result

# 使用标准输入读取数据
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
keys = list(map(int, data[2:2 + N]))

positions = insert_hash_table(keys, M)
print(*positions)

```



## M17975: 用二次探查法建立散列表

hash table, http://cs101.openjudge.cn/practice/17975/

给定一系列整型关键字和素数P，用除留余数法定义的散列函数H（key)=key%M，将关键字映射到长度为M的散列表中，用二次探查法解决冲突.

本题不涉及删除，且保证表长不小于关键字总数的2倍，即没有插入失败的可能。

**输入**

输入第一行首先给出两个正整数N（N<=1000）和M（一般为>=2N的最小素数），分别为待插入的关键字总数以及散列表的长度。
第二行给出N个整型的关键字。数字之间以空格分隔。

**输出**

在一行内输出每个整型关键字的在散列表中的位置。数字间以空格分隔。

样例输入

```
5 11
24 13 35 15 14
```

样例输出

```
2 3 1 4 7 
```

提示

探查增量序列依次为：$1^2，-1^2，2^2 ，-2^2，....,^2$表示平方



需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



```python
def quadratic_probe_insert(keys, M):
    table = [None] * M
    result = []

    for key in keys:
        pos = key % M
        if table[pos] is None or table[pos] == key:
            table[pos] = key
            result.append(pos)
            continue

        # 否则开始二次探查
        i = 1
        instered = False
        while not instered:
            for sign in [1, -1]:
                new_pos = (pos + sign * (i ** 2)) % M
                if table[new_pos] is None or table[new_pos] == key:
                    table[new_pos] = key
                    result.append(new_pos)
                    instered = True
                    break

            i += 1  # 探查次数增加

    return result


import sys

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
keys = list(map(int, data[2:2 + N]))

positions = quadratic_probe_insert(keys, M)
print(*positions)

```



```python
# 2200015507 王一粟
# n, m = map(int, input().split())
# num_list = [int(i) for i in input().split()]
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]

mylist = [0.5] * m

def generate_result():
    for num in num_list:
        pos = num % m
        current = mylist[pos]
        if current == 0.5 or current == num:
            mylist[pos] = num
            yield pos
        else:
            sign = 1
            cnt = 1
            while True:
                now = pos + sign * (cnt ** 2)
                current = mylist[now % m]
                if current == 0.5 or current == num:
                    mylist[now % m] = num
                    yield now % m
                    break
                sign *= -1
                if sign == 1:
                    cnt += 1

result = generate_result()
print(*result)
```





## 19943: 图的拉普拉斯矩阵

http://cs101.openjudge.cn/practice/19943/

在图论中，度数矩阵是一个对角矩阵 ，其中包含的信息为的每一个顶点的度数，也就是说，每个顶点相邻的边数。邻接矩阵是图的一种常用存储方式。如果一个图一共有编号为0,1,2，…n-1的n个节点，那么邻接矩阵A的大小为n*n，对其中任一元素Aij，如果节点i，j直接有边，那么Aij=1；否则Aij=0。

将度数矩阵与邻接矩阵逐位相减，可以求得图的拉普拉斯矩阵。具体可见下图示意。

![img](http://media.openjudge.cn/images/upload/1575881364.jpg)

现给出一个图中的所有边的信息，需要你输出该图的拉普拉斯矩阵。



**输入**

第一行2个整数，代表该图的顶点数n和边数m。
接下m行，每行为空格分隔的2个整数a和b，代表顶点a和顶点b之间有一条无向边相连，a和b均为大小范围在0到n-1之间的整数。输入保证每条无向边仅出现一次（如1 2和2 1是同一条边，并不会在数据中同时出现）。

**输出**

共n行，每行为以空格分隔的n个整数，代表该图的拉普拉斯矩阵。

样例输入

```
4 5
2 1
1 3
2 3
0 1
0 2
```

样例输出

```
2 -1 -1 0
-1 3 -1 -1
-1 -1 3 -1
0 -1 -1 2
```

来源

cs101 2019 Final Exam



```python
class Vertex:	
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

def constructLaplacianMatrix(n, edges):
    graph = Graph()
    for i in range(n):	# 添加顶点
        graph.addVertex(i)
    
    for edge in edges:	# 添加边
        a, b = edge
        graph.addEdge(a, b)
        graph.addEdge(b, a)
    
    laplacianMatrix = []	# 构建拉普拉斯矩阵
    for vertex in graph:
        row = [0] * n
        row[vertex.getId()] = len(vertex.getConnections())
        for neighbor in vertex.getConnections():
            row[neighbor.getId()] = -1
        laplacianMatrix.append(row)

    return laplacianMatrix


n, m = map(int, input().split())	# 解析输入
edges = []
for i in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

laplacianMatrix = constructLaplacianMatrix(n, edges)	# 构建拉普拉斯矩阵

for row in laplacianMatrix:	# 输出结果
    print(' '.join(map(str, row)))
```



## 20018: 蚂蚁王国的越野跑

merge sort, binary search, http://cs101.openjudge.cn/practice/20018

为了促进蚂蚁家族身体健康，提高蚁族健身意识，蚂蚁王国举行了越野跑。假设越野跑共有N个蚂蚁参加，在一条笔直的道路上进行。N个蚂蚁在起点处站成一列，相邻两个蚂蚁之间保持一定的间距。比赛开始后，N个蚂蚁同时沿着道路向相同的方向跑去。换句话说，这N个蚂蚁可以看作x轴上的N个点，在比赛开始后，它们同时向X轴正方向移动。假设越野跑的距离足够远，这N个蚂蚁的速度有的不相同有的相同且保持匀速运动，那么会有多少对参赛者之间发生“赶超”的事件呢？此题结果比较大，需要定义long long类型。请看备注。

<img src="http://media.openjudge.cn/images/upload/1576506586.jpg" alt="img" style="zoom:50%;" />

**输入**

第一行1个整数N。
第2… N +1行：N 个非负整数，按从前到后的顺序给出每个蚂蚁的跑步速度。对于50%的数据，2<=N<=1000。对于100%的数据，2<=N<=100000。

**输出**

一个整数，表示有多少对参赛者之间发生赶超事件。

样例输入

```
5
1
5
10
7
6

5
1
5
5
7
6
```

样例输出

```
7

8
```

提示

我们把这5个蚂蚁依次编号为A,B,C,D,E，假设速度分别为1,5,5,7,6。在跑步过程中：B,C,D,E均会超过A，因为他们的速度都比A快；D,E都会超过B,C，因为他们的速度都比B,C快；D,E之间不会发生赶超，因为速度快的起跑时就在前边；B,C之间不会发生赶超，因为速度一样，在前面的就一直在前面。

考虑归并排序的思想。

此题结果比较大，需要定义long long类型，其输出格式为printf("%lld",x);
long long，有符号 64位整数，所占8个字节(Byte)
-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807



出题人太随意了。题目中的样例数据是 2 组样例数据。



```python
import sys


def merge_sort(arr, temp, left, right):
    if left >= right:
        return 0
    mid = (left + right) // 2
    inv_count = merge_sort(arr, temp, left, mid) + merge_sort(arr, temp, mid + 1, right)

    # 归并过程，同时计算逆序数
    i, j, k = left, mid + 1, left
    while i <= mid and j <= right:
        if arr[i] >= arr[j]:  # 注意这里是 >=，保证稳定性
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            inv_count += (mid - i + 1)  # 统计逆序对
            j += 1
        k += 1

    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1

    # 拷贝回原数组
    for i in range(left, right + 1):
        arr[i] = temp[i]

    return inv_count


if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    arr = [int(sys.stdin.readline().strip()) for _ in range(n)]
    temp = [0] * n
    result = merge_sort(arr, temp, 0, n - 1)
    print(result)

```

主要优化点：

1. **索引传递优化**：避免创建子列表，改为在原数组上进行归并排序，提高空间效率。
2. **减少 `extend` 操作**：直接在 `temp` 中合并排序，最后一次性拷贝回 `arr`，减少内存拷贝开销。
3. **提高稳定性**：使用 `arr[i] <= arr[j]`，确保排序稳定。
4. **使用 `sys.stdin.readline()`**：加速大规模输入读取，提升整体运行效率。

**时间复杂度：** $O(NlogN)$
**空间复杂度：** $O(N)$ (使用 `temp` 作为辅助数组)



```python
import sys
from collections import Counter

sys.setrecursionlimit(200000)

def merge_count(arr):
    # 归并排序，同时计算逆序数
    n = len(arr)
    if n <= 1:
        return arr, 0
    mid = n // 2
    left, inv_left = merge_count(arr[:mid])
    right, inv_right = merge_count(arr[mid:])
    merged = []
    i = j = 0
    inv = inv_left + inv_right
    while i < len(left) and j < len(right):
        # 如果左边元素<=右边元素，不构成逆序对（注意：相等情况不算）
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            # left[i] > right[j]构成逆序对，左边剩余的元素都大于right[j]
            merged.append(right[j])
            inv += len(left) - i
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, inv

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    speeds = list(map(int, input_data[1:]))
    total_pairs = n * (n - 1) // 2

    # 统计相等对数（任意两个相同速度的蚂蚁不会发生赶超）
    cnt = Counter(speeds)
    equal_pairs = sum(v * (v - 1) // 2 for v in cnt.values())

    # 计算传统逆序数：统计满足a[i] > a[j]的(i,j)
    _, inv = merge_count(speeds)
    # 根据分析，赶超事件的对数为：
    result = total_pairs - equal_pairs - inv
    print(result)

if __name__ == '__main__':
    main()

```





```python
#23n2300011505(12号娱乐选手)
def merge_sort(l):
    if len(l) <= 1:
        return l, 0
    mid = len(l) // 2
    left, left_count = merge_sort(l[:mid])
    right, right_count = merge_sort(l[mid:])
    l, merge_count = merge(left, right)
    return l, left_count + right_count + merge_count


def merge(left, right):
    merged = []
    left_index, right_index = 0, 0
    count = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] >= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
            count += len(left) - left_index
    merged += left[left_index:]+right[right_index:]
    return merged, count


n = int(input())
l = []
for i in range(n):
    l.append(int(input()))
l, ans = merge_sort(l)
print(ans)
```



```python
from bisect import bisect_left
n=int(input())
v=[]
ans=0
for i in range(n):
    p=int(input())
    index=bisect_left(v,p)
    v.insert(index,p)
    ans+=index
print(ans)
```



```python
#23n2300011042(Apocalypse)
import bisect

while True:
    try:
        n = int(input())
        ans = 0
        l = []
        for _ in range(n):
            t = int(input())
            dx = len(l) - (bisect.bisect_right(l, -t))
            ans += dx
            bisect.insort_right(l, -t)
        print(ans)
        input()
    except EOFError:
        break
```





## 20449: 是否被5整除

http://cs101.openjudge.cn/dsapre/20449/

给定由0 和 1 组成的字串 A，我们定义 N_i：从 A[0] 到 A[i] 的第 i 个子数组被解释为一个二进制数 

返回0和 1 组成的字串 answer，只有当 N_i 可以被 5 整除时，答案 answer[i] 为 1，否则为 0

具体请看例子 

**输入**

一个0和1组成的字串

**输出**

一行长度等同于输入的0和1组成的字串

样例输入

```
011
```

样例输出

```
100
```

提示

0可以被5整除->1
01不可以被5整除->0
011不可以被5整除->0
结果是100



遍历输入的字符串，然后将每个字符解释为二进制数并检查是否可以被5整除来解决。我们可以使用Python的内置函数int()将二进制字符串转换为整数，并使用模运算符%来检查是否可以被5整除。

```python
def binary_divisible_by_five(binary_string):
    result = ''
    num = 0
    for bit in binary_string:
        num = (num * 2 + int(bit)) % 5
        if num == 0:
            result += '1'
        else:
            result += '0'
    return result

binary_string = input().strip()
print(binary_divisible_by_five(binary_string))
```



## 20453: 和为k的子数组个数

http://cs101.openjudge.cn/practice/20453/

给定一组整数数字和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。

**输入**

第一行:由空格区分的一组数字
第二行:整数k

**输出**

一个整数，代表多少子数组等于k

样例输入

```
1 1 1
2
```

样例输出

```
2
```

提示

有两组1 1 和为2



通过使用一个哈希表来存储前缀和的频率来解决。我们遍历输入的数组，每次迭代时，我们都会更新当前的前缀和。然后，我们检查哈希表中是否存在当前前缀和减去目标值k的条目。如果存在，我们就将其值添加到结果中。最后，我们将当前的前缀和添加到哈希表中。

```python
def subarray_sum(nums, k):
    count = 0
    sums = 0
    d = dict()
    d[0] = 1

    for i in range(len(nums)):
        sums += nums[i]
        count += d.get(sums - k, 0)
        d[sums] = d.get(sums, 0) + 1

    return count

nums = list(map(int, input().split()))
k = int(input().strip())
print(subarray_sum(nums, k))
```



## 20456: 统计封闭岛屿的数目

http://cs101.openjudge.cn/practice/20456/

给定10行，每行有10个数字的方形地图 ，每个位置要么是陆地（记号为 0 ）要么是水域（记号为 1 ）。 我们从一块陆地出发，每次可以往上下左右 4 个方向相邻区域走，能走到的所有陆地区域，我们将其称为一座「岛屿」。 如果一座岛屿 完全 由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为 「封闭岛屿」。 请输出封闭岛屿的数目。

**输入**

10行，每行有10个数字(0或1)

**输出**

一个整数，封闭岛屿的数目

样例输入

```
1,0,0,0,0,0,1,0,1,0
1,1,1,1,1,0,0,0,0,0
1,0,0,0,1,1,1,1,0,0
1,0,0,1,0,1,0,1,1,0
1,0,0,0,0,1,0,1,0,0
0,0,1,0,0,0,0,1,0,0
1,1,1,0,0,0,0,0,0,0
1,0,1,1,0,0,1,1,1,0
1,0,1,0,0,1,0,0,1,0
0,0,0,0,0,0,1,1,1,1
```

样例输出

```
1
```

提示

1个封闭岛屿



```python
def closedIsland(grid):
    rows, cols = len(grid), len(grid[0])

    # 检查岛屿是否封闭的DFS函数
    def dfs(r, c):
        if grid[r][c] == 1:
            return True
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return False
        
        # 标记当前单元格为已访问
        grid[r][c] = 1
        
        # 检查所有方向
        up = dfs(r - 1, c)
        down = dfs(r + 1, c)
        left = dfs(r, c - 1)
        right = dfs(r, c + 1)
        
        return up and down and left and right

    closed_islands = 0
    for r in range(1, rows - 1):  # 从1开始，忽略边界
        for c in range(1, cols - 1):  # 从1开始，忽略边界
            if grid[r][c] == 0 and dfs(r, c):
                closed_islands += 1

    return closed_islands

# 读取输入
grid = []
for _ in range(10):
    row = list(map(int, input().split(',')))
    grid.append(row)

# 输出结果
print(closedIsland(grid))

```





# 20472~23451

## 20472: 死循环的机器人

http://cs101.openjudge.cn/practice/20472/

在无限的平面上，机器人最初位于 (0, 0) 处，面朝北方。机器人可以接受下列三条指令之一： "G"：直走 1 个单位 "L"：左转 90 度 "R"：右转 90 度 机器人按顺序执行指令，并一直重复它们。 只有在平面中存在死循环使得机器人永远无法离开时，返回 1。否则，返回 0。

**输入**

由G L R组成的字符串

**输出**

如果存在死循环输出1，否则0

样例输入

```
GGLLGG
```

样例输出

```
1
```

提示

样例中的机器人会在长度为4的直线徘徊(本来打错成2)



模拟机器人的移动，并检查它是否最终会回到原点并面向原来的方向，因为这是死循环的必要条件。如果在执行一系列指令后，机器人回到原点并且面向原来的方向，那么它将循环执行同样的指令序列，形成一个循环。

下面是一个 Python 函数，它实现了这个逻辑：

```python
def is_robot_making_loop(commands):
    # 初始位置和方向
    x, y = 0, 0
    direction = 'N'

    # 方向变换的规则，用字典表示
    left_turns = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
    right_turns = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}

    # 模拟机器人的移动
    for command in commands:
        if command == 'G':
            if direction == 'N':
                y += 1
            elif direction == 'S':
                y -= 1
            elif direction == 'E':
                x += 1
            elif direction == 'W':
                x -= 1
        elif command == 'L':
            direction = left_turns[direction]
        elif command == 'R':
            direction = right_turns[direction]

    # 如果机器人回到原点，或者不是面向北方（说明它会改变方向然后可能回到原点）
    return (x == 0 and y == 0) or direction != 'N'

# 读取输入并输出结果
commands = input().strip()
print(1 if is_robot_making_loop(commands) else 0)

```

这个函数首先定义了机器人的初始位置和方向。然后，它根据指令移动机器人，并在完成所有指令后检查机器人的位置和方向。

- 如果机器人回到了原点 `(0, 0)` 并且方向不是北（意味着它改变了方向并且可能在执行更多指令后回到原点），函数返回 `True`。
- 如果机器人没有回到原点，或者回到原点时方向是北（意味着它将沿直线移动而不是循环），函数返回 `False`。

最后，程序读取用户输入的指令，调用函数，并输出相应的结果，如果存在死循环输出 `1`，否则 `0`。



```python
def is_robot_making_loop(commands):
    # 初始位置和方向
    x, y = 0, 0
    # 方向变换的规则，用列表表示，0=N, 1=E, 2=S, 3=W
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # 初始方向北
    dir_index = 0

    # 模拟机器人的移动
    for command in commands:
        if command == 'G':
            # 沿着当前方向前进一步
            x += directions[dir_index][0]
            y += directions[dir_index][1]
        elif command == 'L':
            # 左转90度就是方向列表中的前一个方向
            dir_index = (dir_index - 1) % 4
        elif command == 'R':
            # 右转90度就是方向列表中的下一个方向
            dir_index = (dir_index + 1) % 4
    
    # 如果机器人回到原点，或者方向发生改变（不再是北），则会形成循环
    return (x == 0 and y == 0) or (dir_index != 0)

# 读取输入并输出结果
commands = input().strip()
print(1 if is_robot_making_loop(commands) else 0)
```



## 20576: printExp

http://cs101.openjudge.cn/practice/20576/

输出中缀表达式(去除不必要的括号)

**输入**

一个字串

**输出**

一个字串

样例输入

```
( not ( True or False ) ) and ( False or True and True )
```

样例输出

```
not ( True or False ) and ( False or True and True )
```



这三个操作符：`not`：优先级最高，`and`：其次，`or`：优先级最低。

```python
class BinaryTree:
    def __init__(self, root, left=None, right=None):
        self.root = root
        self.leftChild = left
        self.rightChild = right

    def getrightchild(self):
        return self.rightChild

    def getleftchild(self):
        return self.leftChild

    def getroot(self):
        return self.root

def postorder(string):    #中缀改后缀 Shunting yard algorightm
    opStack = []
    postList = []
    inList = string.split()
    prec = { '(': 0, 'or': 1,'and': 2,'not': 3}

    for word in inList:
        if word == '(':
            opStack.append(word)
        elif word == ')':
            topWord = opStack.pop()
            while topWord != '(':
                postList.append(topWord)
                topWord = opStack.pop()
        elif word == 'True' or word == 'False':
            postList.append(word)
        else:
            while opStack and prec[word] <= prec[opStack[-1]]:
                postList.append(opStack.pop())
            opStack.append(word)
    while opStack:
        postList.append(opStack.pop())
    return postList

def buildParseTree(infix):       #以后缀表达式为基础建树
    postList = postorder(infix)
    stack = []
    for word in postList:
        if word == 'not':
            newTree = BinaryTree(word)
            newTree.leftChild = stack.pop()
            stack.append(newTree)
        elif word == 'True' or word == 'False':
            stack.append(BinaryTree(word))
        else:
            right = stack.pop()
            left = stack.pop()
            newTree = BinaryTree(word)
            newTree.leftChild = left
            newTree.rightChild = right
            stack.append(newTree)
    currentTree = stack[-1]
    return currentTree

def printTree(parsetree: BinaryTree):
    if parsetree.getroot() == 'or':
        return printTree(parsetree.getleftchild()) + ['or'] + printTree(parsetree.getrightchild())
    elif parsetree.getroot() == 'not':
        return ['not'] + (
            ['('] + printTree(parsetree.getleftchild()) + [')']
            if parsetree.leftChild.getroot() not in ['True', 'False']
            else printTree(parsetree.getleftchild())
        )
    elif parsetree.getroot() == 'and':
        leftpart = (
            ['('] + printTree(parsetree.getleftchild()) + [')']
            if parsetree.leftChild.getroot() == 'or'
            else printTree(parsetree.getleftchild())
        )
        rightpart = (
            ['('] + printTree(parsetree.getrightchild()) + [')']
            if parsetree.rightChild.getroot() == 'or'
            else printTree(parsetree.getrightchild())
        )
        return leftpart + ['and'] + rightpart
    else:
        return [str(parsetree.getroot())]

def main():
    infix = input()
    Tree = buildParseTree(infix)
    print(' '.join(printTree(Tree)))

main()
```

`printTree`函数是一个递归函数，接收一个`BinaryTree`对象作为参数，然后根据树的结构和节点的值生成一个字符串列表。

函数的工作方式如下：

1. 首先，检查树的根节点的值。根据值的不同，函数会执行不同的操作。

2. 如果根节点的值为"or"，函数会递归地调用自身来处理左子树和右子树，然后将结果合并，并在两个结果之间插入"or"。

3. 如果根节点的值为"not"，函数会递归地调用自身来处理左子树。<mark>如果左子树的根节点的值不是"True"或"False"，则会在左子树的结果周围添加括号。</mark>

4. 如果根节点的值为"and"，函数会递归地调用自身来处理左子树和右子树。<mark>如果左子树或右子树的根节点的值为"or"，则会在相应子树的结果周围添加括号。</mark>

5. 如果根节点的值为"True"或"False"，函数会直接返回一个包含该值的列表。

6. 最后，函数会将生成的字符串列表合并为一个字符串，并返回。

> `printTree` 函数的确是这个程序的关键部分，它负责**把表达式树重新转成中缀表达式**，而且在这个过程中要**去掉不必要的括号，只保留必要的括号**，以保证表达式的优先级正确。
>
> 我们来**一行行分析这个函数**，并且结合优先级规则来详细解释。
>
> ---
>
> ✅ 背景知识（非常重要）
>
> 我们有 3 个操作符：
>
> - `not`：优先级最高（3）
> - `and`：中间（2）
> - `or`：最低（1）
>
> 所以当我们重建中缀表达式时，如果低优先级的操作被嵌套在高优先级的操作里面，那就**需要加括号**来保持原来的优先顺序。
>
> ---
>
> 💡 函数签名
>
> ```python
> def printTree(parsetree: BinaryTree):
> ```
>
> 这是一个递归函数。每次调用会处理一个子树，返回一个**字符串列表**（最后用 `' '.join()` 组合成结果）。
>
> ---
>
> 🧩 情况 1：当前节点是 `or`
>
> ```python
> if parsetree.getroot() == 'or':
>     return printTree(parsetree.getleftchild()) + ['or'] + printTree(parsetree.getrightchild())
> ```
>
> - `or` 是最低优先级，所以它**永远不会被其他操作包裹住**（不需要加括号）
> - 左右子树分别处理就好，中间加上 `'or'`
>
> ---
>
> 🧩 情况 2：当前节点是 `not`
>
> ```python
> elif parsetree.getroot() == 'not':
>     return ['not'] + (
>         ['('] + printTree(parsetree.getleftchild()) + [')']
>         if parsetree.leftChild.getroot() not in ['True', 'False']
>         else printTree(parsetree.getleftchild())
>     )
> ```
>
> - `not` 是单目操作符（只有一个子节点）
>
> - 如果它的子节点是 `'True'` 或 `'False'`，就直接写：
>
>   ```
>   not True
>   ```
>
> - 但如果子节点是像 `True or False` 这样的表达式，需要加括号包住它：
>
>   ```
>   not (True or False)
>   ```
>
> 👉 这是为了防止表达式优先级出错！
>
> ---
>
> 🧩 情况 3：当前节点是 `and`
>
> ```python
> elif parsetree.getroot() == 'and':
>     leftpart = (
>         ['('] + printTree(parsetree.getleftchild()) + [')']
>         if parsetree.leftChild.getroot() == 'or'
>         else printTree(parsetree.getleftchild())
>     )
>     rightpart = (
>         ['('] + printTree(parsetree.getrightchild()) + [')']
>         if parsetree.rightChild.getroot() == 'or'
>         else printTree(parsetree.getrightchild())
>     )
>     return leftpart + ['and'] + rightpart
> ```
>
> - `and` 的优先级高于 `or`
>
> - 如果它的某个子节点是 `or`，就必须加括号：
>
>   ```
>   (A or B) and C
>   ```
>
> 否则表达式优先级就会错了！
>
> **例子：**
>
> - 正确的：`(True or False) and True`
> - 错误的：`True or False and True`（这个意思不一样）
>
> ---
>
> 🧩 情况 4：叶子节点（True 或 False）
>
> ```python
> else:
>     return [str(parsetree.getroot())]
> ```
>
> 这个就是递归到底了，返回布尔值（字符串形式）
>
> ---
>
> 🔁 总结：如何判断加不加括号？
>
> - `or` 最低优先级，从不加括号。
> - `and` 如果操作的是 `or`，要加括号。
> - `not` 如果操作的是复杂表达式（不是布尔常量），要加括号。
>
> ---
>
> ✨举个完整例子
>
> 输入：
>
> ```plaintext
> ( not ( True or False ) ) and ( False or True and True )
> ```
>
> 输出：
>
> ```plaintext
> not ( True or False ) and ( False or True and True )
> ```
>
> 说明：
>
> - `not` 操作 `True or False` → 加括号 ✅
> - `and` 的左右：
>   - 左是 `not (...)`：不需要额外括号
>   - 右是 `False or True and True`，按优先级就是 `False or (True and True)`，不需要额外括号



## 20625: 1跟0数量相等的子字串

http://cs101.openjudge.cn/practice/20625/

给一个由0跟1组成的字串，请问有多少个子字串(非空)的0跟1数量相等而且0跟1分别是连续的
如果一个子字串出现n次记作n

**输入**

一个1跟0组成的字串

**输出**

一个整数

样例输入

```
10101
```

样例输出

```
4
```

提示

总个有4个子字串，10 01 10 01
1010不算，因为1跟0不是连续的



考虑到这个问题的特殊性（0和1必须是连续的），我们可以采取另一种方法，即只计算每段连续的0或1结束时的子串数量。我们不需要关心整个串的子串，只要关心局部的连续部分即可。

在遍历字符串时，需要统计当前连续相同字符的数量，并在遇到不同字符时，检查之前的连续字符部分可以组成多少合法子串。

解释代码逻辑：

- 我们用 `curr_count` 来跟踪当前字符连续出现的次数，用 `prev_count` 来跟踪上一组字符连续出现的次数。
- 每次字符发生变化时，我们可以创建 `min(curr_count, prev_count)` 个子字符串，因为新的字符将断开之前的连续性。
- 然后我们更新 `prev_count` 为 `curr_count`（因为我们要开始统计新的字符了），并将 `curr_count` 重置为1。
- 在字符串遍历结束后，我们还需要再加上最后一组字符可以形成的子字符串数。

```python
def count_balanced_substrings(s):
    # 初始化当前字符和前一个字符的计数器
    curr_count = 1
    prev_count = 0
    result = 0

    # 遍历字符串的每个字符
    for i in range(1, len(s)):
        # 如果当前字符和前一个字符相同，增加当前计数器
        if s[i] == s[i - 1]:
            curr_count += 1
        else:
            # 如果当前字符和前一个字符不同，那么我们可以创建
            # min(curr_count, prev_count) 个子串
            result += min(curr_count, prev_count)
            # 将当前计数器值赋给前一个计数器，并重置当前计数器为1
            prev_count = curr_count
            curr_count = 1

    # 出循环后，处理最后一组字符
    result += min(curr_count, prev_count)

    return result

# 测试样例输入
#print(count_balanced_substrings("10101"))  # 输出应该是4
#print(count_balanced_substrings("00110011"))  # 输出应该是6
print(count_balanced_substrings(input()))
```



## 20626: 对子数列做XOR运算

http://cs101.openjudge.cn/practice/20626/

给定一个正整数数列V，V的下标从零开始。

对V的子数列W进行XOR查询，输入的查询指令有2个数L,R，L<=R，分别为W中第一个和最后一个元素在V中的下标。计算W中所有元素的XOR值，即：V[L] xor V[L+1] xor ... xor V[R]

输入不同的L, R，对V进行10000次查询。



**输入**

第一行是一个空格分开的正整数数列V
第2-10001行每行有2个数L, R，中间用空格分开

**输出**

10000行整数

样例输入

```
1 3 4 8
0 1
1 2
0 3
3 3
```

样例输出

```
2
7
14
8
```

提示

对照样例输入：数列为1,3,4,8。它们用二进制表示：1 = 0001，3 = 0011， 4 = 0100 ，8 = 1000；当L, R的值依次为
0，1时，求得 1 xor 3 = 2
1，2时，求得 3 xor 4 = 7
0，3时，求得 1 xor 3 xor 4 xor 8 = 14 
3，3时，求得 8
顾输出 2 7 14 8。
实际上会有10000行查询指令，请按照样例格式按行输出查询结果。



```python
def precompute_xor_prefixes(values):
    xor_prefixes = [0] * (len(values) + 1)
    for i in range(len(values)):
        xor_prefixes[i+1] = xor_prefixes[i] ^ values[i]
    return xor_prefixes

# 读取输入并处理
values = list(map(int, input().split()))
xor_prefixes = precompute_xor_prefixes(values)

# 读取查询并处理
for _ in range(10000):
    L, R = map(int, input().split())
    result = xor_prefixes[R+1] ^ xor_prefixes[L]
    print(result)
```



i/o优化啊，1w输入输出

```python
# 23n2300017735(夏天明BrightSummer)
import sys
input = sys.stdin.readline

V = [int(i) for i in input().split()]
preV = [0]*(len(V)+1)
for i in range(len(V)):
    preV[i+1] = preV[i] ^ V[i]

results = []
for i in range(10000):
    L, R = map(int, input().split())
    results.append(str(preV[R+1] ^ preV[L]))

sys.stdout.write('\n'.join(results) + '\n')
```



## 20644: 统计全为 1 的正方形子矩阵

http://cs101.openjudge.cn/practice/20644/

给一个 m * n 的矩阵，矩阵中的元素不是 0 就是 1，

请你统计并输出其中完全由 1 组成的 正方形 子矩阵的个数。

备注:请尽量用动态规划

**输入**

第一行是m n 两个数字，空格分开
m行，每行有n个数

**输出**

一个非负整数

样例输入

```
3 4
0111
1111
0111
```

样例输出

```
15
```

提示

边为1的矩阵有10个
边为2的矩阵有4个
边为3的矩阵有1个
总共15个



Dp

```python
#23n2300017735(夏天明BrightSummer)
m, n = map(int, input().split())
mat = [[int(k) for k in input()] for i in range(m)]
dp = [[0 for j in range(n+1)] for i in range(m+1)]
for i in range(m):
    for j in range(n):
        if mat[i][j]:
            dp[i+1][j+1] = min(dp[i][j], dp[i][j+1], dp[i+1][j])+1
print(sum(dp[i][j] for j in range(n+1) for i in range(m+1)))
```



Brute force

```python
m,n = map(int, input().split())
matrix = []
for i in range(m):
    matrix.append(list(map(int, list(input()))))

def check(matrix, i, j, step):
    for x in range(i, i+step+1):
        for y in range(j, j+step+1):
            if matrix[x][y] == 0:
                return False
    return True

cnt = 0
step = 0

while step <= min(m, n):
    for i in range(m-step):
        for j in range(n-step):
            if check(matrix, i, j, step):
                cnt += 1
    step += 1

print(cnt)
```



## 20650: 最长的公共子序列的长度

http://cs101.openjudge.cn/practice/20650/

我们称一个字符的数组S为一个序列。对于另外一个字符数组Z,如果满足以下条件，则称Z是S的一个子序列：（1）Z中的每个元素都是S中的元素（2）Z中元素的顺序与在S中的顺序一致。例如：当S = (E,R,C,D,F,A,K)时，（E，C，F）和（E，R）等等都是它的子序列。而（R，E）则不是。 

现在我们给定两个序列，求它们最长的公共子序列的长度。 

**输入**

一共两行，分别输入两个序列。

**输出**

一行，输出最长公共子序列的长度。

样例输入

```
ABCBDAB

BDCABA
```

样例输出

```
4
```



```python
def longest_common_subsequence(s1, s2):
    dp = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[len(s1)][len(s2)]

s1 = input()
s2 = input()
print(longest_common_subsequence(s1, s2))
```





## 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/

剧组为了整演员，提供给他们的提词本是经过加工的

提词本内容由英文字母跟括号组成，而且括号必定合法，左括号一定有对应的右括号

演员必须从最里层开始翻转括号内的字母

例如(dcba) 要翻转成abcd

最终演员所念的台词不能含有括号

请输出演员应该念出来的台词

**输入**

一个字串s

**输出**

一个字串s2

样例输入

```
(eg(en(duj))po)
```

样例输出

```
openjudge
```

提示

先反转duj
再反转enjud
最后反转全部台词



use a stack to keep track of the characters inside each pair of parentheses. When you encounter a closing parenthesis, you pop characters from the stack and reverse them until you reach an opening parenthesis, then push the reversed characters back onto the stack. Continue this process until you've processed the entire string. Finally, join the characters in the stack to form the final string.

用stack做，碰到后括号说明到了一层结尾了，进栈，用temp做反向处理，然后去掉头括号，合并
stack，继续，最后弹出

```python
def reverse_parentheses(s):
    stack = []
    for char in s:
        if char == ')':
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())
            # remove the opening parenthesis
            if stack:
                stack.pop()
            # add the reversed characters back to the stack
            stack.extend(temp)
        else:
            stack.append(char)
    return ''.join(stack)

# 读取输入并处理
s = input().strip()
print(reverse_parentheses(s))
```





## 20746: 满足合法工时的最少人数

http://cs101.openjudge.cn/practice/20746/

若干个工作任务，需要在一天内完成。给一个正整数数列，存储每个任务所需的工时。

国家法律规定，员工的日工作时长不能超过t。

公司决定雇佣k个员工，每个任务都会让所有员工一同分担，于是每个任务执行的时间等于它所需的工时除以k。

所有任务执行的时间累加起来得到s。

为了满足合法工作不加班，请在s<=t的前提下，找出所需的最少员工数量k。



分担说明:每个任务分担后的时间都是小数点无条件进位取整

7个工时/3个员工 = 3小时, 10个工时/2个员工=5小时



必定存在结果(不用考虑t<数列长度的状况)



**输入**

一个逗号分隔的数列
一个正整数

**输出**

一个正整数

样例输入

```
1,2,5,9
5
```

样例输出

```
5
```

提示

如果员工数是4，sum(1+1+2+3)=7
如果员工数是5，sum(1+1+1+2)=5
如果员工数是6，sum(1+1+1+2)=5
所以答案是5



use a binary search approach. The minimum number of employees can be 1 and the maximum can be the maximum work hours in the tasks. For each mid value in the binary search, calculate the total work hours and compare it with the legal work hours. If it's more, increase the number of employees, else decrease it.

```python
def min_employees(tasks, t):
    left, right = 1, max(tasks)
    while left < right:
        mid = (left + right) // 2
        total_hours = sum((task + mid - 1) // mid for task in tasks)
        if total_hours > t:
            left = mid + 1
        else:
            right = mid
    return left

# 读取输入并处理
tasks = list(map(int, input().split(',')))
t = int(input())
print(min_employees(tasks, t))
```



## 21006: 放苹果（盘子相同）

http://cs101.openjudge.cn/practice/21006/

这个题目后台没有数据，提交什么都是AC。

正确的题目是

http://cs101.openjudge.cn/practice/01664/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional problems



~~把M个同样的苹果放在N个同样的盘子里，允许有的盘子空着不放，问共有多少种不同的分法？（用K表示）5，1，1和1，5，1 是同一种分法。~~

~~**输入**~~

~~苹果个数m 和盘子个数n(0<=M，1<=N<=10)~~

~~**输出**~~

~~不同的放法数目~~



~~这是一个经典的组合数学问题，也被称为“球和盒子模型”。使用动态规划来解决这个问题。定义dp\[i][j]为将i个苹果放入j个盘子的方法数。那么，我们有两种情况：~~

1. ~~所有的盘子都至少放一个苹果，那么问题就变成了如何将i-j个苹果放入j个盘子。这就是dp\[i-j][j]。~~

2. ~~至少有一个盘子是空的，那么问题就变成了如何将i个苹果放入j-1个盘子。这就是dp\[i][j-1]。~~

~~因此，我们有$dp[i][j] = dp[i-j][j] + dp[i][j-1]$。我们可以使用一个二维数组来存储dp值，并使用嵌套循环来计算所有的dp值。~~

> 
>
> ```python
> # 23n2300011427
> m,n=map(int,input().split())
> dp=[[0]*(m+1) for i in range(n+1)]
> for i in range(1,n+1):
>     dp[i][0]=1
> 
> dp[1]=[1]*(m+1)
> for i in range(1,n+1):
>     dp[i][1]=1
> for i in range(1,n+1):
>     for j in range(1,m+1):
>         if i>j:
>             dp[i][j]=dp[j][j]
>         else:
>             dp[i][j]=dp[i-1][j]+dp[i][j-i]
> print(dp[n][m])
> ```
>



## 21515: 电话线路

http://cs101.openjudge.cn/practice/21515/

有N座通信基站，P条双向电缆，第i条电缆连接基站Ai和Bi。特别地，1号基站是通信公司的总站，N号基站位于一座农场中。现在，农场主希望对通信线路进行升级，其中升级第i条电缆需要花费Li。

电话公司正在举行优惠活动。农场主可以指定一条从1号基站到N号基站的路径，然后，农场主可以指定路径上不超过K条电缆，先由电话公司免费提供升级服务。农场主只需要支付在该路径上剩余的电缆中，升级价格最贵的那条电缆的花费即可。支付完成后，其余电缆也将由电话公司免费升级。求至少用多少钱能完成升级。

**输入**

第一行三个整数， N，P，K。
接下来P行，每行三个整数Ai，Bi，Li。

**输出**

若不存在从1到N的路径，输出-1。否则输出所需最小费用。

样例输入

```
5 7 1
1 2 5
3 1 4
2 4 8
3 2 3
5 2 9
3 4 7
4 5 6
```

样例输出

```
4
```

提示

0 ≤ K < N ≤ 1000
1 ≤ P ≤ 2000



```python
#2200015507 王一粟
from heapq import *
n,p,k = map(int,input().split())
graph = {i:{} for i in range(1,n+1)}
h = 0
for _ in range(p):
    a,b,l = map(int,input().split())
    graph[a][b] = graph[b][a] = l
    h = max(h,l)
l = 0

def search(lim):
    heap = [(-1,-k)]
    heapify(heap)
    vis = {}
    while heap:
        idx,free = heappop(heap)
        idx,free = -idx,-free
        if idx == n:
            return 1
        if idx not in vis or vis[idx] < free:
            vis[idx] = free
        else:
            continue
        for t,length in graph[idx].items():
            new_free = free
            if length > lim:
                if new_free > 0:
                    new_free -= 1
                else:
                    continue
            if t in vis and vis[t] > new_free:
                continue
            heappush(heap,(-t,-new_free))
    return 0
while l < h:
    if l +1 == h:
        ans_l,ans_h = search(l),search(h)
        if ans_l == ans_h == 0:
            print(-1)
        else:
            print(l if ans_l else h)
        exit()
    mid = (l+h)//2
    if search(mid):
        h = mid
    else:
        l = mid
```





## 22067: 快速堆猪

辅助栈, http://cs101.openjudge.cn/practice/22067/

小明有很多猪，他喜欢玩叠猪游戏，就是将猪一头头叠起来。猪叠上去后，还可以把顶上的猪拿下来。小明知道每头猪的重量，而且他还随时想知道叠在那里的猪最轻的是多少斤。

输入

有三种输入
1) push n
n是整数(0<=0 <=20000)，表示叠上一头重量是n斤的新猪
2) pop
表示将猪堆顶的猪赶走。如果猪堆没猪，就啥也不干
3) min
表示问现在猪堆里最轻的猪多重。如果猪堆没猪，就啥也不干

输入总数不超过100000条

输出

对每个min输入，输出答案。如果猪堆没猪，就啥也不干

样例输入

```
pop
min
push 5
push 2
push 3
min
push 4
min
```

样例输出

```
2
2
```

来源: Guo wei



用辅助栈：用一个单调栈维护最小值，再用另外一个栈维护其余的值。

每次push时，在辅助栈中加入当前最轻的猪的体重，pop时也同步pop，这样栈顶始终是当前猪堆中最轻的体重，查询时直接输出即可

```python
a = []
m = []

while True:
    try:
        s = input().split()
    
        if s[0] == "pop":
            if a:
                a.pop()
                if m:
                    m.pop()
        elif s[0] == "min":
            if m:
                print(m[-1])
        else:
            h = int(s[1])
            a.append(h)
            if not m:
                m.append(h)
            else:
                k = m[-1]
                m.append(min(k, h))
    except EOFError:
        break
```

 

```python
pig, pigmin = [], []
while True:
    try:
        *line, = input().split()
        if "pop" in line:
            if len(pig) == 0:
                continue

            val = pig.pop()
            if len(pigmin) > 0 and val == pigmin[-1]:
                pigmin.pop()
        elif "push" in line:
            val = int(line[1])
            pig.append(val)
            if len(pigmin) == 0 or val <= pigmin[-1]:
                pigmin.append(val)
        elif "min" in line:
            if len(pig) == 0:
                continue
            else:
                print(pigmin[-1])
    except EOFError:
        break
```



字典标记，懒删除

```python
import heapq
from collections import defaultdict

out = defaultdict(int)
pigs_heap = []
pigs_stack = []

while True:
    try:
        s = input()
    except EOFError:
        break

    if s == "pop":
        if pigs_stack:
            out[pigs_stack.pop()] += 1
    elif s == "min":
        if pigs_stack:
            while True:
                x = heapq.heappop(pigs_heap)
                if not out[x]:
                    heapq.heappush(pigs_heap, x)
                    print(x)
                    break
                out[x] -= 1
    else:
        y = int(s.split()[1])
        pigs_stack.append(y)
        heapq.heappush(pigs_heap, y)
```



集合标记，懒删除。如果有重复项就麻烦了，可能刚好赶上题目数据友好。

```python
import heapq

class PigStack:
    def __init__(self):
        self.stack = []
        self.min_heap = []
        self.popped = set()

    def push(self, weight):
        self.stack.append(weight)
        heapq.heappush(self.min_heap, weight)

    def pop(self):
        if self.stack:
            weight = self.stack.pop()
            self.popped.add(weight)

    def min(self):
        while self.min_heap and self.min_heap[0] in self.popped:
            self.popped.remove(heapq.heappop(self.min_heap))
        if self.min_heap:
            return self.min_heap[0]
        else:
            return None

pig_stack = PigStack()

while True:
    try:
        command = input().split()
        if command[0] == 'push':
            pig_stack.push(int(command[1]))
        elif command[0] == 'pop':
            pig_stack.pop()
        elif command[0] == 'min':
            min_weight = pig_stack.min()
            if min_weight is not None:
                print(min_weight)
    except EOFError:
        break
```



## 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/

给定一个由大小写字母和数字构成的，没有重复字符的长度不超过62的字符串x，现在要将该字符串的字符依次压入栈中，然后再全部弹出。

要求左边的字符一定比右边的字符先入栈，出栈顺序无要求。

再给定若干字符串，对每个字符串，判断其是否是可能的x中的字符的出栈序列。



**输入**

第一行是原始字符串x
后面有若干行(不超过50行)，每行一个字符串，所有字符串长度不超过100

**输出**

对除第一行以外的每个字符串，判断其是否是可能的出栈序列。如果是，输出"YES"，否则，输出"NO"

样例输入

```
abc
abc
bca
cab
```

样例输出

```
YES
YES
NO
```

来源: Guo wei



```python
def is_valid_pop_sequence(origin, output):
    if len(origin) != len(output):
        return False  # 长度不同，直接返回False

    stack = []
    bank = list(origin)
    
    for char in output:
        # 如果当前字符不在栈顶，且bank中还有字符，则继续入栈
        while (not stack or stack[-1] != char) and bank:
            stack.append(bank.pop(0))
        
        # 如果栈为空，或栈顶字符不匹配，则不是合法的出栈序列
        if not stack or stack[-1] != char:
            return False
        
        stack.pop()  # 匹配成功，弹出栈顶元素
    
    return True  # 所有字符都匹配成功

# 读取原始字符串
origin = input().strip()

# 循环读取每一行输出序列并判断
while True:
    try:
        output = input().strip()
        if is_valid_pop_sequence(origin, output):
            print('YES')
        else:
            print('NO')
    except EOFError:
        break

```



```python
# 23n2300011406(cry_QAQ)
origin = input()
while True:
    try:
        outout = input()
        stack,bank = [],list(origin) #stack 用于模拟栈操作，bank 存放原始字符串尚未入栈的字符
        l = len(origin)
        flag = False
        if len(outout) == l:
            for i in range(l):
                # 如果bank不为空且stack为空，则将bank的第一个字符入栈
                if bank and not stack: 
                    stack.append(bank.pop(0))
                
                 # 将bank中的字符入栈，直到栈顶字符和出栈序列的当前字符相同
                while bank and stack[-1] != outout[i]:
                    stack.append(bank.pop(0))
                if stack.pop() != outout[i]:
                    print('NO')
                    flag = True
                    break
            if not flag:
                print('YES')
        else:
            print('NO')
    except EOFError:
        break
```



卢卓然-23-生命科学学院，思路：  

**利用stack的FILO的性质**

入栈的顺序是降序排列（如6,5,4,3,2,1)，由于栈是FILO，那么出栈序列任意数A的后面比A大的数都是按照升序排列的；

入栈的顺序是升序排列（如1,2,3,4,5,6)，由于栈是FILO，那么出栈序列任意数A的后面比A小的数都是按照降序排列的。

以第二种情况为例，因为在出栈序列任意数A的后面比A小的数，都具有的特点是，比A早进栈而且比A晚出栈。那么这些数组成的序列就必然是恰好倒序的。

```python
#23 生科 卢卓然
#将x中每个字符正序编号1~n
x=input()
L=len(x)
dic=dict()
i=1
for string in x:
    dic[string]=i
    i+=1
#提前声明一个maximum，防止oj特有的一种CE
maximum=-1

def check(index,length):#index当前字符的位置,length是s的长度
    '''鉴定该位置之后所有编号比他小的字符是否全为降序排列'''
    
    global s,maximum#maximum该位置之前字符中出现的最大编号
    number=dic[s[index]]
    if number<=maximum:
        return True
    '''之前的最大的编号的字符之后的所有编号比他小的字符全为降序排列,
    所以编号比maximum小的字符之后的就更是降序排列,剪枝'''
    
    tempmax=number#tempmax暂时的最大符号，不断更新以判断是否为降序排列
    flag=True#标记是否为降序排列
    for k in range(index+1,length):
        tempstr=s[k]
        tempnum=dic[tempstr]
        if tempnum<=number:#编号比number小
            if tempnum<=tempmax:#是否为降序排列
                tempmax=tempnum#更新tempmax
                continue
            else:
                flag=False
                break
    if flag:
        maximum=number#更新maximum
        return True
    else:
        return False

output=[]
while True:
    try:
        s = input()
        if set(s)==set(x) and len(set(s))==len(s):#防止蛇皮数据导致RE
            f=True#YES还是NO
            maximum=-1#初始化为一个比较小的值
            for j in range(L-2):#L-2:最后两位不用看，没有意义
                if not check(j,L):
                    f=False
                    break
            if f:
                output.append('YES')
            else:
                output.append("NO")
        else:
            output.append("NO")
    except EOFError:
        break

for o in output:
    print(o)
```



## 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/

假设二叉树的节点里包含一个大写字母，每个节点的字母都不同。

给定二叉树的前序遍历序列和中序遍历序列(长度均不超过26)，请输出该二叉树的后序遍历序列

**输入**

多组数据
每组数据2行，第一行是前序遍历序列，第二行是中序遍历序列

**输出**

对每组序列建树，输出该树的后序遍历序列

样例输入

```
DURPA
RUDPA
XTCNB
CTBNX
```

样例输出

```
RUAPD
CBNTX
```

来源：郭炜



也是递归方式处理，利用前序遍历第一个为根节点可以很容易找到跟，进而区分左右子树。

The problem is asking to construct a binary tree from given preorder and inorder traversal sequences, and then output the postorder traversal sequence of the constructed tree.  

Here is a step-by-step plan:  
1.Create a TreeNode class to represent each node in the tree.
2.Create a function build_tree that takes the preorder and inorder sequences as input and returns the root of the constructed tree.
	The first character of the preorder sequence is the root of the tree.
	Find the position of the root in the inorder sequence.
	Recursively construct the left subtree using the left part of the inorder sequence and the corresponding part of the preorder sequence.
	Recursively construct the right subtree using the right part of the inorder sequence and the corresponding part of the preorder sequence.
3.Create a function postorder_traversal that takes the root of the tree as input and returns the postorder traversal sequence of the tree.
4.For each pair of preorder and inorder sequences, construct the tree and output the postorder traversal sequence.
Here is the Python code that implements this plan:

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```



## 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/

根据字符使用频率(权值)生成一棵唯一的哈夫曼编码树。生成树时需要遵循以下规则以确保唯一性：

选取最小的两个节点合并时，节点比大小的规则是:

1) 权值小的节点算小。权值相同的两个节点，字符集里最小字符小的，算小。

例如 （{'c','k'},12) 和 ({'b','z'},12)，后者小。

2) 合并两个节点时，小的节点必须作为左子节点
3) 连接左子节点的边代表0,连接右子节点的边代表1

然后对输入的串进行编码或解码



**输入**

第一行是整数n，表示字符集有n个字符。
接下来n行，每行是一个字符及其使用频率（权重）。字符都是英文字母。
再接下来是若干行，有的是字母串，有的是01编码串。

**输出**

对输入中的字母串，输出该字符串的编码
对输入中的01串,将其解码，输出原始字符串

样例输入

```
3
g 4
d 8
c 10
dc
110
```

样例输出

```
110
dc
```

提示: 数据规模很小，不用在乎效率

来源: 郭炜



建树：主要利用最小堆，每次取出weight最小的两个节点，weight相加后创建节点，连接左右孩子，再入堆，直至堆中只剩一个节点。

编码：跟踪每一步走的是左还是右，用0和1表示，直至遇到有char值的节点，说明到了叶子节点，将01字串添加进字典。

解码：根据01字串决定走左还是右，直至遇到有char值的节点，将char值取出。

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        #merged = Node(left.weight + right.weight) #note: 合并后，char 字段默认值是空
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        #if node.char:
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        #if node.char:
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

#string = input().strip()
#encoded_string = input().strip()

# 构建哈夫曼编码树
huffman_tree = build_huffman_tree(characters)

# 编码和解码
codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        strings.append(line)

    except EOFError:
        break

results = []
#print(strings)
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```





## 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/

给出一棵二叉搜索树的前序遍历，求它的后序遍历

**输入**

第一行一个正整数n（n<=2000）表示这棵二叉搜索树的结点个数
第二行n个正整数，表示这棵二叉搜索树的前序遍历
保证第二行的n个正整数中，1~n的每个值刚好出现一次

**输出**

一行n个正整数，表示这棵二叉搜索树的后序遍历

样例输入

```
5
4 2 1 3 5
```

样例输出

```
1 3 2 5 4
```

提示

树的形状为
   4  
  / \ 
  2  5 
 / \  
 1  3  



```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_bst(preorder):
    if not preorder:
        return None
    root = Node(preorder[0])
    i = 1
    while i < len(preorder) and preorder[i] < root.val:
        i += 1
    root.left = build_bst(preorder[1:i])
    root.right = build_bst(preorder[i:])
    return root

def post_order(root):
    if not root:
        return []
    # 先左子树，后右子树，最后根节点
    return post_order(root.left) + post_order(root.right) + [root.val]


# Read input
n = int(input())
preorder = list(map(int, input().split()))

# Build BST and get postorder traversal
root = build_bst(preorder)
res = post_order(root)
print(' '.join(map(str, res)))
```



```python
"""
王昊 光华管理学院。思路：
建树思路：数组第一个元素是根节点，紧跟着是小于根节点值的节点，在根节点左侧，直至遇到大于根节点值的节点，
后续节点都在根节点右侧，按照这个思路递归即可
"""
class Node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def buildTree(preorder):
    if len(preorder) == 0:
        return None

    node = Node(preorder[0])

    idx = len(preorder)
    for i in range(1, len(preorder)):
        if preorder[i] > preorder[0]:
            idx = i
            break
    node.left = buildTree(preorder[1:idx])
    node.right = buildTree(preorder[idx:])

    return node


def postorder(node):
    if node is None:
        return []
    output = []
    output.extend(postorder(node.left))
    output.extend(postorder(node.right))
    output.append(str(node.val))

    return output


n = int(input())
preorder = list(map(int, input().split()))
print(' '.join(postorder(buildTree(preorder))))
```



```python
# 管骏杰 生命科学学院
# 中序遍历就是顺序排列，进而通过上次作业的思路根据前序中序推出后序
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def build(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = Node(root_val)
    root_index = inorder.index(root_val)
    root.left = build(preorder[1:root_index + 1], inorder[:root_index])
    root.right = build(preorder[root_index + 1:], inorder[root_index + 1:])
    return root


def postorder(root):
    if not root:
        return []
    if root.left is None and root.right is None:
        return [root.val]
    result = []
    result += postorder(root.left)
    result += postorder(root.right)
    result += [root.val]
    return result


input()
preorder = list(map(int, input().split()))
inorder = sorted(preorder)
root = build(preorder, inorder)
result = postorder(root)
print(' '.join(map(str, result)))
```



```python
def post_order(pre_order):
    if not pre_order:
        return []
    root = pre_order[0]
    left_subtree = [x for x in pre_order if x < root]
    right_subtree = [x for x in pre_order if x > root]
    return post_order(left_subtree) + post_order(right_subtree) + [root]

n = int(input())
pre_order = list(map(int, input().split()))
print(' '.join(map(str, post_order(pre_order))))
```



```python
class Treenode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# 构建二叉搜索树
def buildtree(p):
    n = len(p)
    stack = []
    root = Treenode(p[0])
    stack.append(root)

    for i in range(1, n):
        cur = Treenode(p[i])
        if p[i] < stack[-1].val:
            stack[-1].left = cur
            stack.append(cur)
        else:
            pre = None
            while stack and p[i] > stack[-1].val:
                pre = stack.pop()
            pre.right = cur
            stack.append(cur)

    return root

# 后序遍历
def hou(root):
    if not root:
        return []
    return hou(root.left) + hou(root.right) + [root.val]

# 输入处理
n = int(input())
p = list(map(int, input().split()))

# 构建BST并获取后序遍历结果
root = buildtree(p)
ans = hou(root)

# 输出结果
print(' '.join(map(str, ans)))
```



## 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/

Given the sum of prime A and prime B, find A and B.

**输入**

One positive integer indicating the sum (<= 10000).

**输出**

Two integers A and B.

样例输入

```
10
```

样例输出

```
3 7
```



```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def goldbach(n):
    for i in range(2, n):
        if is_prime(i) and is_prime(n - i):
            return i, n - i

n = int(input())
a, b = goldbach(n)
print(a, b)
```



```python
# 23n2300011075(才疏学浅)
from math import sqrt
n=10000
ls,x,y=[True]*(n+1),2,int(sqrt(n))+1
while x<y:
    if ls[x]==True:
        for i in range(x*2,n+1,x):
            ls[i]=False
    x+=1
ls=set([i for i in range(2,n+1) if ls[i]==True])

n=int(input())
for i in ls:
    if (n-i) in ls:
        print(i,n-i)
        break
```





## 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/

生态文明建设是关系中华民族永续发展的根本大计。近年来，为了响应环保号召，商家们研制出了环保烟花。这类烟花在烟花配方中不采用含有重金属和硫元素的物质，从而减少了硫化物的生成。

为了庆祝院庆，A大学计算机院燃放了一批环保烟花。从正面看，烟花的构成了二叉树的形状。那么从侧面看，烟花又是什么样子的呢？

对于一个二叉树形状的烟花，它有N个节点，每个节点都有一个1~N之间的颜色编号，不同节点的编号互不相同。除了根节点的编号固定为1，其他节点的编号都是随机分配的。

我们需要按照从顶部到底部的顺序，输出从右侧能看到的节点的颜色编号，即**输出广度优先搜索中每一层最后一个节点**。



例如对于如下的二叉树烟花，从右侧看到的结果为[1, 3, 4]。

![img](http://media.openjudge.cn/images/upload/5167/1622035106.png)

再如，对于如下的二叉树烟花，从右侧看到的结果为[1, 7, 5, 6, 2]。

![img](http://media.openjudge.cn/images/upload/6408/1622728596.png)

**输入**

输入共N+1行。
第1行为一个整数N（1<=N<=1000），表示二叉树中的节点个数。这N个节点的颜色编号分别为1到N，其中1号节点为根节点。
接下来N行每行有两个整数，分别为1~N号节点的左子节点和右子节点的颜色编号，如果子节点为空，则用-1表示。

**输出**

按从顶到底的顺序，输出从右侧看二叉树看到的各个节点的颜色编号（即广度优先搜索中每一层最后一个节点），每个编号之间用空格隔开。

样例输入

```
5
2 3
-1 5
-1 4
-1 -1
-1 -1
```

样例输出

```
1 3 4
```

提示

（1）一种处理本题的输入形式的方式：可先开辟一个大小为N的数组，存储这N个二叉树节点，然后根据每行输入，将相关节点用左右子节点指针连接起来。
（2）BFS可以借助队列实现，可以使用STL

来源

TA Zhang C.



This problem is about traversing a binary tree in a breadth-first manner and printing the last node of each level when viewed from the right side. 

Here is a Python solution using a queue for the breadth-first traversal:

```python
from collections import deque

def right_view(n, tree):
    queue = deque([(1, tree[1])])  # start with root node
    right_view = []

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node, children = queue.popleft()
            if children[0] != -1:
                queue.append((children[0], tree[children[0]]))
            if children[1] != -1:
                queue.append((children[1], tree[children[1]]))
        right_view.append(node)

    return right_view

n = int(input())
tree = {1: [-1, -1] for _ in range(n+1)}  # initialize tree with -1s
for i in range(1, n+1):
    left, right = map(int, input().split())
    tree[i] = [left, right]

result = right_view(n, tree)
print(' '.join(map(str, result)))
```

This script first reads the number of nodes and the tree structure from the input. It then calls the `right_view` function to compute the right view of the tree. The `right_view` function uses a queue to perform a breadth-first traversal of the tree. It processes all nodes at the current level before moving on to the next level. After processing all nodes at a level, it appends the last node of the level to the `right_view` list. Finally, it prints the right view of the tree.



```python
# 蒋子轩 工院
def dfs(node,level):
    if ans[level]==0:
        ans[level]=node
    for next in tree[node][::-1]:
        if next!=-1:
            dfs(next,level+1)
n=int(input())
tree={}
ans=[0]*n
for i in range(n):
    tree[i+1]=list(map(int,input().split()))
dfs(1,0)
res=[]
for i in ans:
    if i: res.append(i)
    else: break
print(*res)
```



直接每次把下一层的入栈，输出最后一个即可

```python
# 童溯 数学科学学院
n = int(input())
tree = [0]
for i in range(n):
    tree.append(list(map(int, input().split())))
stack = [1]
ans = []
while stack:
    ans.append(str(stack[-1]))
    temp = []
    for x in stack:
        if tree[x][0] != -1:
            temp.append(tree[x][0])
        if tree[x][1] != -1:
            temp.append(tree[x][1])
    stack = temp
print(" ".join(ans))
```



## T22508:最小奖金方案

DAG 上的 DP, http://cs101.openjudge.cn/practice/22508/

现在有n个队伍参加了比赛，他们进行了m次PK。现在赛事方需要给他们颁奖（奖金为整数），已知参加比赛就可获得100元，由于比赛双方会比较自己的奖金，所以获胜方的奖金一定要比败方奖金高。请问赛事方要准备的最小奖金为多少？奖金数额一定是整数。

**输入**

一组数据，第一行是两个整数n(1≤n≤1000)和m(0≤m≤2000)，分别代表n个队伍和m次pk，队伍编号从0到n-1。接下来m行是pk信息，具体信息a，b，代表编号为a的队伍打败了编号为b的队伍。
输入保证队伍之间的pk战胜关系不会形成有向环

**输出**

给出最小奖金w

样例输入

```
5 6
1 0
2 0
3 0
4 1
4 2
4 3
```

样例输出

```
505
```

来源: 陈鑫



这个问题可以建模为一个 **有向无环图（DAG）中的最小奖金分配问题**，其本质是：

- 每个队伍初始获得 100 元。
- 如果 a 打败了 b，则 a 的奖金必须 **严格大于** b 的奖金。
- 我们的目标是满足所有约束的前提下，使总奖金最小。

---

**解题思路**

这是一个典型的 **拓扑排序 + 动态规划** 的问题：

1. **构建图：**
   - 将每个比赛结果 `(a, b)` 看作一条从 b 到 a 的边（因为 a 要比 b 高）。
2. **拓扑排序：**
   - 因为题目保证没有环，所以我们可以进行拓扑排序来处理依赖关系。
3. **动态规划：**
   - 初始化每个人的奖金为 100。
   - 按照拓扑序处理每个节点，更新获胜队伍的奖金：
     - 对于每条边 `b -> a`（即 a 打败了 b），如果 `bonus[a] <= bonus[b]`，则设置 `bonus[a] = bonus[b] + 1`。

这样就能确保每个人只被处理一次，并且满足所有胜负关系。



```python
import sys
from collections import defaultdict, deque

def min_bonus(n, m, matches):
    # 图结构：记录谁打败了谁（反向边）
    graph = defaultdict(list)
    indegree = [0] * n
    
    for a, b in matches:
        graph[b].append(a)  # a > b，所以 b 是 a 的前驱
        indegree[a] += 1

    # 初始化奖金为 100
    bonus = [100] * n

    # 拓扑排序队列
    queue = deque([i for i in range(n) if indegree[i] == 0])

    while queue:
        curr = queue.popleft()
        for neighbor in graph[curr]:
            # 如果邻居的奖金不大于当前的，就调整它
            if bonus[neighbor] <= bonus[curr]:
                bonus[neighbor] = bonus[curr] + 1
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return sum(bonus)

# 读取输入
if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    matches = []
    idx = 2
    for _ in range(m):
        a = int(data[idx])
        b = int(data[idx+1])
        matches.append((a, b))
        idx += 2

    result = min_bonus(n, m, matches)
    print(result)
```

时间复杂度分析

- 拓扑排序时间复杂度为 O(n + m)
- 每个边最多处理一次，奖金更新也是 O(1) 操作





## 22509: 解方程

http://cs101.openjudge.cn/practice/22509/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional problems



## 22510: 皮卡丘的冒险

dijkstra, http://dsbpython.openjudge.cn/2024hw12/002/

某天火箭队终于成功抓到了皮卡丘，但是因为主角光环加身，他又双叒叕逃脱了。他想赶紧回到
小智身边，但是前面有一片森林挡住了去路。
用一张地图表示这片森林，其中顶点代表森林的隘口，隘口之间有路相连，皮卡丘想要穿过隘口
就必须打败这里的宝可梦（只用打败一次就可以，第二次再来的时候不用再打），打败宝可梦需
要时间，在路上跑也需要花费时间，假设小智一直在原地等待，皮卡丘想知道他最短需要花费多
长时间才能和小智重逢。

**输入**

输入的第一行为整数N（1≤N≤2000）和整数M（2≤M≤5000），用空格分开，分别代表隘口的数量和路的数量，隘口的编号为2..N+1，编号0和1用来分别代表皮卡丘和小智所在的位置。
输入的第2..N+1行为一个自然数，表示击败该编号对应的隘口的宝可梦需要的时间。
接下来M行，每行三个自然数，用空格分开，代表一条路两端的顶点编号和经过这条路需要的时间。路是双向的。

**输出**

输出一行，即皮卡丘和小智重逢所需的最短时间。测试用例保证皮卡丘和小智一定能再次相逢。

样例输入

```
2 5
1
1
0 2 4
3 0 1
2 3 1
2 1 2
1 3 5
```

样例输出

```
6
```

来源

刘宇航



copilot把击败时间直接建图的花费里面了，挺方便

```python
# GitHub Copilot
# 22510皮卡丘的冒险, http://dsbpython.openjudge.cn/2024hw12/002/
import heapq

def dijkstra(graph, start):
    N = len(graph)
    dist = [float('inf')] * N
    dist[start] = 0
    queue = [(0, start)]

    while queue:
        v_dist, v = heapq.heappop(queue)
        if v_dist != dist[v]:
            continue
        for w, w_dist in graph[v]:
            if dist[v] + w_dist < dist[w]:
                dist[w] = dist[v] + w_dist
                heapq.heappush(queue, (dist[w], w))
    return dist

N, M = map(int, input().split())
N += 2
graph = [[] for _ in range(N)]
time_to_defeat = [0, 0] + [int(input()) for _ in range(N-2)]

for _ in range(M):
    u, v, t = map(int, input().split())
    graph[u].append((v, t + time_to_defeat[v]))
    graph[v].append((u, t + time_to_defeat[u]))

distances = dijkstra(graph, 0)
print(distances[1])
```



## M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

看到大家在数算课学的很努力，谢老师决定想把大家最终的成绩调高一点。谢老师打算把每人的分数由x分调整为ax+1.1^ax分，其中 a 是一个 0 到 1 之间的常数。

为了简单起见，我们假设调整后分数超过100分也没有问题。

为了方便输出格式，我们假设 a = b / 1000000000，其中 b 是不超过1000000000的正整数。

给定全班同学的原始成绩，请帮谢老师求出一个最小的正整数 b ，使得调分之后优秀（85分及以上）的人数比例不小于 60% 。

（本题为假想情形，调分方法与现实无关）

**输入**

一行以空格分隔的浮点数，分别代表每个学生的原始成绩。学生数量不超过100000，每个学生的原始成绩是一个在 [40, 100] 区间的浮点数。

**输出**

一个整数，使得至少60%学生不小于85分的正整数b的最小值。

样例输入

```
50.5 100.0 40.0
```

样例输出

```
791111236
```

提示

1. 调分所用函数比较复杂，我们难以得到它的逆的解析解，但可以看出它是单调的
2. 程序的运行时间可能受到两个方面的影响：b的范围大、学生数量多
3. 使用整数b除以10^9的值来代替浮点数a做运算，主要是为了避免浮点数误差和四舍五入导致可能openjudge评判答案不对的问题（因为openjudge要求答案完全一致才算对），可能略微提高了题目难度。现实生活中可以全程用浮点数a进行运算
4. 语法提示：

float(x)可以把字符串x转为浮点数，int(x)可以把浮点数x向下取整转为整数
输入可用[float(x) for x in input().split()]来得到成绩的数组
a的b次方在python的写法是 a ** b



【李佳聪 24工学院】二分查找，只需找到60%节点处的学生成绩并与85分比较即可。

```python
grade = [float(x) for x in input().split()]
le = len(grade)
grade.sort()
targ = grade[int(le * 0.4)]
left = 0
right = 1000000000 + 1
ans = 0
while left < right:
    mid = (left + right) // 2
    gd = targ * mid / 1000000000 + 1.1 ** (targ * mid / 1000000000)
    if gd >= 85:
        ans = mid
        right = mid
    else:
        left = mid + 1
print(left)

```





**总结题目要求**：

- 给定原始成绩数组。
- 调整公式：`new_score = a * x + 1.1 ** (a * x)`
- `a = b / 1e9`
- 找最小的正整数 `b`，使得调整后成绩 ≥ 85 分的学生比例 ≥ 60%。

------

**思路：**

- `new_score` 随 `b` 增大是**单调增加**的（因为 `a` 增大，`a*x`和`1.1^(a*x)`都会增大）。
- 所以可以**二分答案b**：从1到10^9之间找最小的满足条件的b。
- 每次给定一个b，计算每个学生的调后分数，看够不够 60% 达到85。

------

**代码直接上：**

```python
def is_ok(b, scores):
    a = b / 1e9
    cnt = 0
    threshold = 0.6 * len(scores)
    for x in scores:
        new_score = a * x + 1.1 ** (a * x)
        if new_score >= 85:
            cnt += 1
    return cnt >= threshold


def main():
    scores = [float(x) for x in input().split()]

    l, r = 1, 10 ** 9 + 1
    ans = -1
    while l < r:
        mid = (l + r) // 2
        if is_ok(mid, scores):
            ans = mid
            r = mid
        else:
            l = mid + 1

    print(ans)


if __name__ == "__main__":
    main()
```

------

**解释一下：**

- `is_ok(b, scores)`：判断当前这个b，调整后有没有60%人达到85分。
- 二分法不断缩小范围，找到最小的符合要求的b。



**补充小知识**（很重要！）：

- `(n * 6 + 9) // 10` 是为了模拟**向上取整的60%**。
  - 比如 `n=5`，(5*6+9)//10 = 3，要求3个人及以上。
  - 比如 `n=100000`，需要60000人及以上。
- 避免直接用浮点数比较比例，因为浮点数有误差。

```python
import math

def is_ok(b, scores, needed, log_1p1):
    a = b / 1e9
    count = 0
    for x in scores:
        ax = a * x
        new_score = ax + math.exp(ax * log_1p1)
        if new_score >= 85:
            count += 1
            if count >= needed:
                return True
    return False

def main():
    scores = [float(x) for x in input().split()]
    n = len(scores)
    needed = (n * 6 + 9) // 10  # 向上取整，不小于60%

    log_1p1 = math.log(1.1)  # 只算一次，1.1的自然对数

    l, r = 1, 10**9
    ans = -1
    while l <= r:
        mid = (l + r) // 2
        if is_ok(mid, scores, needed, log_1p1):
            ans = mid
            r = mid - 1
        else:
            l = mid + 1

    print(ans)

if __name__ == "__main__":
    main()
```





## 22636: 修仙之路

http://cs101.openjudge.cn/practice/22636/

修仙之路长漫漫，逆水行舟，不进则退！你过五关斩六将，终于来到了仙界。仙界是一个r行c列的二维格子空间，每个单元格是一个”境界“，每个境界都有等级。你需要任意选择其中一个境界作为起点，从一个境界可以前往上下左右相邻四个境界之一 ，当且仅当新到达的境界等级增加。你苦苦行走，直到所在的境界等级比相邻四个境界的等级都要高为止，一览众山小。请问包括起始境界在内最长修仙路径需要经过的境界数是多少？

**输入**

第一行为两个正整数，分别为r和c（1<=r,c<=100）。
接下来有r行，每行有c个0到100000000之间的整数，代表各境界的等级。

**输出**

输出一行，为最长修仙路径需要经过的境界数（包括起始境界）。

样例输入

```
5 5
1 2 3 4 5
16 17 18 19 6
15 24 25 20 7
14 23 22 21 8
13 12 11 10 9
```

样例输出

```
25
```



```python
# 23n2300011075(才疏学浅)
def dfs(i,j):
    if dp[i][j]>0:
        return dp[i][j]
    else:
        for k in range(4):
            if 0<=i+d[k][0]<r and 0<=j+d[k][1]<c and maze[i][j]>maze[i+d[k][0]][j+d[k][1]]:
                dp[i][j]=max(dp[i][j],dfs(i+d[k][0],j+d[k][1])+1)
    return dp[i][j]

r,c=map(int,input().split())
maze=[]
for i in range(r):
    l=list(map(int,input().split()))
    maze.append(l)
dp=[[0]*c for _ in range(r)]
d=[[-1,0],[1,0],[0,1],[0,-1]]
ans=0
for i in range(r):
    for j in range(c):
        ans=max(ans,dfs(i,j))
print(ans+1)
```



```python
#23n2300011072(X)
from functools import lru_cache
@lru_cache(maxsize=None)
def dfs(x,y):
    ans=0
    for dx,dy in dir:
        nx,ny=x+dx,y+dy
        if 0<=nx<m and 0<=ny<n and h[nx][ny]<h[x][y]:
            ans=max(ans,dfs(nx,ny)+1)
    return ans
m,n=map(int,input().split())
h=[list(map(int,input().split())) for _ in range(m)]
dir=[(0,1),(1,0),(-1,0),(0,-1)]
res=0
for i in range(m):
    for j in range(n):
        res=max(res,dfs(i,j))
print(res+1)
```



## 22642: 括号生成

http://cs101.openjudge.cn/practice/22642/

Paul是一名数学专业的同学，在课余选修了C++编程课，现在他能够自己写程序判断判断一个给定的由'('和')'组成的字符串是否是正确匹配的。可是他不满足于此，想反其道而行之，设计一个程序，能够生成所有合法的括号组合，请你帮助他解决这个问题。

**输入**

输入只有一行N，代表生成括号的对数（1 ≤ N ≤ 10)。

**输出**

输出所有可能的并且有效的括号组合，按照字典序进行排列，每个组合占一行。

样例输入

```
3
```

样例输出

```
((()))
(()())
(())()
()(())
()()()
```



```python
# 23n2300011072(蒋子轩)
def add(n, left, right, string):
    # 终止条件：如果已经放置了所有的括号
    if left == n and right == n:
        print(string)
        return

    # 如果我们仍然可以放置左括号，则添加左括号
    if left < n:
        add(n, left+1, right, string+'(')

    # 如果右括号数量小于左括号数量，则添加右括号
    if right < left:
        add(n, left, right+1, string+')')

n = int(input())
add(n, 0, 0, '')
```



## 23451: 交互四则运算计算器_带错误表达式版

http://cs101.openjudge.cn/practice/23451/

实现一个人机交互四则运算计算器。该程序可以根据用户输入的 ***含任意多层圆括号\*** 的四则运算表达式给出计算结果，其中合法的输入数字包括正负整数和小数（例如10，-10，10.5，-10.5，***每个数最多带有一个符号***），运算符包括 +、-、*、/。

相关要求和说明如下：

  程序应该允许用户在任何位置添加任意多个空格，比如 -10 * 3.4、0.1 + 1.0 + -2 / 3、0.85 * (10 / -2) * 05 都是合法的表达式

  程序应该允许用户多次输入运算表达式并完成计算，直到用户输入 "quit"；每个表达式占一行

  程序输出的所有计算结果都保留小数点后 3 位数字，例如，当用户输入 -10.1 + 4.3 * 8.5 - 6 / 4 时，程序输出计算结果24.950

  除括号以外，所有等运算优先级的运算都是左结合的。比如 2 / 4 / 2 应当视为 (2 / 4) / 2 = 0.250，而不是 2 / (4 / 2) = 1.000

  输入保证小数点的两侧均有数字。即 1 + 1. 、 .5 + 0 等表达式都是不存在的

  数据保证运算的中间量和结果均在[-1000, 1000]的区间里。

  数据保证不存在精度损失问题。

  本题中输入的表达式可能非法。非法情况可能包括：

​    括号不匹配。此时应输出单个字符串 "Unmatched bracket."

​    表达式里出现不正常的算符（除四则运算符、数字、小数点之外的任何字符）。此时应输出单个字符串 "Unknown operator."

​    表达式不完整，也即二元运算符的两侧未按要求分别为两个数字。此时应输出单个字符串 "Not implemented."

​    空表达式。此时应输出单个字符串 "No expression." 注意，此情况可能表示多组匹配的括号，但没有任何数值。比如().

祝你好运！——Pasu

**输入**

输入 N+1 行
其中 N 行为待运算的表达式
最后一行为 quit

**输出**

N 行计算结果
计算结果保留小数点后 3 位数字

样例输入

```
 (((-10.1 + 4.3) * 8.5) - 6) / 4   
((1+     2)*3
      (1+1+1.   1) /    3

1 ++ 1
1 +++ 1
1^2
quit
```

样例输出

```
-13.825
Unmatched bracket.
1.033
No expression.
2.000
Not implemented.
Unknown operator.
```

提示

本题目禁用eval函数。

而且，请注意，本题目明确给出了要求“每个数最多带有一个符号”，因此类似“1+++1”的表达式是不合法的，中间总会有一个“+”满足非法情况中的第三条。
但是“1++1”是合法的，因为后一个“+”可以被理解为后一个“1”的符号。
上述这一点与python内置的eval函数的规则不同，eval函数允许单个数带有多个符号。因此仅使用eval函数也是无法通过该题目的。

如果觉得有困难，可以考虑先转成对应的“逆波兰表达式”。

来源: Pasu



```python
class stack():
    def __init__(self):
        self.val=[]
    def isempty(self):
        return len(self.val)==0
    def push(self,item):
        self.val.append(item)
    def top(self):
        return self.val[-1]
    def pop(self):
        del self.val[-1]

def operatorcheck():
    for i in range(len(exp)):
        if exp[i] not in ch:
            return 0
    return 1

def bracketcheck():
    bracket=stack()
    for i in range(len(exp)):
        if exp[i]=='(':
            bracket.push('(')
        if exp[i]==')':
            if bracket.isempty():
                return 0
            else:
                bracket.pop()
    if bracket.isempty():
        return 1
    else:
        return 0

def onlybracket():
    for i in range(len(exp)):
        if exp[i]!='(' and exp[i]!=')':
            return 0
    return 1
            
def cut():
    i=0
    while i<=len(exp)-1:
        if exp[i]=='*' or exp[i]=='/' or exp[i]=='(' or exp[i]==')':
            expression.append(exp[i])
            i+=1
            continue
        if exp[i]=='+' or exp[i]=='-':
            if i==0 or exp[i-1] not in ch[5:]:
                temp=''+exp[i]
                i+=1
                while i<=len(exp)-1 and exp[i] in ch[6:]:
                    temp=temp+exp[i]
                    i+=1
                expression.append(float(temp))
                continue
            else:
                expression.append(exp[i])
                i+=1
                continue
        if exp[i] in ch[6:]:
            temp=''
            while i<=len(exp)-1 and exp[i] in ch[6:]:
                temp=temp+exp[i]
                i+=1
            expression.append(float(temp))
            continue
def value(s,x,y):
    if s=='+':
        return x+y
    if s=='*':
        return x*y
    if s=='-':
        return x-y
    if s=='/':
        return x/y

def calc():
    operator=stack()
    operand=stack()
    for i in range(len(expression)):
        if expression[i] not in ch[0:6]:
            operand.push(expression[i])
        elif expression[i]=='(':
            operator.push('(')
        elif expression[i]==')':
            while operator.top()!='(':
                b=operand.top()
                operand.pop()
                a=operand.top()
                operand.pop()
                operand.push(value(operator.top(),a,b))
                operator.pop()
            operator.pop()
        elif expression[i] in ch[0:4]:
            while not operator.isempty() and prior[operator.top()]>=prior[expression[i]]:
                b=operand.top()
                operand.pop()
                a=operand.top()
                operand.pop()
                operand.push(value(operator.top(),a,b))
                operator.pop()
            operator.push(expression[i])
    while not operator.isempty():
        b=operand.top()
        operand.pop()
        a=operand.top()
        operand.pop()
        operand.push(value(operator.top(),a,b))
        operator.pop()
    print('{:.3f}'.format(operand.top()))
                
        
ch=['+','-','*','/','(',')','.','0','1','2','3','4','5','6','7','8','9']
prior={'*':3,'/':3,'+':2,'-':2,'(':1}
while True:
    s=list(map(str,input().split()))
    if s==["quit"]:
        break
    if len(s)==0:
        print("No expression.")
        continue
    exp=""
    for i in range(len(s)):
        exp=exp+s[i]
    if operatorcheck()==False:
        print("Unknown operator.")
        continue
    if bracketcheck()==False:
        print("Unmatched bracket.")
        continue
    if onlybracket()==True:
        print("No expression.")
        continue
    expression=[]
    try:
        cut()
        calc()
    except:
        print("Not implemented.")
        continue
```





# 23563~30000

## E23563: 多项式时间复杂度

string, implementation, http://cs101.openjudge.cn/practice/23563/

请参看 2020fall_cs101.openjudge.cn_problems.md 的 Easy 部分的相同题目。



## 23568: 幸福的寒假生活

http://cs101.openjudge.cn/practice/23568/

请参看 2020fall_cs101.openjudge.cn_problems.md 的 Optional problems部分的相同题目。



## 23570: 特殊密码锁

http://cs101.openjudge.cn/practice/23570/

有一种特殊的二进制密码锁，由n个相连的按钮组成（1<=n<30），按钮有凹/凸两种状态，用手按按钮会改变其状态。

然而让人头疼的是，当你按一个按钮时，跟它相邻的两个按钮状态也会反转。当然，如果你按的是最左或者最右边的按钮，该按钮只会影响到跟它相邻的一个按钮。

当前密码锁状态已知，需要解决的问题是，你至少需要按多少次按钮，才能将密码锁转变为所期望的目标状态。

**输入**

两行，给出两个由0、1组成的等长字符串，表示当前/目标密码锁状态，其中0代表凹，1代表凸。

**输出**

至少需要进行的按按钮操作次数，如果无法实现转变，则输出impossible。

样例输入

```
011
000
```

样例输出

```
1
```



```python
"""
the toggle function is used to flip the bit, which simplifies the flip function. 
using a for-loop to iterate over the two cases: pressing the first button or not. 
"""
def toggle(bit):
    return '0' if bit == '1' else '1'

def flip(lock, i):
    if i > 0:
        lock[i-1] = toggle(lock[i-1])
    lock[i] = toggle(lock[i])
    if i + 1 < len(lock):
        lock[i+1] = toggle(lock[i+1])

def main():
    s = input()
    fin = input()
    n = len(s)
    ans = float('inf')

    for press_first in [False, True]:
        tmp = 0
        lock = list(s)
        if press_first:
            flip(lock, 0)
            tmp += 1
        for i in range(1, n):
            if lock[i-1] != fin[i-1]:
                flip(lock, i)
                tmp += 1
        if lock[n-1] == fin[n-1]:
            ans = min(ans, tmp)

    if ans == float('inf'):
        print("impossible")
    else:
        print(ans)

if __name__ == "__main__":
    main()
```



## 23660: 7的倍数取法有多少种

http://cs101.openjudge.cn/practice/23660/

在n个不同的正整数里，任意取若干个，不能重复取,要求它们的和是7的倍数，问有几种取法。

**输入**

第一行是整数t，表示有t组数据(t<10)。接下来有t行，每行是一组数据，每组数据第一个数是n（1 <= n <= 16），表示要从n个整数里取数,接下来就是n个整数。

**输出**

对每组数据，输出一行，表示取法的数目（一个都不取也算一种取法）。 

样例输入

```
3
3 1 2 4
5 1 2 3 4 5
12 1 2 3 4 5 6 7 8 9 10 11 12
```

样例输出

```
2
5
586
```

来源

郭炜



```python
def count_combinations(numbers, index, current_sum, count):
    if index >= len(numbers):
        if current_sum % 7 == 0:
            return count + 1
        else:
            return count
    
    # 选择取当前位置的数
    count = count_combinations(numbers, index + 1, current_sum + numbers[index], count)
    
    # 选择不取当前位置的数
    count = count_combinations(numbers, index + 1, current_sum, count)
    
    return count


# 主程序
t = int(input())
for _ in range(t):
    data = list(map(int, input().split()))
    n = data[0]
    numbers = data[1:]
    
    result = count_combinations(numbers, 0, 0, 0)
    print(result)
```





## 24375: 小木棍

http://cs101.openjudge.cn/practice/24375/

小明将一批等长的木棍随机切成最长为50单位的小段。现在他想要将木棍还原成原来的状态，但是却忘记了原来的木棍数量和长度。请写一个程序帮助他计算如果还原成原来的等长木棍，其长度可能的最小值。所有的长度均大于0。

**输入**

输入包含多个实例。每个实例有两行，第一行是切割后的木棍数量n（最多64个），第二行为n个以空格分开的整数，分别为每根木棍的长度。输入的最后以n为 0 结束。

**输出**

对于每个实例，输出一行其长度的可能的最小值。

样例输入

```
9
5 2 1 5 2 1 5 2 1
4
1 2 3 4
0
```

样例输出

```
6
5
```

来源：来自计算概论B期末考试，本题对数据进行了弱化



与 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

中 Optional problems的 01011: Sticks 一样的题目，算法说明也在 01011。



```python
#蒋子轩
def dfs(rem_sticks,rem_len,target):
    if rem_sticks==0 and rem_len==0:
        return True
    if rem_len==0:
        rem_len=target
    for i in range(n):
        if not used[i] and lens[i]<=rem_len:
            used[i]=True
            if dfs(rem_sticks-1,rem_len-lens[i],target):
                return True
            else:
                used[i]=False
                if lens[i]==rem_len or rem_len==target:
                    return False
    return False
while True:
    n=int(input())
    if n==0:
        break
    lens=list(map(int,input().split()))
    lens.sort(reverse=True)
    total_len=sum(lens)
    for l in range(lens[0],total_len//2+1):
        if total_len%l!=0:
            continue
        used=[False]*n
        if dfs(n,l,l):
            print(l)
            break
    else:
        print(total_len)
```





## 24588: 后序表达式求值

http://cs101.openjudge.cn/practice/24588/

后序表达式由操作数和运算符构成。操作数是整数或小数，运算符有 + - * / 四种，其中 * / 优先级高于 + -。后序表达式可用如下方式递归定义：

1) 一个操作数是一个后序表达式。该表达式的值就是操作数的值。
2) 若a,b是后序表达式，c是运算符，则"a b c"是后序表达式。“a b c”的值是 (a) c (b),即对a和b做c运算，且a是第一个操作数，b是第二个操作数。下面是一些后序表达式及其值的例子(操作数、运算符之间用空格分隔)：

3.4       值为：3.4
5        值为：5
5 3.4 +     值为：5 + 3.4
5 3.4 + 6 /   值为：(5+3.4)/6
5 3.4 + 6 * 3 + 值为：(5+3.4)*6+3



**输入**

第一行是整数n(n<100)，接下来有n行，每行是一个后序表达式，长度不超过1000个字符

**输出**

对每个后序表达式，输出其值，保留小数点后面2位

样例输入

```
3
5 3.4 +
5 3.4 + 6 /
5 3.4 + 6 * 3 +
```

样例输出

```
8.40
1.40
53.40
```

来源

Guo wei



```python
def compute(stack, operator):
    op1 = stack.pop()
    op2 = stack.pop()
    if operator == '+':
        return op2 + op1
    elif operator == '-':
        return op2 - op1
    elif operator == '*':
        return op2 * op1
    elif operator == '/':
        return op2 / op1

def post_eva(formula):
    comp = '+-*/'
    wordlist = formula.split()
    opStack = []
    for word in wordlist:
        if word not in comp:
            opStack.append(float(word))
        else:
            op = compute(opStack, word)
            opStack.append(op)
    return opStack[0]

n = int(input())
for _ in range(n):
    result = post_eva(input())
    print(f"{result:.2f}")
```





## T24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

中序表达式是运算符放在两个数中间的表达式。乘、除运算优先级高于加减。可以用"()"来提升优先级 --- 就是小学生写的四则算术运算表达式。中序表达式可用如下方式递归定义：

1）一个数是一个中序表达式。该表达式的值就是数的值。

2) 若a是中序表达式，则"(a)"也是中序表达式(引号不算)，值为a的值。
3) 若a,b是中序表达式，c是运算符，则"acb"是中序表达式。"acb"的值是对a和b做c运算的结果，且a是左操作数，b是右操作数。

输入一个中序表达式，要求转换成一个后序表达式输出。

**输入**

第一行是整数n(n<100)。接下来n行，每行一个中序表达式，数和运算符之间没有空格，长度不超过700。

**输出**

对每个中序表达式，输出转成后序表达式后的结果。后序表达式的数之间、数和运算符之间用一个空格分开。

样例输入

```
3
7+8.3 
3+4.5*(7+2)
(3)*((3+4)*(2+3.5)/(4+5)) 
```

样例输出

```
7 8.3 +
3 4.5 7 2 + * +
3 3 4 + 2 3.5 + * 4 5 + / *
```

来源

Guo wei



接收浮点数，是number buffer技巧。

```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)

n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```



接收数据，还可以用re处理。

```python
# 24591:中序表达式转后序表达式
# http://cs101.openjudge.cn/practice/24591/

def inp(s):
    #s=input().strip()
    import re
    s=re.split(r'([\(\)\+\-\*\/])',s)
    s=[item for item in s if item.strip()]
    return s

exp = "(3)*((3+4)*(2+3.5)/(4+5)) "
print(inp(exp))
```



```python
def tokenize(expression):
    import re
    tokens = re.findall(r'\d+.\d+|\d+|\D', expression)
    tokens = [token.strip() for token in tokens if token.strip()]
    return tokens

exp = "(3)*((3+4)*(2+3.5)/(4+5)) "
print(tokenize(exp))
# ['(', '3', ')', '*', '(', '(', '3', '+', '4', ')', '*', '(', '2', '+', '3.5', ')', '/', '(', '4', '+', '5', ')', ')']


```



卢卓然-23-生命科学学院，思路：

首先我们来考虑一个括号内的转化。对于一个括号内的表达式，可以写为这种格式：**乘除 加减 乘除 加减 乘除**。其中，乘除可以包括连乘连除或先乘后除。注意：中序和后序的**数字相对位置**是不变的。

对于**连续同级运算**的转换，例如1+2-3转化为12+3-，其规律为：从左向右遍历表达式，若遇到运算符，则证明前一个运算符已经算完了，所以在后序中添加前一个运算符。例如遍历到减号，则说明1+2的运算已经结束，按照同级运算从左到右的原则，应该在后序表达式里写12+。然后将减号储存起来。遍历完3后，储存器内剩了一个减号，把这个减号加到表达式最后。

接着考虑如加粗字体所示的表达式形式（没有括号），倘若遍历到一个加号或减号，就说明**其前面的乘除表达式已经计算完毕了**，根据上一段的内容，应该将储存器内剩余的一个乘除号添加到末尾。而后，倘若储存器中还剩余加减运算（当前加减号之前的加减号），那么说明**这个加减运算已经运算完毕**（因为这个加减运算的两个因子都计算完了），所以要把这个加减号写到表达式的末尾。最后，将当前加减号加入储存器。就可以一直运行下去。



如果**带括号**，那么带括号的部分要做到**单独运算**，括号内部的规则与上面是一样的。因此，可以界定一个“边界”，也就是左括号的位置。如果遇到了右括号，就需要把括号内剩余的加减乘除完成。相当于在每一个括号内运行了上面的代码。具体实现如下，函数`trans_input()`是将输入数据改成易于操作的列表形式。

代码

```python
'''带括号的部分独立运算
（表达式）->后序表达式，再与其他组合在一起
遇到(入栈，遇到操作符同上操作，遇到）弹出（之后所有操作符，相当于独立处理了这部分。
'''
def trans_input(string):
    '''处理输入数据,返回所需列表l'''
    operators={'+','-','*','/','(',')'}
    l=[]
    i=0
    string=string.strip()
    length=len(string)
    flag=False
    while i<length:
        letter=string[i]
        if letter!='.' and letter not in operators:
            if not flag:
                flag=True
                l.append(letter)
                i+=1
            else:
                l[-1]=l[-1]+letter
                i+=1
        elif letter in operators:
            flag=False
            l.append(letter)
            i+=1            
        elif letter=='.':
            temp='.'
            j=i+1
            while j<length and string[j].isdigit()==True:
                temp=temp+string[j]
                j+=1
            i=j
            l[-1]=l[-1]+temp
    return l
case=int(input())
output=[]
for _ in range(case):
    infix_notation=input()
    l=trans_input(infix_notation)
    operators={'+','-','*','/','(',')'}
    result_stack=[]
    operator_stack=[]
    n=len(l)
    for i in range(n):
        letter=l[i]
        if letter not in operators:
            result_stack.append(letter)
        else:
            if letter=='(':
                operator_stack.append(letter)
            elif letter=="*" or letter=='/':
                while operator_stack and operator_stack[-1]!='(' and (operator_stack[-1]=='*' or operator_stack[-1]=='/'):
                    p=operator_stack.pop(-1)
                    result_stack.append(p)
                operator_stack.append(letter)
            elif letter==')':#将括号内剩余的加减写出来
                while operator_stack[-1]!='(':
                    p=operator_stack.pop(-1)
                    result_stack.append(p)
                operator_stack.pop(-1)#删除'('
            else:#letter是加减的情况
                while operator_stack and operator_stack[-1]!='(':
                    p=operator_stack.pop(-1)
                    result_stack.append(p)
                operator_stack.append(letter)
    while operator_stack:#整体剩余的加减写出来
        p=operator_stack.pop(-1)
        result_stack.append(p)
    output.append(' '.join(result_stack))
for o in output:
    print(o)
```



## M24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

探险家小B发现了一颗宝藏二叉树。这棵树的树根为Root，除了Root节点之外，每个节点均只有一个父节点，因此形成了一颗二叉树。宝藏二叉树的每个节点都有宝藏，每个宝藏具有相应的价值。小B希望摘取这些宝藏，使自己的收益最大。可是，宝藏二叉树有一个奇怪的性质，在摘取宝藏的时候，如果两个节点之间有边，那么最多只能摘取其中一个节点上的宝藏，如果因为贪婪而把两个节点上的宝藏都摘取，二叉树就会立即消失，丧失所有奖励。为此，小B求助于你，希望你能给出，小B在不使宝藏二叉树消失的前提下，能够获得宝藏的最大价值。

为了简化题目，规定宝藏二叉树均为完全二叉树，树中节点如图所示自上而下，自左向右，从1-N编号。<img src="http://media.openjudge.cn/images/upload/8154/1652369676.png" alt="img" style="zoom:33%;" />

**输入**

输入分为两行
第一行为一个整数N，代表二叉树中节点的个数。
第二行为一个N个非负整数。第i个数代表二叉树中编号为i的节点上的宝藏价值。

**输出**

输出为一个整数，代表小B的最大收益。

样例输入

```
6
3 4 5 1 3 1
```

样例输出

```
9
```



```python
class Solution:
    def rob(self, values):
        from functools import lru_cache

        n = len(values)

        #@lru_cache(None)
        def dfs(i):
            if i > n:
                return 0, 0  # (rob, not_rob)

            # 左右孩子编号
            left = 2 * i
            right = 2 * i + 1

            l_rob, l_not_rob = dfs(left)
            r_rob, r_not_rob = dfs(right)

            rob_i = values[i - 1] + l_not_rob + r_not_rob
            not_rob_i = max(l_rob, l_not_rob) + max(r_rob, r_not_rob)

            return rob_i, not_rob_i

        return max(dfs(1))  # 根节点编号为1

sol = Solution()
n = int(input())
values = list(map(int, input().split()))
print(sol.rob(values))  
```



可以把这棵完全二叉树看成一个树形上的「最大权独立集」问题——也就是在树上选若干个节点，使得没有一条边的两个端点同时被选中，且被选中节点权值之和最大。

对于任意一个节点 \(i\)，定义两个状态：

- $\mathit{f}[i][0]$：在以 \(i\) 为根的子树中，**不**选节点 i 时能获得的最大权值；
- $\mathit{f}[i][1]$：在以 \(i\) 为根的子树中，**选**节点 i 时能获得的最大权值。

考虑节点 i 的左右孩子编号分别为 l=2i 与 r=2i+1（如果超过 N 则不存在）：

1. **不选** i （状态 `f[i][0]`）时，左右孩子可以选也可以不选，取最大：
2. $
   f[i][0] = 
     \begin{cases}
       \max(f[l][0],\,f[l][1]) & \text{若 } l\le N,\\
       0 & \text{否则}
     \end{cases}
   \;+\;
     \begin{cases}
       \max(f[r][0],\,f[r][1]) & \text{若 } r\le N,\\
       0 & \text{否则.}
     \end{cases}
   $
3. **选** i （状态 `f[i][1]`）时，左右孩子都不能选，只能取它们 “不选” 的状态：
   $
   f[i][1] = v_i 
     \;+\;
     \begin{cases}
       f[l][0] & \text{若 } l\le N,\\
       0 & \text{否则}
     \end{cases}
     \;+\;
     \begin{cases}
       f[r][0] & \text{若 } r\le N,\\
       0 & \text{否则.}
     \end{cases}
   $

答案即为整棵树根节点 1 的这两种状态的最大值：

$
\max\bigl(f[1][0],\,f[1][1]\bigr).
$

由于编号是 1 到 N 的完全二叉树，我们可以直接从 i=N 递减到 1，用数组滚动完成 DP，时间复杂度 O(N)、空间复杂度 O(N)。

下面给出一个 Python 实现：

```python
import sys
sys.setrecursionlimit(10**7)

def max_treasure(N, vals):
    # vals: 1-based list of length N+1
    # f[i][0], f[i][1]
    f0 = [0] * (N + 1)
    f1 = [0] * (N + 1)
    
    for i in range(N, 0, -1):
        l, r = 2*i, 2*i+1
        
        # 状态1：选i
        include = vals[i]
        if l <= N:
            include += f0[l]
        if r <= N:
            include += f0[r]
        f1[i] = include
        
        # 状态0：不选i
        not_include = 0
        if l <= N:
            not_include += max(f0[l], f1[l])
        if r <= N:
            not_include += max(f0[r], f1[r])
        f0[i] = not_include
    
    return max(f0[1], f1[1])

if __name__ == "__main__":
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    vals = [0] + list(map(int, data[1:]))
    print(max_treasure(N, vals))
```

**说明：**

1. 由于是完全二叉树，孩子编号固定为 2i 和 2i+1，方便用数组索引处理。
2. 我们从下往上（即从编号最大的叶子节点向上）计算，每个节点只需要看它的两棵子树的两个状态，不会有环，安全又高效。
3. 最终答案是根节点不选或选的两种情况中较大的一个。



【珂有为 24工学院】思路：这个题给的标签竟然是dp,有点在意料之外，我的第一思路就是把二叉树构造出来，然后递归做(觉得就是典型的能把树构造好就能秒的树的题目)。

考试时把编号(从1开始算，不然没规律)和二叉树位置的对应关系忘了，现想的……把编号转为二进制编码，从第二位开始遍历，假如是`0`就往左，是`1`就往右，就可以找到编号对应的二叉树中的位置。

考虑以节点`node`为根节点的子树的答案。有两种情况，一种是不取该节点的值，那么最大值为以其左右子节点为根节点的子树的答案的和；另一种是取该节点的值，则其左右节点的值不能取。为了方便起见，额外用一个参量表示能否取根节点的值。dfs(node, flag)，flag为`False`，表示不能取`node`的值，反之则可以取。

构造好树以后`dfs(root, True)`即可得到答案。

```python
class Node:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

def InsertNode(root, v, code):
    if code == '0':
        root.left = Node(v)
        return
    elif code == '1':
        root.right = Node(v)
        return
    
    if code[0] == '0':
        InsertNode(root.left, v, code[1:])
    elif code[0] == '1':
        InsertNode(root.right, v, code[1:])

def dfs(root, flag):
    if not root:
        return 0
    
    if flag == False:
        return dfs(root.left, True) + dfs(root.right, True)
    
    return max(root.val + dfs(root.left, False) + dfs(root.right, False), dfs(root.left, True) + dfs(root.right, True))

n = int(input())
values = [0] + list(map(int,input().split()))
root = Node(values[1])
for i in range(2, n + 1):
    InsertNode(root, values[i], bin(i)[3:])

print(dfs(root, True))
```



【白晨旭 24工学院】很简洁，选了当前点就不能选左右儿子，相当于就是左右儿子的notchoose + 当前点；不选当前点就看左右儿子最大能有多少宝藏然后加起来。树和递归融合起来确实简洁又漂亮

```python
def max_value(n, tree):
    def dfs(i):
        if i >= n:
            return 0, 0
        
        left = dfs(2 * i + 1)
        right = dfs(2 * i + 2)
        not_choose = max(left) + max(right)
        choose = left[0] + right[0] + tree[i]
        
        return not_choose, choose
    return max(dfs(0))
    
n = int(input())
values = list(map(int, input().split()))
print(max_value(n, values))
```





```python
# 黄鑫源 24地空学院
# 用递归比较节点与孙子节点的和与子节点和的大小
n = int(input())
l = list(map(int, input().split()))


def dfs(i):
    if i >= n:
        return 0
    else:
        return max(l[i] + dfs(4 * i + 3) + dfs(4 * i + 4) + dfs(4 * i + 5) + dfs(4 * i + 6),
                   dfs(2 * i + 1) + dfs(2 * i + 2))


print(dfs(0))

```



## 24676: 共同富裕

http://cs101.openjudge.cn/practice/24676/

Z&Z公司设计了一种发奖金的规则：把n个人的总奖金分成n x n份，放入一个矩阵中，每一份都为正整数，每个人最终拿到的奖金是矩阵中某一列的和。

但财务认为其中运气成分太高，所以提出了一种平衡性调整：可以对奖金矩阵的任意一行进行右移。具体来说，如果对某一行ai1, ai2, ..., ain进行一次右移，最右侧的奖金移动到这一行的开头：ain, ai1, ai2, ..., ai(n-1)。每一行都可以进行任意次右移操作。

最终的目标是希望在对奖金矩阵的每一行经过若干次右移后，个人拿到奖金的最高值最小，即每列和的最大值最小。

**输入**

输入包括多组数据，每一组数据的第一行包含一个正整数n（n不大于5），代表有n个人参与奖金发放，接下来的n行，每行包含n个正整数，代表奖金矩阵。输入数据以一个0为结尾代表结束

**输出**

对于每组数据，输出一行，包括一个正整数，为奖金最高值的最小值

样例输入

```
2
4 6
3 7
3
1 2 3
4 5 6
7 8 9
0
```

样例输出

```
11
15
```



itertools product permutation区别？

itertools.product和itertools.permutation是Python标准库中的两个模块，用于处理组合和排列的操作。

itertools.product函数返回两个或多个可迭代对象的笛卡尔积。它接受多个可迭代对象作为输入，并返回一个迭代器，该迭代器生成包含所有输入可迭代对象元素的元组。换句话说，它返回了所有可能的组合情况。例如，itertools.product('ABCD', 'xy')将生成('A', 'x')，('A', 'y')，('B', 'x')，('B', 'y')，('C', 'x')和('C', 'y')。

itertools.permutation函数返回一个可迭代对象，该对象生成指定长度的所有可能排列。它接受一个可迭代对象和一个整数作为参数，整数表示每个排列的长度。它生成的排列中的元素顺序不同，但保持了原始集合中元素的顺序。例如，itertools.permutation('ABCD', 2)将生成('A', 'B')，('A', 'C')，('A', 'D')，('B', 'A')，('B', 'C')，('B', 'D')，('C', 'A')，('C', 'B')，('C', 'D')，('D', 'A')，('D', 'B')和('D', 'C')。

因此，itertools.product用于生成多个可迭代对象的所有可能组合，而itertools.permutation用于生成一个可迭代对象的所有可能排列。



为了解决这个问题，我们可以用暴力搜索的方法来尝试每一行的所有可能的右移操作，然后寻找每一列的最大值，最终得到所有情况中的最小值。

为了进行暴力搜索，我们可以利用Python的`itertools`库中的`product`函数来产生所有可能的行移动组合。对于每一种组合，我们计算每一列的和，并保留所有组合中的最大列和的最小值。

下面是对应的Python代码实现：

```python
from itertools import product

def right_shift(row, shift):
    return row[-shift:] + row[:-shift]

def calculate_max_column_sum(matrix):
    n = len(matrix)
    column_sums = [0] * n
    for row in matrix:
        for i, val in enumerate(row):
            column_sums[i] += val
    return max(column_sums)

def find_min_max_column_sum(n, original_matrix):
    min_max_sum = float('inf')

    # 产生所有行可能的移动方式
    all_shifts = list(product(range(n), repeat=n))
    for shifts in all_shifts:
        # 应用移动
        shifted_matrix = [
            right_shift(original_matrix[i], shifts[i]) for i in range(n)
        ]
        # 计算当前移动方式下的最大列和
        max_column_sum = calculate_max_column_sum(shifted_matrix)
        # 更新最小的最大列和
        min_max_sum = min(min_max_sum, max_column_sum)
    
    return min_max_sum

# 输入处理
results = []
while True:
    n = int(input())
    if n == 0:
        break
    
    original_matrix = [list(map(int, input().split())) for _ in range(n)]
    result = find_min_max_column_sum(n, original_matrix)
    results.append(result)

# 输出结果
for result in results:
    print(result)
```



## 24677: 安全位置

http://cs101.openjudge.cn/practice/24677/

公元2200年，人类和外星人开始了一场宇宙大战，你作为百京大学的一名本科小盆友和外星人在一个四维空间展开了一场殊死搏斗。现在给出一串密码，希望你能从中解锁出所有的安全位置。具体来说，密码是一个字符串，你可以将其分为四个部分，每个部分依次代表四维空间中该维度的坐标。如果这四个坐标均在0到500之间（包含0和500）则是一个安全位置。**注意坐标不能含有前导0，即001是不合法的坐标。**

**输入**

输入只有一行，是一个字符串S，0<=len(S)<=30。

**输出**

输出共1行，是一个数字，代表从该密码中解锁出的安全位置的个数。

样例输入

```
010010
```

样例输出

```
2
# ['0.10.0.10', '0.100.1.0']
```



```python
"""
GitHub Copilot Chat:
This solution works by recursively splitting the string into four parts and 
checking if each part is a valid coordinate. 
The safe_locations function takes the remaining string, the current parts, 
and the current depth as arguments. 
If the depth is 4, it checks if the string is empty and if all parts are 
valid coordinates. If so, it returns 1, otherwise it returns 0. 
If the depth is less than 4, it tries to split the string at every possible 
position and recursively calls itself with the new parts and increased depth. 
"""


def safe_locations(s, parts, depth=0):
    if depth == 4:
        if not s and all(0 <= int(part) <= 500 and 
                (part == '0' or not part.startswith('0')) for part in parts):
            return 1
        return 0
    return sum(safe_locations(s[i:], parts + [s[:i]], depth + 1) 
               for i in range(1, len(s) + 1))


s = input().strip()
print(safe_locations(s, []))

```



## 24678: 任性买房

http://cs101.openjudge.cn/practice/24678/

在刚刚过去的5月20日，唐老板抽到了价值为W的买房优惠券，且该优惠券的使用条件是实际支付金额不小于W。正巧618即将来临，他希望在中关村北大街买房，经中介介绍，从南至北总共有n套房，每套房价格为pi，他有一些想法：

1. 能用掉优惠券，多余的钱他自己能出，这样怎么想都很赚
2. 所购买的房屋都是相邻的，这样就能够直接打通（例如在购买k套房时，购买的是i,i+1,i+2,...,i+k-1，其中i >= 1, i + k -1 <= n）
3. 购买的房屋数量尽可能少，使得留下尽可能多的房

请你编写一个程序帮唐老板想想是否存在符合他怪异想法的方案

**输入**

总共两行，第一行是两个整数W和n，0 < W < 10^9, 0 < n < 10^5,中间用空格分开，分别表示优惠券的金额与房子数量；第二行是n个整数，表示第i套房的价格pi, 0 < pi < 10^5

**输出**

如果存在满足条件的方案，请输出购房的最小数量；如果没有，则输出0

样例输入

```
7 6
1 3 5 2 1 4
```

样例输出

```
2
```



使用一个滑动窗口算法。从左到右扫描一遍房价数组，同时维护一个窗口，使得这个窗口中的房价总和大于等于优惠券金额W。我们的目标是找到满足条件的最小窗口长度。

```python
def min_houses_to_buy(W, n, prices):
    min_length = n + 1  # 初始化为最大长度+1，表示不可能的情况
    current_sum = 0     # 当前窗口的价格总和
    left = 0            # 窗口的左边界

    # 遍历房屋价格数组
    for right in range(n):
        current_sum += prices[right]  # 扩展窗口的右边界

        # 当当前总和大于等于W时，尝试缩小窗口的大小
        while current_sum >= W and left <= right:
            min_length = min(min_length, right - left + 1)
            current_sum -= prices[left]  # 缩小窗口的左边界
            left += 1

    # 如果min_length没有更新，说明没有找到满足条件的窗口
    return min_length if min_length <= n else 0

# 读取输入
W, n = map(int, input().split())
prices = list(map(int, input().split()))

# 计算结果并打印
print(min_houses_to_buy(W, n, prices))

```





## E24684: 直播计票

dict, sorting, http://cs101.openjudge.cn/pctbook/E24684/



## 24686: 树的重量

http://cs101.openjudge.cn/practice/24686/

有一棵 k 层的满二叉树（一共有2k-1个节点，且从上到下从左到右依次编号为1, 2, ..., 2k-1），最开始每个节点的重量均为0。请编程实现如下两种操作：

1 x y：给以 x 为根的子树的每个节点的重量分别增加 y（ y 是整数且绝对值不超过100）

2 x：查询（此时的）以 x 为根的子树的所有节点重量之和



**输入**

输入有n+1行。第一行是两个整数k, n，分别表示满二叉树的层数和操作的个数。接下来n行，每行形如1 x y或2 x，表示一个操作。

k<=15（即最多32767个节点），n<=50000。

**输出**

输出有若干行，对每个查询操作依次输出结果，每个结果占一行。

样例输入

```
3 7
1 2 1
2 4
1 6 3
2 1
1 3 -2
1 4 1
2 3
```

样例输出

```
1
6
-3
```

提示

可以通过对数计算某节点的深度：

import math

math.log2(x) #以小数形式返回x的对数值，注意x不能为0



满二叉树是一种特殊的二叉树，其中每个节点要么是叶子节点，要么有两个子节点。  

变量k和n分别代表满二叉树的层数和操作的个数。f和g是两个列表，用于存储每个节点的权重和懒惰标记。dep列表用于存储每个节点的深度。  

如果操作的长度为2，那么这是一个查询操作，需要计算以给定节点为根的子树的所有节点的权重之和。如果操作的长度为3，那么这是一个更新操作，需要更新以给定节点为根的子树的所有节点的权重。  



初始化了三个列表 `f`, `g`, 和 `dep` 来存储关于树的信息。`f` 用于记录懒惰传播的值，`g` 可能是用于存储临时的累积更新，`dep` 存储每个节点的深度。`tot` 是树中节点的总数。

- `f` could represent some aggregated value at each node (like a lazy propagation value).
- `g` could represent some other value that needs to be propagated down the tree (potentially a modification that applies to all child nodes).

计算深度：从下到上计算每个节点的深度。对于满二叉树来说，如果一个节点编号为 `i`，则它的子节点编号为 `2i` 和 `2i + 1`。深度是从最底层叶子节点开始反向计算的。

查询操作，首先获取根节点的权重，然后逐层向上，获取每一层父节点的权重，最后加上懒惰标记的权重。  查询操作(2 x)：如果操作有两个数字，它是一个查询操作。它从根开始累积 `f` 中的值，沿着树向上移动直到达到节点 `x`。然后计算以 `x` 为根的子树中所有节点的重量之和，考虑到懒惰传播的值和直接更新的值，最后打印结果。

更新操作，首先更新给定节点的权重，然后逐层向上，更新每一层父节点的懒惰标记。  增加操作(1 x y)：将 `y` 增加到 `f` 中对应节点 `x` 的值，并且将 `w`（`y` 乘以以 `x` 为根的子树的节点总数）累积到 `g` 中对应节点 `x` 的父节点中。然后它继续沿树向上更新 `g`，直到根节点。

主要思想是使用懒惰标记来优化查询和更新操作的时间复杂度。



问：查询时候，是计算以 x 为根的子树的所有节点重量之和，为什么要向上找根节点，一路计算？

答：查询操作的目标是计算以x为根的子树的所有节点重量之和。这是通过向上找根节点并一路计算来实现的。这种方法的原因是，代码中的更新操作是延迟的，也就是说，当我们对一个区间进行更新操作时，并不立即更新区间中的所有元素，而是将更新的值存储在一个特定的数据结构中（在这个例子中是数组f和g）。然后，当我们进行查询操作时，我们需要检查这个区间是否有待更新的值，如果有，我们就需要在查询的过程中，一路向上找到根节点，将这些待更新的值加入到查询结果中。  

具体来说，对于每一个节点u，我们都存储了一个值f[u]，表示这个节点及其所有子节点需要增加的值。然后，当我们进行查询操作时，我们需要从目标节点开始，一路向上找到根节点，将这些待更新的值加入到查询结果中。这就是为什么我们在查询操作中需要一路向上找根节点的原因。  

同时，我们还需要注意，由于我们的更新操作是延迟的，所以在查询操作中，我们还需要处理那些还没有被实际更新的节点。这就是为什么我们在查询操作中，除了加入f[u]之外，还需要加入g[u]。g[u]存储的是这个节点及其所有子节点由于之前的更新操作而增加的值，但是这些值还没有被实际加入到这些节点中。所以，在查询操作中，我们需要将这些值也加入到查询结果中。

```python
k, n = [int(x) for x in input().split()]
f, g, dep = [], [], []
tot = (1 << k) - 1
for _ in range(tot+1):
    f.append(0)
    g.append(0)
    dep.append(0)
for i in range(tot, 0, -1):
    dep[i] = 1 if i * 2 > tot else dep[i * 2] + 1
for _ in range(n):
    a = [int(x) for x in input().split()]
    if len(a) == 2:
        u = a[1]
        s = f[1]
        while u != 1:
            s += f[u]
            u >>= 1
        ans = s * ((1 << dep[a[1]]) - 1) + g[a[1]]
        print(ans)
    elif len(a) == 3:
        u = a[1]
        w = a[2] * ((1 << dep[u]) - 1)
        f[u] += a[2]
        while u != 1:
            u >>= 1
            g[u] += w
```



## 24687: 封锁管控

http://cs101.openjudge.cn/practice/24687/

为减少人员流动，降低疫情传播风险，某城市决定在内部施加封锁管控措施。

为方便讨论，假设城市为一条线段，从左至右排布了 n 个居民区，第 i 个居民区中住有 ai 个人。现在要建设 m(m<n) 个“管控点”（可视为墙），每个管控点设在相邻两个居民区之间，使得居民的活动不能跨越该管控点。

定义“人口流动指数”为每个居民（从其原住区）能到达的居民区个数的总和。求在建设 m 个管控点后，人口流动指数最小为多少？



例如，5个居民区被1个管控点隔开（数字表示居民区的人数）：

10 50 | 20 30 40

则此时的人口流动指数为 (10 + 50) * 2 + (20 + 30 + 40) * 3 = 390 。

**输入**

输入有两行。第一行为两个正整数n, m（n<=100）；第二行有n个数，表示每个居民区的人数ai（ai<=1000），用空格隔开。

**输出**

输出只有一行。一个正整数表示人口流动指数的最小值。

样例输入

```
5 1
10 50 20 30 40
```

样例输出

```
380
```

提示

对样例的解释：在第三个和第四个居民区间设管控点，此时人口流动指数为
(10+50+20)\*3+(30+40)\*2=380。



为了找到最小的人口流动指数，我们需要确定在哪里建立管控点才能最大限度地减少人口流动。一个朴素的方法是考虑所有可能的管控点设置，然后选择人口流动指数最小的设置。但是，这样做的时间复杂度是非常高的，特别是当居民区数量较多时。

我们可以使用动态规划来解决这个问题。我们可以定义一个动态规划数组 `dp[i][j]` 表示前 `i` 个居民区建立 `j` 个管控点后的最小人口流动指数。

状态转移方程如下：

`dp[i][j] = min(dp[k][j-1] + sum[k+1 to i] * (i-k))` 对于所有 `k < i`

其中 `sum[k+1 to i]` 表示从居民区 `k+1` 到居民区 `i` 的人口数总和。

这样，最终答案将是 `dp[n][m]`。

```python
def min_population_flow(n, m, populations):
    # Initialize the prefix sum array for fast range sum computation
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + populations[i - 1]
    
    # Initialize the DP table
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    
    # Base case: with 0 control points, the flow index is just the sum of all populations times their district count
    for i in range(1, n + 1):
        dp[i][0] = prefix_sum[i] * i
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, min(i, m) + 1):
            for k in range(j-1, i):
                dp[i][j] = min(dp[i][j], dp[k][j-1] + (prefix_sum[i] - prefix_sum[k]) * (i - k))
    
    # The answer is the minimum flow index after setting up m control points
    return dp[n][m]

# Input
n, m = map(int, input().split())
populations = list(map(int, input().split()))

# Output
print(min_population_flow(n, m, populations))
```



## 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/

可以用括号嵌套的方式来表示一棵树。表示方法如下：

1) 如果一棵树只有一个结点，则该树就用一个大写字母表示，代表其根结点。
2) 如果一棵树有子树，则用“树根(子树1,子树2,...,子树n)”的形式表示。树根是一个大写字母，子树之间用逗号隔开，没有空格。子树都是用括号嵌套法表示的树。

给出一棵不超过26个结点的树的括号嵌套表示形式，请输出其前序遍历序列和后序遍历序列。

输入样例代表的树如下图：

![img](http://media.openjudge.cn/images/upload/5805/1653472173.png)

**输入**

一行，一棵树的括号嵌套表示形式

**输出**

两行。第一行是树的前序遍历序列，第二行是树的后序遍历序列



样例输入

```
A(B(E),C(F,G),D(H(I)))
```

样例输出

```
ABECFGDHI
EBFGCIHDA
```

来源：Guo Wei





下面两个代码。先给出用类表示node。

思路：对于括号嵌套树，使用stack记录进行操作中的父节点，node记录正在操作的节点。每当遇见一个字母，将其设为node，并存入stack父节点中；遇到'('，即对当前node准备添加子节点，将其append入stack中，node重新设为None；遇到')'，stack父节点操作完毕，将其弹出并作为操作中的节点node，不断重复建立树，同时最后返出的父节点为树的根root。

前序遍历和后序遍历只要弄清楚意思，用递归很好写，注意这道题并不是二叉树，需要遍历解析树。

```python
class TreeNode:
    def __init__(self, value): #类似字典
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())  # 去掉所有空白字符
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```



用字典表示node

```python
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = {'value': char, 'children': []}
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1]['children'].append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node['value']]
    for child in node['children']:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node['children']:
        output.extend(postorder(child))
    output.append(node['value'])
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())  # 去掉所有空白字符
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```



实现了括号嵌套表示树的解析以及前序、后序遍历：

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

def parse_tree(s, i=0):
    # 当前字符为结点的值（大写字母）
    node = Node(s[i])
    i += 1
    # 如果下一个字符是'('，说明有子树
    if i < len(s) and s[i] == '(':
        i += 1  # 跳过'('
        while True:
            child, i = parse_tree(s, i)  # 解析一个子树
            node.children.append(child)
            # 子树之间用逗号隔开
            if i < len(s) and s[i] == ',':
                i += 1  # 跳过逗号，继续解析下一个子树
            else:
                break
        i += 1  # 跳过')'
    return node, i

def preorder(node, res):
    if node is None:
        return
    res.append(node.val)
    for child in node.children:
        preorder(child, res)

def postorder(node, res):
    if node is None:
        return
    for child in node.children:
        postorder(child, res)
    res.append(node.val)

def main():
    # 读入一行树的括号嵌套表示形式
    s = input().strip()
    root, _ = parse_tree(s)
    
    pre_res = []
    preorder(root, pre_res)
    
    post_res = []
    postorder(root, post_res)
    
    print("".join(pre_res))
    print("".join(post_res))

if __name__ == '__main__':
    main()
```

代码说明

- **Node 类**：定义了树的结点，包含结点值 `val` 和子结点列表 `children`。
- **parse_tree 函数**：采用递归下降的方式解析字符串。遇到大写字母创建结点；若后续遇到 '(' 则说明存在子树，解析所有子树直到遇到 ')'。
- **preorder 和 postorder 函数**：分别实现前序遍历（先访问结点，再遍历所有子树）和后序遍历（先遍历所有子树，最后访问结点）。
- **main 函数**：读取输入，构造树，然后输出前序遍历和后序遍历的结果。





## 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/

假设二叉树的节点里包含一个大写字母，每个节点的字母都不同。

给定二叉树的中序遍历序列和后序遍历序列(长度均不超过26)，请输出该二叉树的前序遍历序列。

**输入**

2行，均为大写字母组成的字符串，表示一棵二叉树的中序遍历序列与后序遍历排列。

**输出**

表示二叉树的前序遍历序列。

样例输入

```
BADC
BDCA
```

样例输出

```
ABCD
```

来源

Lou Yuke



```python
"""
后序遍历的最后一个元素是树的根节点。然后，在中序遍历序列中，根节点将左右子树分开。
可以通过这种方法找到左右子树的中序遍历序列。然后，使用递归地处理左右子树来构建整个树。
"""
def build_tree(inorder, postorder):
    if not inorder or not postorder:
        return []

    root_val = postorder[-1]
    root_index = inorder.index(root_val)

    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]

    left_postorder = postorder[:len(left_inorder)]
    right_postorder = postorder[len(left_inorder):-1]

    root = [root_val]
    root.extend(build_tree(left_inorder, left_postorder))
    root.extend(build_tree(right_inorder, right_postorder))

    return root


def main():
    inorder = input().strip()
    postorder = input().strip()
    preorder = build_tree(inorder, postorder)
    print(''.join(preorder))


if __name__ == "__main__":
    main()
```



可以利用这幅图，理解下面代码。图源自这里：https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/solutions/426738/cong-zhong-xu-yu-hou-xu-bian-li-xu-lie-gou-zao-14/

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240324122940268.png" alt="image-20240324122940268" style="zoom: 50%;" />



```python
"""
定义一个递归函数。在这个递归函数中，我们将后序遍历的最后一个元素作为当前的根节点，然后在中序遍历序列中找到这个根节点的位置，
这个位置将中序遍历序列分为左子树和右子树。
"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None

    # 后序遍历的最后一个元素是当前的根节点
    root_val = postorder.pop()
    root = TreeNode(root_val)

    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_val)

    # 构建右子树和左子树
    root.right = buildTree(inorder[root_index + 1:], postorder)
    root.left = buildTree(inorder[:root_index], postorder)

    return root


def preorderTraversal(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorderTraversal(root.left))
        result.extend(preorderTraversal(root.right))
    return result


# 读取输入
inorder = input().strip()
postorder = input().strip()

# 构建树
root = buildTree(list(inorder), list(postorder))

# 输出前序遍历序列
print(''.join(preorderTraversal(root)))
```



## 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/

后序算术表达式可以通过栈来计算其值，做法就是从左到右扫描表达式，碰到操作数就入栈，碰到运算符，就取出栈顶的2个操作数做运算(先出栈的是第二个操作数，后出栈的是第一个)，并将运算结果压入栈中。最后栈里只剩下一个元素，就是表达式的值。

有一种算术表达式不妨叫做“队列表达式”，它的求值过程和后序表达式很像，只是将栈换成了队列：从左到右扫描表达式，碰到操作数就入队列，碰到运算符，就取出队头2个操作数做运算（先出队的是第2个操作数，后出队的是第1个），并将运算结果加入队列。最后队列里只剩下一个元素，就是表达式的值。

给定一个后序表达式，请转换成等价的队列表达式。例如，`3 4 + 6 5 * -`的等价队列表达式就是`5 6 4 3 * + -` 。

**输入**

第一行是正整数n(n<100)。接下来是n行，每行一个由字母构成的字符串，长度不超过100,表示一个后序表达式，其中小写字母是操作数，大写字母是运算符。运算符都是需要2个操作数的。

**输出**

对每个后序表达式，输出其等价的队列表达式。

样例输入

```
2
xyPzwIM
abcABdefgCDEF
```

样例输出

```
wzyxIPM
gfCecbDdAaEBF
```

提示

建立起表达式树，按层次遍历表达式树的结果前后颠倒就得到队列表达式

来源：Guo Wei modified from Ulm Local 2007



The problem is asking to convert a postfix expression to an equivalent queue expression. The queue expression is obtained by reversing the level order traversal of the expression tree built from the postfix expression.  

Here is a step-by-step plan:  
1.Create a TreeNode class to represent each node in the tree.
2.Create a function build_tree that takes the postfix expression as input and returns the root of the constructed tree.
	Use a stack to store the nodes.
	Iterate over the characters in the postfix expression.
	If the character is an operand, create a new node and push it onto the stack.
	If the character is an operator, pop two nodes from the stack, make them the children of a new node, and push the new node onto the stack.
3.Create a function level_order_traversal that takes the root of the tree as input and returns the level order traversal of the tree.
	Use a queue `traversal` to store the nodes to be visited.
	While the queue is not empty, dequeue a node, visit it, and enqueue its children.
4.For each postfix expression, construct the tree, perform the level order traversal, reverse the result, and output it.

```python
from collections import deque

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(postfix):
    stack = []
    for char in postfix:
        node = TreeNode(char)
        if char.isupper():
            node.right = stack.pop()
            node.left = stack.pop()
        stack.append(node)
    return stack[0]

def level_order_traversal(root):
    dq = [root]
    traversal = []
    while dq:
        node = dq.pop(0)
        traversal.append(node.value)
        if node.left:
            dq.append(node.left)
        if node.right:
            dq.append(node.right)
    return traversal

n = int(input().strip())
for _ in range(n):
    postfix = input().strip()
    root = build_tree(postfix)
    queue_expression = level_order_traversal(root)[::-1]
    print(''.join(queue_expression))
```



## 25145: 猜二叉树（按层次遍历）

http://cs101.openjudge.cn/practice/25145/

一棵二叉树，结点都是大写英文字母，且不重复。

给出它的中序遍历序列和后序遍历序列，求其按层次遍历的序列。

 

**输入**

第一行是整数n, n <=30，表示有n棵二叉树
接下来每两行代表一棵二叉树，第一行是其中序遍历序列，第二行是后序遍历序列

**输出**

对每棵二叉树输出其按层次遍历序列

样例输入

```
2
LZGD
LGDZ
BKTVQP
TPQVKB
```

样例输出

```
ZLDG
BKVTQP
```

来源: Guo Wei



```python
from collections import deque

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def build_tree(inorder, postorder):
    if inorder:
        root = Node(postorder.pop())
        root_index = inorder.index(root.data)
        root.right = build_tree(inorder[root_index+1:], postorder)
        root.left = build_tree(inorder[:root_index], postorder)
        return root

def level_order_traversal(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.data)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

n = int(input())
for _ in range(n):
    inorder = list(input().strip())
    postorder = list(input().strip())
    root = build_tree(inorder, postorder)
    print(''.join(level_order_traversal(root)))
```



## 25655: 核酸检测统计

http://cs101.openjudge.cn/practice/25655/

因疫情防控需要，每名同学需要遵守“三天一检”的核酸要求，即每三天至少需要做一次核酸检测。现在需要统计学校同学近9天内的核酸检测完成情况，其中每名同学第一天必须完成一次核酸检测。请输出有多少名同学没有按时完成核酸检测，并输出完成情况（未按时完成核酸的学生数量除以院系总人数）最差的院系编号。

**输入**

第一行是整数n，为学生数量；
第二行是整数m，为核酸检测信息数量；
接下来先有n行，每行为学生的基本信息，即学生编号和院系编号，用空格隔开；
最后为m行核酸检测信息，每行为检测日期和学生编号，用空格隔开。其中，检测日期为1～9的数字。

**输出**

第一行为没有按时完成核酸检测的学生数量；
第二行为完成情况最差的院系编号

样例输入

```
3
10
1001 101
1003 101
1004 102
1 1001
3 1001
6 1001
6 1003
1 1003
8 1003
4 1003
4 1004
7 1004
2 1004
```

样例输出

```
2
102
```



```python
# 真不玩原
from collections import defaultdict

n = int(input())  # 学生数量
m = int(input())  # 核酸检测信息数量

# 学生基本信息，以及核酸检测信息
student_info = [list(map(int, input().split())) for _ in range(n)]
test_info = [list(map(int, input().split())) for _ in range(m)]

# 统计每名学生的核酸检测情况
test_record = defaultdict(list)
for day, student_id in test_info:
    test_record[student_id].append(day)

# 统计未按时完成核酸检测的学生数量
late_count = 0
department_uncompletion = defaultdict(int)
department_total_students = defaultdict(int)

for student in student_info:
    student_id, department = student
    sign = False
    a = sorted(test_record[student_id])
    if a[0] != 1 or max(a) < 7:
        sign = True
    for i in range(len(a)-1):
        if a[i+1] - a[i] > 3:
            sign = True
            break
    if sign:
        late_count += 1
        department_uncompletion[department] += 1
    department_total_students[department] += 1

# 计算每个院系未按时完成核酸检测的学生数量占比
department_ratio = {}
for department in department_uncompletion.keys():
    ratio = department_uncompletion[department] / department_total_students[department]
    department_ratio[department] = ratio

# 输出结果
worst_department = max(department_ratio, key=department_ratio.get)

print(late_count)
print(worst_department)
```



## 25815: 回文字符串

http://cs101.openjudge.cn/practice/25815/

给定一个字符串 S ，最少需要几次增删改操作可以把 S 变成一个回文字符串？

一次操作可以在任意位置插入一个字符，或者删除任意一个字符，或者把任意一个字符修改成任意其他字符。

**输入**

字符串 S。S 的长度不超过100, 只包含'A'-'Z'。

**输出**

最少的修改次数。

样例输入

```
ABAD
```

样例输出

```
1
```

来源: hihoCoder



```python
# 2300011335
S = list(input())
n = len(S)
dp = [[0 for _ in range(n)] for _ in range(n)]
for length in range(1,n):
    for i in range(n-length):
        j = i+length
        if S[i] == S[j]:
            dp[i][j] = dp[i+1][j-1]
        else:
            dp[i][j] = min(dp[i+1][j],dp[i][j-1],dp[i+1][j-1])+1
print(dp[0][-1])
```



## 26495: 素数和

http://hxsjjg.openjudge.cn/2024finaltest/7/

给定 n 个整数和一个整数 k（k < n）. 从这 n 个整数中任选 k 个整数相加，可分别得到一系列的和。现在，要求你计算出和为素数共有多少种。

**输入**

输入包含两行内容
第一行两个空格隔开的整数 n 和 k. (1 ≤ n ≤ 20, k < n)
第二行有 n 个整数，每个整数的大小范围 [1, 5×10^6]，可能存在相同的整数

**输出**

输出一个整数，表示和为素数的情况共有多少种

样例输入

```
4 3
3 7 12 19
```

样例输出

```
1
```



Python实现

```python
import itertools

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def count_prime_sums(nums, k):
    count = 0
    for combination in itertools.combinations(nums, k):
        if is_prime(sum(combination)):
            count += 1
    return count

n, k = map(int, input().split())
nums = list(map(int, input().split()))
print(count_prime_sums(nums, k))
```



C++实现

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// 辅助函数：判断一个数是否为素数
bool isPrime(int num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    int sqrt_n = static_cast<int>(sqrt(num));
    for (int i = 3; i <= sqrt_n; i += 2) {
        if (num % i == 0) return false;
    }
    return true;
}

// 递归函数：生成所有组合并计算和
void findCombinations(const vector<int>& numbers, int k, int start, int current_sum, int& prime_count, vector<int>& combination) {
    if (combination.size() == k) {
        if (isPrime(current_sum)) {
            prime_count++;
        }
        return;
    }

    for (int i = start; i < numbers.size(); ++i) {
        combination.push_back(numbers[i]);
        findCombinations(numbers, k, i + 1, current_sum + numbers[i], prime_count, combination);
        combination.pop_back();
    }
}

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> numbers(n);
    for (int i = 0; i < n; ++i) {
        cin >> numbers[i];
    }

    int prime_count = 0;
    vector<int> combination;
    findCombinations(numbers, k, 0, 0, prime_count, combination);

    cout << prime_count << endl;
    return 0;
}
```





## 26518: 最大无环图扩展

http://cs101.openjudge.cn/practice/28334/

给定一个有向无环图（DAG），你的任务是在保持图的有向无环性质的前提下，尽可能多地添加边，使得加完边之后的图仍然是一个有向无环图。

**输入**

输入包含两部分：
第一部分是两个整数 n，m，n表示图中节点的总数（1 ≤ n ≤ 10），m表示边的总数（1 ≤ m ≤ 100）。
第二部分有m行，每行的两个数a，b表示存在一条从节点 a 指向节点 b 的有向边（0 ≤ a, b < n）。
重边算做一条。

**输出**

输出一个整数k，表示添加的边的数量

样例输入

```
6 6
5 2
5 0
4 0
4 1
2 3
3 1
```

样例输出

```
9
```

提示

拓扑排序！



Python实现

```python
from collections import deque

def max_dag(n, edges):
    adj = [[] for _ in range(n)]
    indegree = [0]*n
    for a, b in edges:
        adj[a].append(b)
        indegree[b] += 1

    topo_order = []
    queue = deque([i for i in range(n) if indegree[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    added_edges = 0
    for i in range(n):
        for j in range(i+1, n):
            if topo_order[j] not in adj[topo_order[i]]:
                added_edges += 1

    return added_edges

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]
print(max_dag(n, edges))
```



C++实现

```c++
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>

using namespace std;

int max_dag(int n, vector<vector<int>>& edges) {
    vector<vector<int>> adj(n);
    vector<int> indegree(n, 0);

    for (auto& edge : edges) {
        int a = edge[0];
        int b = edge[1];
        adj[a].push_back(b);
        indegree[b]++;
    }

    vector<int> topo_order;
    deque<int> queue;
    for (int i = 0; i < n; ++i) {
        if (indegree[i] == 0)
            queue.push_back(i);
    }

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();
        topo_order.push_back(node);
        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0)
                queue.push_back(neighbor);
        }
    }

    int added_edges = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (find(adj[topo_order[i]].begin(), adj[topo_order[i]].end(), topo_order[j]) == adj[topo_order[i]].end())
                added_edges++;
        }
    }

    return added_edges;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> edges(m, vector<int>(2));
    for (int i = 0; i < m; ++i) {
        cin >> edges[i][0] >> edges[i][1];
    }
    cout << max_dag(n, edges) << endl;
    return 0;
}

```





## 26572: 多余的括号

http://cs101.openjudge.cn/practice/26572/

小明总是记不清四则运算的优先级关系，为了保险起见，他总是在算式中加上许多冗余的括号，但层层嵌套的括号可苦了批改作业的老师。现在想请你编写一个程序，在不改变算式运算顺序的前提下，删除其中多余的括号。

为了简单起见，我们只考虑加法和乘法两种运算，其中乘法优先级高于加法。题目保证给出的算式是合法的，且所有出现的运算数都是非负整数，不含正负号。

输入是若干行只含有非负整数、加号、乘号和括号的四则运算表达式。对于每一行输入，你的程序需要输出一行结果，即删去表达式中所有冗余括号后得到的简化表达式。

**输入**

(1+11)
((1+2))+3*(4+5)

**输出**

1+11
1+2+3*(4+5)

样例输入

```
(1+11)
((1+2))+3*(4+5)
1+(2+3)
```

样例输出

```
1+11
1+2+3*(4+5)
1+(2+3)
```

来源: lxp



2025/6/4 修正过测试数据。正确方法应该是中缀转后缀，后缀再转中缀表达式（同时满足题面要求：不改变算式运算顺序）。

基于 中缀表达式转后缀表达式（也叫逆波兰表示法，Reverse Polish Notation，RPN）的思想，并在后缀表达式的基础上再转回中缀表达式，同时去除冗余的括号。下面代码也可以 AC。

```python
import re

def infix_to_postfix(tokens):
    """
    中缀（tokens）→ 后缀：
    - 使用栈存运算符，根据优先级（+：1，*：2）和左结合性来出栈与入栈。
    - 数字直接输出到 postfixList，遇到 '(' 入栈，遇到 ')' 则依次弹出运算符直到遇到 '('。
    """
    op_stack = []
    postfix = []
    prec = {"+": 1, "*": 2}

    for tk in tokens:
        if tk.isdigit():
            # 数字（包括多位）直接放到后缀列表
            postfix.append(tk)
        elif tk == "(":
            op_stack.append(tk)
        elif tk == ")":
            # 右括号时，把栈顶运算符都弹出直到遇到左括号
            while op_stack and op_stack[-1] != "(":
                postfix.append(op_stack.pop())
            op_stack.pop()  # 弹出 '('
        else:
            # 运算符：比较优先级，优先级低（或相等）的运算符先出栈
            while (op_stack and op_stack[-1] != "("
                   and prec[tk] <= prec[op_stack[-1]]):
                postfix.append(op_stack.pop())
            op_stack.append(tk)

    # 最后把栈里剩余的运算符都弹出
    while op_stack:
        postfix.append(op_stack.pop())

    return postfix

def postfix_to_infix(postfix):
    """
    后缀 → 中缀（去冗余括号）：
    - 每次遇到数字就把 (字符串, “自身优先级”) 入栈。这里设计：
        数字的“自身优先级”设为 3，
        '+' 的优先级设为 1，
        '*' 的优先级设为 2。
    - 每次遇到 '+' 或 '*'：
        从栈顶弹出右操作数 (右子表达式, right_prec)，
                   再弹出左操作数 (左子表达式, left_prec)。
        设当前运算符 op 的优先级为 prec_op（'+'→1，'*'→2）。
        
        1) 左子表达式需要加括号的条件是：left_prec < prec_op  
           （也就是子表达式的根运算符优先级更低，一定要加括号）。
        2) 右子表达式需要加括号的条件是：
           - 子表达式的根运算符优先级更低： right_prec < prec_op  
             （相当于 “1 * (2+3)” 这种情况，子表达式是加法、优先级 < 乘法，必须加括号），
           - 或者 子表达式的根运算符优先级等于当前运算符且 “当前运算符是左结合”：
             right_prec == prec_op  
             （例如 “a + (b + c)” 里，右边如果也是 ‘+’，没有括号会变成 “a + b + c” 
              （即 (a+b)+c），这改变了原先“先算 b+c 再加 a”的顺序，必须保留）。  
              
        这样就能保证：在重建中缀字符串时，只加必要的括号，不会破坏原始运算顺序。
    """
    stack = []
    for tk in postfix:
        if tk.isdigit():
            # 数字本身优先级设为 3
            stack.append((tk, 3))
        else:
            # tk 是 '+' 或 '*'
            right_str, right_prec = stack.pop()
            left_str, left_prec = stack.pop()
            prec_op = 2 if tk == "*" else 1

            # 左子表达式加括号的条件：left_prec < prec_op
            if left_prec < prec_op:
                left_str = f"({left_str})"
            # 右子表达式加括号的条件：
            #   right_prec < prec_op    或
            #   right_prec == prec_op  （因为 + 和 * 均为左结合，
            #                           同级右侧子表达式必须加括号）
            if right_prec < prec_op or right_prec == prec_op:
                right_str = f"({right_str})"

            merged = f"{left_str}{tk}{right_str}"
            stack.append((merged, prec_op))

    return stack.pop()[0]

def simplify_expression(expr):
    # 用正则把多位数字和运算符/括号拆开
    tokens = re.findall(r'\d+|[()+*]', expr)
    postfix = infix_to_postfix(tokens)
    return postfix_to_infix(postfix)

# 主流程：读每一行、输出简化后的结果
if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(simplify_expression(line))
```





```python
import sys

# 运算符优先级和结合性（左结合 = 'L'，右结合 = 'R'）
PRECEDENCE = {
    '+': (1, 'L'),
    '-': (1, 'L'),
    '*': (2, 'L'),
    '/': (2, 'L'),
    'NEG': (3, 'R'),  # 一元负号
}

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value  # 运算符或数值
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

def tokenize(expr):
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in '+-*/()':
            tokens.append(expr[i])
            i += 1
        elif expr[i].isdigit():
            num = ''
            while i < len(expr) and expr[i].isdigit():
                num += expr[i]
                i += 1
            tokens.append(num)
        else:
            raise ValueError(f"Invalid character: {expr[i]}")
    return tokens

def parse_expr(tokens):
    i = 0

    def parse_primary():
        nonlocal i
        if tokens[i] == '-':
            # 一元负号
            i += 1
            return Node('NEG', right=parse_primary())
        elif tokens[i] == '(':
            i += 1
            node = parse_add_sub()
            i += 1  # skip ')'
            return node
        else:
            val = tokens[i]
            i += 1
            return Node(val)

    def parse_unary():
        return parse_primary()

    def parse_mul_div():
        nonlocal i
        node = parse_unary()
        while i < len(tokens) and tokens[i] in ('*', '/'):
            op = tokens[i]
            i += 1
            node = Node(op, node, parse_unary())
        return node

    def parse_add_sub():
        nonlocal i
        node = parse_mul_div()
        while i < len(tokens) and tokens[i] in ('+', '-'):
            op = tokens[i]
            i += 1
            node = Node(op, node, parse_mul_div())
        return node

    return parse_add_sub()

def to_string(node, parent_op=None, is_right=False):
    if node.is_leaf():
        return node.value

    if node.value == 'NEG':
        expr = to_string(node.right, 'NEG', True)
        if not node.right.is_leaf() and PRECEDENCE.get(node.right.value, (0,))[0] < PRECEDENCE['NEG'][0]:
            expr = f'({expr})'
        return f'-{expr}'

    left_expr = to_string(node.left, node.value, False) if node.left else ''
    right_expr = to_string(node.right, node.value, True)

    def need_paren(child, parent, is_right_child):
        if child.is_leaf():
            return False
        child_prec, _ = PRECEDENCE.get(child.value, (0, 'L'))
        parent_prec, parent_assoc = PRECEDENCE.get(parent, (0, 'L'))

        if child_prec < parent_prec:
            return True
        if child_prec == parent_prec:
            if parent_assoc == 'L' and is_right_child:
                return True
            if parent_assoc == 'R' and not is_right_child:
                return True
        return False

    if need_paren(node.left, node.value, False):
        left_expr = f'({left_expr})'
    if need_paren(node.right, node.value, True):
        right_expr = f'({right_expr})'

    return f'{left_expr}{node.value}{right_expr}'

def simplify_expression(expr):
    tokens = tokenize(expr)
    ast = parse_expr(tokens)
    return to_string(ast)

if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(simplify_expression(line))
```











## 26573: 康托集的图像表示

http://cs101.openjudge.cn/practice/26573/

在数学上具有重要意义的康托集(cantor set)是用如下方法构造的。考虑区间[0,1]，我们第一步要做的是，将区间三等分，然后删去中间的部分(1/3, 2/3)。在后面的每一步中，取出所有剩下的小区间，将每一个小区间都三等分后删去中间的部分，这样操作无穷次，最后剩下的点即为康托集。

在本题中，对于输入n，我们假设**操作n步之后剩下的每个小区间为一个单位长度**。请你用线段图表示出这些剩下的小区间。每个小区间使用一个字符‘*’表示，而[0,1]区间的其余位置按照其单位长度用相应个数的‘-’表示。

例如：对于输入n=3，最后剩下的小区间为1个单位长度，则整个[0,1]区间的单位长度为27，各步删去的区间如下表示：

第一步删去中间一段（一个区间，9个单位长度）：

`---------*********---------`

第二步删去左右区间的中间一段（两个区间，分别有3个单位长度）：

`---***---------------***---`

最后一步删去的小区间是（四个区间，分别有1个单位长度）：

`-*-----*-----------*-----*-`

剩下的没有删去的小区间是（8个区间，分别有一个单位长度）：

`*-*---*-*---------*-*---*-*`

注意：你的程序需要输出的只是上面示例中的最后一行，即操作n步之后剩余的小区间

**输入**

3

**输出**

`*-*---*-*---------*-*---*-*`

样例输入

```
3
```

样例输出

```
*-*---*-*---------*-*---*-*
```

来源：lxp



```python
def print_cantor_set(n):
    def cantor(start, end, level):
        if level == 0:
            for i in range(start, end):
                cantor_set[i] = '*'  # Mark the segment as occupied
        else:
            segment_length = (end - start) // 3
            # Recursively mark the first third and the last third
            cantor(start, start + segment_length, level - 1)
            cantor(end - segment_length, end, level - 1)

    # Initialize the list with dashes, representing an empty line
    cantor_set = ['-' for _ in range(3 ** n)]
    cantor(0, 3 ** n, n)
    return ''.join(cantor_set)

# Read the input
n = int(input())

# Generate and print the Cantor set
print(print_cantor_set(n))
```



## T27018: 康托展开

http://cs101.openjudge.cn/practice/27018/

总时间限制: 3000ms 单个测试点时间限制: 2000ms 内存限制: 90112kB
描述
求 1∼N 的一个给定全排列在所有 1∼N 全排列中的排名。结果对 998244353取模。

**输入**
第一行一个正整数 N。

第二行 N 个正整数，表示 1∼N 的一种全排列。
**输出**
一行一个非负整数，表示答案对 998244353 取模的值。
样例输入

```
Sample1 in:
3
2 1 3

Sample1 output:
3
```

样例输出

```
Sample2 in:
4
1 2 4 3

Sample2 output:
2
```

提示: 对于100%数据，$1≤N≤1000000$。
来源: https://www.luogu.com.cn/problem/P5367



思路：容易想到的方法是把所有排列求出来后再进行排序，但事实上有更简单高效的算法来解决这个问题，那就是康托展开。

> **康托展开**是一个全排列到一个自然数的双射，常用于构建特定哈希表时的空间压缩。 康托展开的实质是计算当前排列在所有由小到大全排列中的次序编号，因此是可逆的。即由全排列可得到其次序编号（康托展开），由次序编号可以得到对应的第几个全排列（逆康托展开）。
>
> 康托展开的**表达式为**：
>
> $X＝a_n×(n-1)!＋a_{n-1}×(n-2)!＋…＋a_i×(i-1)!＋…＋a_2×1!＋a_1×0!$
>
> 其中：X 为比当前排列小的全排列个数（X+1即为当前排列的次序编号）；n 表示全排列表达式的字符串长度；$a_i$ 表示原排列表达式中的第 i 位（由右往左数），前面（其右侧） i-1 位数有多少个数的值比它小。

例如求 5 2 3 4 1 在 {1, 2, 3, 4, 5} 生成的排列中的次序可以按如下步骤计算。
从右往左数，i 是5时候，其右侧比5小的数有1、2、3、4这4个数，所以有4×4！。
是2，比2小的数有1一个数，所以有 1×3！。
是3，比3小的数有1一个数，为1×2！。
是4，比4小的数有1一个数，为1×1！。
最后一位数右侧没有比它小的数，为 0×0！＝0。
则 4×4！＋1×3！＋1×2！＋1×1！＝105。
这个 X 只是这个排列之前的排列数，而题目要求这个排列的位置，即 5 2 3 4 1排在第 106 位。

同理，4 3 5 2 1的排列数：3×4!＋2×3!＋2×2!＋1×1!＝89，即 4 3 5 2 1 排在第90位。
因为比4小的数有3个：3、2、1；比3小的数有2个：2、1；比5小的数有2个：2、1；比2小的数有1个：1。

参考代码如下。



```python
MOD = 998244353								# Time Limit Exceeded, 内存7140KB, 时间18924ms
fac = [1]

def cantor_expand(a, n):
    ans = 0
    
    for i in range(1, n + 1):
        count = 0
        for j in range(i + 1, n + 1):
            if a[j] < a[i]:
                count += 1				# 计算有几个比他小的数
        ans = (ans + (count * fac[n - i]) % MOD) % MOD
    return ans + 1

a = [0]
N = int(input())		# 用大写N，因为spyder的debug，执行下一条指令的命令是 n/next。与变量n冲突。

for i in range(1, N + 1):
    fac.append((fac[i - 1] * i) % MOD)		# 整数除法具有分配律

*perm, = map(int, input().split())
a.extend(perm)

print(cantor_expand(a, N))
```



用C++也是超时

```c++
#include<iostream>							// Time Limit Exceeded, 内存960KB, 时间1986ms
using namespace std;

const long long MOD = 998244353;
long long fac[1000005]={1};

int cantor_expand (int a[],int n){
    int i, j, count;
    long long ans = 0 ;

    for(i = 1; i <= n; i ++){
        count = 0;
        for(j = i + 1; j <= n; j ++){
            if(a[j] < a[i]) count ++;						// 计算有几个比它小的数
        }
        ans = (ans + (count * fac[n-i]) % MOD ) % MOD;
    }
    return ans + 1;
}


int a[1000005];

int main()
{
  int N;
  //cin >> N;
  scanf("%d", &N);
  for (int i=1; i<=N; i++){
      fac[i] = (fac[i-1]*i)%MOD;
  }

  for (int i=1; i<=N; i++)
      //cin >> a[i];
      scanf("%d",&a[i]);
  cout << cantor_expand(a,N) << endl;
  return 0;
}
```



### 优化

康托展开用 $O(n^2)$ 算法超时，需要把时间复杂度降到$O(nLogn)$。“计算有几个比他小的数”，时间复杂度由 $O(n)$ 降到 $O(Logn)$。

### 树状数组（Binary Indexed Tree）

实现树状数组的核心部分，包括了三个重要的操作：lowbit、修改和求和。

1. lowbit函数：`lowbit(x)` 是用来计算 `x` 的二进制表示中最低位的 `1` 所对应的值。它的运算规则是利用位运算 `(x & -x)` 来获取 `x` 的最低位 `1` 所对应的值。例如，`lowbit(6)` 的结果是 `2`，因为 `6` 的二进制表示为 `110`，最低位的 `1` 所对应的值是 `2`。

   > `-x` 是 `x` 的补码表示。
   >
   > 对于正整数 `x`，`-x` 的二进制表示是 `x` 的二进制表示取反后加 1。
   >
   > `6` 的二进制表示为 `110`，取反得到 `001`，加 1 得到 `010`。
   >
   > `-6` 的二进制表示为 `11111111111111111111111111111010`（假设 32 位整数）。
   >
   > `6 & -6` 的结果：
   >
   > `110` 与 `11111111111111111111111111111010` 按位与运算，结果为 `010`，即 `2`。

2. update函数：这个函数用于修改树状数组中某个位置的值。参数 `x` 表示要修改的位置，参数 `y` 表示要增加/减少的值。函数使用一个循环将 `x` 的所有对应位置上的值都加上 `y`。具体的操作是首先将 `x` 位置上的值与 `y` 相加，然后通过 `lowbit` 函数找到 `x` 的下一个需要修改的位置，将该位置上的值也加上 `y`，然后继续找下一个位置，直到修改完所有需要修改的位置为止。这样就完成了数组的修改。

3. getsum函数：这个函数用于求解树状数组中某个范围的前缀和。参数 `x` 表示要求解前缀和的位置。函数使用一个循环将 `x` 的所有对应位置上的值累加起来，然后通过 `lowbit` 函数找到 `x` 的上一个位置（即最后一个需要累加的位置），再将该位置上的值累加起来，然后继续找上一个位置，直到累加完所有需要累加的位置为止。这样就得到了从位置 `1` 到位置 `x` 的前缀和。

这就是树状数组的核心操作，通过使用这三个函数，我们可以实现树状数组的各种功能，如求解区间和、单点修改等。

```python
n, MOD, ans = int(input()), 998244353, 1						# 内存69832KB, 时间2847ms
a, fac = list(map(int, input().split())), [1]

tree = [0] * (n + 1)

def lowbit(x):
    return x & -x

def update(x, y):
    while x <= n:
        tree[x] += y
        x += lowbit(x)

def getsum(x):
    tot = 0
    while x:
        tot += tree[x]
        x -= lowbit(x)
    return tot


for i in range(1, n):
    fac.append(fac[i-1] * i % MOD)

for i in range(1, n + 1):
    cnt = getsum(a[i-1])
    update(a[i-1], 1)
    ans = (ans + ((a[i-1] - 1 - cnt) * fac[n - i]) % MOD) % MOD
    
print(ans)
```





### 线段树（Segment tree）

线段树 segment tree 来计算第i位右边比该数还要小的数的个数。

```python
n, MOD, ans = int(input()), 998244353, 1					# 内存69900KB, 时间5162ms
a, fac = list(map(int, input().split())), [1]

tree = [0] * (2*n)


def build(arr):

    # insert leaf nodes in tree
    for i in range(n):
        tree[n + i] = arr[i]

    # build the tree by calculating parents
    for i in range(n - 1, 0, -1):
        tree[i] = tree[i << 1] + tree[i << 1 | 1]


# function to update a tree node
def updateTreeNode(p, value):

    # set value at position p
    tree[p + n] = value
    p = p + n

    # move upward and update parents
    i = p
    while i > 1:

        tree[i >> 1] = tree[i] + tree[i ^ 1]
        i >>= 1


# function to get sum on interval [l, r)
def query(l, r):

    res = 0

    l += n
    r += n

    while l < r:

        if (l & 1):
            res += tree[l]
            l += 1

        if (r & 1):
            r -= 1
            res += tree[r]

        l >>= 1
        r >>= 1

    return res


#build([0]*n)

for i in range(1, n):
    fac.append(fac[i-1] * i % MOD)

for i in range(1, n + 1):
    cnt = query(0, a[i-1])
    updateTreeNode(a[i-1]-1, 1)
    
    ans = (ans + (a[i-1] -1 - cnt) * fac[n - i]) % MOD
    
print(ans)

```





## 27205: 护林员盖房子 加强版

monotonous stack, http://cs101.openjudge.cn/practice/27205

在一片保护林中，护林员想要盖一座房子来居住，但他不能砍伐任何树木。
现在请你帮他计算：保护林中所能用来盖房子的矩形空地的最大面积。

**输入**
保护林用一个二维矩阵来表示，长宽都不超过20（即<=20）。
第一行是两个正整数m,n，表示矩阵有m行n列。
然后是m行，每行n个整数，用1代表树木，用0表示空地。

**输出**
一个正整数，表示保护林中能用来盖房子的最大矩形空地面积。

样例输入

```
4 5
0 1 0 1 1
0 1 0 0 1
0 0 0 0 0
0 1 1 0 1
```

样例输出

```
5
```

提示

子矩阵边长可以为1，也就是说：
0 0 0 0 0
依然是一个可以盖房子的子矩阵。



看成是以不同的底竖直摆放的矩形的高度。

LeetCode 85 | 如何从矩阵当中找到数字围成的最大矩形的面积？https://zhuanlan.zhihu.com/p/162834671

```python
def maximalRectangle(matrix) -> int:
    if (rows := len(matrix)) == 0:
        return 0

    cols = len(matrix[0])
    # 存储每一层的高度
    height = [0 for _ in range(cols + 1)]
    res = 0

    for i in range(rows):  # 遍历以哪一层作为底层
        stack = [-1]
        for j in range(cols + 1):
            # 计算j位置的高度，如果遇到1则置为0，否则递增
            h = 0 if j == cols or matrix[i][j] == '1' else height[j] + 1
            height[j] = h
            # 单调栈维护长度
            while len(stack) > 1 and h < height[stack[-1]]:
                res = max(res, (j - stack[-2] - 1) * height[stack[-1]])
                stack.pop()
            stack.append(j)
    return res


rows, _ = map(int, input().split())
a = [input().split() for _ in range(rows)]

print(maximalRectangle(a))


```



## T27256: 当前队列中位数

data structures, OOP, http://cs101.openjudge.cn/practice/27256/

中位数是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

现在，假设我有一个空的list，我会对他进行以下三种操作：

1. 在list最后添加一个数：add x
2. 删除当前list的第一个数：del
3. 查询当前list的中位数：query

**输入**

输入为若干行，第一行为一个整数n（n <= 100000)，表示操作的次数，接下来n行表示n次操作。数据保证在删除或查询的时候，队列长度大于0

**输出**

针对于每次query，输出当前的中位数

样例输入

```
sample1 in:
5
add 1
add 2
query
del
query

sample1 out:
1.5
2
```

样例输出

```
sample2 input:
5
add 1
query
add 3
query
query

sample2 out:
1
2
2
```

提示

tags: sorting, data structure。注意输出格式如样例所示。

来源

2023fall zzr



给出一种基于双堆（大根堆和小根堆）及延迟删除的做法，既能在队列操作（FIFO）下维护数据，又能高效查询中位数。

```python
import sys
import heapq
from collections import deque, defaultdict

input = sys.stdin.readline

class DualHeap:
    def __init__(self):
        # 大根堆（存放较小的一半，用负数模拟）
        self.small = []  
        # 小根堆（存放较大的一半）
        self.large = []  
        # 延迟删除的记录
        self.delayed = defaultdict(int)
        # 两个堆中有效数据的个数
        self.small_size = 0  
        self.large_size = 0  

    def prune(self, heap):
        # 清理堆顶已经被延迟删除的元素，不再调整size（因为在remove中已扣减）
        if heap is self.small:
            while heap and self.delayed[-heap[0]] > 0:
                num = -heapq.heappop(heap)
                self.delayed[num] -= 1
        else:
            while heap and self.delayed[heap[0]] > 0:
                num = heapq.heappop(heap)
                self.delayed[num] -= 1

    def balance(self):
        # 保持 small 比 large 多 0 或 1 个有效元素
        if self.small_size > self.large_size + 1:
            self.prune(self.small)
            num = -heapq.heappop(self.small)
            self.small_size -= 1
            heapq.heappush(self.large, num)
            self.large_size += 1
        elif self.small_size < self.large_size:
            self.prune(self.large)
            num = heapq.heappop(self.large)
            self.large_size -= 1
            heapq.heappush(self.small, -num)
            self.small_size += 1

    def add(self, num):
        # 插入新数时，根据大小决定进入哪个堆
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.small_size += 1
        else:
            heapq.heappush(self.large, num)
            self.large_size += 1
        self.balance()

    def remove(self, num):
        # 延迟删除：标记待删除，同时减少对应堆的有效元素数量
        self.delayed[num] += 1
        if self.small and num <= -self.small[0]:
            self.small_size -= 1
            if num == -self.small[0]:
                self.prune(self.small)
        else:
            self.large_size -= 1
            if self.large and num == self.large[0]:
                self.prune(self.large)
        self.balance()

    def median(self):
        # 查询中位数前先清理堆顶
        self.prune(self.small)
        self.prune(self.large)
        total = self.small_size + self.large_size
        if total % 2 == 1:
            return -self.small[0]
        else:
            return (-self.small[0] + self.large[0]) / 2

if __name__ == '__main__':
    n = int(input())
    dh = DualHeap()
    # 使用 deque 记录入队顺序，确保del操作删除最先添加的数
    q = deque()
    results = []
    for _ in range(n):
        parts = input().split()
        op = parts[0]
        if op == 'add':
            x = int(parts[1])
            dh.add(x)
            q.append(x)
        elif op == 'del':
            x = q.popleft()
            dh.remove(x)
        elif op == 'query':
            med = dh.median()
            # 若中位数为整数则去除小数部分
            if med == int(med):
                results.append(str(int(med)))
            else:
                results.append(str(med))
    print("\n".join(results))

```

**代码说明**

1. **DualHeap 类：**
   - 使用两个堆：`small`（大根堆，用负数存储）和 `large`（小根堆）。
   - 通过延迟删除字典 `delayed` 记录待删除的元素，避免在堆中直接查找删除，节省时间。
   - `small_size` 和 `large_size` 分别记录堆中实际有效的元素数量。
2. **操作函数：**
   - **add(num):** 根据数值大小将 `num` 插入到对应的堆中，然后调用 `balance` 进行平衡。
   - **remove(num):** 将 `num`标记为延迟删除，同时减少相应堆的计数，再进行堆顶清理和平衡。
   - **median():** 先清理堆顶，再根据当前元素个数奇偶返回中位数（若为偶数则返回两个堆顶的平均值）。
3. **主函数：**
   - 通过 `deque` 保存添加的顺序，保证“del”操作始终删除最先添加的数。
   - 每次执行 `query` 时调用 `median()` 并按照要求输出。



```python
import collections
import heapq


class DualHeap:
    def __init__(self):
        # 大根堆，维护较小的一半元素，注意 python 没有大根堆，需要将所有元素取相反数并使用小根堆
        self.small = list()
        # 小根堆，维护较大的一半元素
        self.large = list()
        # 哈希表，记录「延迟删除」的元素，key 为元素，value 为需要删除的次数
        self.delayed = collections.Counter()

        # small 和 large 当前包含的元素个数，需要扣除被「延迟删除」的元素
        self.smallSize = 0
        self.largeSize = 0

    # 不断地弹出 heap 的堆顶元素，并且更新哈希表
    def prune(self, heap):
        while heap:
            num = heap[0]
            if heap is self.small:
                num = -num
            if num in self.delayed:
                self.delayed[num] -= 1
                if self.delayed[num] == 0:
                    self.delayed.pop(num)
                heapq.heappop(heap)
            else:
                break

    # 调整 small 和 large 中的元素个数，使得二者的元素个数满足要求
    def make_balance(self):
        if self.smallSize > self.largeSize + 1:
            # small 比 large 元素多 2 个
            heapq.heappush(self.large, -self.small[0])
            heapq.heappop(self.small)
            self.smallSize -= 1
            self.largeSize += 1
            # small 堆顶元素被移除，需要进行 prune
            self.prune(self.small)
        elif self.smallSize < self.largeSize:
            # large 比 small 元素多 1 个
            heapq.heappush(self.small, -self.large[0])
            heapq.heappop(self.large)
            self.smallSize += 1
            self.largeSize -= 1
            # large 堆顶元素被移除，需要进行 prune
            self.prune(self.large)

    def insert(self, num):
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.smallSize += 1
        else:
            heapq.heappush(self.large, num)
            self.largeSize += 1
        self.make_balance()

    def erase(self, num):
        self.delayed[num] += 1
        if num <= -self.small[0]:
            self.smallSize -= 1
            if num == -self.small[0]:
                self.prune(self.small)
        else:
            self.largeSize -= 1
            if num == self.large[0]:
                self.prune(self.large)
        self.make_balance()

    def get_median(self):
        return -self.small[0] if self.smallSize != self.largeSize else (-self.small[0] + self.large[0]) / 2


n = int(input())
q = DualHeap()
l = []
start_idx = 0
for _ in range(n):
    operation = input()
    if operation == 'query':
        ans = q.get_median()
        if round(ans) == ans:
            print(int(ans))
        else:
            print(ans)
    elif operation == 'del':
        q.erase(l[start_idx])
        start_idx += 1
    else:
        t = int(operation.split()[1])
        q.insert(t)
        l.append(t)
```



```python
# 2300011742 张展皓 化院
from bisect import bisect_left
a, b, cnt, now = [], [], 0, 0
for _ in range(int(input())):
    opt = input().split()
    if opt[0] == 'query':
        l = len(a)
        if l & 1: print(a[l >> 1][0])
        else:
            ans = (a[l >> 1][0] + a[l - 1 >> 1][0]) / 2 
            print(ans if int(ans) != ans else int(ans))
    if opt[0] == 'add':
        v = int(opt[1])
        a.insert(bisect_left(a, [v, 0]), [v, cnt])
        b.append(v)
        cnt += 1
    if opt[0] == 'del': 
        v = b[now]
        now += 1
        a.pop(bisect_left(a, [v, 0]))
```



沈俊丞25。AC后，用AI修了一下。

```python
from heapq import heappop, heappush
from collections import deque
import sys
input = sys.stdin.readline

def lazy_delete(heap, del_num):
    # 清理堆顶那些逻辑上已被删除（index < del_num）的元素
    while heap and heap[0][1] < del_num:
        heappop(heap)

n = int(input().strip())
arr = deque()        # 存 (value, idx)
heap_low = []        # 小根堆：存“较大的一半”（直接存 value, idx）
heap_high = []       # 用负值表示的大根堆：存“较小的一半”（存 -value, idx）

del_num = 0          # 已删除计数阈值（所有 index < del_num 的元素视为已删除）
addi = 0             # 插入编号，用作唯一索引

# 记录每个 idx 当前属于哪一侧（'low' 或 'high'）
pos = {}

# 有效计数（不含堆中尚未物理弹出的过期项）
size_low = 0
size_high = 0

for _ in range(n):
    order = input().strip()
    if not order:
        continue

    # 每次操作前先清理堆顶过期项（保证堆顶可用）
    lazy_delete(heap_low, del_num)
    lazy_delete(heap_high, del_num)

    if order[0] == 'a':  # add x
        x = int(order.split()[1])
        arr.append((x, addi))

        # 决定放哪一堆：保持 heap_low 存较大一半（min-heap）， heap_high 存较小一半（max via -v）
        if not heap_low or x >= heap_low[0][0]:
            heappush(heap_low, [x, addi])
            pos[addi] = 'low'
            size_low += 1
        else:
            heappush(heap_high, [-x, addi])
            pos[addi] = 'high'
            size_high += 1

        # 根据**有效计数**平衡两堆：允许 heap_low 比 heap_high 多 1
        if size_low > size_high + 1:
            # 从 low 移动到 high
            lazy_delete(heap_low, del_num)
            v, i = heappop(heap_low)
            heappush(heap_high, [-v, i])
            pos[i] = 'high'
            size_low -= 1
            size_high += 1
        elif size_high > size_low:
            # 从 high 移动到 low
            lazy_delete(heap_high, del_num)
            vneg, i = heappop(heap_high)
            v = -vneg
            heappush(heap_low, [v, i])
            pos[i] = 'low'
            size_high -= 1
            size_low += 1

        addi += 1

    elif order[0] == 'd':  # del
        # 题目保证删除时队列非空
        val, idx_del = arr.popleft()
        # 标记逻辑删除的阈值
        del_num += 1

        # 减少对应堆的有效计数（我们在插入时用 pos 记录了它属哪堆）
        side = pos.get(idx_del)
        if side == 'low':
            size_low -= 1
        elif side == 'high':
            size_high -= 1
        # 删除 pos 记录（此元素已逻辑删除）
        if idx_del in pos:
            del pos[idx_del]

        # 根据有效计数重新平衡（同上）
        if size_low > size_high + 1:
            lazy_delete(heap_low, del_num)
            v, i = heappop(heap_low)
            heappush(heap_high, [-v, i])
            pos[i] = 'high'
            size_low -= 1
            size_high += 1
        elif size_high > size_low:
            lazy_delete(heap_high, del_num)
            vneg, i = heappop(heap_high)
            v = -vneg
            heappush(heap_low, [v, i])
            pos[i] = 'low'
            size_high -= 1
            size_low += 1

    else:  # query
        # 如果 low 比 high 多一个，则中位数是 low 的堆顶（最小的右侧元素）
        if size_low > size_high:
            # low 存原值
            print(heap_low[0][0])
        else:
            a = heap_low[0][0]
            b = -heap_high[0][0]
            s = a + b
            if s % 2 == 0:
                print(s // 2)
            else:
                # 输出 1.5 之类的一位小数（避免 2.0）
                print(f"{s / 2:.1f}")

```





## 27625: AVL树至少有几个结点

http://cs101.openjudge.cn/practice/27625/

输入n(0<n<50),输出一个n层的AVL树至少有多少个结点。

**输入**

n

**输出**

答案

样例输入

```
4
```

样例输出

```
7
```

来源

http://dsbpython.openjudge.cn/dspythonbook/P1350/



```python
from functools import lru_cache

@lru_cache(maxsize=None)
def avl_min_nodes(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return avl_min_nodes(n-1) + avl_min_nodes(n-2) + 1

n = int(input())
min_nodes = avl_min_nodes(n)
print(min_nodes)
```





## 27626: AVL树最多有几层

http://cs101.openjudge.cn/practice/27626/

n个结点的AVL树最多有多少层？

**输入**

整数n 。 0< n < 50,000,000

**输出**

AVL树最多有多少层

样例输入

```
20
```

样例输出

```
6
```

来源

http://dsbpython.openjudge.cn/dspythonbook/P1360/



```python
from functools import lru_cache

@lru_cache(maxsize=None)
def min_nodes(h):
    if h == 0: return 0
    if h == 1: return 1
    return min_nodes(h-1) + min_nodes(h-2) + 1

def max_height(n):
    h = 0
    while min_nodes(h) <= n:
        h += 1
    return h - 1

n = int(input())
print(max_height(n))
```





## 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/

例题：给定一个无向图，判断是否连通，是否有回路。

**输入**

第一行两个整数n,m，分别表示顶点数和边数。顶点编号从0到n-1。 (1<=n<=110, 1<=m <= 10000)
接下来m行，每行两个整数u和v，表示顶点u和v之间有边。

**输出**

如果图是连通的，则在第一行输出“connected:yes",否则第一行输出“connected:no"。
如果图中有回路，则在第二行输出“loop:yes ",否则第二行输出“loop:no"。

样例输入

```
3 2
0 1
0 2
```

样例输出

```
connected:yes
loop:no
```

来源

http://dsbpython.openjudge.cn/dspythonbook/P1040/



dfs能直接判断是否连通，过程中记录父亲节点，只要当前节点能够去到一个已经遍历过的节点，并且这个节点不是父亲节点，那么必然成环，以及连通和成环是可以同时判断的

```python
#王昊 光华管理学院
n, m = list(map(int, input().split()))
edge = [[]for _ in range(n)]
for _ in range(m):
    a, b = list(map(int, input().split()))
    edge[a].append(b)
    edge[b].append(a)
cnt, flag = set(), False


def dfs(x, y):
    global cnt, flag
    cnt.add(x)
    for i in edge[x]:
        if i not in cnt:
            dfs(i, x)
        elif y != i:
            flag = True


for i in range(n):
    cnt.clear()
    dfs(i, -1)
    if len(cnt) == n:
        break
    if flag:
        break

print("connected:"+("yes" if len(cnt) == n else "no"))
print("loop:"+("yes" if flag else 'no'))
```



```python
def is_connected(graph, n):
    visited = [False] * n  # 记录节点是否被访问过
    stack = [0]  # 使用栈来进行DFS
    visited[0] = True

    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                stack.append(neighbor)
                visited[neighbor] = True

    return all(visited)

def has_cycle(graph, n):
    def dfs(node, visited, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, visited, node):
                    return True
            elif parent != neighbor:
                return True
        return False

    visited = [False] * n
    for node in range(n):
        if not visited[node]:
            if dfs(node, visited, -1):
                return True
    return False

# 读取输入
n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

# 判断连通性和回路
connected = is_connected(graph, n)
has_loop = has_cycle(graph, n)
print("connected:yes" if connected else "connected:no")
print("loop:yes" if has_loop else "loop:no")
```



这题一个dfs就够了

判断连通就是单纯dfs，每个节点拓展出没去过的节点递归，cnt这个集合记录去过的节点

判断成环可以同步进行，只是我们的dfs需要而外记录一下父亲节点，于是只要当前节点能够去到一个已经遍历过的节点，并且这个节点不是父亲节点，那么必然成环——这是因为我们的dfs是单源的遍历，不妨说那个已经遍历的非父亲节点是x，这次dfs的源头是root，当前节点是cur，那么x一定存在一条不经过cur到达root的路径（因为不会遍历去过的节点，所以cur是第一次到，之前遍历到x一定是没到过cur的），而现在cur能到x，说明有一条经过cur到达root的路径，而且x不是父亲节点，于是这两条路径通过root形成了一个环，并且由于不是父亲节点，所以这个环上的节点数>2，于是就必然是环了——这个想法说实话临时想是有点绕的。

还有一个性质，一旦连通，必然成环也判断完毕了，一旦成环，也意味着连通必然判断完毕了，所以两个break都是正确的。

```python
# 熊江凯、元培学院
n,m=list(map(int,input().split()))
edge=[[]for _ in range(n)]
for _ in range(m):
    a,b=list(map(int,input().split()))
    edge[a].append(b)
    edge[b].append(a)
cnt,ok=set(),0
def dfs(x,y):
    global cnt,ok
    cnt.add(x)
    for i in edge[x]:
        if i not in cnt:dfs(i,x)
        elif y!=i:ok=1
for i in range(n):
    cnt.clear()
    dfs(i,-1)
    if len(cnt)==n:break
    if ok:break
print("connected:"+("yes"if len(cnt)==n else "no")+'\n'+"loop:"+('yes'if ok else 'no'))
```



```python
# 2100017777 李鹏辉
# pylint: skip-file

n, m = map(int, input().split())
p = [i for i in range(n)]

def find(i):
    if p[i] != i:
        p[i] = find(p[i])
    return p[i]

def union(x, y):
    p[find(x)] = find(y)

l = False
for _ in range(m):
    x, y = map(int, input().split())
    if find(x) == find(y):
        l = True
    else:
        union(x, y)
p = [find(i) for i in range(n)]
p = set(p)
print('connected:yes' if len(p) == 1 else 'connected:no')
print('loop:yes' if l else 'loop:no')
```



## 27637: 括号嵌套二叉树

dfs, stack, http://cs101.openjudge.cn/practice/27637/

可以用括号嵌套的方式来表示一棵二叉树。

方法如下：`*`表示空的二叉树。

如果一棵二叉树只有一个结点，则该树就用一个非`*`字符表示，代表其根结点。

如果一棵二叉左右子树都非空，则用`树根(左子树,右子树)`的形式表示。树根是一个非`*`字符，左右子树之间用逗号隔开，没有空格。左右子树都用括号嵌套法表示。

如果左子树非空而右子树为空，则用`树根(左子树,*)`形式表示；如果左子树为空而右子树非空，则用`树根(*,右子树)`形式表示。

给出一棵树的括号嵌套表示形式，请输出其前序遍历序列、中序遍历序列、后序遍历序列。例如，`A(B(*,C),D(E))`表示的二叉树如图所示

![img](http://media.openjudge.cn/images/upload/1636/1707558029.jpg)

**输入**

第一行是整数n表示有n棵二叉树(n<100) 接下来有n行，每行是1棵二叉树的括号嵌套表示形式

**输出**

对每棵二叉树，输出其前序遍历序列和中序遍历序列

样例输入

```
2
A
A(B(*,C),D(E,*))
```

样例输出

```
A
A
ABCDE
BCAED
```

来源

http://dsbpython.openjudge.cn/dspythonbook/P0680/



将输入的括号嵌套形式转换成二叉树，然后实现了前序和中序遍历。

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def parse_tree(s):
    """ 解析括号嵌套格式的二叉树 """
    if s == '*':  # 处理空树
        return None
    if '(' not in s:  # 只有单个根节点
        return TreeNode(s)

    root_value = s[0]  # 根节点值
    subtrees = s[2:-1]  # 去掉根节点和外层括号

    # 使用栈找到逗号位置
    stack = []
    comma_index = None
    for i, char in enumerate(subtrees):
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
        elif char == ',' and not stack:
            comma_index = i
            break

    left_subtree = subtrees[:comma_index] if comma_index is not None else subtrees
    right_subtree = subtrees[comma_index + 1:] if comma_index is not None else None

    root = TreeNode(root_value)
    root.left = parse_tree(left_subtree)  # 解析左子树
    root.right = parse_tree(right_subtree) if right_subtree else None  # 解析右子树
    return root


def preorder_traversal(root):
    """前序遍历：根 -> 左 -> 右"""
    return root.value + preorder_traversal(root.left) + preorder_traversal(root.right) if root else ""


def inorder_traversal(root):
    """中序遍历：左 -> 根 -> 右"""
    return inorder_traversal(root.left) + root.value + inorder_traversal(root.right) if root else ""


# 读取输入
n = int(input().strip())  
results = []

for _ in range(n):
    tree_string = input().strip().replace(" ", "")  # 去掉可能的空格
    tree = parse_tree(tree_string)
    results.append(preorder_traversal(tree))
    results.append(inorder_traversal(tree))

print("\n".join(results))  # 按格式输出

```



## 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/

给定一棵二叉树，求该二叉树的高度和叶子数目二叉树高度定义：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的结点数减1为树的高度。只有一个结点的二叉树，高度是0。

**输入**

第一行是一个整数n，表示二叉树的结点个数。二叉树结点编号从0到n-1，根结点n <= 100 接下来有n行，依次对应二叉树的编号为0,1,2....n-1的节点。 每行有两个整数，分别表示该节点的左儿子和右儿子的编号。如果第一个（第二个）数为-1则表示没有左（右）儿子

**输出**

在一行中输出2个整数，分别表示二叉树的高度和叶子结点个数

样例输入

```
3
-1 -1
0 2
-1 -1
```

样例输出

```
1 2
```

来源

http://dsbpython.openjudge.cn/dspythonbook/P0610/



由于输入无法分辨谁为根节点，所以写寻找根节点语句。

```python
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None

def tree_height(node):
    if node is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(node.left), tree_height(node.right)) + 1

def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

n = int(input())  # 读取节点数量
nodes = [TreeNode() for _ in range(n)]
has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index]
        has_parent[left_index] = True
    if right_index != -1:
        #print(right_index)
        nodes[i].right = nodes[right_index]
        has_parent[right_index] = True

# 寻找根节点，也就是没有父节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]

# 计算高度和叶子节点数
height = tree_height(root)
leaves = count_leaves(root)

print(f"{height} {leaves}")
```



## 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/

实现用户定义的类，一个常用的例子是构建实现抽象数据类型Fraction的类。我们已经看到，Python提供了很多数值类。但是在有些时候，需要创建“看上去很像”分数的数据对象。

像 3/5 这样的分数由两部分组成。斜线左侧的值称作分子，可以是任意整数。斜线右侧的值称作分母，可以是任意大于0的整数（负的分数带有负的分子）。尽管可以用浮点数来近似表示分数，但我们在此希望能精确表示分数的值。

Fraction对象的表现应与其他数值类型一样，针对分数进行加、减、乘、除等运算，也能够使用标准的斜线形式来显示分数，比如3/5。此外，所有的分数方法都应该返回结果的最简形式。这样一来，不论进行何种运算，最后的结果都是最简分数。

**出于练习考虑，只需要支持一种分数加运算。**

**输入**

空格分割的一行字符串。第一个分数的分子 第一个分数的分母 第二个分数的分子 第二个分数的分母

**输出**

输出相加的结果。使用标准的斜线形式来显示分数，并且要求是最简分数。

样例输入

```
1 4 1 2
```

样例输出

```
3/4
```

来源

https://runestone.academy/ns/books/published/pythonds/index.html



```python
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

class Fraction:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self.simplify()

    def simplify(self):
        common = gcd(self.numerator, self.denominator)
        self.numerator //= common
        self.denominator //= common

    def __add__(self, other):
        numerator = self.numerator * other.denominator + self.denominator * other.numerator
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

a, b, c, d = map(int, input().split())
x = Fraction(a, b)
y = Fraction(c, d)
print(x+y)
```





## 27862: 博弈树分析与最优策略确定:获得最大收益的抉择路径

http://cs101.openjudge.cn/practice/27862/

博弈论是当今的一个热门领域，如下图是一棵博弈树，从博弈者1开始2人轮流做出抉择，叶子结点下方是相应博弈路径的两人的收益，左边第一个叶子结点代表当博弈者1选择C，博弈者2选择E时博弈者1收获为2，博弈者2收获为1。

我们知道每个人做抉择时都希望获得此时的最大收益，所以可以从博弈树中逆推求出2人的策略。如下图中左边子树博弈者2选择E可以获得最大收益1，右边子树博弈者2选择H可以获得最大收益3，而博弈者1在C,D中选择的时候应该考虑到自己选C则博弈者2会选择E，自己选D博弈者2会选择H，所以他为了自己收益最大会选择C-->E可以获得2的收益。

<img src="http://media.openjudge.cn/images/upload/6442/1710986373.jpg" alt="img" style="zoom:67%;" />

**输入**

每组测试数据第一行给出节点数n，分别标为1--n，接下来n-1行给出n-1条边所连的2个节点。

下一行给出叶子节点数k，接下来k行每一行三个数a,b,c，分别代表叶子节点编号和博弈者1、博弈者2的收益。

**输出**

最后2人选择的博弈路径2人的收益各是多少。数据保证根节点为博弈者1的抉择，且根节点编号一定为1，每人每次的策略一定有2种选择，2种选择的收益不相等。

样例输入

```
7
1 2
1 3
2 4
2 5
3 6
3 7
4
4 2 1
5 3 0
6 0 2
7 1 3
```

样例输出

```
2 1
```

来源: HYS 结合通选课知识出了一道树题。



SPNE代表"Sequentially Perfect Nash Equilibrium"（顺序完美纳什均衡），是博弈论中的一个概念。
在博弈论中，纳什均衡是指在一个博弈中，每个参与者选择的策略是相互协调的，即在其他参与者选择其策略的情况下，没有参与者有动机单独改变自己的策略。纳什均衡是一种稳定的策略组合。
顺序完美纳什均衡是在考虑博弈的顺序和时间因素的基础上定义的。它要求在博弈的每个子阶段中，参与者的策略都是最优的，即在当前阶段的最佳反应。这意味着每个参与者都能够根据先前的动作和信息做出最佳决策，并且在整个博弈过程中没有后悔的动作。顺序完美纳什均衡是纳什均衡的一个更严格的概念。
顺序完美纳什均衡在博弈论中具有重要的理论和应用价值，特别是在分析顺序博弈、动态博弈和信息博弈等方面。它帮助人们理解和预测参与者在博弈中的决策行为，并为博弈策略的设计和分析提供了理论基础。

```python
# 23 元培 夏天明
class Node:
    def __init__(self):
        self.val = None
        self.child = []
        self.color = 0
    
    def getSPNE(self):
        if self.child:
            for nd in self.child:
                nd.getSPNE()
            # 选择子结点的SPNE中，自己颜色收益最大的收益数对，作为这个节点的收益数对
            # 根据color。代表该节点是几号博弈者进行抉择
            self.val = max([nd.val for nd in self.child], key=lambda x:x[self.color])


n = int(input())
nodes = [Node() for i in range(n+1)]
for o in range(n-1):
    a, b = map(int, input().split())
    nodes[a].child.append(nodes[b])
this = [nodes[1]]
col = 0
for o in range(int(input())):
    leaf, v1, v2 = map(int, input().split())
    nodes[leaf].val = (v1, v2)
while this:
    new = []
    for nd in this:
        nd.color = col
        new.extend(nd.child)
    this = new
    col = 1-col
nodes[1].getSPNE()
print(*nodes[1].val)
```





```python
# 黄源森23工院，从下往上
from collections import defaultdict
def f(x,layer):
    if x in leaf:
        return leaf[x]
    l=dic[x]
    for u in l:
        if u in vis:
            l.remove(u)
    vis.update(l)
    t=[]
    for u in l:
        t.append(f(u,layer+1))
        
    if layer%2==1:
        t.sort()
        return t[-1]
    else:
        t.sort(key=lambda x:x[1])
        return t[-1]
n=int(input())
vis=set([1])
dic=defaultdict(list)
for _ in range(n-1):
    a,b=map(int,input().split())
    dic[a].append(b)
    dic[b].append(a)
leaf={}
k=int(input())
for _ in range(k):
    a,b,c=map(int,input().split())
    leaf[a]=(b,c)
print(*f(1,1))
```



## 27880: 繁忙的厦门

MST, http://cs101.openjudge.cn/practice/27880/

城市 XM 是一个非常繁忙的大都市，城市中的道路十分的拥挤，于是市长决定对其中的道路进行改造。城市 XM 的道路是这样分布的：城市中有 *n* 个交叉路口，有些交叉路口之间有道路相连，两个交叉路口之间最多有一条道路相连接。这些道路是双向的，且把所有的交叉路口直接或间接的连接起来了。每条道路都有一个分值，分值越小表示这个道路越繁忙，越需要进行改造。但是市政府的资金有限，市长希望进行改造的道路越少越好，于是他提出下面的要求：

1. **改造的那些道路能够把所有的交叉路口直接或间接的连通起来。**
2. **在满足要求 1 的情况下，改造的道路尽量少。**
3. **在满足要求 1、2 的情况下，改造的那些道路中分值最大的道路分值尽量小。**

**任务：作为市规划局的你，应当作出最佳的决策，选择哪些道路应当被修建。**

输入

第一行有两个整数 n,m 表示城市有 n 个交叉路口，m 条道路。

接下来 m 行是对每条道路的描述，u,v,c 表示交叉路口 u 和 v 之间有道路相连，分值为 c。

输出

两个整数 s,max，表示你选出了几条道路，分值最大的那条道路的分值是多少。

样例输入

```
4 5
1 2 3
1 4 5
2 4 7
2 3 6
3 4 8
```

样例输出

```
3 6
```

提示

对于全部数据，满足 1≤ n ≤300，1≤ c ≤10^4，1≤ m ≤8000。



```python
import sys

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.rank[px] > self.rank[py]:
                self.parent[py] = px
            else:
                self.parent[px] = py
                if self.rank[px] == self.rank[py]:
                    self.rank[py] += 1

def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    mst, max_edge = 0, 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst += 1
            max_edge = max(max_edge, w)
            if mst == n - 1:
                break
    return mst, max_edge

def main():
    n, m = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(m):
        u, v, c = map(int, sys.stdin.readline().split())
        edges.append((u - 1, v - 1, c))
    mst, max_edge = kruskal(n, edges)
    print(f"{mst} {max_edge}")

if __name__ == "__main__":
    main()
```





## 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/

有 n个小组要排成一个队列，每个小组中有若干人。
当一个人来到队列时，如果队列中已经有了自己小组的成员，他就直接插队排在自己小组成员的后面，否则就站在队伍的最后面。
请你编写一个程序，模拟这种小组队列。

注意：**每个人的编号不重复，另外可能有散客**。

**输入**

第一行：小组数量 t (t<100)。
接下来 t 行，每行输入一个小组描述，表示这个小组的人的编号。
编号是 0 到 999999 范围内的整数，一个小组最多可包含 1000 个人。
最后，命令列表如下。 有三种不同的命令：
1、ENQUEUE x - 将编号是 x 的人插入队列；
2、DEQUEUE - 让整个队列的第一个人出队；
3、STOP - 测试用例结束
每个命令占一行，不超过50000行。

**输出**

对于每个 DEQUEUE 命令，输出出队的人的编号，每个编号占一行。

样例输入

```
Sample1 input:
2
101 102 103
201 202 203
ENQUEUE 101
ENQUEUE 201
ENQUEUE 102
ENQUEUE 202
ENQUEUE 103
ENQUEUE 203
DEQUEUE
DEQUEUE
DEQUEUE
DEQUEUE
DEQUEUE
DEQUEUE
STOP

Sample1 output:
101
102
103
201
202
203
```

样例输出

```
Sample2 input:
2
259001 259002 259003 259004 259005
260001 260002 260003 260004 260005 260006
ENQUEUE 259001
ENQUEUE 260001
ENQUEUE 259002
ENQUEUE 259003
ENQUEUE 259004
ENQUEUE 259005
DEQUEUE
DEQUEUE
ENQUEUE 260002
ENQUEUE 260003
DEQUEUE
DEQUEUE
DEQUEUE
DEQUEUE
STOP

Sample2 output:
259001
259002
259003
259004
259005
260001
```

来源

acwing 小组队列 https://www.acwing.com/problem/content/description/134/





```python
from collections import deque					# 时间: 105ms

# Initialize groups and mapping of members to their groups
t = int(input())
groups = {}
member_to_group = {}



for _ in range(t):
    members = list(map(int, input().split()))
    group_id = members[0]  # Assuming the first member's ID represents the group ID
    groups[group_id] = deque()
    for member in members:
        member_to_group[member] = group_id

# Initialize the main queue to keep track of the group order
queue = deque()
# A set to quickly check if a group is already in the queue
queue_set = set()


while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        x = int(command[1])
        group = member_to_group.get(x, None)
        # Create a new group if it's a new member not in the initial list
        if group is None:
            group = x
            groups[group] = deque([x])
            member_to_group[x] = group
        else:
            groups[group].append(x)
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0] == 'DEQUEUE':
        if queue:
            group = queue[0]
            x = groups[group].popleft()
            print(x)
            if not groups[group]:  # If the group's queue is empty, remove it from the main queue
                queue.popleft()
                queue_set.remove(group)
```





## 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/

请你对输入的树做遍历。遍历的规则是：遍历到每个节点时，按照该节点和所有子节点的值从小到大进行遍历，例如：

```
        7
    /   |   \
  10    3     6
```

对于这个树，你应该先遍历值为3的子节点，然后是值为6的子节点，然后是父节点7，最后是值为10的子节点。

本题中每个节点的值为互不相同的正整数，最大不超过9999999。

**输入**

第一行：节点个数n (n<500)

接下来的n行：第一个数是此节点的值，之后的数分别表示它的所有子节点的值。每个数之间用空格隔开。如果没有子节点，该行便只有一个数。

**输出**

输出遍历结果，一行一个节点的值。

样例输入

```
sample1 input:
4
7 10 3 6
10
6
3

sample1 output:
3
6
7
10
```

样例输出

```
sample2 input:
6
10 3 1
7
9 2 
2 10
3 7
1

sample2 output:
2
1
3
7
10
9
```

来源

2024spring zht



思路：

- 读入节点个数 n，然后依次读入 n 行，每一行的第一个数为该节点的值，后续数为它的所有直接孩子的值。
- 使用字典存储各节点与它们子节点的关系。同时记录所有出现在“孩子”位置的节点，这样可以通过集合差运算找到根节点（根节点不会作为孩子出现）。
- 如果某个元素等于当前节点，则直接输出该节点的值；如果某个元素是子节点，则递归地遍历该子节点。



```python
# 李思哲 物理学院
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def traverse_print(root, nodes):
    if root.children == []:
        print(root.value)
        return
    pac = {root.value: root}
    for child in root.children:
        pac[child] = nodes[child]
    for value in sorted(pac.keys()):
        if value in root.children:
            traverse_print(pac[value], nodes)
        else:
            print(root.value)


n = int(input())
nodes = {}
children_list = []
for i in range(n):
    info = list(map(int, input().split()))
    nodes[info[0]] = TreeNode(info[0])
    for child_value in info[1:]:
        nodes[info[0]].children.append(child_value)
        children_list.append(child_value)
root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
traverse_print(root, nodes)

```







总体思路分为三步：1.通过字典建立输入数据的父子关系；2.找到树的根（这里我将父节点和子节点分别用两个列表记录，最后使用集合减法）；3.通过递归实现要求的从小到大遍历。

感觉这种题目用字典写会更简洁，而且不再需要考虑如何将输入的值全部归入TreeNode类并建立父子关系的问题。

```python
# 王铭健，工学院 2300011118
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parents = []
children = []
for i in range(n):
    t = list(map(int, input().split()))
    parents.append(t[0])
    if len(t) > 1:
        ch = t[1::]
        children.extend(ch)
        tree[t[0]].extend(ch)


def traversal(node):
    seq = sorted(tree[node] + [node])
    for x in seq:
        if x == node:
            print(node)
        else:
            traversal(x)


traversal((set(parents) - set(children)).pop())

```



```python
from collections import defaultdict
import sys
sys.setrecursionlimit(10000)

def main():
    n = int(sys.stdin.readline())
    tree = defaultdict(list)
    all_nodes = set()
    child_nodes = set()
    
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        parent, *children = parts
        tree[parent].extend(children)
        all_nodes.add(parent)
        all_nodes.update(children)
        child_nodes.update(children)
    
    # 根节点 = 出现在 all_nodes 但没出现在 child_nodes 的那个
    root = (all_nodes - child_nodes).pop()
    
    def traverse(u):
        # 把 u 自己和它的所有直接孩子放一起排序
        group = tree[u] + [u]
        group.sort()
        for x in group:
            if x == u:
                print(u)
            else:
                traverse(x)
    
    traverse(root)

if __name__ == "__main__":
    main()

```



```python
# 刘华君 物理学院
class Tree:
    def __init__(self, val):
        self.val = val
        self.children = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def traverse(self):
        if self.children == []:
            print(self.val)
        else:
            tmp_nodes = self.children + [self]
            tmp_nodes.sort(key=lambda x: x.val)
            for node in tmp_nodes:
                if node.val != self.val:
                    node.traverse()
                else:
                    print(node.val)

def build_tree(n, nodes):
    for _ in range(n):
        values = list(map(int, input().split()))
        root_val = values[0]
        if root_val not in nodes:
            nodes[root_val] = Tree(root_val)
        t = nodes[root_val]
        for child_val in values[1:]:
            if child_val not in nodes:
                nodes[child_val] = Tree(child_val)
            child = nodes[child_val]
            t.add_child(child)
            child.parent = t

    root = None
    for root_val in nodes:
        if not nodes[root_val].parent:
            root = nodes[root_val]
            break

    return root

if __name__ == "__main__":
    nodes = {}
    n = int(input())
    root = build_tree(n, nodes)
    if root:
        root.traverse()

```



```python
def dfs(node, graph, result):
    if node not in graph:  # 如果节点没有子节点
        result.append(node)
        return
    children = graph[node]
    temp = [node] + children
    temp.sort()
    for child in temp:
        if child == node:
            result.append(node)
        elif child in graph:  # 仅当子节点存在于图中时才进行递归
            dfs(child, graph, result)

def main():
    n = int(input())  # 节点个数
    graph = {}
    all_nodes = set()
    child_nodes = set()

    for _ in range(n):
        line = list(map(int, input().split()))
        node = line[0]
        all_nodes.add(node)
        if len(line) > 1:
            children = line[1:]
            graph[node] = children
            child_nodes.update(children)
        else:
            graph[node] = []  # 确保没有子节点的情况也在图中表示出来

    # 可能不只有一个根节点，找到所有顶层节点
    top_level_nodes = list(all_nodes - child_nodes)
    top_level_nodes.sort()

    result = []
    for node in top_level_nodes:
        dfs(node, graph, result)

    for node in result:
        print(node)

if __name__ == "__main__":
    main()
```





## 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/

给定一个长度为n和整数k的整数序列。请你打印出`[1,10^9]`范围内的**最小整数**x(即1≤x≤10^9)，使得给定序列中恰好有k个元素**小于或等于x**。

注意，序列可以包含相等的元素。
如果没有这样的x，打印"-1"(不带引号)。

**输入**

输入的第一行包含整数 n 和 k ( 1≤n≤2·10^5, 0≤k≤n)。
输入的第二行包含n个整数 a_1,a_2，…，a_n (1≤a_i≤10^9) ——序列本身。

**输出**

输出最小整数 x (1≤x≤10^9)，使得给定序列中恰好有k个元素小于或等于x。
如果没有这样的x，打印"-1"(不带引号)。

样例输入

```
sample1 input:
7 4
3 7 5 1 10 3 20

sample1 output:
5
```

样例输出

```
sample2 input:
7 2
3 7 5 1 10 3 20

sample2 output:
-1
```

提示

tags: sortings, *1200

来源

tcy, https://codeforces.com/contest/977/problem/C，略作改动



```python
n, k = map(int, input().split())

a = list(map(int, input().split()))
a.sort()

# 寻找 x
if k == 0:
    x = 1 if a[0] > 1 else -1
elif k == n:
    x = a[-1]
else:
    # 检查第 k 个元素是否是唯一满足条件的
    x = a[k-1] if a[k-1] < a[k] else -1

print(x)
```



## 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/

依次读入一个整数序列，每当已经读入的整数个数为奇数时，输出已读入的整数构成的序列的中位数。

**输入**

第一行输入一个整数 T（1<=T<=100），代表后面数据集的个数。
接下来每行一个数据集，包含 M 个空格分隔的整数。1<=M<=99999 且所有 M 相加之和不超过500000。

**输出**

对于每个数据集输出两行，第一行输出中位数的个数N。

第二行输出空格分隔的N个整数，表示中位数。

样例输入

```
3
1 2 3 4 5 6 7 8 9
9 8 7 6 5 4 3 2 1
23 41 13 22 -3 24 -31 -11 -8 -7 3 5 103 211 -311 -45 -67 -73 -81 -99 -33 24 56
```

样例输出

```
5
1 2 3 4 5
5
9 8 7 6 5
12
23 23 22 22 13 3 5 5 3 -3 -7 -3
```

提示

tags: heap

来源

AcWing 106, https://www.acwing.com/problem/content/description/108/



```python
import heapq

def dynamic_median(nums):
    # 维护小根和大根堆（对顶），保持中位数在大根堆的顶部
    min_heap = []  # 存储较大的一半元素，使用最小堆
    max_heap = []  # 存储较小的一半元素，使用最大堆

    median = []
    for i, num in enumerate(nums):
        # 根据当前元素的大小将其插入到对应的堆中
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)

        # 调整两个堆的大小差，使其不超过 1
        if len(max_heap) - len(min_heap) > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

        if i % 2 == 0:
            median.append(-max_heap[0])

    return median

T = int(input())
for _ in range(T):
    #M = int(input())
    nums = list(map(int, input().split()))
    median = dynamic_median(nums)
    print(len(median))
    print(*median)
```





## 27948: FBI树

http://cs101.openjudge.cn/practice/27948/

我们可以把由 0 和 1 组成的字符串分为三类：全 0 串称为 B 串，全 1 串称为 I 串，既含 0 又含 1 的串则称为 F 串。
FBI 树是一种二叉树，它的结点类型也包括 F 结点，B 结点和 I 结点三种。
由一个长度为 2^N 的 01 串 S 可以构造出一棵 FBI 树 T，递归的构造方法如下：

1. T 的根结点为 R，其类型与串 S 的类型相同；
2. 若串 S 的长度大于 1，将串 S 从中间分开，分为等长的左右子串 S1 和 S2；由左子串 S1 构造 R 的左子树 T1，由右子串 S2 构造 R 的右子树 T2。

现在给定一个长度为 2^N 的 01 串，请用上述构造方法构造出一棵 FBI 树，并输出它的后序遍历序列。

**输入**

第一行是一个整数 N，0<= N <= 10。
第二行是一个长度为 2^N 的 01 串。

**输出**

包含一行，这一行只包含一个字符串，即 FBI 树的后序遍历序列。

样例输入

```
sample1 input:
3
10001011

sample1 output:
IBFBBBFIBFIIIFF
```

样例输出

```
sample2 input:
2
0000

sample2 output:
BBBBBBB
```

提示

tags: binary tree

来源

AcWing 419, https://www.acwing.com/problem/content/421/





```python
def construct_FBI_tree(s):
    # 判断当前字符串的类型
    if '0' in s and '1' in s:
        node_type = 'F'
    elif '1' in s:
        node_type = 'I'
    else:
        node_type = 'B'
    
    if len(s) > 1:  # 如果字符串长度大于1，则继续分割
        mid = len(s) // 2
        # 递归构建左右子树，并将结果按后序遍历拼接
        left_tree = construct_FBI_tree(s[:mid])
        right_tree = construct_FBI_tree(s[mid:])
        return left_tree + right_tree + node_type
    else:  # 如果字符串长度为1，直接返回该节点类型
        return node_type

N = int(input())
s = input()
print(construct_FBI_tree(s))
```



```python
# ==谭訸 生命科学学院==
class Node:
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

def build_FBI(string):
    root = Node()
    if '0' not in string:
        root.value = 'I'
    elif '1' not in string:
        root.value = 'B'
    else:
        root.value = 'F'
    l = len(string) // 2
    if l > 0:
        root.left = build_FBI(string[:l])
        root.right = build_FBI(string[l:])
    return root

def post_traverse(node):
    ans = []
    if node:
        ans.extend(post_traverse(node.left))
        ans.extend(post_traverse(node.right))
        ans.append(node.value)
    return ''.join(ans)

n = int(input())
string = input()
root = build_FBI(string)
print(post_traverse(root))
```





## 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/

小晨的电脑上安装了一个机器翻译软件，他经常用这个软件来翻译英语文章。

这个翻译软件的原理很简单，它只是从头到尾，依次将每个英文单词用对应的中文含义来替换。对于每个英文单词，软件会先在内存中查找这个单词的中文含义，如果内存中有，软件就会用它进行翻译；如果内存中没有，软件就会在外存中的词典内查找，查出单词的中文含义然后翻译，并将这个单词和译义放入内存，以备后续的查找和翻译。

假设内存中有M个单元，每单元能存放一个单词和译义。每当软件将一个新单词存入内存前，如果当前内存中已存入的单词数不超过M−1，软件会将新单词存入一个未使用的内存单元；若内存中已存入M个单词，软件会清空最早进入内存的那个单词，腾出单元来，存放新单词。

假设一篇英语文章的长度为N个单词。给定这篇待译文章，翻译软件需要去外存查找多少次词典？假设在翻译开始前，内存中没有任何单词。

**输入**

共2行。每行中两个数之间用一个空格隔开。

第一行为两个正整数M，N，代表内存容量和文章的长度。M<=100,N<=1000

第二行为N个非负整数，按照文章的顺序，每个数（大小不超过1000）代表一个英文单词。文章中两个单词是同一个单词，当且仅当它们对应的非负整数相同。

**输出**

一个整数，为软件需要查词典的次数。

样例输入

```
sample1 input:
3 7
1 2 1 5 4 4 1

sample1 output:
5

#整个查字典过程如下：
每行表示一个单词的翻译，冒号前为本次翻译后的内存状况：

1：查找单词 1 并调入内存。
1 2：查找单词 2 并调入内存。
1 2：在内存中找到单词 1。
1 2 5：查找单词 5 并调入内存。
2 5 4：查找单词 4 并调入内存替代单词 1。
2 5 4：在内存中找到单词 4。
5 4 1：查找单词 1 并调入内存替代单词 2。
共计查了5次词典。
```

样例输出

```
sample2 input:
3 8
1 5 2 7 1 4 2 1

sample2 output:
7
```

提示

tags: queue, implementation
难度1100

来源

luogu P1540[NOIP2010]



```python
from collections import deque

M, N = map(int, input().split())
words = list(map(int, input().split()))

memory = deque()
lookups = 0

for word in words:
    if word not in memory:
        if len(memory) == M:
            memory.popleft()
        memory.append(word)
        lookups += 1

print(lookups)
```



## 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/

词梯问题是由“爱丽丝漫游奇境”的作者 Lewis Carroll 在1878年所发明的单词游戏。从一个单词演变到另一个单词，其中的过程可以经过多个中间单词。要求是相邻两个单词之间差异只能是1个字母，如fool -> pool -> poll -> pole -> pale -> sale -> sage。与“最小编辑距离”问题的区别是，中间状态必须是单词。目标是找到最短的单词变换序列。

假设有一个大的单词集合（或者全是大写单词，或者全是小写单词），集合中每个元素都是四个字母的单词。采用图来解决这个问题，如果两个单词的区别仅在于有一个不同的字母，就用一条边将它们相连。如果能创建这样一个图，那么其中的任意一条连接两个单词的路径就是词梯问题的一个解，我们要找最短路径的解。下图展示了一个小型图，可用于解决从 fool 到sage的词梯问题。

注意，它是无向图，并且边没有权重。

![img](http://media.openjudge.cn/images/upload/2596/1712744630.jpg)



**输入**

输入第一行是个正整数 n，表示接下来有n个四字母的单词，每个单词一行。2 <= n <= 4000。
随后是 1 行，描述了一组要找词梯的起始单词和结束单词，空格隔开。

**输出**

输出词梯对应的单词路径，空格隔开。如果不存在输出 NO。
如果有路径，保证有唯一解。

样例输入

```
25
bane
bank
bunk
cane
dale
dunk
foil
fool
kale
lane
male
mane
pale
pole
poll
pool
quip
quit
rain
sage
sale
same
tank
vain
wane
fool sage
```

样例输出

```
fool pool poll pole pale sale sage
```

来源

https://runestone.academy/ns/books/published/pythonds/Graphs/TheWordLadderProblem.html



按照单词随机替换一个字母建立桶，构建桶内各单词的联系，然后从起点广度优先遍历和起点相连的
点，过程中记录每个词的前一个词，直至遇到终止词，然后倒序往前追溯即可

```python
import sys
from collections import deque

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex

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

    def get_vertices(self):
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
        self.disc = 0
        self.fin = 0

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    # def __str__(self):
    #     return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
    #         self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"



def build_graph(all_words):
    buckets = {}
    the_graph = Graph()

    # 创建词桶 create buckets of words that differ by 1 letter
    for line in all_words:
        word = line.strip()
        for i, _ in enumerate(word):
            bucket = f"{word[:i]}_{word[i + 1:]}"
            buckets.setdefault(bucket, set()).add(word)

    # 为同一个桶中的单词添加顶点和边
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words - {word1}:
                the_graph.add_edge(word1, word2)

    return the_graph


def bfs(start, end):
    start.distnce = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()  # 取队首作为当前顶点

        if current == end:
            return True

        for neighbor in current.get_neighbors():  # 遍历当前顶点的邻接顶点
            if neighbor.color == "white":
                neighbor.color = "gray"
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = "black"  # 当前顶点已经处理完毕，设黑色

    return False

"""
BFS 算法主体是两个循环的嵌套: while-for
    while 循环对图中每个顶点访问一次，所以是 O(|V|)；
    嵌套在 while 中的 for，由于每条边只有在其起始顶点u出队的时候才会被检查一次，
    而每个顶点最多出队1次，所以边最多被检查次，一共是 O(|E|)；
    综合起来 BFS 的时间复杂度为 0(V+|E|)

词梯问题还包括两个部分算法
    建立 BFS 树之后，回溯顶点到起始顶点的过程，最多为 O(|V|)
    创建单词关系图也需要时间，时间是 O(|V|+|E|) 的，因为每个顶点和边都只被处理一次
"""


def traverse(starting_vertex):
    ans = []
    current = starting_vertex
    while (current.previous):
        ans.append(current.key)
        current = current.previous
    ans.append(current.key)

    return ans


n = int(input())
all_words = []
for _ in range(n):
    all_words.append(input().strip())

g = build_graph(all_words)
# print(len(g))

s, e = input().split()
start, end = g.get_vertex(s), g.get_vertex(e)
if start is None or end is None:
    print('NO')
    exit(0)

if bfs(start, end):
    ans = traverse(end)
    print(' '.join(ans[::-1]))
else:
    print('NO')
```



```python
# 周添 物理学院
from collections import deque

def construct_graph(words):
    graph = {}
    for word in words:
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i + 1:]
            if pattern not in graph:
                graph[pattern] = []
            graph[pattern].append(word)
    return graph

def bfs(start, end, graph):
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        word, path = queue.popleft()
        if word == end:
            return path
        for i in range(len(word)):
            pattern = word[:i] + '*' + word[i + 1:]
            if pattern in graph:
                neighbors = graph[pattern]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
    return None

def word_ladder(words, start, end):
    graph = construct_graph(words)
    return bfs(start, end, graph)

n = int(input())
words = [input().strip() for _ in range(n)]
start, end = input().strip().split()

result = word_ladder(words, start, end)

if result:
    print(' '.join(result))
else:
    print("NO")
```



焦晨航 数学科学学院：最最最高兴的一集！零碎看了一天，看了题解没直接ctrl c+ctrl v，而是狠狠洞察思路用计概手段拿下！长度短，能看懂，好操作，爽完了。

```python
# 焦晨航 数学科学学院
from collections import defaultdict
dic=defaultdict(list)
n,lis=int(input()),[]
for i in range(n):
    lis.append(input())
for word in lis:
    for i in range(len(word)):
        bucket=word[:i]+'_'+word[i+1:]
        dic[bucket].append(word)
def bfs(start,end,dic):
    queue=[(start,[start])]
    visited=[start]
    while queue:
        currentword,currentpath=queue.pop(0)
        if currentword==end:
            return ' '.join(currentpath)
        for i in range(len(currentword)):
            bucket=currentword[:i]+'_'+currentword[i+1:]
            for nbr in dic[bucket]:
                if nbr not in visited:
                    visited.append(nbr)
                    newpath=currentpath+[nbr]
                    queue.append((nbr,newpath))
    return 'NO'
start,end=map(str,input().split())    
print(bfs(start,end,dic))
```



将有三个字母确定的单词存到固定的桶中，进行BFS时只要将同一个桶中未入队的单词入队即可。对每个单词存储其BFS过程中的“父节点”，最后逆序找出路径输出即可。

```python
# 蔡沐轩 数学科学学院
from collections import defaultdict,deque

buckets=defaultdict(list)
for _ in range(int(input())):
    word=input()
    for k in range(4):
        buckets[word[:k]+' '+word[k+1:]].append(word)
x,y=input().split()
father={x:x}
q=deque([x])
while q:
    word=q.popleft()
    if word==y:break
    for k in range(4):
        for i in buckets[word[:k]+' '+word[k+1:]]:
            if i not in father:
                q.append(i)
                father[i]=word
if word==y:
    ans=[y]
    while y!=x:
        y=father[y]
        ans.append(y)
    print(' '.join(reversed(ans)))
else:print('NO')
```





## 28050: 骑士周游

Warnsdorff, 回溯, http://cs101.openjudge.cn/practice/28050/

在一个国际象棋棋盘上，一个棋子“马”（骑士），按照“马走日”的规则，从一个格子出发，要走遍所有棋盘格恰好一次。把一个这样的走棋序列称为一次“周游“。在 8 × 8 的国际象棋棋盘上，合格的“周游”数量有 1.305×10^35这么多，走棋过程中失败的周游就更多了。

采用图搜索算法，是解决骑士周游问题最容易理解和编程的方案之一，解决方案分为两步： 首先用图表示骑士在棋盘上的合理走法； 采用图搜索算法搜寻一个长度为（行 × 列-1）的路径，路径上包含每个顶点恰一次。

<img src="http://media.openjudge.cn/images/upload/9136/1712843793.jpg" alt="img" style="zoom:67%;" />

**输入**

两行。
第一行是一个整数n，表示正方形棋盘边长，3 <= n <= 19。

第二行是空格分隔的两个整数sr, sc，表示骑士的起始位置坐标。棋盘左上角坐标是 0 0。0 <= sr <= n-1, 0 <= sc <= n-1。

**输出**

如果是合格的周游，输出 success，否则输出 fail。

样例输入

```
5
0 0
```

样例输出

```
success
```

来源

https://runestone.academy/ns/books/published/pythonds3/Graphs/KnightsTourAnalysis.html



和马走日思路一样，不过需要优化搜索算法，先找到当前点能够走的所有下一个点，然后计算每个下一个点能走的下下一个点数量，优先搜索数量少的，一旦发现一条周游路径就返回True。

采用 **Warnsdorff’s Rule** 进行搜索，优先选择出度最小的下一步路径，从而提高找到完整骑士周游路径的成功率。需要**回溯**的功能来确保即使某个方向走不通，仍然可以回到上一步，尝试其他可能的路径。**结合了 Warnsdorff 规则 和 回溯**，确保最大程度提高找到骑士周游的概率

```python
import sys

def is_valid_move(x, y, board, n):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def get_degree(x, y, board, n, moves):
    count = 0
    for dx, dy in moves:
        if is_valid_move(x + dx, y + dy, board, n):
            count += 1
    return count

def knights_tour_warnsdorff(n, sr, sc):
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]
    board = [[-1 for _ in range(n)] for _ in range(n)]
    board[sr][sc] = 0
    
    def backtrack(x, y, move_count):
        if move_count == n * n:
            return True
        
        next_moves = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny, board, n):
                degree = get_degree(nx, ny, board, n, moves)
                next_moves.append((degree, nx, ny))
        
        next_moves.sort()  # 按 Warnsdorff 规则选择最少可行移动的方向
        
        for _, nx, ny in next_moves:
            board[nx][ny] = move_count
            if backtrack(nx, ny, move_count + 1):
                return True
            board[nx][ny] = -1  # 回溯
        
        return False
    
    if backtrack(sr, sc, 1):
        print("success")
    else:
        print("fail")

if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    sr, sc = map(int, sys.stdin.readline().strip().split())
    knights_tour_warnsdorff(n, sr, sc)
```



```python
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
        self.disc = 0
        self.fin = 0

    def __lt__(self,o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight


    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
            self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"



def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):           #遍历每一行
        for col in range(board_size):       #遍历行上的每一个格子
            node_id = pos_to_node_id(row, col, board_size) #把行、列号转为格子ID
            new_positions = gen_legal_moves(row, col, board_size) #按照 马走日，返回下一步可能位置
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size) #下一步的格子ID
                kt_graph.add_edge(node_id, other_node_id) #在骑士周游图中为两个格子加一条边
    return kt_graph

def pos_to_node_id(x, y, bdSize):
    return x * bdSize + y

def gen_legal_moves(row, col, board_size):
    new_moves = []
    move_offsets = [                        # 马走日的8种走法
        (-1, -2),  # left-down-down
        (-1, 2),  # left-up-up
        (-2, -1),  # left-left-down
        (-2, 1),  # left-left-up
        (1, -2),  # right-down-down
        (1, 2),  # right-up-up
        (2, -1),  # right-right-down
        (2, 1),  # right-right-up
    ]
    for r_off, c_off in move_offsets:
        if (                                # #检查，不能走出棋盘
            0 <= row + r_off < board_size
            and 0 <= col + c_off < board_size
        ):
            new_moves.append((row + r_off, col + c_off))
    return new_moves

# def legal_coord(row, col, board_size):
#     return 0 <= row < board_size and 0 <= col < board_size


def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)              #当前顶点涂色并加入路径
    if n < limit:
        neighbors = ordered_by_avail(u) #对所有的合法移动依次深入
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0

        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):   #选择“白色”未经深入的点，层次加一，递归深入
                return True
        else:                       #所有的“下一步”都试了走不通
            path.pop()              #回溯，从路径中删除当前顶点
            u.color = "white"       #当前顶点改回白色
            return False
    else:
        return True

def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c,v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

# class DFSGraph(Graph):
#     def __init__(self):
#         super().__init__()
#         self.time = 0                   #不是物理世界，而是算法执行步数
# 
#     def dfs(self):
#         for vertex in self:
#             vertex.color = "white"      #颜色初始化
#             vertex.previous = -1
#         for vertex in self:             #从每个顶点开始遍历
#             if vertex.color == "white":
#                 self.dfs_visit(vertex)  #第一次运行后还有未包括的顶点
#                                         # 则建立森林
# 
#     def dfs_visit(self, start_vertex):
#         start_vertex.color = "gray"
#         self.time = self.time + 1       #记录算法的步骤
#         start_vertex.discovery_time = self.time
#         for next_vertex in start_vertex.get_neighbors():
#             if next_vertex.color == "white":
#                 next_vertex.previous = start_vertex
#                 self.dfs_visit(next_vertex)     #深度优先递归访问
#         start_vertex.color = "black"
#         self.time = self.time + 1
#         start_vertex.closing_time = self.time


def main():
    def NodeToPos(id):
       return ((id//8, id%8))

    bdSize = int(input())  # 棋盘大小
    *start_pos, = map(int, input().split())  # 起始位置
    g = knight_graph(bdSize)
    start_vertex = g.get_vertex(pos_to_node_id(start_pos[0], start_pos[1], bdSize))
    if start_vertex is None:
        print("fail")
        exit(0)

    tour_path = []
    done = knight_tour(0, tour_path, start_vertex, bdSize * bdSize-1)
    if done:
        print("success")
    else:
        print("fail")

    exit(0)

    # 打印路径
    cnt = 0
    for vertex in tour_path:
        cnt += 1
        if cnt % bdSize == 0:
            print()
        else:
            print(vertex.key, end=" ")
            #print(NodeToPos(vertex.key), end=" ")   # 打印坐标

if __name__ == '__main__':
    main()
```



王镜廷，数学学院，发现了不能周游的规律。

Q: 在一个类似国际象棋棋盘上，一个棋子“马”（骑士），按照“马走日”的规则，从一个格子出发，要走遍所有棋盘格恰好一次。把一个这样的走棋序列称为一次“周游“。 

如何证明：整数n，表示正方形棋盘边长。如果n是奇数，骑士出发点坐标是(x,y)，x + y也是奇数，这样的周游不存在。

A: 证明这个问题的方法主要基于对棋盘的二分色彩与骑士跳跃特性的分析。我们可以将这个问题简化如下：

1. **棋盘的着色**: 首先，我们可以将一个 n x n 的棋盘视为黑白相间的格子，类似于国际象棋棋盘。如果 n 是奇数，棋盘会有 (n^2) 个格子，因为 n^2 也是奇数，所以黑白格子数量不相等。具体来说，其中一种颜色的格子将比另一种多一个。

2. **骑士的跳跃**: 骑士（马）的移动可以看作是在这样的格子间的跳跃。每次移动，骑士从一个颜色的格子跳到另一个颜色的格子。具体来说，如果它在一个黑色格子上，下一步必然是白色格子，反之亦然。

3. **起始点的颜色与奇数特性**: 如果骑士起始点的坐标 (x, y) 满足 x + y 为奇数，在标准黑白棋盘上，这意味着骑士位于一种特定颜色的格子上（比如说白格）。由于总格子数为奇数，且黑白格子数目不相等，假设黑格比白格多一个。因此，如果骑士始终从白格跳到黑格，那么它无法再次跳回到黑格，因为最终会有一个黑格没有白格可供跳转。

4. **周游的可能性**: 周游意味着每个格子都恰好走一次，没有遗漏也没有重复。如果棋盘的一个颜色的格子比另一个多，那么从总格数为奇数的格子颜色开始的周游是不可能的，因为骑士将无法找到足够的跳转点完成最后一步。

结合以上几点，可以得出结论：在一个奇数边长的棋盘上，如果骑士起始点的坐标 (x, y) 使得 x + y 为奇数，那么无法实现一个完整的周游，因为从颜色较少的格子开始，最终将无法匹配到足够的颜色格子完成最后一跳。因此，这样的周游不存在。



为了更好地理解这一概念，让我们考虑一个具体的例子，以 5x5 的棋盘为例。**5x5 棋盘的着色**，B 代表黑色格，W 代表白色格

```
B W B W B
W B W B W
B W B W B
W B W B W
B W B W B
```

**一个不可能完成周游的例子**，骑士的起始点到 (3, 2)：**颜色**：白色（W），**坐标和 (x + y)**：3 + 2 = 5 (奇数)

在这种情况下，因为 5 是奇数，我们预计无法完成周游。理由是起始于白色格子，并且由于黑色格子（13个）比白色格子（12个）多一个，所以最终将会有一个黑色格子无法被访问完成最后一步跳跃。

**验证理论**

尝试规划一个从 (3, 2) 开始的骑士周游路径会发现，由于棋盘的不对称性（黑格多于白格），骑士最终将被迫尝试从一个黑格跳至另一个黑格，这违反了骑士的跳跃规则（每次跳跃必须从一种颜色的格子跳到另一种颜色的格子），从而导致周游失败。

此理论和例子展示了如何使用棋盘的二分色彩性质以及格子数量的奇偶性来分析和判断骑士周游的可能性。在任何边长为奇数的棋盘上，如果起始点的坐标和为奇数，则无法完成周游，因为总会有多出的一个同色格子无法被合规跳跃覆盖。



Q: 如何判断是否存在哈密顿通路？

A: 判断是否存在哈密顿通路（一条通过图中每个顶点恰好一次的路径）是一个在图论中广泛研究的问题，尤其是在国际象棋棋盘中骑士的移动规则下。这个问题属于NP-完全问题，即没有已知的能在多项式时间内解决的算法。尽管如此，有一些启发式方法和特殊情况下的理论可以帮助判断某些图是否存在哈密顿通路。

对于骑士周游问题，特别是在 n x n 棋盘上的骑士的移动，可以使用以下方法或考虑以下条件来估计或判断哈密顿通路的存在：

1. **图的结构和大小**：
   - 对于某些特定大小的棋盘，如 8x8 棋盘，已经通过计算机算法证明可以找到哈密顿通路（即完整的骑士周游）。
   - 对于较小或不规则大小的棋盘，情况可能会变得复杂，例如 3x3、4x4 和部分偶数棋盘，已知是不可能的。

2. **计算机搜索算法**：
   - 使用回溯法、深度优先搜索（DFS）或启发式搜索（如A*搜索算法）来尝试构建通路。这些方法可能需要大量的计算时间，尤其是当棋盘较大时。
   - 对于大棋盘，通常需要利用某些启发式方法来减少搜索空间，例如优先考虑边缘或难以达到的格子。

3. **数学定理和规则**：
   - 存在一些定理，如Dirac定理和Ore定理，它们提供了判断一般图是否存在哈密顿回路的充分条件，但这些通常难以直接应用于国际象棋棋盘上的骑士移动问题。
   - 对于特定的起点和棋盘布局，可以进行数学归纳或构造性证明来判断哈密顿通路的存在。

4. **实验和模拟**：
   - 对不同的棋盘配置进行实验模拟，尝试找到通路。实际上，许多现存的哈密顿通路是通过计算机模拟和实验发现的。

5. **理论上的限制**：
   - 如上文中提到的，对于奇数x奇数的棋盘以及某些起始点条件，可以数学上证明骑士周游问题的解不存在。

综上所述，判断骑士的哈密顿通路存在与否需要依赖具体情况和实验，对于大型棋盘通常依赖计算机算法和启发式搜索。针对特定棋盘和条件，可能需要定制特定的算法或利用已有的理论成果。





【李炳锋 24 级】观察到，随着棋盘总格数的增加，需要搜索的分支总数呈指数级增长。这意味着在较大的棋盘上，尝试搜完所有可能的分支必然导致时间超限（TLE）。基于这一点，我们可以得出一个结论：当无法通过简单的判据来确定某一情形是否有解时，该情形实际上是可解的。否则，我们将不得不依赖于逐一分支搜索的方法来证明其无解，而这在规定的时间内是不可行的。

一种简便有效的判别方式是对棋盘进行交叉染色处理。骑士每走一步都会从一种颜色的格子跳到另一种颜色的格子。因此，如果棋盘上某一种颜色的格子比另一种多出一个，则骑士必须从较多的那种颜色的格子开始才能完成周游；若某一种颜色的格子比另一种多出超过一个，则可以断定无法完成周游。

```python
# 李炳锋24级
n = int(input())  # 输入棋盘的大小 n x n
p, q = map(int, input().split())  # 输入骑士的起始位置 (p, q)

if n < 5:
    print("fail")  # 如果棋盘大小小于5x5，直接判断为失败
else:
    if n % 2 == 1:  # 如果棋盘大小是奇数
        if (p + q) % 2 == 0:
            print("success")  # 如果起始位置的坐标和为偶数，可以成功
        else:
            print("fail")  # 否则失败
    else:
        print("success")  # 如果棋盘大小是偶数，总是可以成功
```





## 28170: 算鹰

dfs, http://cs101.openjudge.cn/practice/28170/

注意看，这个男人叫小帅。他今天在云顶之弈中有所感悟，决定和捷豹在围棋领域扳扳手腕。他注意到，对于自己的棋子(如下O)，其上下左右四个位置（如下'!'所示）极其关键。如果四个位置都被对方棋子占领（如下） 那么自己的棋子就会因为没有“气”而被提子。

\#!#

!O!

\#!#

而如果这四个位置有己方的子，则可以连成整体，整体的“气”更多，更难被提子。小帅决定称自己通过上下左右四个方向连成一起的整体为“鹰”，以表示其蓬勃的生机。现在他有一个10x10的小棋盘，其中'-'表示空位，'.'表示自己的落子，他想写一个程序，看看自己有多少“鹰”。

**输入**

输入为 10x10的字符，表示棋盘局势，'-'表示空位，'.'表示己方落子

**输出**

输出为一个int数，表示己方“鹰”的数量。

样例输入

```
---.--.-..
-..-.-....
...--....-
----......
--.---....
-.-..-.---
....-.-..-
-..-----..
-.......-.
.....--.--
```

样例输出

```
8
```

提示

可以利用深度优先遍历进行求解。





这个题目描述不清楚。不如 02386:Lake Counting 题目好。http://cs101.openjudge.cn/practice/02386/

```python
def dfs(x,y):
    graph[x][y] = "-"
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        if 0<=x+dx<10 and 0<=y+dy<10 and graph[x+dx][y+dy] == ".":
            dfs(x+dx,y+dy)
graph = []
result = 0
for i in range(10):
    graph.append(list(input()))
for i in range(10):
    for j in range(10):
        if graph[i][j] == ".":
            result += 1
            dfs(i,j)
print(result)
```





## 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/

奶牛在熊大妈的带领下排成了一条直队。显然，不同的奶牛身高不一定相同……

现在，奶牛们想知道，如果找出一些连续的奶牛，要求最左边的奶牛 A 是最矮的，最右边的 B 是最高的，且 B 高于 A 奶牛。中间如果存在奶牛，则身高不能和 A,B 奶牛相同。问这样的奶牛最多会有多少头？

从左到右给出奶牛的身高，请告诉它们符合条件的最多的奶牛数（答案可能是 0,2，但不会是 1）。



**输入**

第一行一个正整数 N，表示奶牛的头数。(2<=N<=10000)

接下来 N 行，每行一个正整数，从上到下表示从左到右奶牛的身高 hi (1<=hi<=50000000) 。

**输出**

一行一个整数，表示最多奶牛数。

样例输入

```
sample input1:
5
1
2
3
4
1

sample output1:
4

#取第 1 头到第 4 头奶牛，满足条件且为最多。
```

样例输出

```
sample input2:
10
15
15
2
13
7
4
11
5
11
12

sample output2:
5
```

提示

tags: monotonous-stack

来源

hhy, https://www.luogu.com.cn/problem/P6510



利用单调栈， left_bound用于记录以当前点为最右端，满足条件的最左端的索引减1； right_bound用于记录以当前节点为最左端，满足条件的最右端的索引加1，最终答案就是两段拼起来之后的最长长度。

```python
"""
https://www.luogu.com.cn/problem/solution/P6510
简化题意：求一个区间，使得区间左端点最矮，区间右端点最高，且区间内不存在与两端相等高度的奶牛，输出这个区间的长度。
我们设左端点为 A ,右端点为 B
因为 A 是区间内最矮的，所以 [A.B]中，都比 A 高。所以只要 A 右侧第一个 ≤A的奶牛位于 B 的右侧，则 A 合法
同理，因为B是区间内最高的，所以 [A.B]中，都比 B 矮。所以只要 B 左侧第一个 ≥B 的奶牛位于 A的左侧，则 B合法
对于 “ 左/右侧第一个 ≥/≤ ” 我们可以使用单调栈维护。用单调栈预处理出 zz数组表示左，r 数组表示右。
然后枚举右端点 B寻找 A，更新 ans 即可。

这个算法的时间复杂度为 O(n)，其中 n 是奶牛的数量。
"""

N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []  # 单调栈，存储索引

# 求左侧第一个≥h[i]的奶牛位置
for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()

    if stack:
        left_bound[i] = stack[-1]

    stack.append(i)

stack = []  # 清空栈以供寻找右边界使用

# 求右侧第一个≤h[i]的奶牛位
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()

    if stack:
        right_bound[i] = stack[-1]

    stack.append(i)

ans = 0

# for i in range(N-1, -1, -1):  # 从大到小枚举是个技巧
#     for j in range(left_bound[i] + 1, i):
#         if right_bound[j] > i:
#             ans = max(ans, i - j + 1)
#             break
#
#     if i <= ans:
#         break

for i in range(N):  # 枚举右端点 B寻找 A，更新 ans
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)
```





因为有单调栈的提示，所以优先思考单调栈的性质：能够找到（向左）一个最长的区间，其上面的值都比当前位置要小，于是当前位置就是这个区间的最大值——这算是想到了一半；

后一半就着重解决最左端最小值的性质，其实和前面是对称的想法，我要找到一个位置，这个位置必须要前面最长的区间内，且是区间最小值，这利用了单调栈中的另一个性质：单调栈内的元素，是单调的，意味着每个元素都是其到下一个元素之间的最小值——而最小值是具有传递性的，也就是每个元素是该位置当前遍历到的位置之间的最小值——恰好满足我们的要求。

于是我们只要找第一步中区间是否包含第二步中单调栈的元素即可（这里可以选择线性遍历或者二分，二分查找是因为单调栈是单调的），选择被包含元素中最左端的元素，就是以当前遍历到的位置为最大值的最长连续区间。

综上，我们只需要两个单调栈，一个是递增栈，一个是非增栈就好了。

```python
# 熊江凯、元培学院
from bisect import bisect_right as bl
lis,q1,q2,ans=[int(input())for _ in range(int(input()))],[-1],[-1],0
for i in range(len(lis)):
    while len(q1)>1 and lis[q1[-1]]>=lis[i]:q1.pop()
    while len(q2)>1 and lis[q2[-1]]<lis[i]:q2.pop()
    id=bl(q1,q2[-1])
    if id<len(q1):ans=max(ans,i-q1[id]+1)
    q1.append(i)
    q2.append(i)
print(ans)
```





## 28203:【模板】单调栈

monotonous-stack, http://cs101.openjudge.cn/practice/28203/



给出项数为 n 的整数数列 a1...an。

定义函数 f(i) 代表数列中第 i 个元素之后第一个大于 ai 的元素的**下标**，。若不存在，则 f(i)=0。

试求出 f(1...n)。

**输入**

第一行一个正整数 n。
第二行 n 个正整数 a1...an​。

**输出**

一行 n 个整数表示 f(1), f(2), ..., f(n) 的值。

样例输入

```
5
1 4 2 3 5
```

样例输出

```
2 5 4 5 0
```

提示

【数据规模与约定】

对于 30% 的数据，n <= 100；

对于 60% 的数据，n <= 5 * 10^3 ；

对于 100% 的数据，1 <= n <= 3 * 10^6，1 <= ai <= 10^9。

来源

P5788 【模板】单调栈，https://www.luogu.com.cn/problem/P5788



```python
n = int(input())
a = list(map(int, input().split()))
stack = []

#f = [0]*n
for i in range(n):
    while stack and a[stack[-1]] < a[i]:
        #f[stack.pop()] = i + 1
        a[stack.pop()] = i + 1


    stack.append(i)

while stack:
    a[stack[-1]] = 0
    stack.pop()

print(*a)
```



```python
n = int(input())
ans = [0 for _ in range(n)]
l = list(map(int, input().split()))
stack = []
i = 0
while i < n:
    while stack and l[i] > l[stack[-1]]:
        ans[stack.pop()] = i + 1
    stack.append(i)
    i += 1
print(*ans)

```



## 28810:是否同一棵二叉排序树

http://cs101.openjudge.cn/practice/28810/

给定一个插入序列就可以唯一确定一棵二叉排序树。然而，一棵给定的二叉排序树却可以由多种不同的插入序列得到。例如分别按照序列{2, 1, 3}和{2, 3, 1}插入初始为空的二叉排序树，都得到一样的结果。于是对于输入的各种插入序列，你需要判断它们是否能生成一样的二叉排序树。



输入

输入数据的第1行给出两个正整数N (≤10)和L，分别是每个序列插入元素的个数和需要检查的序列个数。第2行给出N个以空格分隔的正整数，作为初始插入序列。随后L行，每行给出N个插入的元素，属于L个需要检查的序列。

输出

对每一个需要检查的序列，如果其生成的二叉搜索树跟对应的初始序列生成的一样，输出“Yes”，否则输出“No”。

样例输入

```
4 2
3 1 4 2
3 4 1 2
3 2 4 1
```

样例输出

```
Yes
No
```

来源

浙江大学 DS课程组



这个问题是典型的 **二叉搜索树（BST）结构判等问题**：我们需要判断多个序列在插入到 BST 中时，是否生成了和初始序列一致的结构。

🧠 解题思路：

1. 根据初始序列构建一棵标准 BST。
2. 对于每个待检查序列，重新构建 BST，并与标准 BST 的结构进行比较。
3. 若结构一致，输出 "Yes"，否则输出 "No"。

✅ 判断结构一致性方法：

- 一种做法是对 BST 结构进行**序列化**（例如前序+中序），然后进行字符串比较。
- 也可以直接写一个**结构相等判断函数**，递归比较两个 BST 是否在每个结点上数值相同、左右子树结构相同。

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root

def build_bst(sequence):
    root = None
    for num in sequence:
        root = insert(root, num)
    return root

def is_same_tree(t1, t2):
    if not t1 and not t2:
        return True
    if not t1 or not t2:
        return False
    if t1.val != t2.val:
        return False
    return is_same_tree(t1.left, t2.left) and is_same_tree(t1.right, t2.right)

# 主程序
def main():
    while True:
        try:
            N, L = map(int, input().split())
            if N == 0:
                break
            base_seq = list(map(int, input().split()))
            base_tree = build_bst(base_seq)

            for _ in range(L):
                check_seq = list(map(int, input().split()))
                check_tree = build_bst(check_seq)
                print("Yes" if is_same_tree(base_tree, check_tree) else "No")
        except EOFError:
            break

# 示例输入运行
if __name__ == "__main__":
    main()

```





## 29468.实现散列表

http://cs101.openjudge.cn/practice/29468/

给定一个指定大小**N**的散列表，并输入一系列数字：

若找到空槽，则插入该数字，并返回槽位置；若该数字在散列表中存在，则直接输出其位置。

注1：使用下标增加的二次探测法解决散列冲突

注2：散列表实际大小应确定为不小于用户输入**N**的最小**质数**

注3：散列函数使用**除余法**



**输入**

两行
第一行为用户指定散列表大小整数N
第二行为一系列数字，以空格分隔

**输出**

逐个输出对应数字在散列表中位置，以空格分隔
若该数字无法插入，则输出“-”

样例输入

```
4
10 6 4 10 15
```

样例输出

```
0 1 4 0 -
```

来源

yh



按照以下步骤进行：

**1. 确定散列表的实际大小**

- 用户输入的 `N` 是散列表的建议大小。
- 我们需要找到不小于 `N` 的最小质数作为散列表的实际大小。

**2. 散列函数**

- 使用除余法：`hash(key) = key % table_size`

**3. 解决冲突**

- 使用二次探测法（Quadratic Probing）解决冲突：
  - 如果槽位被占用，则依次尝试 `(hash(key) + i^2) % table_size` 和 `(hash(key) - i^2) % table_size`，其中 `i` 从 1 开始递增。

> 题目中的注1：“下标增加的二次探测法”。可能正确的做法是只沿着一个方向探测，比如每次加i²，而不是同时正反两个方向。比如，二次探测的标准做法是每次跳跃的步长是i²，但可能只在正方向上。

**4. 插入和查找逻辑**

- 如果数字已经在散列表中，直接输出其位置。
- 如果找到空槽，则插入该数字并返回槽位置。
- 如果所有槽位都被占满且无法插入，输出 `-`。

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def next_prime(n):
    while not is_prime(n):
        n += 1
    return n

def quadratic_probing_insert(table, key, table_size):
    hash_val = key % table_size
    if table[hash_val] == key:
        return hash_val
    elif table[hash_val] is None:
        table[hash_val] = key
        return hash_val
    else:
        i = 1
        while i < table_size:
            new_pos = (hash_val + i * i) % table_size
            if table[new_pos] == key:
                return new_pos
            elif table[new_pos] is None:
                table[new_pos] = key
                return new_pos
            i += 1
        return -1  # 插入失败

# 输入处理
import sys
input = sys.stdin.read().splitlines()

N = int(input[0])
numbers = list(map(int, input[1].split()))

table_size = next_prime(N)
hash_table = [None] * table_size

result = []
for num in numbers:
    pos = quadratic_probing_insert(hash_table, num, table_size)
    if pos == -1:
        result.append("-")
    else:
        result.append(str(pos))

print(" ".join(result))
```



## T29803:穿越火线

Dijkstra, binary search, http://cs101.openjudge.cn/practice/29803/

你是战场上的一名司机,现在需要在时限T内把物资从大本营送到前线。 
从战场的地图可以视为一张无向图(可能有重边),其中每条边代表一条道路,每条道路除了通过时间t以外,还有一个交火等级a,这里a是0~100之间的整数。一般来说,车辆的防护等级要大于等于交火等级才能通过这条道路。
不过,你有一个一次性的光学迷彩,可以让你无视交火等级地通过某一条道路。
请计算为了按时把物资运到指定地点,车辆的防护等级至少要是多少.

输入

第一行三个整数n,m,T，表示战场的节点数，边数和要求的时限。大本营是1号，前线是n号节点。
接下来m行，每行四个整数u,v,t,a，表示节点u和节点v之间有一条道路，通过时间为t，交火等级为m。

输出

一个正整数，表示车辆需要的最小防护等级。数据保证100的防护等级下可以按时到达。

样例输入

```
4 4 6
1 2 4 0
2 4 4 10
1 3 3 50
3 4 3 60
```

样例输出

```
50
```

提示

从1到4有1-2-4和1-3-4两条路径，但是路线1-2-4需要总时间8，超过时间要求。所以只能走1-3-4.两条道路需要的防护等级分别为50和60，可以在后一条道路上使用光学迷彩，所以防护等级只需要是50.



这是一个结合了 **最短路径 + 二分答案 + 特殊处理一次性道具** 的问题，具体思路如下：

------

✅ **问题理解**

我们要：

- 从节点 1 到 n；
- 时间总花费 ≤ T；
- 道路有交火等级 a，只有当 `防护等级 ≥ a` 才能通过；
- 我们可以 **一次性无视** 一条边的交火等级。

目标是求：**能按时到达的最小防护等级**。

------

🧠 解题思路

步骤一：二分答案

我们设防护等级为 `x`，我们要找满足以下条件的最小 `x`：

- 存在一条从 1 到 n 的路径，总时间 ≤ T；
- 在这条路径上：
  - 所有边要么 `a ≤ x`；
  - 最多一条边 `a > x`（我们用光学迷彩）。

所以我们可以 **二分防护等级 x ∈ [0, 100]**，判断在当前 `x` 下是否可行。

------

步骤二：判定函数 check(x)

对于某个防护等级 `x`，我们要判断：能否在总时间 ≤ T 的情况下，从 1 到 n？

方法：

- 枚举所有边，把它们分类：
  - **普通边**：`a ≤ x`，可直接走；
  - **特权边**：`a > x`，可以最多选择其中一条使用光学迷彩。
- 对于每个 **特权边**，试图在图中加上这条特权边，其它只能走普通边，跑一次 Dijkstra，看从 1 到 n 是否可达且时间 ≤ T。

------

✅ 实现代码（Python）

```python
import heapq

def dijkstra(n, graph, allowed_edges):
    dist = [float('inf')] * (n + 1)
    dist[1] = 0
    pq = [(0, 1)]
    while pq:
        time, u = heapq.heappop(pq)
        if time > dist[u]:
            continue
        for v, t in allowed_edges[u]:
            if dist[v] > time + t:
                dist[v] = time + t
                heapq.heappush(pq, (dist[v], v))
    return dist[n]

def check(x, n, T, edges):
    graph = [[] for _ in range(n + 1)]
    special_edges = []

    for u, v, t, a in edges:
        if a <= x:
            graph[u].append((v, t))
            graph[v].append((u, t))
        else:
            special_edges.append((u, v, t))

    # 尝试不使用光学迷彩
    if dijkstra(n, edges, graph) <= T:
        return True

    # 尝试每一条特权边作为迷彩边
    for u, v, t in special_edges:
        # 暂时加上这条边
        graph[u].append((v, t))
        graph[v].append((u, t))
        if dijkstra(n, edges, graph) <= T:
            return True
        # 撤销
        graph[u].pop()
        graph[v].pop()

    return False

def min_armor(n, m, T, edge_list):
    left, right = 0, 100
    answer = 100
    while left <= right:
        mid = (left + right) // 2
        if check(mid, n, T, edge_list):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    return answer

# 读入样例输入
n, m, T = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]

print(min_armor(n, m, T, edges))
```



## T29702:二叉的水管

topological order, tree, http://cs101.openjudge.cn/practice/29702/

现在有一根形状为完全二叉树的水管，根结点为总进水口。由于重力朝向原因，这根水管的流量非常不均衡。这根水管总共有k层，从根节点开始，分别是第0层，第1层，...，第k-1层。具体来说，第i层的结点会把它$10^{-2^{(k-i)}}$的流量分配给左子树，余下的给右子树。若只有左子树，视为右侧全部流入泥土中。

现在有一位管道工被安排来维护这条管道。问题是，这条管道中众多节点的对应图已经遗失不见。他手里的检测工具也非常有限，只能对两个结点的流量进行大小比较，希望以此来判断结点位置。请你帮助他重建这个二叉树形状的水管。

**输入**

第一行是节点数m和关系数n
第2~n+1行给出n个关系，以“A > B”的形式表示结点A的流量大于结点B的。

**输出**

如果你能够通过检测数据重建二叉树，则输出二叉树的中序遍历结果
如果你发现检测结果不合逻辑，说明仪器坏了，请输出"Device error."
如果不能重建二叉树，则请输出"Not determined."

样例输入

```
3 3
1 > 2
2 > 3
1 > 3
```

样例输出

```
3 1 2
```

提示

按这样的流量分配，左子树的根结点都要小于右子树的最小结点。



用逐行读取、`split('>')` 来解析 `A > B`，避免了“`1>2`”或多余空格导致的拆分错误。

```python
import sys
from collections import defaultdict, deque


def solve():
    data = sys.stdin.readline().strip().split()
    if not data:
        return
    m, n = map(int, data)

    # 1) 读入所有 “A > B” 关系，建图
    edges = defaultdict(list)
    indegree = [0] * (m + 1)
    for _ in range(n):
        line = sys.stdin.readline().strip()
        if not line:
            continue
        left_str, right_str = line.split('>')
        A = int(left_str.strip())
        B = int(right_str.strip())
        edges[A].append(B)
        indegree[B] += 1

    # 2) 拓扑排序：检查矛盾（环）和是否唯一
    q = deque()
    for u in range(1, m + 1):
        if indegree[u] == 0:
            q.append(u)

    topo_list = []
    multiple = False
    while q:
        if len(q) > 1:
            multiple = True
        u = q.popleft()
        topo_list.append(u)
        for v in edges[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    if len(topo_list) < m:
        print("Device error.")
        return
    if multiple:
        print("Not determined.")
        return

    # 3) 生成“位置从大到小”的序列 pos_order（前序 根→右→左）
    pos_order = []

    def dfs(u):
        if u > m:
            return
        pos_order.append(u)
        dfs(2 * u + 1)
        dfs(2 * u)

    dfs(1)

    # 4) 给这些位置分配流量编号（topo_list 为从大到小的编号）
    assigned = [0] * (m + 1)
    for i in range(m):
        assigned[pos_order[i]] = topo_list[i]

    # 5) 使用递归方式中序遍历 assigned[]
    res = []

    def inorder(u):
        if u > m:
            return
        inorder(2 * u)
        res.append(str(assigned[u]))
        inorder(2 * u + 1)

    inorder(1)
    print(" ".join(res))


if __name__ == "__main__":
    solve()
```

------

关键点说明

1. **输入解析更健壮**
   - 用 `line.split('>')` 而不是按空格拆 `A > B`，可以兼容用户输入 `1>2`、`1 > 2`、` 1 > 2` 等多种格式，避免因空格不一致导致的 ValueError 或拆分错误。
2. **拓扑排序判断“有环”与“是否唯一”**
   - `edges[A].append(B)` 表示 “A 的流量要大于 B（A→B）”。
   - 维护 `indegree[B] += 1`，把所有 `indegree[u]==0` 的节点先入队。
   - 如果某一步队列中超过 1 个节点，就说明有多种合法的拓扑结果（`multiple = True`）。
   - 最终若拿出的节点数 `< m`，说明有环 → `Device error.`
   - 否则若 `multiple == True` → `Not determined.`
   - 否则，`topo_list` 就是“标签从大到小”的**唯一**线性序列。
3. **“位置从大到小”的序列 pos_order**
   - 题目保证这颗水管形状是一棵“完全二叉树”共 `k` 层，节点数正好是 $2^k−1$。在堆式编号下，节点编号是 `1..m`，且父节点编号比子节点都要小。
   - 根据题意：
     - 根节点的流量最大；
     - “任意一个节点” 的右子树里所有节点都要大于它的左子树里所有节点；
     - 而且子节点肯定流量小于父节点。
   - 由此可以推导，整棵树的“从大到小”的**位置次序**便是“先访问根→再访问右子树（整棵）→再访问左子树（整棵）”。
4. **把“标签从大到小”的拓扑序，填入“位置从大到小”**
   - `topo_list[0]`（流量最大的标签）对应 `pos_order[0]`（位置最大的编号1）；
   - `topo_list[1]`（第二大）对应 `pos_order[1]`（位置第二大的编号，是根的右孩子）；
   - ……依此类推，最后最小的标签放在最小位置（最深最左）。
   - 这样，所有关系 `A > B`（即标签 A 的流量大于标签 B）都会自动满足，因为如果 A 比 B 大，A 在 `topo_list` 中下标更靠前，就会被放到“更靠前（流量更大）的树位置”。
5. **中序遍历输出**
   - 最后，把 `assigned[]` 看作“堆式”存放了标签值的数组。我们需要按照“左→根→右”的顺序打印标签。







# 部分题目渐进性分类

http://cs101.openjudge.cn/dsapre/

把 http://xzmdsa.openjudge.cn/ 2023年春季的几次作业陆续加到cs101，其中有些题目在 cs101已经有。

因为缺少图的题目，我们额外增加了 H7: 图应用



## H1: Python入门

22359: Goldbach Conjecture

http://cs101.openjudge.cn/dsapre/22359/

02039: 反反复复

http://cs101.openjudge.cn/dsapre/02039/

22548: 机智的股民老张

http://cs101.openjudge.cn/dsapre/22548/

23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



## H2: 线性表

05345: 位查询

http://cs101.openjudge.cn/dsapre/05345/

 05344: 最后的最后

http://cs101.openjudge.cn/dsapre/05344/

05467: 多项式加法

http://cs101.openjudge.cn/dsapre/05467/

07297: 神奇的幻方

http://cs101.openjudge.cn/dsapre/07297/

21006: 放苹果（盘子相同）

http://cs101.openjudge.cn/dsapre/21006/

22068: 合法出栈序列

http://cs101.openjudge.cn/dsapre/22068/

23451: 交互四则运算计算器_带错误表达式版

http://cs101.openjudge.cn/dsapre/23451/



## H3: 递归与动态规划

04117: 简单的整数划分问题

http://cs101.openjudge.cn/practice/04117/

02773: 采药

http://cs101.openjudge.cn/practice/02773/

08780: 拦截导弹

http://cs101.openjudge.cn/dsapre/02945/

题目出现在  https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md，

提交Optional problems 部分的 02945: 拦截导弹 的代码直接AC了。

```python
k=int(input())
l=list(map(int,input().split()))
dp=[0]*k
for i in range(k-1,-1,-1):
    maxn=1
    for j in range(k-1,i,-1):
        if l[i]>=l[j] and dp[j]+1>maxn:
            maxn=dp[j]+1
    dp[i]=maxn
print(max(dp))
```



22636: 修仙之路

http://cs101.openjudge.cn/dsapre/22636/

24375: 小木棍

http://cs101.openjudge.cn/dsapre/24375/

25815: 回文字符串

http://cs101.openjudge.cn/dsapre/25815/

20650: 最长的公共子序列的长度

http://cs101.openjudge.cn/dsapre/20650/



## H4: 查找与排序

07745: 整数奇偶排序

http://cs101.openjudge.cn/dsapre/07745/

04143: 和为给定数

http://cs101.openjudge.cn/practice/04143/

04135: 月度开销

http://cs101.openjudge.cn/practice/04135/

09201: Freda的越野跑

http://cs101.openjudge.cn/dsapre/09201/

20741: 两座孤岛最短距离

http://cs101.openjudge.cn/practice/20741/

23568: 幸福的寒假生活

http://cs101.openjudge.cn/practice/23568/

04136: 矩形分割

http://cs101.openjudge.cn/practice/04136/



## H5: 树及算法-上

01145:Tree Summing

http://cs101.openjudge.cn/practice/01145/

02255:重建二叉树

http://cs101.openjudge.cn/practice/02255/

02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/

02788: 二叉树

http://cs101.openjudge.cn/dsapre/02788/

04081: 树的转换

http://cs101.openjudge.cn/practice/04081/

04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/

14683: 合并果子

http://cs101.openjudge.cn/dsapre/14683/



## H6: 树及算法-下

02756: 二叉树

http://cs101.openjudge.cn/practice/02756/

06646: 二叉树的深度

http://cs101.openjudge.cn/dsapre/06646/

06648: Sequence

http://cs101.openjudge.cn/practice/06648/

01760: Disk Tree

http://cs101.openjudge.cn/practice/01760

04079: 二叉搜索树

http://cs101.openjudge.cn/practice/04079/

 04089: 电话号码

http://cs101.openjudge.cn/practice/04089/

05430: 表达式·表达式树·表达式求值

http://cs101.openjudge.cn/practice/05430/



## H7: 图应用

05442: 兔子与星空

http://cs101.openjudge.cn/practice/05442/

05443: 兔子与樱花

http://cs101.openjudge.cn/practice/05443/

01178: Camelot

http://cs101.openjudge.cn/practice/01178/

01376: Robot

http://cs101.openjudge.cn/practice/01376/

02049: Finding Nemo

http://cs101.openjudge.cn/practice/02049/



### 01324: Holedox Moving

http://cs101.openjudge.cn/practice/01324/

During winter, the most hungry and severe time, Holedox sleeps in its lair. When spring comes, Holedox wakes up, moves to the exit of its lair, comes out, and begins its new life.
Holedox is a special snake, but its body is not very long. Its lair is like a maze and can be imagined as a rectangle with n*m squares. Each square is either a stone or a vacant place, and only vacant places allow Holedox to move in. Using ordered pair of row and column number of the lair, the square of exit located at (1,1). 

Holedox's body, whose length is L, can be represented block by block. And let B1(r1,c1) B2(r2,c2) .. BL(rL,cL) denote its L length body, where Bi is adjacent to Bi+1 in the lair for 1 <= i <=?L-1, and B1 is its head, BL is its tail. 

To move in the lair, Holedox chooses an adjacent vacant square of its head, which is neither a stone nor occupied by its body. Then it moves the head into the vacant square, and at the same time, each other block of its body is moved into the square occupied by the corresponding previous block. 

For example, in the Figure 2, at the beginning the body of Holedox can be represented as B1(4,1) B2(4,2) B3(3,2)B4(3,1). During the next step, observing that B1'(5,1) is the only square that the head can be moved into, Holedox moves its head into B1'(5,1), then moves B2 into B1, B3 into B2, and B4 into B3. Thus after one step, the body of Holedox locates in B1(5,1)B2(4,1)B3(4,2) B4(3,2) (see the Figure 3).

Given the map of the lair and the original location of each block of Holedox's body, your task is to write a program to tell the minimal number of steps that Holedox has to take to move its head to reach the square of exit (1,1).
![img](http://media.openjudge.cn/images/g326/1324_1.jpg)

**输入**

The input consists of several test cases. The first line of each case contains three integers n, m (1<=n, m<=20) and L (2<=L<=8), representing the number of rows in the lair, the number of columns in the lair and the body length of Holedox, respectively. The next L lines contain a pair of row and column number each, indicating the original position of each block of Holedox's body, from B1(r1,c1) to BL(rL,cL) orderly, where 1<=ri<=n, and 1<=ci<=m,1<=i<=L. The next line contains an integer K, representing the number of squares of stones in the lair. The following K lines contain a pair of row and column number each, indicating the location of each square of stone. Then a blank line follows to separate the cases.

The input is terminated by a line with three zeros.

Note: Bi is always adjacent to Bi+1 (1<=i<=L-1) and exit square (1,1) will never be a stone.

**输出**

For each test case output one line containing the test case number followed by the minimal number of steps Holedox has to take. "-1" means no solution for that case.

样例输入

```
5 6 4
4 1
4 2
3 2
3 1
3
2 3
3 3
3 4

4 4 4
2 3
1 3
1 4
2 4
4

2 1
2 2
3 4
4 2

0 0 0
```

样例输出

```
Case 1: 9
Case 2: -1
```

提示

In the above sample case, the head of Holedox can follows (4,1)->(5,1)->(5,2)->(5,3)->(4,3)->(4,2)->(4,1)->(3,1)->(2,1)->(1,1) to reach the square of exit with minimal number of step, which is nine.

来源

Beijing 2002



题目大意：一个蛇，长度为l，要爬到(1,1)点，问你最短的路径是多少。如果不存在最短路径就输出-1

解题思路：BFS+哈希（状态压缩BFS）。将蛇的身体当做状态，记录蛇头位置，然后以蛇头位置开始，向后面计算。比如说第一个样例。

我们约定(0,1)为状态00，(1,0)为状态01，(0,-1)为状态10，(-1,0)为状态11，注意这个部分与后面运算的过程要一一对应。

蛇头位置为(4,1)记录下来，然后考虑第二个位置(4,2)，发现两者差别为(0,1)，所以状态为00，第三个位置(3,2)，与第二个位置差距为(-1,0)，所以状态为11，第四个位置(3,1)，与第三个位置差距为(0,-1)，所以状态为10。

那么把这三个状态组合在一起，就成为状态101100，同样的也可以轻松的按照这个把状态还原为蛇的位置。

判断这个题目只需要考虑是否满足蛇的下个位置：1）不会与当前身体的任何部分相碰撞，2）没有另外一个相同的状态在下个蛇头的位置出现过。在满足这两个条件之后，就可以得到最后的结果了。

```python
# https://www.cnblogs.com/wiklvrain/p/8179443.html
from collections import deque

# Constants for the maximum grid size
maxn = 21
# Directions representing right, down, left, up
dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]


# Function to judge if a move is valid
def judge(p, t, l):
    a, b = p[0], p[1]
    row, col = a + dir[t][0], b + dir[t][1]
    if row == a and col == b:
        return False
    k = l - 1
    while k:
        q = p[2] & 3
        p = (p[0], p[1], p[2] >> 2)
        nx, ny = a + dir[q][0], b + dir[q][1]
        if nx == row and ny == col:
            return False
        a, b = nx, ny
        k -= 1
    return True


# BFS function to find the shortest path for the snake
def bfs(s, n, m, l, g):
    q = deque()
    vis = [[[0] * (1 << 14) for _ in range(maxn)] for _ in range(maxn)]

    q.append(s)
    vis[s[0]][s[1]][s[2]] = 1

    while q:
        p = q.popleft()
        if p[0] == 1 and p[1] == 1:
            return vis[p[0]][p[1]][p[2]] - 1
        for i in range(4):
            nx, ny = p[0] + dir[i][0], p[1] + dir[i][1]
            st = (p[2] & ((1 << (2 * (l - 2))) - 1)) << 2
            st |= (i + 2) % 4
            if 1 <= nx <= n and 1 <= ny <= m and not vis[nx][ny][st] and not g[nx][ny] and judge(p, i, l):
                vis[nx][ny][st] = vis[p[0]][p[1]][p[2]] + 1
                q.append((nx, ny, st))
    return -1


def main():
    cas = 1
    while True:
        n, m, l = map(int, input().split())
        if n == 0 and m == 0 and l == 0:
            break

        # Initialize the snake
        ss = (0, 0, 0)
        tmp1, tmp2 = 0, 0
        for i in range(l):
            a, b = map(int, input().split())
            if i == 0:
                ss = (a, b, 0)
            else:
                for j in range(4):
                    nx = tmp1 + dir[j][0]
                    ny = tmp2 + dir[j][1]
                    if nx == a and ny == b:
                        ss = (ss[0], ss[1], ss[2] | (j << (2 * (i - 1))))
                        break
            tmp1, tmp2 = a, b

        # Read obstacles
        k = int(input())
        g = [[0] * maxn for _ in range(maxn)]
        #for _ in range(k):
        while k:
            try:
                a, b = map(int, input().split())
            except ValueError:
                continue
            k -= 1

            g[a][b] = 1

        # Perform BFS
        result = bfs(ss, n, m, l, g)
        print(f"Case {cas}: {result}")
        cas += 1
        input()


if __name__ == "__main__":
    main()
```





# 其他题目

## 20555: evaluate

http://cs101.openjudge.cn/practice/20555/

逻辑表达式求值

**输入**

一行空格隔开的字串

**输出**

1(True)或是0(False)

样例输入

```
( not ( True or False ) ) and ( False or True and True )
```

样例输出

```
0
```



```python
def evaluate_expression(expression):
    # Replace logical operators with Python equivalents
    expression = expression.replace("not", "not ").replace("and", " and ").replace("or", " or ")
    # Evaluate the expression
    return int(eval(expression))

# 读取输入并处理
expression = input()
print(evaluate_expression(expression))
```



## P0020.删除链表元素

http://dsbpython.openjudge.cn/dspythonbook/P0020/

程序填空，删除链表元素

```python
class Node:
	def __init__(self, data, next=None):
		self.data, self.next = data, next
class LinkList:  #循环链表
	def __init__(self):
		self.tail = None
		self.size = 0
	def isEmpty(self):
		return self.size == 0
	def pushFront(self,data):
		nd = Node(data)
		if self.tail == None:
			self.tail = nd
			nd.next = self.tail
		else:
			nd.next = self.tail.next
			self.tail.next = nd
		self.size += 1
	def pushBack(self,data):
		self.pushFront(data)
		self.tail = self.tail.next
	def popFront(self):
		if self.size == 0:
			return None
		else:
			nd = self.tail.next
			self.size -= 1
			if self.size == 0:
				self.tail = None
			else:
				self.tail.next = nd.next
		return nd.data
	def printList(self):
		if self.size > 0:
			ptr = self.tail.next
			while True:
				print(ptr.data,end = " ")
				if ptr == self.tail:
					break
				ptr = ptr.next
			print("")

	def remove(self,data):
// 在此处补充你的代码
t = int(input())
for i in range(t):
	lst = list(map(int,input().split()))
	lkList = LinkList()
	for x in lst:
		lkList.pushBack(x)
	lst = list(map(int,input().split()))
	for a in lst:
		result = lkList.remove(a)
		if result == True:
			lkList.printList()
		elif result == False:
			print("NOT FOUND")
		else:
			print("EMPTY")
	print("----------------")
```

输入

第一行为整数t，表示有t组数据。
每组数据2行
第一行是若干个整数，构成了一张链表
第二行是若干整数，是要从链表中删除的数。

输出

对每组数据第二行中的每个整数x:

1) 如果链表已经为空，则输出 "EMPTY"
2) 如果x在链表中，则将其删除，并且输出删除后的链表。如果删除后链表为空，则没输出。如果有重复元素，则删前面的。

3）如果链表不为空且x不在链表中，则输出"NOT FOUND"

样例输入

```
2
1 2 3
3 2 2 9 5 1 1 4
1
9 88 1 23
```

样例输出

```
1 2 
1 
NOT FOUND
NOT FOUND
NOT FOUND
EMPTY
EMPTY
----------------
NOT FOUND
NOT FOUND
EMPTY
----------------
```

来源

郭炜



程序填空题目，需要掌握“补充代码”题型，例如写出某个函数的实现代码，如 def remove(self,data):

```python
class Node:
    def __init__(self, data, next=None):
        self.data, self.next = data, next


class LinkList:  # 循环链表
    def __init__(self):
        self.tail = None
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def pushFront(self, data):
        nd = Node(data)
        if self.tail == None:
            self.tail = nd
            nd.next = self.tail
        else:
            nd.next = self.tail.next
            self.tail.next = nd
        self.size += 1

    def pushBack(self, data):
        self.pushFront(data)
        self.tail = self.tail.next

    def popFront(self):
        if self.size == 0:
            return None
        else:
            nd = self.tail.next
            self.size -= 1
            if self.size == 0:
                self.tail = None
            else:
                self.tail.next = nd.next
        return nd.data

    def printList(self):
        if self.size > 0:
            ptr = self.tail.next
            while True:
                print(ptr.data, end=" ")
                if ptr == self.tail:
                    break
                ptr = ptr.next
            print("")

    def remove(self, data):  # 填空：实现函数
        if self.size == 0:
            return None
        else:
            ptr = self.tail
            while ptr.next.data != data:
                ptr = ptr.next
                if ptr == self.tail:
                    return False
            self.size -= 1
            if ptr.next == self.tail:
                self.tail = ptr
            ptr.next = ptr.next.next
            return True


t = int(input())
for i in range(t):
    lst = list(map(int, input().split()))
    lkList = LinkList()
    for x in lst:
        lkList.pushBack(x)
    lst = list(map(int, input().split()))
    for a in lst:
        result = lkList.remove(a)
        if result == True:
            lkList.printList()
        elif result == False:
            print("NOT FOUND")
        else:
            print("EMPTY")
    print("----------------")

"""
样例输入
2
1 2 3
3 2 2 9 5 1 1 4
1
9 88 1 23

样例输出
1 2 
1 
NOT FOUND
NOT FOUND
NOT FOUND
EMPTY
EMPTY
----------------
NOT FOUND
NOT FOUND
EMPTY
----------------
"""
```



## 004.插入链表元素

http://dsbpython.openjudge.cn/2024allhw/004/

很遗憾，一意孤行的Y君没有理会你告诉他的饮食计划并很快吃完了他的粮食储备。
但好在他捡到了一张校园卡，凭这个他可以偷偷混入领取物资的队伍。
为了不被志愿者察觉自己是只猫，他想要插到队伍的最中央。（插入后若有偶数个元素则选取靠后的位置）
于是他又找到了你，希望你能帮他修改志愿者写好的代码，在发放顺序的中间加上他的学号6。
你虽然不理解志愿者为什么要用链表来写这份代码，但为了不被发现只得在此基础上进行修改：

```python
class Node:
	def __init__(self, data, next=None):
		self.data, self.next = data, next

class LinkList:
	def __init__(self):
		self.head = None

	def initList(self, data):
		self.head = Node(data[0])
		p = self.head
		for i in data[1:]:
			node = Node(i)
			p.next = node
			p = p.next

	def insertCat(self):
// 在此处补充你的代码
########            
	def printLk(self):
		p = self.head
		while p:
			print(p.data, end=" ")
			p = p.next
		print()

lst = list(map(int,input().split()))
lkList = LinkList()
lkList.initList(lst)
lkList.insertCat()
lkList.printLk()
```

输入

一行，若干个整数，组成一个链表。

输出

一行，在链表中间位置插入数字6后得到的新链表

样例输入

```
### 样例输入1
8 1 0 9 7 5
### 样例输入2
1 2 3
```

样例输出

```
### 样例输出1
8 1 0 6 9 7 5
### 样例输出2
1 2 6 3
```

来源

Lou Yuke



程序填空题目，需要掌握“补充代码”题型，例如写出某个函数的实现代码，如 def insertCat(self):

```python
class Node:
    def __init__(self, data, next=None):
        self.data, self.next = data, next

class LinkList:
    def __init__(self):
        self.head = None

    def initList(self, data):
        self.head = Node(data[0])
        p = self.head
        for i in data[1:]:
            node = Node(i)
            p.next = node
            p = p.next

    def insertCat(self):
        # 计算链表的长度
        length = 0
        p = self.head
        while p:
            length += 1
            p = p.next

        # 找到插入位置
        position = length // 2 if length % 2 == 0 else (length // 2) + 1
        p = self.head
        for _ in range(position - 1):
            p = p.next

        # 在插入位置处插入数字6
        node = Node(6)
        node.next = p.next
        p.next = node

    def printLk(self):
        p = self.head
        while p:
            print(p.data, end=" ")
            p = p.next
        print()

lst = list(map(int, input().split()))
lkList = LinkList()
lkList.initList(lst)
lkList.insertCat()
lkList.printLk()

"""
### 样例输入1
8 1 0 9 7 5
### 样例输入2
1 2 3

### 样例输出1
8 1 0 6 9 7 5
### 样例输出2
1 2 6 3
"""
```



## P0560:文本缩进二叉树

http://dsbpython.openjudge.cn/dspythonbook/P0560/

文本缩进二叉树就是由若干行文本来表示的一棵二叉树。其定义如下: 

1) 若一行由若干个制表符("\t")和一个字母构成，则该行表示一个二叉树的结点。该结点的层次就是制表符的数量（根是0层）
2) 每个结点的父结点，就是它上方，离它最近的，比它往左偏移了一个制表符的那个结点。没有父结点的结点，是树根。
3) 如果一个结点的左子树为空但右子树不为空，则在其下面的一行用一个向右多缩进了一个制表符的'*'表示其有空的左子树，然后再表示右子树。若右子树为空，则右子树无须表示。若右子树不为空，则表示完左子树后再表示右子树。若左右子树都为空，则左右子树都不需要表示。

给定一个文本缩进二叉树，求其前序、中序、后序遍历序列。

输入样例的二叉树如下图:

![img](http://media.openjudge.cn/images/upload/1709/1662288791.png)

输入

一棵文本缩进二叉树, 不超过100个结点。

输出

该二叉树的前序、中序、后序遍历序列。

样例输入

```
A
	B
		*
		D
			E
	F
		G
			*
			I
		H
```

样例输出

```
ABDEFGIH
BEDAGIFH
EDBIGHFA
```





思路很：

1. 当用字母节点挂父节点的 **左子树** 时，挂完之后就应该把 `parent.next_child` 从 `"left"` 切换到 `"right"`。
2.  `stack` 用列表，遇到更浅的行就 `stack = stack[:depth]`，自动丢弃深度更大的旧节点。

```python
import sys
sys.setrecursionlimit(10000)

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        # 表示下一个插入到该节点的是 'left' 还是 'right'
        self.next_child = 'left'

def build_tree(lines):
    stack = []   # stack[d] 是深度 d 的最新节点
    root = None

    for line in lines:
        # 计算深度
        depth = 0
        while depth < len(line) and line[depth] == '\t':
            depth += 1
        c = line[depth]

        # 保持 stack 长度等于当前深度
        stack = stack[:depth]

        if c == '*':
            # 标记父节点接下来要挂右子树
            parent = stack[-1]
            parent.next_child = 'right'
        else:
            node = Node(c)
            if depth == 0:
                root = node
            else:
                parent = stack[-1]
                if parent.next_child == 'left':
                    parent.left = node
                    # 挂完左子树后，下一次就该挂右子树了
                    parent.next_child = 'right'
                else:
                    parent.right = node
            # 新节点默认先挂左子树
            node.next_child = 'left'
            # 推入栈
            stack.append(node)

    return root

def preorder(node, res):
    if not node: return
    res.append(node.val)
    preorder(node.left, res)
    preorder(node.right, res)

def inorder(node, res):
    if not node: return
    inorder(node.left, res)
    res.append(node.val)
    inorder(node.right, res)

def postorder(node, res):
    if not node: return
    postorder(node.left, res)
    postorder(node.right, res)
    res.append(node.val)

if __name__ == "__main__":
    lines = [line.rstrip('\n') for line in sys.stdin if line.strip()!='']
    root = build_tree(lines)

    pre, ino, post = [], [], []
    preorder(root, pre)
    inorder(root, ino)
    postorder(root, post)

    print(''.join(pre))
    print(''.join(ino))
    print(''.join(post))
```









## P1260: 火星大工程

AOE，拓扑排序，关键路径，http://dsbpython.openjudge.cn/dspythonbook/P1260/

中国要在火星上搞个大工程，即建造n个科考站

建科考站需要很专业的设备，不同的科考站需要不同的设备来完成

有的科考站必须等另外一些科考站建好后才能建。

每个设备参与建完一个科考站后，都需要一定时间来保养维修，才能参与到下一个科考站的建设。

所以，会发生科考站A建好后，必须至少等一定时间才能建科考站B的情况。因为B必须在A之后建，且建B必需的某个设备，参与了建A的工作，它需要一定时间进行维修保养。

一个维修保养任务用三个数a b c表示，意即科考站b必须等a建完才能建。而且，科考站a建好后，建a的某个设备必须经过时长c的维修保养后，才可以开始参与建科考站b。

假设备都很牛，只要设备齐全可用，建站飞快就能完成，建站时间忽略不计。一开始所有设备都齐全可用。

给定一些维修保养任务的描述，求所有科考站都建成，最快需要多长时间。

有的维修保养任务，能开始的时候也可以先不开始，往后推迟一点再开始也不会影响到整个工期。问在不影响最快工期的情况下，哪些维修保养任务的开始时间必须是确定的。按字典序输出这些维修保养工任务，输出的时候不必输出任务所需的时间。

  

**输入**

第一行两个整数n,m，表示有n个科考站，m个维修保养任务。科考站编号为1，2.....n
接下来m行，每行三个整数a b c，表示一个维修保养任务
1 < n,m <=3000

**输出**

先输出所有科考站都建成所需的最短时间
然后按字典序输出开始时间必须确定的维修保养任务

样例输入

```
9 11
1 2 6
1 3 4
1 4 5
2 5 1
3 5 1
4 6 2
5 7 9
5 8 7
6 8 4
7 9 2
8 9 4
```

样例输出

```
18
1 2
2 5
5 7
5 8
7 9
8 9
```

来源

郭炜



```python
# 李宗远 白衣者
# 拓扑排序和AOE网络问题
# 首先建立edge对象，依据数据得到邻接矩阵
# 得到拓扑排序序列
# 依据拓扑排序序列得到时间最早和最晚开始时间和最快时长确定关键事件
# 进而确定关键活动
from collections import defaultdict, deque


class Edge:
    def __init__(self, end, weight):
        self.end = end
        self.weight = weight

    def __lt__(self, other):
        return self.end < other.end

def find_critical_activities(n, m, edges):
    # 构建邻接表和入度数组
    graph = defaultdict(list)
    in_degree = [0] * n
    for s, e, w in edges:
        graph[s - 1].append(Edge(e - 1, w))
        in_degree[e - 1] += 1

    # 拓扑排序
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topological_order = []
    while queue:
        node = queue.popleft()
        topological_order.append(node)
        for edge in graph[node]:
            in_degree[edge.end] -= 1
            if in_degree[edge.end] == 0:
                queue.append(edge.end)

    # 计算最早开始时间
    earliest = [0] * n
    for i in topological_order:
        for edge in graph[i]:
            earliest[edge.end] = max(earliest[edge.end], earliest[i] + edge.weight)
    T = max(earliest)

    # 计算最晚开始时间
    latest = [T] * n
    for j in reversed(topological_order):
        for edge in graph[j]:
            latest[j] = min(latest[j], latest[edge.end] - edge.weight)

    # 确定关键事件
    critical_events = [i for i in range(n) if earliest[i] == latest[i]]

    # 确定关键活动
    critical_activities = []
    for i in critical_events:
        graph[i].sort()
        for edge in graph[i]:
            #关键活动通常指的是导致关键事件发生的活动，而不是所有指向关键事件的活动都是关键活动。
            if edge.end in critical_events and earliest[edge.end] - earliest[i] == edge.weight:
                critical_activities.append((i + 1, edge.end + 1))

    return T, critical_activities


n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# 求解关键活动
T, critical_activities = find_critical_activities(n, m, edges)

print(T)
for activity in critical_activities:
    print(*activity)
```



```python
from collections import deque, defaultdict
from dataclasses import dataclass


@dataclass
class Edge:
    end: int
    weight: int


def topo_sort(graph, in_degrees, n):
    queue = deque([i for i in range(n) if in_degrees[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for edge in graph[node]:
            in_degrees[edge.end] -= 1
            if in_degrees[edge.end] == 0:
                queue.append(edge.end)
    return topo_order


def find_critical_activities(n, m, edges):
    graph = defaultdict(list)
    in_degrees = [0] * n
    for u, v, w in edges:
        graph[u - 1].append(Edge(v - 1, w))
        in_degrees[v - 1] += 1

    topo_order = topo_sort(graph, in_degrees[:], n)
    if not topo_order:
        return ["No"]

    est = [0] * n
    for node in topo_order:
        for edge in graph[node]:
            est[edge.end] = max(est[edge.end], est[node] + edge.weight)

    T = max(est)

    # 计算最晚开始时间
    lst = [T] * n
    for node in reversed(topo_order):
        for edge in graph[node]:
            lst[node] = min(lst[node], lst[edge.end] - edge.weight)

    # 确定关键事件
    critical_events = [i for i in range(n) if est[i] == lst[i]]

    # 确定关键活动
    critical_activities = []
    for node in critical_events:
        for edge in graph[node]:
            # 关键活动通常指的是导致关键事件发生的活动，而不是所有指向关键事件的活动都是关键活动。
            if edge.end in critical_events and est[node] == est[edge.end] - edge.weight:
                critical_activities.append((node + 1, edge.end + 1))

    # critical_activities.sort()

    return T, ["{} {}".format(u, v) for u, v in critical_activities]


n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]
T, critical_activities = find_critical_activities(n, m, edges)
print(T)
print('\n'.join(critical_activities))

```













