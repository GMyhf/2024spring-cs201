# 树的编程题目 continue

Updated 2026 GMT+8 March 20, 2024

2024 spring, Complied by Hongfei Yan





题目练习网址，https://sunnywhy.com/sfbj



# 树与二叉树 1题

## 1 树的判定

https://sunnywhy.com/sfbj/9/1

现有一个由个结点连接而成的**连通**结构，已知这个结构中存在的边数，问这个连通结构是否是一棵树。

输入描述

两个整数、（），分别表示结点数和边数。

输出描述

如果是一棵树，那么输出`Yes`，否则输出`No`。

样例1

输入

复制

```
2 1
```

输出

复制

```
Yes
```

解释

两个结点，一条边，显然是一棵树。

样例2

输入

复制

```
2 0
```

输出

复制

```
No
```

解释

两个结点，没有边，显然不是树。



```python
def is_tree(nodes, edges):
    if nodes - 1 == edges:
        return 'Yes'
    else:
        return 'No'

if __name__ == "__main__":
    n, m = map(int, input().split())
    print(is_tree(n, m))
```



# 二叉树的遍历 16题

## 1 二叉树的先序遍历

https://sunnywhy.com/sfbj/9/2

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵二叉树的先序遍历序列。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
0 2 1 4 5 3
```

解释

对应的二叉树如下图所示，先序序列为`0 2 1 4 5 3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
from collections import deque

class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def traversal(self, mode):
        result = []
        if mode == "preorder":
            result.append(self.val)
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "postorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            result.append(self.val)
            return result
        elif mode == "inorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            result.append(self.val)
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "levelorder":
            queue = deque([self])
            while queue:
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            return result
          
	def tree_height(self):
    if self is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(self.left), tree_height(self.right)) + 1


n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

# "preorder", "postorder", "inorder, "levelorder"
mode = "levelorder"
pt = nodes[0]
result = pt.traversal(mode)

print(*result)
```





## 2 二叉树的中序遍历



mode = "preorder"



## 3 二叉树的后序遍历



mode = "postorder"



## 4 二叉树的层次遍历



mode = "levelorder"

```python

```





## 5 二叉树的高度

层级 Level：从根节点开始到达一个节点的路径，所包含的边的数量，称为这个节点的层级。根节点的层级为 0。

高度 Height：树中所有节点的最大层级称为树的高度。因此空树的高度是-1。



```python
from collections import deque

class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def traversal(self, mode):
        result = []
        if mode == "preorder":
            result.append(self.val)
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "postorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            result.append(self.val)
            return result
        elif mode == "inorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            result.append(self.val)
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "levelorder":
            queue = deque([self])
            while queue:
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            return result

    def height(self):
        if self.left is None and self.right is None:
            return 0
        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0
        return max(left_height, right_height) + 1


n = int(input())
nodes = [Node(i) for i in range(n)]
has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left, right = map(int, input().split())
    if left != -1:
        nodes[i].left = nodes[left]
        has_parent[left] = True
    if right != -1:
        nodes[i].right = nodes[right]
        has_parent[right] = True

# 寻找根节点，也就是没有父节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]

# "preorder", "postorder", "inorder, "levelorder"
# mode = "levelorder"
# result = root.traversal(mode)
# print(*result)

"""
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
"""
print(root.height())
# 2
```



## 6 二叉树的结点层号

https://sunnywhy.com/sfbj/9/2/334

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵二叉树所有结点的层号（假设根结点的层号为`1`）。

输入

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出个整数，分别表示编号从`0`到`n-1`的结点的层号，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
1 3 2 3 3 2
```

解释

对应的二叉树如下图所示，层号为`1`的结点编号为`0`，层号为`2`的结点编号为`2`、`5`，层号为`3`的结点编号为`1`、`4`、`3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
from collections import deque

def node_levels(n, nodes):
    levels = [0] * n
    queue = deque([(0, 1)])  # (node, level)

    while queue:
        node, level = queue.popleft()
        levels[node] = level
        left, right = nodes[node]
        if left != -1:
            queue.append((left, level + 1))
        if right != -1:
            queue.append((right, level + 1))

    return levels

n = int(input())
nodes = [[left, right] for left, right in [map(int, input().split()) for _ in range(n)]]

print(*node_levels(n, nodes))
```



## 7 翻转二叉树

https://sunnywhy.com/sfbj/9/2/335

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），将这棵二叉树中每个结点的左右子树交换，输出新的二叉树的先序序列和中序序列。

输入

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出两行，第一行为先序序列，第二行为中序序列。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
0 5 3 2 4 1
3 5 0 4 2 1
```

解释

对应的二叉树和翻转后的二叉树如下图所示。

![翻转二叉树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/93380dc9-4690-45ac-8b42-d8347bc14fc4.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def flip_tree(node):
    if node is None:
        return
    node.left, node.right = node.right, node.left
    flip_tree(node.left)
    flip_tree(node.right)

def preorder_traversal(node):
    if node is None:
        return []
    return [node.val] + preorder_traversal(node.left) + preorder_traversal(node.right)

def inorder_traversal(node):
    if node is None:
        return []
    return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

flip_tree(nodes[0])

print(*preorder_traversal(nodes[0]))
print(*inorder_traversal(nodes[0]))
```





## 8 先序中序还原二叉树

https://sunnywhy.com/sfbj/9/2/336

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`），已知其先序序列和中序序列，求后序序列。

输入

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

第二行为个整数，表示二叉树的先序序列；

第三行为个整数，表示二叉树的中序序列。

输出

输出个整数，表示二叉树的后序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
0 2 1 4 5 3
1 2 4 0 5 3
```

输出

```
1 4 2 3 5 0
```

解释

对应的二叉树如下图所示，其后序序列为`1 4 2 3 5 0`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/aaaa2905-d60b-4ca6-b445-d7c2600df176.png)

```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder, inorder):
    if inorder:
        index = inorder.index(preorder.pop(0))
        node = Node(inorder[index])
        node.left = build_tree(preorder, inorder[0:index])
        node.right = build_tree(preorder, inorder[index+1:])
        return node

def postorder_traversal(node):
    if node is None:
        return []
    return postorder_traversal(node.left) + postorder_traversal(node.right) + [node.val]

n = int(input())
preorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))

root = build_tree(preorder, inorder)
print(*postorder_traversal(root))
```





## 9 后序中序还原二叉树

https://sunnywhy.com/sfbj/9/2/337

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`），已知其后序序列和中序序列，求先序序列。

输入

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

第二行为个整数，表示二叉树的后序序列；

第三行为个整数，表示二叉树的中序序列。

输出

输出个整数，表示二叉树的先序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
1 4 2 3 5 0
1 2 4 0 5 3
```

输出

```
0 2 1 4 5 3
```

解释

对应的二叉树如下图所示，其先序序列为`0 2 1 4 5 3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/3a704577-f914-4fa8-812c-13b79ac9d104.png)

```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(postorder, inorder):
    if inorder:
        index = inorder.index(postorder.pop())
        node = Node(inorder[index])
        node.right = build_tree(postorder, inorder[index+1:])
        node.left = build_tree(postorder, inorder[0:index])
        return node

def preorder_traversal(node):
    if node is None:
        return []
    return [node.val] + preorder_traversal(node.left) + preorder_traversal(node.right)

n = int(input())
postorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))

root = build_tree(postorder, inorder)
print(*preorder_traversal(root))
```



## 10 层序中序还原二叉树

https://sunnywhy.com/sfbj/9/2/338

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`），已知其层序序列和中序序列，求先序序列。

输入描

第一行一个整数n (1<=n<=50)，表示二叉树的结点个数；

第二行为个整数，表示二叉树的层序序列；

第三行为个整数，表示二叉树的中序序列。

输出描

输出个整数，表示二叉树的先序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
0 2 5 1 4 3
1 2 4 0 5 3
```

输出

```
0 2 1 4 5 3
```

解释

对应的二叉树如下图所示，其先序序列为`0 2 1 4 5 3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/7d88fa4c-28ef-4aca-84d4-5f16bc147517.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(levelorder, inorder):
    if inorder:
        for i in range(0, len(levelorder)):
            if levelorder[i] in inorder:
                node = Node(levelorder[i])
                io_index = inorder.index(levelorder[i])
                break
        node.left = build_tree(levelorder, inorder[0:io_index])
        node.right = build_tree(levelorder, inorder[io_index+1:])
        return node

def preorder_traversal(node):
    if node is None:
        return []
    return [node.val] + preorder_traversal(node.left) + preorder_traversal(node.right)

n = int(input())
levelorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))

root = build_tree(levelorder, inorder)
print(*preorder_traversal(root))
```



## 11 二叉树的最近公共祖先

https://sunnywhy.com/sfbj/9/2/339

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求两个指定编号结点的最近公共祖先。

注：二叉树上两个结点A、B的最近公共祖先是指：二叉树上存在的一个结点，使得既是的祖先，又是的祖先，并且需要离根结点尽可能远（即层号尽可能大）。

输入

第一行三个整数$n、k_1、k_2 (1 \le n \le 50, 0 \le k_1 \le n-1, 0 \le k_2 \le n-1)$，分别表示二叉树的结点个数、需要求最近公共祖先的两个结点的编号；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出一个整数，表示最近公共祖先的编号。

样例1

输入

```
6 1 4
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
2
```

解释

对应的二叉树如下图所示，结点`1`和结点`4`的公共祖先有结点`2`和结点`0`，其中结点`2`是最近公共祖先。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find_path(root, path, k):
    if root is None:
        return False
    path.append(root.val)
    if root.val == k:
        return True
    if ((root.left != None and find_path(root.left, path, k)) or
            (root.right!= None and find_path(root.right, path, k))):
        return True
    path.pop()
    return False

def find_LCA(root, n1, n2):
    path1 = []
    path2 = []
    if (not find_path(root, path1, n1) or not find_path(root, path2, n2)):
        return -1
    i = 0
    while(i < len(path1) and i < len(path2)):
        if path1[i] != path2[i]:
            break
        i += 1
    return path1[i-1]

n, n1, n2 = map(int, input().split())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print(find_LCA(nodes[0], n1, n2))
```



## 12 二叉树的路径和

https://sunnywhy.com/sfbj/9/2/340

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径和是指，从根结点到该结点的路径上所有结点的权值之和；
2. 二叉树的路径和是指，二叉树所有叶结点的路径和之和。

求这棵二叉树的路径和。

输入

第一行一个整数n (1<=n<=50)，表示二叉树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值（）；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出一个整数，表示二叉树的路径和。

样例1

输入

```
6
3 2 1 5 1 2
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
21
```

解释

对应的二叉树如下图所示，其中黑色数字为结点编号，编号右下角的灰色数字为结点权值。由此可得叶结点`1`的路径和为，叶结点`4`的路径和为，叶结点`3`的路径和为，因此二叉树的路径和为。

![二叉树的路径和.png](https://raw.githubusercontent.com/GMyhf/img/main/img/061c3f04-4557-4ab1-aec3-5c563c7e1e5d.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def path_sum(node, current_sum=0):
    if node is None:
        return 0
    current_sum += node.val
    if node.left is None and node.right is None:
        return current_sum
    return path_sum(node.left, current_sum) + path_sum(node.right, current_sum)

n = int(input())
values = list(map(int, input().split()))
nodes = [Node(values[i]) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print(path_sum(nodes[0]))
```



## 13 二叉树的带权路径长度

https://sunnywhy.com/sfbj/9/2/341

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径长度是指，从根结点到该结点的边数；
2. 结点的带权路径长度是指，结点权值乘以结点的路径长度；
3. 二叉树的带权路径长度是指，二叉树所有叶结点的带权路径长度之和。

求这棵二叉树的带权路径长度。

输入

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值（）；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出一个整数，表示二叉树的带权路径长度。

样例1

输入

```
5
2 3 1 2 1
2 3
-1 -1
1 4
-1 -1
-1 -1
```

输出

```
10
```

解释

对应的二叉树如下图所示，其中黑色数字为结点编号，编号右下角的格式为`结点权值*结点路径长度=结点带权路径长度`。由此可得二叉树的带权路径长度为。

![二叉树的带权路径长度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/835e6c0a-d265-4d6b-b484-c2261915cc22.png)



```python
class TreeNode:
    def __init__(self, value=0):
        self.value = value
        self.left = None
        self.right = None

def build_tree(weights, edges):
    # 根据边构建二叉树，并返回根节点
    nodes = [TreeNode(w) for w in weights]
    for i, (left, right) in enumerate(edges):
        if left != -1:
            nodes[i].left = nodes[left]
        if right != -1:
            nodes[i].right = nodes[right]
    return nodes[0] if nodes else None

def weighted_path_length(node, depth=0):
    # 计算带权路径长度
    if not node:
        return 0
    # 如果是叶子节点，返回其带权路径长度
    if not node.left and not node.right:
        return node.value * depth
    # 否则递归计算左右子树的带权路径长度
    return weighted_path_length(node.left, depth + 1) + weighted_path_length(node.right, depth + 1)

# 输入处理
n = int(input())  # 节点个数
weights = list(map(int, input().split()))  # 各节点权值
edges = [list(map(int, input().split())) for _ in range(n)]  # 节点边

# 构建二叉树
root = build_tree(weights, edges)

# 计算并输出带权路径长度
print(weighted_path_length(root))

```



## 14 二叉树的左视图序列

https://sunnywhy.com/sfbj/9/2/342

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），从二叉树的左侧看去，同一层的多个结点只能看到这层中最左边的结点，这些能看到的结点从上到下组成的序列称为左视图序列。求这棵二叉树的左视图序列。

输入

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出若干个整数，表示左视图序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
0 2 1
```

解释

对应的二叉树如下图所示，从左侧看去，第一层可以看到结点`0`，第二层可以看到结点`2`，第三层可以看到结点`1`，因此左视图序列是`0 2 1`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def left_view(root):
    if root is None:
        return []
    queue = [root]
    view = [root.val]
    while queue:
        level = []
        for node in queue:
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        if level:
            view.append(level[0].val)
        queue = level
    return view

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print(*left_view(nodes[0]))
```



## 15 满二叉树的判定

https://sunnywhy.com/sfbj/9/2/343

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），判断这个二叉树是否是满二叉树。

注：如果一棵二叉树每一层的结点数都达到了当层能达到的最大结点数（即如果二叉树的层数为，且结点总数为 $2^k-1$），那么称这棵二叉树为满二叉树。

输入

第一行一个整数n (1<=n<=64)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

如果是满二叉树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
7
2 5
-1 -1
1 4
-1 -1
-1 -1
6 3
-1 -1
```

输出

```
Yes
```

解释

对应的二叉树如下图所示，是满二叉树。

![满二叉树的判定.png](https://raw.githubusercontent.com/GMyhf/img/main/img/f5ce47ce-813f-47bc-badb-1bdf7e2b95e2.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_full(node):
    if node is None:
        return True
    if (node.left is None and node.right is None) or (node.left is not None and node.right is not None):
        return is_full(node.left) and is_full(node.right)
    return False

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print("Yes" if is_full(nodes[0]) else "No")
```



## 16 完全二叉树的判定

https://sunnywhy.com/sfbj/9/2/344

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），判断这个二叉树是否是完全二叉树。

注：如果一棵二叉树除了最下面一层之外，其余层的结点个数都达到了当层能达到的最大结点数，且最下面一层只从左至右连续存在若干结点，而这些连续结点右边不存在别的结点，那么就称这棵二叉树为完全二叉树。

输入

第一行一个整数n (1<=n<=64)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

如果是完全二叉树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
3 -1
```

输出

```
Yes
```

解释

对应的二叉树如下图所示，是完全二叉树。

![完全二叉树的判定.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b51c5d62-9753-46dd-a57c-60658c32847a.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_complete(root):
    if root is None:
        return True
    queue = [root]
    flag = False
    while queue:
        node = queue.pop(0)
        if node.left:
            if flag:  # If we have seen a node with a missing right or left child
                return False
            queue.append(node.left)
        else:
            flag = True
        if node.right:
            if flag:  # If we have seen a node with a missing right or left child
                return False
            queue.append(node.right)
        else:
            flag = True
    return True

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print("Yes" if is_complete(nodes[0]) else "No")
```





# 树的遍历 7题

## 1 树的先根遍历

https://sunnywhy.com/sfbj/9/3

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的先根遍历序列。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出

输出个整数，表示先根遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
0 2 1 6 4 5 3
```

解释

对应的树如下图所示，先根遍历序列为`0 2 1 6 4 5 3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)





```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def pre_order(node):
    if node is None:
        return []
    result = [node.val]
    for child in node.children:
        result.extend(pre_order(child))
    return result

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(*pre_order(nodes[0]))
```



## 2 树的后根遍历

https://sunnywhy.com/sfbj/9/3/346

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的后根遍历序列。

输入

第一行一个整数 $n (1 \le n \le 50)$， 表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出

输出个整数，表示后根遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
1 6 2 4 3 5 0
```

解释

对应的树如下图所示，后根遍历序列为`1 6 2 4 3 5 0`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def post_order(node):
    if node is None:
        return []
    result = []
    for child in node.children:
        result.extend(post_order(child))
    result.append(node.val)
    return result

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(*post_order(nodes[0]))
```



## 3 树的层序遍历

https://sunnywhy.com/sfbj/9/3/347

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的层序遍历序列。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出

输出个整数，表示层序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
0 2 4 5 1 6 3
```

解释

对应的树如下图所示，层序遍历序列为`0 2 4 5 1 6 3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def level_order(root):
    if root is None:
        return []
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.val)
        queue.extend(node.children)
    return traversal

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(*level_order(nodes[0]))
```



## 4 树的高度

https://sunnywhy.com/sfbj/9/3/348

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的高度。

输入描述

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出

输出一个整数，表示树的高度。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
3
```

解释

对应的树如下图所示，高度为`3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def height(node):
    if node is None:
        return 0
    if not node.children:
        return 1
    return max(height(child) for child in node.children) + 1

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(height(nodes[0]))
```







# 二叉查找树（BST）5题





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







# 平衡二叉树（AVL树）3题







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





# 并查集 5题





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







# 堆 6题





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







# 哈夫曼树 3题





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







