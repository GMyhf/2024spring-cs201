# 晴问编程题目

Updated 2022 GMT+8 March 22, 2024

2024 spring, Complied by Hongfei Yan





题目练习网址，https://sunnywhy.com/sfbj



# 树与二叉树 1题

## 1 树的判定

https://sunnywhy.com/sfbj/9/1

现有一个由个结点连接而成的**连通**结构，已知这个结构中存在的边数，问这个连通结构是否是一棵树。

输入描述

两个整数$n、m（1 \le n \le 100, 0 \le m \le 100）$，分别表示结点数和边数。

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



## 5 树的结点层号

https://sunnywhy.com/sfbj/9/3/349

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的所有结点的层号（假设根结点的层号为`1`）。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出

输出个整数，分别表示编号从`0`到`n-1`的结点的层号，中间用空格隔开，行末不允许有多余的空格。

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
1 3 2 3 2 2 3
```

解释

对应的树如下图所示，层号为`1`的结点编号为`0`，层号为`2`的结点编号为`2`、`4`、`5`，层号为`3`的结点编号为`1`、`6`、`3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
n = int(input().strip())
tree = [[] for _ in range(n)]
levels = [0 for _ in range(n)]

for i in range(n):
    line = list(map(int, input().strip().split()))
    k = line[0]
    for j in range(1, k + 1):
        tree[i].append(line[j])

q = [(0, 1)]
while q:
    node, level = q.pop(0)
    levels[node] = level
    for child in tree[node]:
        q.append((child, level + 1))

print(' '.join(map(str, levels)))
```





```python
class Tree:
    def __init__(self, n):
        self.n = n
        self.tree = [[] for _ in range(n)]
        self.levels = [0 for _ in range(n)]

    def add_node(self, node, children):
        self.tree[node] = children

    def bfs(self):
        q = [(0, 1)]
        while q:
            node, level = q.pop(0)
            self.levels[node] = level
            for child in self.tree[node]:
                q.append((child, level + 1))

    def print_levels(self):
        print(' '.join(map(str, self.levels)))


n = int(input().strip())
tree = Tree(n)

for i in range(n):
    line = list(map(int, input().strip().split()))
    k = line[0]
    children = line[1:k+1]
    tree.add_node(i, children)

tree.bfs()
tree.print_levels()
```



## 6 树的路径和

https://sunnywhy.com/sfbj/9/3/350

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径和是指，从根结点到该结点的路径上所有结点的权值之和；
2. 树的路径和是指，树所有叶结点的路径和之和。

求这棵树的路径和。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值$w (1 \le w \le 100)$，；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出

输出一个整数，表示树的路径和。

样例1

输入

```
7
3 5 1 1 2 4 2
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
28
```

解释

对应的树如下图所示，其中黑色数字为结点编号，编号右下角的灰色数字为结点权值。由此可得叶结点`1`的路径和为，叶结点`6`的路径和为，叶结点`4`的路径和为，叶结点`3`的路径和为，因此二叉树的路径和为。

![树的路径和.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202120669.png)





```python
n = int(input().strip())
weights = list(map(int, input().strip().split()))
tree = [[] for _ in range(n)]

for i in range(n):
    line = list(map(int, input().strip().split()))
    k = line[0]
    for j in range(1, k + 1):
        tree[i].append(line[j])

def dfs(node, path_sum):
    path_sum += weights[node]
    if not tree[node]:  # if the node is a leaf node
        return path_sum
    return sum(dfs(child, path_sum) for child in tree[node])

result = dfs(0, 0)
print(result)
```



## 7 树的带权路径长度

https://sunnywhy.com/sfbj/9/3/351

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径长度是指，从根结点到该结点的边数；
2. 结点的带权路径长度是指，结点权值乘以结点的路径长度；
3. 树的带权路径长度是指，树的所有叶结点的带权路径长度之和。

求这棵树的带权路径长度。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值（）；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

输出描述

输出一个整数，表示树的带权路径长度。

样例1

输入

```
7
3 5 1 1 2 4 2
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
18
```

解释

对应的树如下图所示，其中黑色数字为结点编号，编号右下角的格式为`结点权值*结点路径长度=结点带权路径长度`。由此可得树的带权路径长度为。

![树的带权路径长度.png](https://cdn.sunnywhy.com/202203/2d112f11-2165-4105-ab75-3c0782aa4572.png)



```python
class TreeNode:
    def __init__(self, weight=0):
        self.weight = weight
        self.children = []

def build_tree(weights, edges):
    nodes = [TreeNode(weight=w) for w in weights]
    for i, children in enumerate(edges):
        for child in children:
            nodes[i].children.append(nodes[child])
    return nodes[0]  # 返回根节点

def dfs(node, depth):
    # 如果当前节点是叶子节点，则返回其带权路径长度
    if not node.children:
        return node.weight * depth
    # 否则，递归遍历其子节点
    total_weight_path_length = 0
    for child in node.children:
        total_weight_path_length += dfs(child, depth + 1)
    return total_weight_path_length

def weighted_path_length(n, weights, edges):
    # 构建树
    root = build_tree(weights, edges)
    # 从根节点开始深度优先搜索
    return dfs(root, 0)

# 输入处理
n = int(input().strip())
weights = list(map(int, input().strip().split()))
edges = []
for _ in range(n):
    line = list(map(int, input().strip().split()))
    if line[0] != 0:  # 忽略没有子节点的情况
        edges.append(line[1:])
    else:
        edges.append([])

# 计算带权路径长度
print(weighted_path_length(n, weights, edges))

```









# 二叉查找树（BST）5题

## 1 二叉查找树的建立

https://sunnywhy.com/sfbj/9/4

将n个互不相同的正整数先后插入到一棵空的二叉查找树中，求最后生成的二叉查找树的先序序列。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行 n 个整数 $a_i (1 \le a_i \le 100)$，表示插入序列。

输出

输出个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
5 2 3 6 1 8
```

输出

```
5 2 1 3 6 8
```

解释

插入的过程如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403202151255.png" alt="二叉查找树的建立.png" style="zoom:67%;" />







```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if node is None:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
values = list(map(int, input().strip().split()))
bst = BST()
for value in values:
    bst.insert(value)
print(' '.join(map(str, bst.preorder())))
```



## 2 二叉查找树的判定

https://sunnywhy.com/sfbj/9/4/353

现有一棵二叉树的中序遍历序列，问这棵二叉树是否是二叉查找树。

二叉查找树的定义：在二叉树定义的基础上，满足左子结点的数据域小于或等于根结点的数据域，右子结点的数据域大于根结点的数据域。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行 n 个整数 $a_i (1 \le a_i \le 100)$，表示中序遍历序列。数据保证序列元素互不相同。

输出

如果是二叉查找树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
3
1 2 3
```

输出

```
Yes
```

解释

对应的二叉树如下所示，是二叉查找树。

![二叉查找树的判定.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202221291.png)

样例2

输入

```
3
2 1 3
```

输出

```
No
```

解释

对应的二叉树如下所示，不是二叉查找树。

![二叉查找树的判定_2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202222819.png)





```python
n = int(input().strip())
sequence = list(map(int, input().strip().split()))

if sequence == sorted(sequence):
    print("Yes")
else:
    print("No")
```



## 3 还原二叉查找树

https://sunnywhy.com/sfbj/9/4/354

现有一棵二叉查找树的先序遍历序列，还原这棵二叉查找树，并输出它的后序序列。

输入描述

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行 n 个整数 $a_i (1 \le a_i \le 100)$，表示先序遍历序列。数据保证序列元素互不相同。

输出

输出个整数，表示后序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
5 2 1 3 6 8
```

输出

```
1 3 2 8 6 5
```

解释

对应的二叉查找树如下所示，后序序列为`1 3 2 8 6 5`。

![还原二叉查找树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202231311.png)





```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self, preorder):
        if not preorder:
            self.root = None
        else:
            self.root = self.build(preorder)

    def build(self, preorder):
        if not preorder:
            return None
        root = Node(preorder[0])
        i = 1
        while i < len(preorder) and preorder[i] < root.value:
            i += 1
        root.left = self.build(preorder[1:i])
        root.right = self.build(preorder[i:])
        return root

    def postorder(self):
        return self._postorder(self.root)

    def _postorder(self, node):
        if node is None:
            return []
        return self._postorder(node.left) + self._postorder(node.right) + [node.value]

n = int(input().strip())
preorder = list(map(int, input().strip().split()))
bst = BST(preorder)
print(' '.join(map(str, bst.postorder())))
```



## 4 相同的二叉查找树

https://sunnywhy.com/sfbj/9/4/355

将第一组个互不相同的正整数先后插入到一棵空的二叉查找树中，得到二叉查找树；再将第二组个互不相同的正整数先后插入到一棵空的二叉查找树中，得到二叉查找树。判断和是否是同一棵二叉查找树。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行个 n 个整数 $a_i (1 \le a_i \le 100)$，表示第一组插入序列；

第三行个 n 个整数 $b_i (1 \le b_i \le 100)$，表示第二组插入序列。

输出

如果是同一棵二叉查找树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
6
5 2 3 6 1 8
5 6 8 2 1 3
```

输出

```
Yes
```

解释

两种插入方式均可以得到下面这棵二叉查找树。

![还原二叉查找树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202231311.png)

样例2

输入

```
6
5 2 3 6 1 8
5 6 8 3 1 2
```

输出

```
No
```

解释

两种插入方式分别得到下图的两种二叉查找树。

![相同的二叉查找树_2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202341265.png)



先定义了`TreeNode`类用于表示二叉树的节点，然后定义了`insert_into_bst`函数用于将一个新值插入到二叉查找树中。`build_bst_from_sequence`函数接收一个序列，依次调用`insert_into_bst`来构建出一棵二叉查找树。`is_same_tree`函数用于比较两棵二叉树是否结构相同（即形状相同且对应位置的节点值相等）。

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def insert_into_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    return root

def build_bst_from_sequence(sequence):
    root = None
    for val in sequence:
        root = insert_into_bst(root, val)
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

# 输入处理
n = int(input().strip())
seq1 = list(map(int, input().strip().split()))
seq2 = list(map(int, input().strip().split()))

# 构建二叉查找树
tree1 = build_bst_from_sequence(seq1)
tree2 = build_bst_from_sequence(seq2)

# 判断是否为同一棵树
if is_same_tree(tree1, tree2):
    print("Yes")
else:
    print("No")

```



## 5 填充二叉查找树

https://sunnywhy.com/sfbj/9/4/356

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），将个互不相同的正整数填入这棵二叉树结点的数据域中，使其成为二叉查找树。求填充后二叉查找树的先序序列。

输入

第一行一个整数 $n (1 \le n \le 50)$，表示二叉树的结点个数；

第二行 n 个整数，表示需要填入二叉树中的数 $val_i$, 其中填入数字的范围为 $1 \le val_i \le 100$。

接下来 n 行，每行一个结点，按顺序给出编号为从`0`到`n-1`的个结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

输出

输出 n 个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
1 2 3 5 6 8
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
5 2 1 3 6 8
```

解释

下左图为输入的二叉树，填入`6`个整数后变为下右图的二叉查找树。

![填充二叉查找树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202353104.png)





To solve this problem, you can follow these steps:

1. Read the input values and the structure of the binary tree.
2. Sort the input values in ascending order.
3. Perform an inorder traversal of the binary tree and fill the nodes with the sorted values. This will make the binary tree a binary search tree (BST) because the inorder traversal of a BST is a sorted sequence.
4. Perform a preorder traversal of the BST and print the result.

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self, n):
        self.nodes = [Node() for _ in range(n)]
        self.index = 0

    def insert_values(self, values):
        self.values = iter(sorted(values))
        self._insert_values(self.nodes[0])

    def _insert_values(self, node):
        if node is not None:
            self._insert_values(node.left)
            node.value = next(self.values)
            self._insert_values(node.right)

    def preorder(self):
        return self._preorder(self.nodes[0])

    def _preorder(self, node):
        if node is None:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
values = list(map(int, input().strip().split()))
bst = BST(n)
for i in range(n):
    left, right = map(int, input().strip().split())
    if left != -1:
        bst.nodes[i].left = bst.nodes[left]
    if right != -1:
        bst.nodes[i].right = bst.nodes[right]
bst.insert_values(values)
print(' '.join(map(str, bst.preorder())))
```

This code reads the input values and the structure of the binary tree from the input, sorts the input values, fills the nodes of the binary tree with the sorted values, performs a preorder traversal of the BST, and then prints the result.





# 平衡二叉树（AVL树）3题

## 1 二叉查找树的平衡因子

https://sunnywhy.com/sfbj/9/5

对一棵二叉查找树来说，我们把结点的左子树和右子树的高度之差称为结点的平衡因子。

现将 n 个互不相同的正整数先后插入到一棵空的二叉查找树中，然后按中序遍历的顺序，给出二叉查找树中各结点的平衡因子。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行个整数 $a_i (1 \le a_i \le 100)$，表示插入序列。

**输出**

输出 n 个整数，表示 n 个结点的平衡因子，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
5 2 3 6 1 8 7
```

输出

```
0 0 0 -1 -2 0 1
```

解释

生成的二叉查找树和每个结点的平衡因子计算过程如图所示。

![二叉查找树的平衡因子.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210006174.png)



To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into a binary search tree (BST).
3. Perform an inorder traversal of the BST and calculate the balance factor for each node. The balance factor of a node is the height of its left subtree minus the height of its right subtree.
4. Print the balance factors in the order they were visited during the inorder traversal.

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height

    def balance_factors(self):
        return self._balance_factors(self.root)

    def _balance_factors(self, node):
        if node is None:
            return []
        balance_factor = self._get_height(node.left) - self._get_height(node.right)
        return self._balance_factors(node.left) + [balance_factor] + self._balance_factors(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

bst = BST()
for value in sequence:
    bst.insert(value)

print(' '.join(map(str, bst.balance_factors())))
```

This code reads the sequence from the input, inserts its values into a BST, calculates the balance factors of the nodes during an inorder traversal of the BST, and then prints the results.



## 2 平衡二叉树的判定

https://sunnywhy.com/sfbj/9/5/358

将 n 个互不相同的正整数先后插入到一棵空的二叉查找树中，判断最后生成的二叉查找树是否是平衡二叉树（AVL树）。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行个 n 整数 $a_i (1 \le a_i \le 100)$，表示插入序列。数据保证序列元素互不相同。

**输出**

如果是平衡二叉树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5
5 2 3 6 1
```

输出

```
Yes
```

解释

对应的二叉查找树如下所示，是平衡二叉树。

![平衡二叉树的判定_样例1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210034791.png)

样例2

输入

```
4
5 2 3 1
```

输出

```
No
```

解释

对应的二叉查找树如下所示，不是平衡二叉树。

![平衡二叉树的判定_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210034500.png)



To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into a binary search tree (BST).
3. Check if the BST is balanced. A BST is balanced if the absolute difference between the heights of the left and right subtrees of every node is at most 1.
4. Print "Yes" if the BST is balanced, otherwise print "No".

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height

    def is_balanced(self):
        return self._is_balanced(self.root)

    def _is_balanced(self, node):
        if node is None:
            return True
        balance_factor = abs(self._get_height(node.left) - self._get_height(node.right))
        return balance_factor <= 1 and self._is_balanced(node.left) and self._is_balanced(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

bst = BST()
for value in sequence:
    bst.insert(value)

if bst.is_balanced():
    print("Yes")
else:
    print("No")
```

This code reads the sequence from the input, inserts its values into a BST, checks if the BST is balanced, and then prints the result.





## 3 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359

将 n 个互不相同的正整数先后插入到一棵空的AVL树中，求最后生成的AVL树的先序序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示AVL树的结点个数；

第二行 n 个整数$a_i (1 \le a_i \le 100)$，表示表示插入序列。

**输出**

输出 n 个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
5
5 2 3 6 8
```

输出

```
3 2 6 5 8
```

解释

插入的过程如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210041932.png" alt="平衡二叉树的建立.png" style="zoom:67%;" />



To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into an AVL tree. An AVL tree is a self-balancing binary search tree, and the heights of the two child subtrees of any node differ by at most one.
3. Perform a preorder traversal of the AVL tree and print the result.

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)

    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```

This code reads the sequence from the input, inserts its values into an AVL tree, performs a preorder traversal of the AVL tree, and then prints the result.







# 并查集 5题

## 1 学校的班级个数（1）

https://sunnywhy.com/sfbj/9/6/360

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），并给出 m 组学生关系（指定两个学生处于一个班级），问总共有多少个班级。

**输入**

第一行两个整数 $m、n (1 \le n \le 100, 1 \le m \le 100)$，分别表示学生个数、学生关系个数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n, a \ne b)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

**输出**

输出一个整数，表示班级个数。

样例1

输入

```
5 3
4 2
1 3
2 5
```

输出

```
2
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此共有两个班级。



To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. The number of disjoint sets at the end will be the number of classes.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. For each student relationship, unite their sets using the union operation.
3. Count the number of unique parents. This will be the number of classes.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x: # 如果不是根结点，继续循环
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))	# parent[i] == i，则说明元素i是该集合的根结点

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

classes = set(find(x) for x in range(1, n + 1))
print(len(classes))
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, and then counts and prints the number of unique classes.





## 2 学校的班级人数（2）

https://sunnywhy.com/sfbj/9/6/361

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），并给出 m 组学生关系（指定两个学生处于一个班级），问总共有多少个班级，并按降序给出每个班级的人数。

输入

第一行两个整数 $m、n (1 \le n \le 100, 1 \le m \le 100)$，分别表示学生个数、学生关系个数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n, a \ne b)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

输出

第一行输出一个整数，表示班级个数；

第二行若干个整数，按降序给出每个班级的人数。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
5 3
4 2
1 3
2 5
```

输出

```
2
3 2
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此共有两个班级，人数分别是`3`和`2`。





To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. The number of disjoint sets at the end will be the number of classes. You can also maintain a count of the number of students in each class.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. Initialize an array `size` of size `n+1` where `size[i]` is initially `1`. This means each class initially has one student.
3. For each student relationship, unite their sets using the union operation and update the size of the new set.
4. Count the number of unique parents. This will be the number of classes.
5. Print the sizes of the classes in descending order.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_x] = root_y
        size[root_y] += size[root_x]

n, m = map(int, input().split())
parent = list(range(n + 1))
size = [1] * (n + 1)

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

#classes = [size[find(x)] for x in range(1, n + 1) if x == parent[x]]
classes = [size[x] for x in range(1, n + 1) if x == parent[x]]
print(len(classes))
print(' '.join(map(str, sorted(classes, reverse=True))))
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, counts and prints the number of unique classes, and prints the sizes of the classes in descending order.





## 3 是否相同班级

https://sunnywhy.com/sfbj/9/6/362

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），并给出 m 组学生关系（指定两个学生处于一个班级）。然后给出 k 个查询，每个查询询问两个学生是否在同一个班级。

**输入**

第一行两个整数 $n、m (1 \le n \le 10^5, 1 \le m \le 10^5)$，分别表示学生个数、学生关系个数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n, a \ne b)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

然后一个整数 $k (1 \le k \le 10^3)$，表示查询个数；

接下来 k 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n)$，表示询问编号为`a`的学生和编号为`b`的学生是否在同一个班级。

**输出**

每个查询输出一行，如果在同一个班级，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5 3
4 2
1 3
2 5
2
4 5
1 2
```

输出

```
Yes
No
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此编号`4`和`5`的学生在同一个班级，编号`1`和`2`的学生不在同一个班级。





To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. For each query, you check if the two students are in the same set.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. For each student relationship, unite their sets using the union operation.
3. For each query, check if the two students are in the same set using the find operation.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

k = int(input())
for _ in range(k):
    a, b = map(int, input().split())
    if find(a) == find(b):
        print('Yes')
    else:
        print('No')
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, and then for each query, checks and prints whether the two students are in the same class.



## 4 迷宫连通性

https://sunnywhy.com/sfbj/9/6/363

现有一个迷宫，迷宫中有 n 个房间（编号为从`1`到`n`），房间与房间之间可能连通。如果房间`A`和房间`B`连通，房间`B`和房间`C`连通，那么我们称房间`A`和房间`C`也连通。给定 m 组连通关系（指定两个房间连通），问迷宫中的所有房间是否连通。

**输入**

第一行两个整数$n、m (1 \le n \le 100, 1 \le m \le 100)$，分别表示房间个数、连通关系个数；

接下来行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n)$，表示编号为`a`的房间和编号为`b`的房间是连通的。

**输出**

如果所有房间连通，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5 4
4 2
1 3
2 5
1 5
```

输出

```
Yes
```

解释

所有房间都连通，因此输出`Yes`。

样例2

输入

```
5 3
4 2
1 3
2 5
```

输出

```
No
```

解释

编号`2`、`4`、`5`的房间互相连通，编号`1`、`3`的房间互相连通，因此没有全部互相连通，输出`No`。



To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each room is initially in its own set. When a connection between two rooms is given, you unite their sets. If at the end there is only one set, then all rooms are connected.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each room is in its own set.
2. For each connection, unite their sets using the union operation.
3. Check if all rooms are in the same set.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

sets = set(find(x) for x in range(1, n + 1))
if len(sets) == 1:
    print('Yes')
else:
    print('No')
```

This code reads the number of rooms and connections from the input, initializes each room to be in its own set, unites the sets of each pair of rooms that have a connection, and then checks and prints whether all rooms are in the same set.





## 5 班级最高分

https://sunnywhy.com/sfbj/9/6/364

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），每个学生有一个考试分数，再给出 m 组学生关系（指定两个学生处于一个班级），问总共有多少个班级，并按降序给出每个班级的最高考试分数。

**输入**

第一行两个整数 $n、m (1 \le n \le 100, 1 \le m \le 100)$，分别表示学生个数、学生关系个数；

第二行为用空格隔开的 n 个整数（$0 \le 每个整数 \le 100$），表示个学生的考试分数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

**输出**

第一行输出一个整数，表示班级个数；

第二行若干个整数，按降序给出每个班级的最高考试分数。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入



```
5 3
88 90 86 92 95
4 2
1 3
2 5
```

输出

```
2
95 88
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此共有两个班级，最高分数分别是编号`1`的学生的`88`分、编号`5`的学生的`95`分。



To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. The number of disjoint sets at the end will be the number of classes. You can also maintain a list of scores for each class and find the maximum score for each class.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. Initialize an array `scores` of size `n+1` to store the scores of each student.
3. For each student relationship, unite their sets using the union operation.
4. For each class, find the maximum score.
5. Count the number of unique parents. This will be the number of classes.
6. Print the maximum scores of the classes in descending order.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_x] = root_y
        scores[root_y] = max(scores[root_y], scores[root_x])

n, m = map(int, input().split())
parent = list(range(n + 1))
scores = list(map(int, input().split()))
scores.insert(0, 0)  # to make the scores 1-indexed

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

classes_scores = [scores[find(x)] for x in range(1, n + 1) if x == parent[x]]
print(len(classes_scores))
print(' '.join(map(str, sorted(classes_scores, reverse=True))))
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, finds the maximum score for each class, counts and prints the number of unique classes, and prints the maximum scores of the classes in descending order.





# 堆 6题

## 1 向下调整构建大顶堆

https://sunnywhy.com/sfbj/9/7

现有个不同的正整数，将它们按层序生成完全二叉树，然后使用**向下调整**的方式构建一个完整的大顶堆。最后按层序输出堆中的所有元素。

**输入**

第一行一个整数$n (1 \le n \le 10^3)$，表示正整数的个数；

第二行 n 个整数$a_i (1 \le a_i \le 10^4) $​，表示正整数序列。

**输出**

输出 n 个整数，表示堆的层序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
3 2 6 5 8 7
```

输出

```
8 5 7 3 2 6
```

解释

调整前的完全二叉树和调整后的堆如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210116556.png" alt="向下调整构建大顶堆.png" style="zoom:67%;" />



解法1:

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            #print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        #print(f'i = {i}, {self.heapList}')


n = int(input().strip())
heap = list(map(int, input().strip().split())) # [9, 5, 6, 2, 3]
heap = [-x for x in heap]

bh = BinHeap()
bh.buildHeap(heap)
ans = [-x for x in bh.heapList[1:]]
print(*ans)
```



解法2:

To solve this problem, you can use the concept of a binary heap. A binary heap is a complete binary tree where each node is either greater than or equal to its children (in a max heap) or less than or equal to its children (in a min heap). In this case, you are asked to create a max heap.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each non-leaf node in the `heap` array (starting from the last non-leaf node and moving to the root), perform a downward adjustment to ensure the max heap property is maintained.
4. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def down_adjust(parent_index, length):
    temp = heap[parent_index]
    child_index = 2 * parent_index + 1
    while child_index < length:
        if child_index + 1 < length and heap[child_index + 1] > heap[child_index]:
            child_index += 1
        if temp > heap[child_index]:
            break
        heap[parent_index] = heap[child_index]
        parent_index = child_index
        child_index = 2 * child_index + 1
    heap[parent_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range((n - 2) // 2, -1, -1):
    down_adjust(i, n)

print(' '.join(map(str, heap)))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs a downward adjustment for each non-leaf node in the `heap` array to ensure the max heap property is maintained, and then prints the elements of the `heap` array in order.





## 2 向上调整构建大顶堆

https://sunnywhy.com/sfbj/9/7/366

现有 n 个不同的正整数，将它们按层序生成完全二叉树，然后使用**向上调整**的方式构建一个完整的大顶堆。最后按层序输出堆中的所有元素。

输入

第一行一个整数 $n (1 \le n \le 10^3)$，表示正整数的个数；

第二行 n 个整数$a_i (1 \le a_i \le 10^4) $​，表示正整数序列。

输出

输出 n 个整数，表示堆的层序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
3 2 6 5 8 7
```

输出

```
8 6 7 2 5 3
```

解释

调整前的完全二叉树和调整后的堆如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210120258.png" alt="向上调整构建大顶堆.png" style="zoom: 67%;" />





To solve this problem, you can use the concept of a binary heap. A binary heap is a complete binary tree where each node is either greater than or equal to its children (in a max heap) or less than or equal to its children (in a min heap). In this case, you are asked to create a max heap.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each inserted node in the `heap` array (starting from the last inserted node and moving to the root), perform an upward adjustment to ensure the max heap property is maintained.
4. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def up_adjust(child_index):
    temp = heap[child_index]
    parent_index = (child_index - 1) // 2
    while child_index > 0 and temp > heap[parent_index]:
        heap[child_index] = heap[parent_index]
        child_index = parent_index
        parent_index = (parent_index - 1) // 2
    heap[child_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range(1, n):
    up_adjust(i)

print(' '.join(map(str, heap)))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs an upward adjustment for each inserted node in the `heap` array to ensure the max heap property is maintained, and then prints the elements of the `heap` array in order.







## 3 删除堆顶元素

https://sunnywhy.com/sfbj/9/7/367

现有 n 个不同的正整数，将它们按层序生成完全二叉树，然后使用**向下调整**的方式构建一个完整的大顶堆。然后删除堆顶元素，并将层序最后一个元素置于堆顶，进行一次向下调整，以形成新的堆。最后按层序输出新堆中的所有元素。

**输入**

第一行一个整数 $n (1 \le n \le 10^3)$，表示正整数的个数；

第二行 n 个整数 $a_i (1 \le a_i \le 10^4) $​，表示正整数序列。

**输出**

输出 n - 1 个整数，表示堆的层序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
3 2 6 5 8 7
```

输出

```
7 5 6 3 2
```

解释

操作过程如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210124838.png" alt="删除堆顶元素.png" style="zoom:67%;" />



解法1:

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            #print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        #print(f'i = {i}, {self.heapList}')


n = int(input().strip())
heap = list(map(int, input().strip().split())) # [9, 5, 6, 2, 3]
heap = [-x for x in heap]

bh = BinHeap()
bh.buildHeap(heap)

bh.delMin()

ans = [-x for x in bh.heapList[1:]]
print(*ans)
```



解法2:

To solve this problem, you can use the concept of a binary heap. A binary heap is a complete binary tree where each node is either greater than or equal to its children (in a max heap) or less than or equal to its children (in a min heap). In this case, you are asked to create a max heap, delete the root, and then adjust the heap.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each non-leaf node in the `heap` array (starting from the last non-leaf node and moving to the root), perform a downward adjustment to ensure the max heap property is maintained.
4. Delete the root of the heap and replace it with the last element in the heap.
5. Perform a downward adjustment on the new root to maintain the max heap property.
6. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def down_adjust(parent_index, length):
    temp = heap[parent_index]
    child_index = 2 * parent_index + 1
    while child_index < length:
        if child_index + 1 < length and heap[child_index + 1] > heap[child_index]:
            child_index += 1
        if temp > heap[child_index]:
            break
        heap[parent_index] = heap[child_index]
        parent_index = child_index
        child_index = 2 * child_index + 1
    heap[parent_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range((n - 2) // 2, -1, -1):
    down_adjust(i, n)

heap[0] = heap[n - 1]
down_adjust(0, n - 1)

print(' '.join(map(str, heap[:-1])))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs a downward adjustment for each non-leaf node in the `heap` array to ensure the max heap property is maintained, deletes the root of the heap and replaces it with the last element in the heap, performs a downward adjustment on the new root to maintain the max heap property, and then prints the elements of the `heap` array in order.





## 4 堆排序

https://sunnywhy.com/sfbj/9/7/368

输入 n 个正整数，使用堆排序算法将它们按从小到大的顺序进行排序。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示正整数的个数；

第二行为用空格隔开的 n 个正整数（每个正整数均不超过`100`）。

**输出**

输出一行，表示排序后的 n 个正整数。整数间用一个空格隔开，行末不允许有多余的空格。

样例1

输入

```
5
2 8 5 1 3
```

输出

```
1 2 3 5 8
```

解释

从小到大排序后可以得到序列`1 2 3 5 8`



解法1:

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            #print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        #print(f'i = {i}, {self.heapList}')


n = int(input().strip())
heap = list(map(int, input().strip().split())) # [9, 5, 6, 2, 3]

bh = BinHeap()
bh.buildHeap(heap)

ans = []
while bh.currentSize > 0:
    ans.append(bh.delMin())
print(*ans)
```



解法2:

To solve this problem, you can use the heap sort algorithm. Heap sort is a comparison-based sorting algorithm that uses a binary heap data structure. It works by building a max heap from the input data, then iteratively removing the maximum element from the heap and inserting it at the end of the sorted section of the array.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each non-leaf node in the `heap` array (starting from the last non-leaf node and moving to the root), perform a downward adjustment to ensure the max heap property is maintained.
4. Repeatedly swap the first element in the heap with the last element in the unsorted section of the array, then perform a downward adjustment on the new root to maintain the max heap property.
5. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def down_adjust(parent_index, length):
    temp = heap[parent_index]
    child_index = 2 * parent_index + 1
    while child_index < length:
        if child_index + 1 < length and heap[child_index + 1] > heap[child_index]:
            child_index += 1
        if temp > heap[child_index]:
            break
        heap[parent_index] = heap[child_index]
        parent_index = child_index
        child_index = 2 * child_index + 1
    heap[parent_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range((n - 2) // 2, -1, -1):
    down_adjust(i, n)

for i in range(n - 1, 0, -1):
    heap[i], heap[0] = heap[0], heap[i]
    down_adjust(0, i)

print(' '.join(map(str, heap)))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs a downward adjustment for each non-leaf node in the `heap` array to ensure the max heap property is maintained, repeatedly swaps the first element in the heap with the last element in the unsorted section of the array and performs a downward adjustment on the new root, and then prints the elements of the `heap` array in order.





## 5 数据流第K大元素

https://sunnywhy.com/sfbj/9/7/369

现有一个初始为空的序列 S，对其执行 n 个操作，每个操作是以下两种操作之一：

1. 往序列 S 中加入一个正整数；
2. 输出当前序列 S 中第大的数。

其中，第大是指将序列从大到小排序后的第 k 个数。

**输入**

第一行两个整数$n、k (1 \le n \le 10^4, 2 \le k \le 100)$，分别表示操作个数、需要输出第几大的数；

接下来 n 行，每行一个操作信息：使用"`Push x`"来表示往序列中加入正整数 $x (1 \le x \le 10^9)$，使用"`Print`"来表示需要输出当前序列中第大的数。

**输出**

每次执行`Print`操作时，输出一行，表示当前序列中第大的数。如果不存在第大的数，那么输出`-1`。

样例1

输入

```
7 2
Push 1
Print
Push 3
Print
Push 7
Push 6
Print
```

输出

```
-1
1
6
```

解释

第一个`Print`时序列中元素为`1`，不存在第`2`大的元素，因此输出`-1`；

第二个`Print`时序列中元素为`1 3`，因此第`2`大的元素为`1`；

第三个`Print`时序列中元素为`1 3 7 6`，因此第`2`大的元素为`6`。





To solve this problem, you can use a priority queue data structure. A priority queue can efficiently insert elements and retrieve the maximum element. In Python, you can use the `heapq` module to implement a priority queue. However, Python's `heapq` module only provides a min-heap, so you need to insert the negative of the numbers to simulate a max-heap.

Here is a step-by-step plan:

1. Initialize an empty list `heap` to store the elements of the heap.
2. For each operation:
   - If the operation is "Push x", insert `-x` into the `heap`.
   - If the operation is "Print", if the size of the `heap` is less than `k`, print `-1`. Otherwise, create a copy of the `heap`, pop `k` elements from the copy, and print the negative of the last popped element.

Here is the Python code that implements this plan:

```python
import heapq

n, k = map(int, input().split())
heap = []

for _ in range(n):
    operation = input().split()
    if operation[0] == "Push":
        heapq.heappush(heap, -int(operation[1]))
    else:  # operation[0] == "Print"
        if len(heap) < k:
            print(-1)
        else:
            temp_heap = heap.copy()
            for _ in range(k):
                result = heapq.heappop(temp_heap)
            print(-result)
```

This code reads the number of operations and the value of `k` from the input, then for each operation, if the operation is "Push x", it inserts `-x` into the `heap`, and if the operation is "Print", it checks if the size of the `heap` is less than `k`, if so, it prints `-1`, otherwise, it creates a copy of the `heap`, pops `k` elements from the copy, and prints the negative of the last popped element.





## 6 数据流中位数

https://sunnywhy.com/sfbj/9/7/370

现有一个初始为空的序列 S，对其执行 n 个操作，每个操作是以下两种操作之一：

1. 往序列 S 中加入一个正整数 x ；
2. 输出当前序列 S 的中位数。

注：序列的中位数是指，将这个序列从小到大排序后最中间的那个元素；如果最中间有两个元素，那么取这两个元素的平均数作为序列的中位数。

**输入**

第一行一个整数 $n (2 \le n \le 10^4)$，表示操作个数；

接下来行，每行一个操作信息：使用"`Push x`"来表示往序列中加入正整数$x (1 \le x \le 10^5)$，使用"`Print`"来表示需要输出当前序列的中位数。

数据保证不会在序列为空时进行`Print`操作。

**输出**

每次执行`Print`操作时，输出一行，表示当前序列的中位数。结果保留一位小数。

样例1

输入

```
6
Push 3
Push 7
Push 6
Print
Push 1
Print
```

输出

```
6.0
4.5
```

解释

第一个`Print`时序列中元素为`3 7 6`，因此中位数是`6`；

第二个`Print`时序列中元素为`3 7 6 1`，因此中位数是。



To solve this problem, you can use two heaps: a max heap to store the smaller half of the numbers, and a min heap to store the larger half. The median is then either the maximum element in the max heap (when the total number of elements is odd) or the average of the maximum element in the max heap and the minimum element in the min heap (when the total number of elements is even).

Here is a step-by-step plan:

1. Initialize an empty max heap `left` and an empty min heap `right`.
2. For each operation:
   - If the operation is "Push x", insert `x` into the appropriate heap. If the size of the heaps differ by more than 1 after the insertion, balance the heaps by moving the top element from the heap with more elements to the heap with fewer elements.
   - If the operation is "Print", print the median. The median is the top element of `left` if the total number of elements is odd, or the average of the top elements of `left` and `right` if the total number of elements is even.

Here is the Python code that implements this plan:

```python
import heapq

n = int(input().strip())
left, right = [], []

for _ in range(n):
    operation = input().split()
    if operation[0] == "Push":
        x = int(operation[1])
        if not left or x <= -left[0]:
            heapq.heappush(left, -x)
        else:
            heapq.heappush(right, x)
        if len(left) < len(right):
            heapq.heappush(left, -heapq.heappop(right))
        elif len(left) > len(right) + 1:
            heapq.heappush(right, -heapq.heappop(left))
    else:  # operation[0] == "Print"
        if len(left) > len(right):
            print(f"{-left[0]:.1f}")
        else:
            print(f"{(-left[0] + right[0]) / 2:.1f}")
```

This code reads the number of operations from the input, then for each operation, if the operation is "Push x", it inserts `x` into the appropriate heap and balances the heaps if necessary, and if the operation is "Print", it prints the median.





# 哈夫曼树 3题

## 1 合并果子

https://sunnywhy.com/sfbj/9/8

有 n 堆果子，每堆果子的质量已知，现在需要把这些果子合并成一堆，但是每次只能把两堆果子合并到一起，同时会消耗与两堆果子质量之和等值的体力。显然，在进行 n - 1 次合并之后，就只剩下一堆了。为了尽可能节省体力，需要使耗费的总体力最小。求需要耗费的最小总体力。

**输入**

第一行一个整数$n (1 \le n \le 100)$，表示果子的堆数；

第二行为用空格隔开的 n 个正整数（每个正整数均不超过`100`），表示每堆果子的质量。

**输出**

输出一个整数，表示需要耗费的最小总体力。

样例1

输入

```
3
1 2 9
```

输出

```
15
```

解释

先将质量为`1`的果堆和质量为`2`的果堆合并，得到质量为`3`的果堆，同时消耗体力值`3`；

接着将质量为`3`的果堆和质量为`9`的果堆合并，得到质量为`12`的果堆，同时消耗体力值`12`；

因此共消耗体力值。



To solve this problem, you can use a priority queue data structure. A priority queue can efficiently insert elements and retrieve the minimum element. In Python, you can use the `heapq` module to implement a priority queue.

Here is a step-by-step plan:

1. Initialize an empty min heap `heap`.
2. For each pile of fruits, insert its weight into the `heap`.
3. While there is more than one pile of fruits, remove the two piles with the smallest weights from the `heap`, add their weights together, add the result to the total energy consumption, and insert the result back into the `heap`.
4. Print the total energy consumption.

Here is the Python code that implements this plan:

```python
import heapq

n = int(input().strip())
heap = list(map(int, input().strip().split()))
heapq.heapify(heap)

energy = 0
while len(heap) > 1:
    a = heapq.heappop(heap)
    b = heapq.heappop(heap)
    energy += a + b
    heapq.heappush(heap, a + b)

print(energy)
```

This code reads the number of piles of fruits and the weights of the piles from the input, inserts each weight into the `heap`, while there is more than one pile of fruits, removes the two piles with the smallest weights from the `heap`, adds their weights together, adds the result to the total energy consumption, and inserts the result back into the `heap`, and then prints the total energy consumption.



## 2 树的最小带权路径长度

https://sunnywhy.com/sfbj/9/8/372

对一棵 n 个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点，每个结点有各自的权值）来说：

1. 结点的路径长度是指，从根结点到该结点的边数；
2. 结点的带权路径长度是指，结点权值乘以结点的路径长度；
3. 树的带权路径长度是指，树的所有叶结点的带权路径长度之和。

现有 n 个不同的正整数，需要寻找一棵树，使得树的所有叶子结点的权值恰好为这 n 个数，并且使得这棵树的带权路径长度是所有可能的树中最小的。求最小带权路径长度。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示正整数的个数；

第二行为用空格隔开的 n 个正整数（每个正整数均不超过`100`），含义如题意所示。

**输出**

输出一个整数，表示最小带权路径长度。

样例1

输入

```
3
1 2 9
```

输出

```
15
```

解释

对应最小带权路径长度的树如下图所示，其中黑色数字为结点编号，编号右下角的格式为`结点权值*结点路径长度=结点带权路径长度`。由此可得树的带权路径长度为，是所有可能的树中最小的。

![树的最小带权路径长度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210139224.png)



To solve this problem, you can use a priority queue data structure. A priority queue can efficiently insert elements and retrieve the minimum element. In Python, you can use the `heapq` module to implement a priority queue.

Here is a step-by-step plan:

1. Initialize an empty min heap `heap`.
2. For each weight, insert it into the `heap`.
3. While there is more than one weight in the `heap`, remove the two weights with the smallest values from the `heap`, add them together, add the result to the total weighted path length, and insert the result back into the `heap`.
4. Print the total weighted path length.

Here is the Python code that implements this plan:

```python
import heapq

n = int(input().strip())
heap = list(map(int, input().strip().split()))
heapq.heapify(heap)

weighted_path_length = 0
while len(heap) > 1:
    a = heapq.heappop(heap)
    b = heapq.heappop(heap)
    weighted_path_length += a + b
    heapq.heappush(heap, a + b)

print(weighted_path_length)
```

This code reads the number of weights from the input, inserts each weight into the `heap`, while there is more than one weight in the `heap`, removes the two weights with the smallest values from the `heap`, adds them together, adds the result to the total weighted path length, and inserts the result back into the `heap`, and then prints the total weighted path length.



## 3 最小前缀编码长度

https://sunnywhy.com/sfbj/9/8/373

现需要将一个字符串 s 使用**前缀编码**的方式编码为 01 串，使得解码时不会产生混淆。求编码出的 01 串的最小长度。

**输入**

一个仅由大写字母组成、长度不超过的字符串。

**输出**

输出一个整数，表示最小长度。

样例1

输入

```
ABBC
```

输出

```
6
```

解释

将`A`编码为`00`，`B`编码为`1`，`C`编码为`01`，可以得到`ABBC`的前缀编码串`001101`，此时达到了所有可能情况中的最小长度`6`。



解法1:

使用一种基于哈夫曼编码的方法。哈夫曼编码是一种用于无损数据压缩的最优前缀编码方法。简单来说，它通过创建一棵二叉树，其中每个叶节点代表一个字符，每个节点的路径长度（从根到叶）代表该字符编码的长度，来生成最短的编码。字符出现的频率越高，其在树中的路径就越短，这样可以保证整个编码的总长度最小。

首先需要统计输入字符串中每个字符的出现频率。然后，根据这些频率构建哈夫曼树。构建完成后，遍历这棵树以确定每个字符的编码长度。最后，将所有字符的编码长度乘以其出现次数，累加起来，就得到了编码后的字符串的最小长度。

```python
from collections import Counter
import heapq

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    # 为了让节点可以在优先队列中被比较，定义比较方法
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    priority_queue = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)
    
    return priority_queue[0]

def calculate_huffman_code_lengths(node, length=0):
    if node is None:
        return {}
    if node.char is not None:
        return {node.char: length}
    left_lengths = calculate_huffman_code_lengths(node.left, length + 1)
    right_lengths = calculate_huffman_code_lengths(node.right, length + 1)
    left_lengths.update(right_lengths)
    return left_lengths

def min_encoding_length(s):
    frequencies = Counter(s)
    huffman_tree = build_huffman_tree(frequencies)
    code_lengths = calculate_huffman_code_lengths(huffman_tree)
    return sum(frequencies[char] * length for char, length in code_lengths.items())


s = input().strip()
print(min_encoding_length(s)) 
```





解法2:

To solve this problem, you can use Huffman coding, which is a common algorithm for constructing prefix codes. Huffman coding is a greedy algorithm that uses a priority queue where the node with the lowest probability is given the highest priority.

Here is a step-by-step plan:

1. Count the frequency of each character in the string.
2. Create a priority queue (min heap) where each node contains a character and its frequency. The priority of a node is determined by its frequency.
3. While there is more than one node in the queue:
   - Dequeue the two nodes with the highest priority (lowest frequency).
   - Create a new node with these two nodes as children and with frequency equal to the sum of the two nodes' frequencies.
   - Enqueue the new node.
4. The remaining node is the root of the Huffman tree. Traverse the Huffman tree and assign a binary code to each character. The binary code of a character is the path from the root to the character, where going to the left child adds a '0' and going to the right child adds a '1'.
5. Replace each character in the string with its binary code. The length of the resulting string is the minimum length of the prefix encoding.

Here is the Python code that implements this plan:

```python
import heapq
from collections import Counter

def huffman_encoding(s):
    # Count the frequency of each character
    freq = Counter(s)

    # Create a priority queue with single node trees
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    # Compute the length of the encoded string
    length = sum(freq[char] * len(code) for char, code in huff)

    return length

s = input().strip()
print(huffman_encoding(s))
```

This code reads a string from the input, counts the frequency of each character, creates a priority queue with single node trees, while there is more than one node in the queue, dequeues the two nodes with the highest priority (lowest frequency), creates a new node with these two nodes as children and with frequency equal to the sum of the two nodes' frequencies, enqueues the new node, the remaining node is the root of the Huffman tree, traverses the Huffman tree and assigns a binary code to each character, replaces each character in the string with its binary code, and then prints the length of the resulting string.

