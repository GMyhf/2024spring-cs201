# Cheeting Sheet

2024/6/5 王昊21光华

# 基础语法

## 字符串

```python
text.upper() # 变全大写
text.lower() # 变全小写
text.capitalize() # 首字母大写
text.title() # 单个字母大写
text.swapcase() # 大小写转换
s.isdigit() # 检查字符串中是否只包含数字字符
s.isnumeric() # 判断是否为数字（包含汉字、阿拉伯数字等）更广泛
s.isalpha()：# 判断是否是字母
s.isupper()/islower()：# 判断大小写
```

```python
list.index(element, start, end)
list(zip(a, b)) # a, b两列表，[1, 2, 4]; [1, 3, 4]=>[[1, 1], [2, 3], [4, 4]]
str.find()
str.index()
dict.get(key, default=None)
dict.setdefault(key, default=None)
union_set = set1 | set2 # 并集
intersection_set = set1 & set2 # 交集
difference_set = set1 - set2 # 差集
symmetric_difference_set = set1 ^ set2 # 对称差集
```

1. `str.lstrip() / str.rstrip()`: 移除字符串左侧/右侧的空白字符。
2. `str.find(sub)`: 返回子字符串`sub`在字符串中首次出现的索引，如果未找到，则返回-1。
3. `str.replace(old, new)`: 将字符串中的`old`子字符串替换为`new`。
4. `str.startswith(prefix) / str.endswith(suffix)`: 检查字符串是否以`prefix`开头或以`suffix`结尾。
5. `str.isalpha() / str.isdigit() / str.isalnum()`: 检查字符串是否全部由字母/数字/字母和数字组成。

## 工具

```python
# pylint: skip-file
import heapq
from collections import defaultdict
from collections import dequeue
import bisect
from functools import lru_cache
@lru_cache(maxsize=None)
import sys
sys.setrecursionlimit(1<<32)
import math
math.ceil()  # 函数进行向上取整
math.floor() # 函数进行向下取整。
math.isqrt() # 开方取整
exit()
import calendar
```

calendar：日历

1. `calendar.month(年, 月)`: 返回一个月份的日历字符串。它接受年份和月份作为参数，并以多行字符串的形式返回该月份的日历。
2. `calendar.calendar(年)`: 返回一个年份的日历字符串。这个函数生成整个年份的日历，格式化为多行字符串。
3. `calendar.monthrange(年, 月)`: 返回两个整数，第一个是该月第一天是周几（0-6表示周一到周日），第二个是该月的天数。
4. `calendar.weekday(年, 月, 日)`: 返回给定日期是星期几。0-6的返回值分别代表星期一到星期日。
5. `calendar.isleap(年)`: 返回一个布尔值，指示指定的年份是否是闰年。
6. `calendar.leapdays(年1, 年2)`: 返回在指定范围内的闰年数量，不包括第二个年份。
7. `calendar.monthcalendar(年, 月)`: 返回一个整数矩阵，表示指定月份的日历。每个子列表表示一个星期；天数为0表示该月份此天不在该星期内。
8. `calendar.setfirstweekday(星期)`: 设置日历每周的起始日。默认情况下，第一天是星期一，但可以通过这个函数更改。
9. `calendar.firstweekday()`: 返回当前设置的每周起始日。

counter：计数

```python
from collections import Counter
a = ['red', 'blue', 'red', 'green', 'blue', 'blue']
a = Counter(a)
```

permutations：全排列

```python
from itertools import permutations as per
elements = [1, 2, 3]
permutations = list(per(elements))
```

combinations：组合

```python
from itertools import combinations as com
elements = ['A', 'B', 'C', 'D']
# 生成所有长度为2的组合
combinations = list(com(elements, 2))
```

bisect

```python
import bisect
# bisect.bisect_left(a, x, lo=0, hi=len(a))
insert_index = bisect.bisect_left(sorted_list, 4)
bisect.insort_left(sorted_list, 4)
```

reduce：累积

```python
import functools
numbers = [1, 2, 3, 4, 5]
# 使用 reduce 计算累积乘积
product = functools.reduce(lambda x, y: x * y, numbers)
```

product：笛卡尔积

```python
from itertools import product
colors = ['red', 'blue']
numbers = [1, 2]
# 生成它们的笛卡尔积
cartesian_product = list(product(colors, numbers))
# [('red', 1), ('red', 2), ('blue', 1), ('blue', 2)]
# 生成它们的重复笛卡尔积
repeat_cartesian_product = list(product(colors, repeat=2))
# [('red', 'red'), ('red', 'blue'), ('blue', 'red'), ('blue', 'blue')]
```

## 转换

```python
b = bin(item)  # 2进制
o = oct(item)  # 8进制
h = hex(item)  # 16进制
```

```python
ord(char) -> ASCII_value
chr(ascii_value) -> char
```

```python
print("%.6f" % x)
print("{:.6f}".format(result))
# 当输出内容很多时：
print('\n'.join(map(str, ans)))
```



# 数据结构与算法

## 排序

逆序对

```python
def ReversePairs(arr):
    def sort(arr, l, r, temp):
        res = 0
        if l < r:
            mid = l + (r - l) // 2
            res += sort(arr, l, mid, temp)
            res += sort(arr, mid + 1, r, temp)
            if arr[mid] > arr[mid + 1]:
                res += merge(arr, l, mid, r, temp)
        return res
    def merge(arr, l, mid, r, temp):
        temp[l:r + 1] = arr[l:r + 1]
        i = l
        j = mid + 1
        res = 0
        for k in range(l, r + 1):
            if i > mid:
                arr[k] = temp[j]
                j += 1
            elif j > r:
                arr[k] = temp[i]
                i += 1
            elif temp[i] <= temp[j]:
                arr[k] = temp[i]
                i += 1
            else:
                arr[k] = temp[j]
                j += 1
                res += mid - i + 1
        return res
    temp = arr * 1
    return sort(arr, 0, len(arr) - 1, temp)
```

归并

```python
def merge_sort(lst):
    l = len(lst)
    if l <= 1:
        return lst,0
    middle = l // 2
    left = lst[:middle]
    right = lst[middle:]
    merged_left, left_inv = merge_sort(left)
    merged_right, right_inv = merge_sort(right)
    merged, merge_inv = merge(merged_left, merged_right)
    return merged, merge_inv + left_inv + right_inv
def merge(left,right):
    i = j = 0
    merge_inv = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            merge_inv += len(left) - i
    merged += left[i:]
    merged += right[j:]
    return merged, merge_inv
```

## 滑动窗口

最小覆盖字串

```python
def minWindow(self, s: str, t: str) -> str:
    ans_left, ans_right = -1, len(s)
    left = 0
    cnt_s = Counter()
    cnt_t = Counter(t)
    for right, c in enumerate(s):
        cnt_s[c] += 1
        while cnt_s >= cnt_t:
            if right - left < ans_right - ans_left:
                ans_left, ans_right = left, right
            cnt_s[s[left]] -= 1
            left += 1
    return "" if ans_left < 0 else s[ans_left: ans_right + 1]
```

## 栈

中序转后序

```python
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
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
```

动态中位数

```python
for _ in range(int(input())):
    q1, q2, cur, cnt = [], [], [], 0
    for i in list(map(int, input().split())):
        cnt += 1
        if not q1:
            heappush(q1, -i)
        elif i >= -q1[0]:
            heappush(q2, i)
        else:
            heappush(q1, -i)
        while len(q2) > len(q1):
            tmp = heappop(q2)
            heappush(q1, -tmp)
        while len(q1) > len(q2) + 1:
            tmp = -heappop(q1)
            heappush(q2, tmp)
        if cnt % 2:
            cur.append(str(-q1[0]))
```

## 单调栈

奶牛排队

```python
from bisect import bisect_right

lst, q1, q2, ans = [int(input())for _ in range(int(input()))], [-1], [-1], 0
for i in range(len(lst)):
    while len(q1) > 1 and lst[q1[-1]] >= lst[i]:
        q1.pop()
    while len(q2) > 1 and lst[q2[-1]] < lst[i]:
        q2.pop()
    id = bisect_right(q1, q2[-1])
    if id < len(q1):
        ans = max(ans, i - q1[id] + 1)
    q1.append(i)
    q2.append(i)
```

接雨水

```python
def trap(self, height: List[int]) -> int:
    ans = 0
    stack = list()
    n = len(height)

    for i, h in enumerate(height):
        while stack and h > height[stack[-1]]:
            top = stack.pop()
            if not stack:
                break
            left = stack[-1]
            currWidth = i - left - 1
            currHeight = min(height[left], height[i]) - height[top]
            ans += currWidth * currHeight
        stack.append(i)

    return ans
```

柱状图中的最大矩形

```python
def largestRectangleArea(self, heights):
    n = len(heights)
    left, right = [0] * n, [0] * n
    mono_stack = list()
    for i in range(n):
        while mono_stack and heights[mono_stack[-1]] >= heights[i]:
            mono_stack.pop()
        left[i] = mono_stack[-1] if mono_stack else -1
        mono_stack.append(i)
    mono_stack = list()
    for i in range(n - 1, -1, -1):
        while mono_stack and heights[mono_stack[-1]] >= heights[i]:
            mono_stack.pop()
        right[i] = mono_stack[-1] if mono_stack else n
        mono_stack.append(i)
    ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
    return ans
```

护林员盖房子

```python
# 寻找最大全0子矩阵
for row in ma:
    stack = []
    for i in range(n):
        h[i] = h[i]+1 if row[i] == 0 else 0
        while stack and h[stack[-1]] > h[i]:
            y = h[stack.pop()]
            w = i if not stack else i-stack[-1]-1
            ans = max(ans, y*w)
        stack.append(i)
    while stack:
        y = h[stack.pop()]
        w = n if not stack else n-stack[-1]-1
        ans = max(ans, y*w)
print(ans)
```

## DP

全为1正方形矩阵

```python
m, n = map(int, input().split())
mat = [[int(k) for k in input()] for i in range(m)]
dp = [[0 for j in range(n+1)] for i in range(m+1)]
for i in range(m):
    for j in range(n):
        if mat[i][j]:
            dp[i+1][j+1] = min(dp[i][j], dp[i][j+1], dp[i+1][j])+1
print(sum(dp[i][j] for j in range(n+1) for i in range(m+1)))
```

最长公共子序列

```python
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if x[i - 1] == y[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
print(dp[m][n])
```

宠物小精灵之收服

```python
# 双限制背包问题
# dp[i][j]为捕获i个小精灵，皮卡丘剩余j体力时，剩余的最大精灵球数量
dp = [[-1]*(m+1) for _ in range(k+1)]
dp[0][m] = n
for i in range(k):
    cost, harm = map(int,input().split())
    for blood in range(m):
        for catch in range(i+1):
            pre_blood = blood+harm
            if pre_blood <= m and dp[catch][pre_blood] != -1:
                dp[catch+1][blood] = max(dp[catch+1][blood], dp[catch][pre_blood]-cost)
for i in range(k, -1, -1):
    for j in range(m, -1, -1):
        if dp[i][j] != -1:
            print(i,j)
            exit()
```

coins

```python
# 多重背包中的方案数问题
dp = [0] * (m + 1)
dp[0] = 1
for i in range(n):
    coin, count = value[i], counts[i]
    for j in range(count):
        for v in range(m, coin - 1, -1):
            dp[v] += dp[v - coin]
print(sum(1 for x in dp[1:] if x > 0))
```

NBA门票

```python
# 多重背包中的最优解问题
dp = [float('inf')]*(n+1)
dp[0] = 0
for i in range(6, -1, -1):
    cur = price[i]
    for k in range(n, cur-1, -1):
        for j in range(1, nums[i]+1):
            if k >= cur*j:
                dp[k] = min(dp[k], dp[k-cur*j]+j)
            else:
                break
if dp[-1] == float('inf'):
    print('Fail')
else:
    print(dp[-1])
```

复杂的整数划分问题

```python
# N划分成K个正整数之和
def divide_k(n,k):
    dp = [[0]*(k+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][1] = 1
    for i in range(1, n+1):
        for j in range(1, k+1):
            if i >= j:
                # dp[i-1][j-1]为包含1的划分的数量
                # 若不包含1，我们对每个数-1仍为正整数，划分数量为dp[i-j][j]
                dp[i][j] = dp[i-j][j] + dp[i-1][j-1]
    return dp[n][k]
# N划分成若干个不同正整数之和
def divide_dif(n):
    # dp[i][j]表示将数字 i 划分，其中最大的数字不大于 j 的方法数量
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i < j:
                dp[i][j] = dp[i][i]
            elif i == j:
                dp[i][j] = dp[i][j - 1] + 1
            # 用/不用j
            else:
                dp[i][j] = dp[i][j - 1] + dp[i - j][j - 1]
    return dp[n][n]
```

## 多指针

分发糖果

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        left = [0] * n
        for i in range(n):
            if i > 0 and ratings[i] > ratings[i - 1]:
                left[i] = left[i - 1] + 1
            else:
                left[i] = 1
        
        right = ret = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and ratings[i] > ratings[i + 1]:
                right += 1
            else:
                right = 1
            ret += max(left[i], right)
        
        return ret
```

## 技巧型

最大子矩阵

```python
def max_submatrix(matrix):
    def kadane(arr):
        max_end_here = max_so_far = arr[0]
        for x in arr[1:]:
            max_end_here = max(x, max_end_here + x)
            max_so_far = max(max_so_far, max_end_here)
        return max_so_far

    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    for left in range(cols):
        temp = [0] * rows
        for right in range(left, cols):
            for row in range(rows):
                temp[row] += matrix[row][right]
            max_sum = max(max_sum, kadane(temp))
    return max_sum
```

## 树

哈夫曼树

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
        merged = Node(left.weight + right.weight)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def encode_huffman_tree(root):
    codes = {}
    def traverse(node, code):
        if node.char:
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
        if node.char:
            decoded += node.char
            node = root
    return decoded
```

Trie

```python
class TrieNode:
    def __init__(self, char):
        self.char = char
        self.is_end = False
        self.children = {}
class Trie(object):
    def __init__(self):
        self.root = TrieNode("")
    def insert(self, word):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        node.is_end = True
    def dfs(self, node, pre):
        if node.is_end:
            self.output.append((pre + node.char))
        for child in node.children.values():
            self.dfs(child, pre + node.char)
    def search(self, x):
        node = self.root
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        self.output = []
        self.dfs(node, x[:-1])
        return self.output
```

括号嵌套树

```python
def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in '+-*/)':
            currentTree.setRootVal(int(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in '+-*/':
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError("Unknown Operator: " + i)
    return eTree
exp = "( ( 7 + 3 ) * ( 5 - 2 ) )"
pt = buildParseTree(exp)
for mode in ["preorder", "postorder", "inorder"]:
    pt.traversal(mode)
    print()
"""
* + 7 3 - 5 2 
7 3 + 5 2 - * 
7 + 3 * 5 - 2 
"""
import operator
def evaluate(parseTree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    leftC = parseTree.getLeftChild()
    rightC = parseTree.getRightChild()
    if leftC and rightC:
        fn = opers[parseTree.getRootVal()]
        return fn(evaluate(leftC),evaluate(rightC))
    else:
        return parseTree.getRootVal()
print(evaluate(pt))
# 30

#后序求值
def postordereval(tree):
    opers = {'+':operator.add, '-':operator.sub,
             '*':operator.mul, '/':operator.truediv}
    res1 = None
    res2 = None
    if tree:
        res1 = postordereval(tree.getLeftChild())
        res2 = postordereval(tree.getRightChild())
        if res1 and res2:
            return opers[tree.getRootVal()](res1,res2)
        else:
            return tree.getRootVal()

print(postordereval(pt))
# 30

#中序还原完全括号表达式
def printexp(tree):
    sVal = ""
    if tree:
        sVal = '(' + printexp(tree.getLeftChild())
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + printexp(tree.getRightChild()) + ')'
    return sVal

print(printexp(pt))
# (((7)+3)*((5)-2))
```

树的转换

```python
def tree_heights(s):
    old_height, max_old, new_height, max_new = 0, 0, 0, 0
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
```

树的直径

```python
def diameterOfBinaryTree(root):
    diameter = [0]
    def depth(node):
        if not node:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        diameter[0] = max(diameter[0], left_depth + right_depth)
        return max(left_depth, right_depth) + 1
    depth(root)
    return diameter[0]
```

多叉树转二叉树

```python
class B_node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
class T_node:
    def __init__(self, value):
        self.value = value
        self.children = []
        
def to_b_tree(t_node):
    if t_node is None: return None
    b_node = B_node(t_node.value)
    if len(t_node.children) > 0: b_node.left = to_b_tree(t_node.children[0])
    current_node = b_node.left
    for child in t_node.children[1:]: current_node.right = to_b_tree(child); current_node = current_node.right
    return b_node

def to_tree(b_node):
    if b_node is None: return None
    t_node, child = T_node(b_node.value), b_node.left
    while child is not None: t_node.children += to_tree(child); child = child.right
    return t_node
```

## 图论

**经典类**

```python
class Vertex:	
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
		self.dist = sys.maxsize
        self.pred = None
    
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
    
    def __lt__(self, other):
        return self.distance < other.distance
```

```python
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
```

```python
def constructLaplacianMatrix(n, edges):
    graph = Graph()
    for i in range(n):
        graph.addVertex(i)

    for edge in edges:
        a, b = edge
        graph.addEdge(a, b)
        graph.addEdge(b, a)

    laplacianMatrix = []
    for vertex in graph:
        row = [0] * n
        row[vertex.getId()] = len(vertex.getConnections())
        for neighbor in vertex.getConnections():
            row[neighbor.getId()] = -1
        laplacianMatrix.append(row)

    return laplacianMatrix
```

**强连通单元**

**Kosaraju算法**

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)

def kosaraju(graph):
    # Step 1: Perform first DFS to get finishing times
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    
    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    
    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs
```

**Tarjan算法**

```python
def tarjan(graph):
    def dfs(node):
        nonlocal index, stack, indices, low_link, on_stack, sccs
        index += 1
        indices[node] = index
        low_link[node] = index
        stack.append(node)
        on_stack[node] = True
        
        for neighbor in graph[node]:
            if indices[neighbor] == 0:  # Neighbor not visited yet
                dfs(neighbor)
                low_link[node] = min(low_link[node], low_link[neighbor])
            elif on_stack[neighbor]:  # Neighbor is in the current SCC
                low_link[node] = min(low_link[node], indices[neighbor])
        
        if indices[node] == low_link[node]:
            scc = []
            while True:
                top = stack.pop()
                on_stack[top] = False
                scc.append(top)
                if top == node:
                    break
            sccs.append(scc)
    
    index = 0
    stack = []
    indices = [0] * len(graph)
    low_link = [0] * len(graph)
    on_stack = [False] * len(graph)
    sccs = []
    
    for node in range(len(graph)):
        if indices[node] == 0:
            dfs(node)
    
    return sccs
```

**最短路径**

**Dijkstra算法**：用于在加权图中找到一个顶点到其他所有顶点的最短路径。

```python
import heapq
def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
```

**Floyd-Warshall算法**

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif j in graph[i]:
                dist[i][j] = graph[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist
```

**有额外限制的最短路径**

道路

```python
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

**最小生成树问题**

**Kruskal算法**: 适合处理稀疏图，遵循贪心策略

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))  # 初始化每个顶点的父节点为自己
        self.rank = [0] * size  # 初始化树的高度
    def find(self, x):
        # 寻找根节点，并进行路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        # 合并两个集合
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
```

```python
def kruskal(graph_edges, num_vertices):
    # graph_edges 是边的列表，每个元素是一个元组 (weight, vertex1, vertex2)
    # 按照边的权重从小到大排序
    graph_edges.sort()
    uf = UnionFind(num_vertices)  # 初始化并查集
    mst = []  # 用来存储最小生成树的边
    for weight, u, v in graph_edges:
        # 如果加入这条边不会形成环
        if uf.find(u) != uf.find(v):
            uf.union(u, v)  # 合并两个集合
            mst.append((u, v, weight))  # 加入到最小生成树中
    return mst
```

**Prim算法：**适合处理稠密图

```python
import heapq
def prim(graph, start_vertex):
    # graph 是一个字典，键是顶点，值是一个列表，列表中的元素是(邻居, 权重)
    mst = []  # 存储最小生成树的边
    visited = set([start_vertex])  # 已访问的顶点
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex]]  # 从起始顶点出发的边
    heapq.heapify(edges)  # 将边转换成最小堆，以便高效地获取最小边
    while edges:
        weight, frm, to = heapq.heappop(edges)  # 获取权重最小的边
        if to not in visited:
            visited.add(to)  # 标记为已访问
            mst.append((frm, to, weight))  # 加入到最小生成树中
            for next_to, next_weight in graph[to]:
                if next_to not in visited:
                    heapq.heappush(edges, (next_weight, to, next_to))  # 将与新顶点相连的边加入堆中
    return mst

def prim(graph,start):
    pq = PriorityQueue()
    for vertex in graph:
        vertex.distance = sys.maxsize
        vertex.previous = None
    start.distance = 0
    pq.buildHeap([(v.distance,v) for v in graph])
    while pq:
        distance, current_v = pq.delete()
        for next_v in current_v.get_eighbors():
          new_distance = current_v.get_neighbor(next_v)
          if next_v in pq and new_distance < next_v.distance:
              next_v.previous = current_v
              next_v.distance = new_distance
              pq.change_priority(next_v,new_distance)
```

**最大流问题**

**Ford-Fulkerson 算法**

```python
# 使用邻接表表示图
class Graph:
    def __init__(self, graph):
        self.graph = graph  # 原始图
        self.ROW = len(graph)
        self.col = len(graph[0])
    # 使用DFS寻找从给定源到汇的路径，并存储路径，如果找到则返回True
    def dfs(self, s, t, parent):
        visited = [False] * self.ROW
        stack = []
        stack.append(s)
        visited[s] = True
        while stack:
            u = stack.pop()
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] is False and val > 0:
                    stack.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        return False
    # 使用Ford-Fulkerson算法返回图中从s到t的最大流量
    def fordFulkerson(self, source, sink):
        parent = [-1] * self.ROW
        max_flow = 0  # 最初最大流量为0
        # 增加流量只要能找到增广路径
        while self.dfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            # 更新残余容量
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            max_flow += path_flow
        return max_flow
```

**图着色问题**

```python
def graph_coloring(graph):
    result = {}
    for node in sorted(graph, key=lambda x: len(graph[x]), reverse=True):
        neighbor_colors = {result[neighbor] for neighbor in graph[node] if neighbor in result}
        color = 1
        while color in neighbor_colors:
            color += 1
        result[node] = color
    return result
```

**拓扑排序（判断环）**

```python
from collections import deque
def topological_sort(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    queue = deque([u for u in in_degree if in_degree[u] == 0])
    visited = 0
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        visited += 1
    if visited != len(graph):
        return []
    return top_order
```

```python
def CheckCycle(graph):
    color = [0 for _ in range(N+1)]
    def dfs(x):
        color[x] = 1
        for i in graph[x]:
            if color[i] == 1:
                return True
            elif not color[i] and dfs(i):
                return True
        color[x] = 2
        return False

    for i in range(1, N+1):
        if not color[i]:
            if dfs(i):
                return True
    return False
```

