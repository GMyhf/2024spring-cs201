

# 2024数算B模拟考试1@gw calss

Updated 0043 GMT+8 May 23, 2024

2024 spring, Complied by Hongfei Yan



2024-05-23 19:40:00 ～ 2024-05-23 21:30:00



## 001/26588: 最长上升子串

http://dsbpython.openjudge.cn/2024moni1/001/

一个由数字‘0’到‘9’构成的字符串，求其中最长上升子串的长度。

一个子串，如果其中每个字符都不大于其右边的字符，则该子串为上升子串

输入

第一行是整数n, 1 < n < 100，表示有n个由数字‘0’到‘9’构成的字符串
接下来n行，每行一个字符串，字符串长度不超过100

输出

对每个字符串，输出其中最长上升子串的长度

样例输入

```
4
112300125239
1
111
1235111111
```

样例输出

```
5
1
3
6
```



```python
def longestIncreasingSubstring(s):
    max_length = 1
    current_length = 1

    for i in range(1, len(s)):
        if s[i] >= s[i - 1]:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1

    return max(max_length, current_length)

if __name__ == "__main__":
    n = int(input())
    for _ in range(n):
        string = input()
        print(longestIncreasingSubstring(string))
```





## 002/26589: 快速排序填空

http://dsbpython.openjudge.cn/2024moni1/002/

输入n ( 1 <= n <= 100000) 个整数，每个整数绝对值不超过100000，将这些整数用快速排序算法排序后输出

请填空

```python
sort = None
def quickSort(a,s,e): 
    if s >= e:
        return
    i,j = s,e
    while i != j:
        while i < j and a[i] <= a[j]:
            j -= 1
        a[i],a[j] = a[j],a[i]
// 在此处补充你的代码
#-----
    quickSort(a,s,i-1)
    quickSort(a, i+1,e)
n = int(input())
a = []
for i in range(n):
    a.append(int(input()))
quickSort(a,0,len(a)-1)
for i in a:
    print(i)
```

输入

第一行是整数n
接下来n行，每行1个整数

输出

n行，输入中n个整数排序后的结果

样例输入

```
3
6
12
3
```

样例输出

```
3
6
12
```


```python
sort = None
def quickSort(a,s,e): 
    if s >= e:
        return
    i,j = s,e
    while i != j:
        while i < j and a[i] <= a[j]:
            j -= 1
        a[i],a[j] = a[j],a[i]
        while i < j and a[i] <= a[j]:
            i += 1
        a[i],a[j] = a[j],a[i]
#-----
    quickSort(a,s,i-1)
    quickSort(a, i+1,e)
n = int(input())
a = []
for i in range(n):
    a.append(int(input()))
quickSort(a,0,len(a)-1)
for i in a:
    print(i)
```





## 003/26590: 检测括号嵌套

http://dsbpython.openjudge.cn/2024moni1/003/

字符串中可能有3种成对的括号，"( )"、"[ ]"、"{}"。请判断字符串的括号是否都正确配对以及有无括号嵌套。无括号也算正确配对。括号交叉算不正确配对，例如"1234[78)ab]"就不算正确配对。一对括号被包含在另一对括号里面，例如"12(ab[8])"就算括号嵌套。括号嵌套不影响配对的正确性。 给定一个字符串: 如果括号没有正确配对，则输出 "ERROR" 如果正确配对了，且有括号嵌套现象，则输出"YES" 如果正确配对了，但是没有括号嵌套现象，则输出"NO"   

输入

一个字符串，长度不超过5000,仅由 ( ) [ ] { } 和小写英文字母以及数字构成

输出

根据实际情况输出 ERROR, YES 或NO

样例输入

```
样例1:
[](){}
样例2:
[(a)]bv[]
样例3:
[[(])]{}
```

样例输出

```
样例1:
NO
样例2:
YES
样例3:
ERROR
```



```python
def check_brackets(s):
    stack = []
    nested = False
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs.keys():
            if not stack or stack.pop() != pairs[ch]:
                return "ERROR"
            if stack:
                nested = True
    if stack:
        return "ERROR"
    return "YES" if nested else "NO"

s = input()
print(check_brackets(s))
```



## 004/19164: 移动办公

http://dsbpython.openjudge.cn/2024moni1/004/

假设你经营着一家公司，公司在北京和南京各有一个办公地点。公司只有你一个人，所以你只能每月选择在一个城市办公。在第i个月，如果你在北京办公，你能获得Pi的营业额，如果你在南京办公，你能获得Ni的营业额。但是，如果你某个月在一个城市办公，下个月在另一个城市办公，你需要支付M的交通费。那么，该怎样规划你的行程（可在任何一个城市开始），才能使得总收入（总营业额减去总交通费）最大？

输入

输入的第一行有两个整数T（1 <= T <= 100）和M（1 <= M <= 100），T代表总共的月数，M代表交通费。接下来的T行每行包括两个在1到100之间（包括1和100）的整数，分别表示某个月在北京和在南京办公获得的营业额。

输出

输出只包括一行，这一行只包含一个整数，表示可以获得的最大总收入。

样例输入

```
4 3
10 9
2 8
9 5
8 2
```

样例输出

```
31
```





```python
n,m=map(int,input().split())
dp1=[0]*(n+1)
dp2=[0]*(n+1)
for i in range(1,n+1):
    a,b=map(int,input().split())
    dp1[i]=max(dp1[i-1],dp2[i-1]-m)+a
    dp2[i]=max(dp1[i-1]-m,dp2[i-1])+b
print(max(dp1[n],dp2[n]))
```



## 005/26570: 我想完成数算作业：沉淀

http://dsbpython.openjudge.cn/2024moni1/005/

小A选了数算课之后从来没去过，有一天他和小B聊起来才知道原来每周都有上机作业！于是小A连忙问了小B题目序号，保险起见他又问了小C作业的题号，不幸的是两人的回答并不一样...

无奈之下小A决定把两个人记的题目都做完，为此他希望先合并两人的题号，并去除其中的重复题目，请你帮他完成下面的程序填空。

```python
class Node:
    def __init__(self, data, next=None):
        self.data, self.next = data, next
class LinkedList:
    def __init__(self):
        self.head = None
    def create(self, data):
        self.head = Node(0)
        cur = self.head
        for i in range(len(data)):
            node = Node(data[i])
            cur.next = node
            cur = cur.next
def printList(head):
    cur = head.next
    while cur:
        print(cur.data, end=' ')
        cur = cur.next

def mergeTwoLists( l1, l2):
    head = cur = Node(0)
    while l1 and l2:
        if l1.data > l2.data:
            cur.next = l2
            l2 = l2.next
        else:
            cur.next = l1
            l1 = l1.next
        cur = cur.next
    cur.next = l1 or l2
    return head
def deleteDuplicates(head):
// 在此处补充你的代码
data1 = list(map(int, input().split()))
data2 = list(map(int, input().split()))
list1 = LinkedList()
list2 = LinkedList()
list1.create(data1)
list2.create(data2)
head = mergeTwoLists(list1.head.next, list2.head.next)
deleteDuplicates(head)
printList(head)
```

输入

输入包括两行，分别代表小B和小C记下的排好序的作业题号

输出

按题号顺序输出一行小A应该完成的题目序号。

样例

```
sample1 input：
1 3
1 2

sample1 output：
1 2 3
```

样例

```
sample2 input：
1 2 4
1 3 4

sample2 output：
1 2 3 4
```

提示

程序中处理的链表是带空闲头结点的



```python
class Node:
    def __init__(self, data, next=None):
        self.data, self.next = data, next
class LinkedList:
    def __init__(self):
        self.head = None
    def create(self, data):
        self.head = Node(0)
        cur = self.head
        for i in range(len(data)):
            node = Node(data[i])
            cur.next = node
            cur = cur.next
def printList(head):
    cur = head.next
    while cur:
        print(cur.data, end=' ')
        cur = cur.next

def mergeTwoLists( l1, l2):
    head = cur = Node(0)
    while l1 and l2:
        if l1.data > l2.data:
            cur.next = l2
            l2 = l2.next
        else:
            cur.next = l1
            l1 = l1.next
        cur = cur.next
    cur.next = l1 or l2
    return head
def deleteDuplicates(head):
    cur = head
    while cur and cur.next:
        if cur.data == cur.next.data:
            cur.next = cur.next.next
        else:
            cur = cur.next
data1 = list(map(int, input().split()))
data2 = list(map(int, input().split()))
list1 = LinkedList()
list2 = LinkedList()
list1.create(data1)
list2.create(data2)
head = mergeTwoLists(list1.head.next, list2.head.next)
deleteDuplicates(head)
printList(head)
```





## 006/26571: 我想完成数算作业：代码

http://dsbpython.openjudge.cn/2024moni1/006/

当卷王小D睡前意识到室友们每天熬夜吐槽的是自己也选了的课时，他距离早八随堂交的ddl只剩下了不到4小时。已经debug一晚上无果的小D有心要分无力做题，于是决定直接抄一份室友的作业完事。万万没想到，他们作业里完全一致的错误，引发了一场全面的作业查重……

假设a和b作业雷同，b和c作业雷同，则a和c作业雷同。所有抄袭现象都会被发现，且雷同的作业只有一份独立完成的原版，请输出独立完成作业的人数

输入

第一行输入两个正整数表示班上的人数n与总比对数m，接下来m行每行均为两个1-n中的整数i和j，表明第i个同学与第j个同学的作业雷同。

输出

独立完成作业的人数

样例输入

```
样例1：
3 2
1 2
1 3
样例2：
4 2
2 4
1 3
```

样例输出

```
样例1：
1
样例2:
2
```



```python
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        parent[xroot] = yroot

n, m = map(int, input().split())
parent = list(range(n + 1))
for _ in range(m):
    i, j = map(int, input().split())
    union(parent, i, j)

count = sum(i == parent[i] for i in range(1, n + 1))
print(count)
```





## 007/04128: 单词序列

http://cs101.openjudge.cn/practice/04128/

给出两个单词（开始单词和结束单词）以及一个词典。找出从开始单词转换到结束单词，所需要的最短转换序列。转换的规则如下：

1、每次只能改变一个字母

2、转换过程中出现的单词(除开始单词和结束单词)必须存在于词典中

例如：

开始单词为：hit

结束单词为：cog

词典为：[hot,dot,dog,lot,log,mot]

那么一种可能的最短变换是： hit -> hot -> dot -> dog -> cog,

所以返回的结果是序列的长度5；

注意：

1、如果不能找到这种变换，则输出0；

2、词典中所有单词长度一样；

3、所有的单词都由小写字母构成；

4、开始单词和结束单词可以不在词典中。

输入

共两行，第一行为开始单词和结束单词（两个单词不同），以空格分开。第二行为若干的单词（各不相同），以空格分隔开来，表示词典。单词长度不超过5,单词个数不超过30。

输出

输出转换序列的长度。

样例输入

```
hit cog
hot dot dog lot log
```

样例输出

```
5
```







```python
from collections import deque

def is_valid_transition(word1, word2):
    """
    Check if two words differ by exactly one character.
    """
    diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
    return diff_count == 1

def shortest_word_sequence(start, end, dictionary):
    if start == end:
        return 0
    
    if len(start) != len(end):
        return 0
    
    if start not in dictionary:
        dictionary.append(start)
    
    if end not in dictionary:
        dictionary.append(end)

    queue = deque([(start, 1)])
    visited = set([start])

    while queue:
        current_word, steps = queue.popleft()
        if current_word == end:
            return steps
        
        for word in dictionary:
            if is_valid_transition(current_word, word) and word not in visited:
                queue.append((word, steps + 1))
                visited.add(word)
    
    return 0

# Example usage:
start, end = input().split()
dictionary = input().split()

result = shortest_word_sequence(start, end, dictionary)
print(result)
```





## 008/26582: 我想成为数算高手：重生

http://dsbpython.openjudge.cn/2024moni1/008/

经历了查重风波的小D被取消了当次作业成绩，但他可是我们这次故事的主角。学习难道只是为了这点作业和分数吗？！小D振作了起来，我不能成为作业的奴隶，我要成为数算高手！

成为数算高手，第一步是要会建QTE二叉树。

我们可以把由 0 和 1 组成的字符串分为三类：全 0 串称为 Q串，全 1 串称为 T 串，既含 0 又含 1 的串则称为 E 串。

QTE二叉树是一种二叉树，它的结点类型包括 Q 结点，T 结点和 E 结点三种。由一个长度为 n 的 01 串 S 可以构造出一棵 QTE 二叉树 T，递归的构造方法如下：

设T 的根结点为root，则root类型与串 S 的类型相同；
若串 S的长度大于1，将串 S 从中间分开，分为左右子串 S1 和 S2，若串S长度为奇数，则让左子串S1的长度比右子串恰好长1，否则让左右子串长度相等；由左子串 S1 构造root的左子树 T1，由右子串 S2 构造root 的右子树 T2。

 

输入

第一行是一个整数 T, T <= 50,表示01串的数目。
接下来T行，每行是一个01串，对应于一棵QTE二叉树。01串长度至少为1，最多为2048。

输出

输出T行，每行是一棵QTE二叉树的后序遍历序列

样例输入

```
3
111
11010
10001011
```

样例输出

```
TTTTT
TTTQETQEE
TQEQQQETQETTTEE
```






```python
#2300011335	邓锦文
class TreeNode:
    def __init__(self, val, letter):
        self.val = val
        self.letter = letter
        self.left = None
        self.right = None

def build(node):
    if len(node.val) > 1:
        mid = (len(node.val)+1) // 2
        l = node.val[:mid]
        if '0' not in l:
            lc = 'T'
        elif '1' not in l:
            lc = 'Q'
        else:
            lc = 'E'
        node.left = TreeNode(l, lc)
        r = node.val[mid:]
        if '0' not in r:
            rc = 'T'
        elif '1' not in r:
            rc = 'Q'
        else:
            rc = 'E'
        node.right = TreeNode(r, rc)
        build(node.left)
        build(node.right)

def postfix(root):
    if root is None:
        return ''
    return postfix(root.left)+postfix(root.right)+root.letter

for _ in range(int(input())):
    s = input().strip()
    if '0' not in s:
        c = 'T'
    elif '1' not in s:
        c = 'Q'
    else:
        c = 'E'
    root = TreeNode(s, c)
    build(root)
    print(postfix(root))
```





## 009/26585: 我想成为数算高手：穿越

http://dsbpython.openjudge.cn/2024moni1/009/

小D拿着地图走进理一楼，发现地图上记录的还不足这大迷宫的冰山一角……

经过考察，理一楼应当用一个树形结构描述，每一个结点上都是一个办公室或者一个机房，当然也可以是一片空地。在确定结点1 为根结点之后，对于每一个非叶子的结点 i，设以它为根的子树中所有的办公室数量为 office_i，机房数目为 computerRoom_i，都有 office_i=computerRoom_i。

小D心想，这偌大的理一里到底最多能容纳多少个机房呀。

 

输入

输入的第一行是一个正整数n, 1 <= n <= 100000，表示用树形结构描述的理一的结点，结点编号为1 到n。
第 2 至 n 行，每行两个正整数 u，v，表示结点 u 与结点 v 间有一条边。

输出

只有一行，即理一最多能容纳多少个机房。

样例输入

```
5
1 2
2 3
3 4
2 5
```

样例输出

```
2
```



```python
#2300011335	邓锦文
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

def build(n, nodes):
    tree = [Node(i) for i in range(n+1)]
    for a, b in nodes:
        tree[a].children.append(tree[b])
    return tree[1]

def cal(root):
    if root.children:
        if root.val == 1:
            cnt = 1
            for child in root.children:
                cnt += cal(child)
            return cnt//2
        cnt = 1
        for child in root.children:
            cnt += cal(child)
        if cnt % 2:
            return cnt-1
        else:
            return cnt
    else:
        return 1

n = int(input())
nodes = [list(map(int, input().split())) for _ in range(n-1)]
root = build(n, nodes)
print(cal(root))
```





## 010/25612: 团结不用排序就是力量

http://dsbpython.openjudge.cn/2024moni1/010/

有n个人，本来互相都不认识。现在想要让他们团结起来。假设如果a认识b，b认识c，那么最终a总会通过b而认识c。如果最终所有人都互相认识，那么大家就算团结起来了。但是想安排两个人直接见面认识，需要花费一些社交成本。不同的两个人要见面，花费的成本还不一样。一个人可以见多个人，但是，有的两个人就是不肯见面。问要让大家团结起来，最少要花多少钱。请注意，认识是相互的，即若a认识b，则b也认识a。

输入

第一行是整数n和m，表示有n个人，以及m对可以安排见面的人(0 < n <100, 0 < m < 5000) 。n个人编号从0到n-1。

接下来的m行,每行有两个整数s,e和一个浮点数w, 表示花w元钱(0 <= w < 100000)可以安排s和e见面。数据保证每一对见面的花费都不一样。

输出

第一行输出让大家团结起来的最小花销，保留小数点后面2位。接下来输出每一对见面的人。输出一对人的时候，编号小在前面。
如果没法让大家团结起来，则输出"NOT CONNECTED"。

样例输入

```
5 9
0 1 10.0
0 3 7.0
0 4 25.0
1 2 8.0
1 3 9.0
1 4 35.0
2 3 11.0
2 4 50.0
3 4 24.0
```

样例输出

```
48.00
0 3
1 2
1 3
3 4
```

来源

Guo Wei



```python
#2300011335	邓锦文
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0 for _ in range(n)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return False
        if self.rank[x] < self.rank[y]:
            self.parent[x] = y
        elif self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[y] = x
            self.rank[x] += 1
        return True

def operate(n, nodes):
    nodes.sort(key=lambda x:x[2])
    f = UnionFind(n)
    cost = 0
    result = []
    for s, e, w in nodes:
        if f.union(s, e):
            cost += w
            result.append([s, e])
    if len(result) == n-1:
        return cost, result
    return 'NOT CONNECTED'

n, m = map(int, input().split())
nodes = []
for _ in range(m):
    s, e, w = input().split()
    nodes.append([int(s), int(e), float(w)])
ans = operate(n, nodes)
if isinstance(ans, tuple):
    cost, result = ans
    print(f'{cost:.2f}')
    for row in result:
        print(*row)
else:
    print(ans)
```



## 011/25156: 判断是否是深度优先遍历序列

http://dsbpython.openjudge.cn/2024moni1/011/

给定一个无向图和若干个序列，判断该序列是否是图的深度优先遍历序列。

输入

第一行是整数n( 0 < n <= 16)和m,表示无向图的点数和边数。顶点编号从0到n-1
接下来有m行，每行两个整数a,b，表示顶点a和b之间有边
接下来是整数k(k<50)，表示下面有k个待判断序列
再接下来有k行，每行是一个待判断序列,每个序列都包含n个整数,整数范围[0,n-1]

输出

对每个待判断序列，如果是该无向图的深度优先遍历序列，则输出YES，否则输出NO

样例输入

```
9 9
0 1
0 2
3 0
2 1
1 5
1 4
4 5
6 3
8 7
3
0 1 2 4 5 3 6 8 7
0 1 5 4 2 3 6 8 7
0 1 5 4 3 6 2 8 7
```

样例输出

```
YES
YES
NO
```

来源

Guo Wei



```python
#2300011335	邓锦文
n, m = map(int, input().split())
graph = {i: [] for i in range(n)}
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)
vis = [False for _ in range(n)]
stack = []
def check(nodes):
    node = nodes.pop(0)
    if not nodes:
        return True
    vis[node] = True
    stack.append(node)
    if nodes[0] in graph[node]:
        if not vis[nodes[0]]:
            if check(nodes):
                return True
        return False
    if not all(vis[i] for i in graph[node]):
        return False
    stack.pop()
    while stack:
        if nodes[0] in graph[stack[-1]]:
            if not vis[nodes[0]]:
                if check(nodes):
                    return True
        if not all(vis[i] for i in graph[stack[-1]]):
            return False
        stack.pop()
    if check(nodes):
        return True
    return False
for _ in range(int(input())):
    vis = [False for _ in range(n)]
    stack = []
    nodes = list(map(int, input().split()))
    print('YES' if check(nodes) else 'NO')
```





## 012/22275: 二叉搜索树的遍历

http://dsbpython.openjudge.cn/2024moni1/012/

给出一棵二叉搜索树的前序遍历，求它的后序遍历

输入

第一行一个正整数n（n<=2000）表示这棵二叉搜索树的结点个数
第二行n个正整数，表示这棵二叉搜索树的前序遍历
保证第二行的n个正整数中，1~n的每个值刚好出现一次

输出

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
def postorder(preorder):
    if not preorder:
        return []
    root = preorder[0]
    i = 1
    while i < len(preorder):
        if preorder[i] > root:
            break
        i += 1
    left = postorder(preorder[1:i])
    right = postorder(preorder[i:])
    return left + right + [root]

n = int(input())
preorder = list(map(int, input().split()))
postorder = postorder(preorder)
print(' '.join(map(str, postorder)))
```

