# 数算（数据结构与算法）pre每日选做

Updated 2142 GMT+8 March 11, 2024

2024 spring, Complied by Hongfei Yan



**说明：**

1）数算课程在春季学期开，前一年计算概论课程结束，正值寒假，同学建议每日推出少许题目练习，因此成为“**数算pre每日选做**”。

2）为避免重复，如果题目出现在  https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md，会给出指引。

3）有同学假期时候完成了这些题目，放在 gitbub上面，可以参考

Wangjie Su's GitHub,  https://github.com/csxq0605/CS101-spring

Zixuan Jiang's summary, https://github.com/GMyhf/2024spring-cs201/blob/main/cheatsheet/DataStructuresAndAlgorithms-WinterBreak-20240214-JIANGZixuan.md



# 1-10

## 01094: Sorting It All Out

http://cs101.openjudge.cn/dsapre/01094/

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





## 01258: Agri-Net

http://cs101.openjudge.cn/dsapre/01258/

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

来源

蔡错@pku



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

http://cs101.openjudge.cn/dsapre/01426/

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



```python
#23n2300011072(X)
from collections import deque
while 1:
    n=int(input())
    if n==0:
        break
    q=deque([(1,'1')])
    vis={1}
    while q:
        remainder,num=q.popleft()
        if remainder==0:
            print(num)
            break
        for digit in [0,1]:
            new_remainder=(remainder*10+digit)%n
            if new_remainder not in vis:
                vis.add(new_remainder)
                q.append((new_remainder,num+str(digit)))
```



## 01611: The Suspects

http://cs101.openjudge.cn/dsapre/01611/

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



## 01760: Disk Tree

http://cs101.openjudge.cn/dsapre/01760/

Hacker Bill has accidentally lost all the information from his workstation's hard drive and he has no backup copies of its contents. He does not regret for the loss of the files themselves, but for the very nice and convenient directory structure that he had created and cherished during years of work. Fortunately, Bill has several copies of directory listings from his hard drive. Using those listings he was able to recover full paths (like "WINNT\SYSTEM32\CERTSRV\CERTCO~1\X86") for some directories. He put all of them in a file by writing each path he has found on a separate line. Your task is to write a program that will help Bill to restore his state of the art directory structure by providing nicely formatted directory tree.

**输入**

The first line of the input file contains single integer number N (1 <= N <= 500) that denotes a total number of distinct directory paths. Then N lines with directory paths follow. Each directory path occupies a single line and does not contain any spaces, including leading or trailing ones. No path exceeds 80 characters. Each path is listed once and consists of a number of directory names separated by a back slash ("\"). 

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
**1/Σ(to,td)d(to,td)**
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

Several currency exchange points are working in our city. Let us suppose that each point specializes in two particular currencies and performs exchange operations only with these currencies. There can be several points specializing in the same pair of currencies. Each point has its own exchange rates, exchange rate of A to B is the quantity of B you get for 1A. Also each exchange point has some commission, the sum you have to pay for your exchange operation. Commission is always collected in source currency.
For example, if you want to exchange 100 US Dollars into Russian Rubles at the exchange point, where the exchange rate is 29.75, and the commission is 0.39 you will get (100 - 0.39) * 29.75 = 2963.3975RUR. 
You surely know that there are N different currencies you can deal with in our city. Let us assign unique integer number from 1 to N to each currency. Then each exchange point can be described with 6 numbers: integer A and B - numbers of currencies it exchanges, and real RAB, CAB, RBA and CBA - exchange rates and commissions when exchanging A to B and B to A respectively.
Nick has some money in currency S and wonders if he can somehow, after some exchange operations, increase his capital. Of course, he wants to have his money in currency S in the end. Help him to answer this difficult question. Nick must always have non-negative sum of money while making his operations.

**输入**

The first line of the input contains four numbers: N - the number of currencies, M - the number of exchange points, S - the number of currency Nick has and V - the quantity of currency units he has. The following M lines contain 6 numbers each - the description of the corresponding exchange point - in specified above order. Numbers are separated by one or more spaces. 1<=S<=N<=100, 1<=M<=100, V is real number, 0<=V<=103.
For each point exchange rates and commissions are real, given with at most two digits after the decimal point, 10-2<=rate<=102, 0<=commission<=102.
Let us call some sequence of the exchange operations simple if no exchange point is used more than once in this sequence. You may assume that ratio of the numeric values of the sums at the end and at the beginning of any simple sequence of the exchange operations will be less than 104.

**输出**

If Nick can increase his wealth, output YES, in other case output NO to the output file.

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

http://cs101.openjudge.cn/dsapre/02039/

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

http://cs101.openjudge.cn/dsapre/02049/

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

http://cs101.openjudge.cn/dsapre/02092/

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



## 02226: Muddy Fields

http://cs101.openjudge.cn/dsapre/02226/

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

http://cs101.openjudge.cn/dsapre/02255/

**输入**

输入可能有多组，以EOF结束。 每组输入包含两个字符串，分别为树的前序遍历和中序遍历。每个字符串中只包含大写字母且互不重复。

**输出**

对于每组输入，用一行来输出它后序遍历结果。

样例输入

`DBACEGF ABCDEFG BCAD CBAD `

样例输出

`ACBFGED CDAB `

提示

以英文题面为准



```python
ls = []
rs = []
root = 0
cnt = 0
cot = 0

def Solve(l,r):
    global cnt
    if cnt >= len(Line1):
        return -1
    Pl = Line2.find(Line1[cnt])
    if Pl < l or Pl > r:
        return -1

    x = ord(Line1[cnt]) - 65
    cnt = cnt + 1
    ls[x] = Solve(l,Pl-1)
    rs[x] = Solve(Pl+1,r)
    return x

def Pout(x):
    if ls[x] != -1:
        Pout(ls[x])
    if rs[x] != -1:
        Pout(rs[x])
    print(chr(x+65),end = '')

while True:
    try:
        Line1,Line2 = input().split()
        ls = [-1]*len(Line1)
        rs = [-1]*len(Line1)
        cnt = 0
        
        root = Solve(0,len(Line1) - 1)

        Pout(root)
        print()

    except:
        break
```





## 02299: Ultra-QuickSort

http://cs101.openjudge.cn/dsapre/02299/

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



## 02337: Catenyms

http://cs101.openjudge.cn/dsapre/02337/

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

**Background**
The knight is getting bored of seeing the same black and white squares again and again and has decided to make a journey
around the world. Whenever a knight moves, it is two squares in one direction and one square perpendicular to this. The world of a knight is the chessboard he is living on. Our knight lives on a chessboard that has a smaller area than a regular 8 * 8 board, but it is still rectangular. Can you help this adventurous knight to make travel plans?

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

世界上有许多宗教，你感兴趣的是你学校里的同学信仰多少种宗教。

你的学校有n名学生（0 < n <= 50000），你不太可能询问每个人的宗教信仰，因为他们不太愿意透露。但是当你同时找到2名学生，他们却愿意告诉你他们是否信仰同一宗教，你可以通过很多这样的询问估算学校里的宗教数目的上限。你可以认为每名学生只会信仰最多一种宗教。



**输入**

输入包括多组数据。
每组数据的第一行包括n和m，0 <= m <= n(n-1)/2，其后m行每行包括两个数字i和j，表示学生i和学生j信仰同一宗教，学生被标号为1至n。输入以一行 n = m = 0 作为结束。

**输出**

对于每组数据，先输出它的编号（从1开始），接着输出学生信仰的不同宗教的数目上限。

样例输入

```
10 9
1 2
1 3
1 4
1 5
1 6
1 7
1 8
1 9
1 10
10 4
2 3
4 5
4 8
5 8
0 0
```

样例输出

```
Case 1: 1
Case 2: 7
```



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

描述

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

http://cs101.openjudge.cn/dsapre/02766/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 02773: 采药

http://cs101.openjudge.cn/dsapre/02773/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 02774: 木材加工

http://cs101.openjudge.cn/dsapre/02774/

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





## 02788: 二叉树（2）

http://cs101.openjudge.cn/dsapre/02788/



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



完全二叉树

​           每个结点

左孩子 2\*i          右孩子 2\*i+1

 

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



## 03720: 文本二叉树

http://cs101.openjudge.cn/dsapre/03720/



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

输入

第一行是树的数目 n

接下来是n棵树，每棵树以'0'结尾。'0'不是树的一部分
每棵树不超过100个节点

输出

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





# 31-40

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

http://cs101.openjudge.cn/dsapre/04082/

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



前序遍历（Preorder Traversal）是二叉树遍历的一种方式，它的遍历顺序是先访问根节点，然后按照左子树和右子树的顺序递归地遍历子树。

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
    if tempList[index][1] == '0' and node.x != '$':
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

# 读取输入
n = int(input())
tempList = input().split(' ')

# 构建多叉树
root, _ = build_tree(tempList, 0)

# 执行宽度优先遍历并打印镜像映射序列
print_tree(root)
```



```c++
#include <iostream>
#include <vector>
using namespace std;

struct node {
	char ele;
	node * l;
	node * r;
	node (char e) :ele(e), l(NULL), r(NULL) {}
};

node * generate() {
	char ele;
	int mark;
	cin>>ele>>mark;
	node * n = NULL;
	if (mark == 0) {
		n = new node(ele);
		n->l = generate();
		n->r = generate();
	} else
		if (ele != '$')
			n = new node(ele);
	return n;
}

void out(node * root) {
	bool isEnd = false;
	vector<node*> now;
	vector<node*> next;
	node * n = root;
	while (n) {
		now.push_back(n);
		n = n->r;
	}
	while (!isEnd) {
		isEnd = true;
		for (int i=now.size()-1; i>=0; --i)
			cout<<now[i]->ele<<" ";
		for (int i=0; i<now.size(); ++i)
			if (now[i]->l) {
				n = now[i]->l;
				while (n) {
					next.push_back(n);
					n = n->r;
				}
				isEnd = false;
			}
		now = next;
		next.clear();
	}

}

int main() {
	int num;
	cin>>num;

	node * root = generate();
	out(root);
	cout<<endl;

	return 0;
}
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



## 05344: 最后的最后

http://cs101.openjudge.cn/dsapre/05344/

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





# 41-50

## 05345: 位查询

http://cs101.openjudge.cn/dsapre/05345/

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

http://cs101.openjudge.cn/dsapre/05442/

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



## 05443: 兔子与樱花

http://cs101.openjudge.cn/dsapre/05443/

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



## 05907: 二叉树的操作

http://cs101.openjudge.cn/dsapre/05907/

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

http://cs101.openjudge.cn/dsapre/06648/

给定m个数字序列，每个序列包含n个非负整数。我们从每一个序列中选取一个数字组成一个新的序列，显然一共可以构造出n^m个新序列。接下来我们对每一个新的序列中的数字进行求和，一共会得到n^m个和，请找出最小的n个和

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



虑到n^m个和的数量可能非常大，我们不能直接存储它们。因此，我们可以通过逐步合并两个序列来找到最小的n个和，而不是一次性生成所有可能的和。

为了找到最小的n个和，我们可以按照以下步骤操作：

1. 对每个序列进行排序，确保我们可以从最小的元素开始处理。
2. 使用一个最小堆（优先队列）来维护当前可能的最小和。最初，我们只将每个序列的最小元素（即每个序列的第一个元素）的和放入最小堆中。
3. 每次从最小堆中取出当前的最小和，然后探索通过替换这个和中的某个元素来得到下一个可能的最小和。
4. 重复这个过程，直到我们找到了n个最小的和。

以下是一个更为内存效率的Python代码解决方案：

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





# 51-60

## 07161: 森林的带度数层次序列存储

http://cs101.openjudge.cn/dsapre/07161/

对于树和森林等非线性结构，我们往往需要将其序列化以便存储。有一种树的存储方式称为带度数的层次序列。我们可以通过层次遍历的方式将森林序列转化为多个带度数的层次序列。

例如对于以下森林：

![img](http://media.openjudge.cn/images/upload/1401904592.png)

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



## 07297: 神奇的幻方

http://cs101.openjudge.cn/dsapre/07297/

幻方是一个很神奇的N*N矩阵，它的每行、每列与对角线，加起来的数字和都是相同的。
我们可以通过以下方法构建一个幻方。（阶数为奇数）
1.第一个数字写在第一行的中间
2.下一个数字，都写在上一个数字的右上方：
  a.如果该数字在第一行，则下一个数字写在最后一行，列数为该数字的右一列
  b.如果该数字在最后一列，则下一个数字写在第一列，行数为该数字的上一行
  c.如果该数字在右上角，或者该数字的右上方已有数字，则下一个数字写在该数字的下方

**输入**

一个数字N（N<=20）

**输出**

按上方法构造的2N-1 * 2N-1的幻方

样例输入

```
3
```

样例输出

```
17 24 1 8 15
23 5 7 14 16
4 6 13 20 22
10 12 19 21 3
11 18 25 2 9
```



```python
# 23n2300011031
n=int(input())
t=n+n-1
l=[[0]*t for _ in range(t)]
i,j=0,n-1
for k in range(1,1+t**2):
    l[i][j]=k
    if (i==0 and j==t-1) or l[(i-1)%t][(j+1)%t]!=0:
        i,j=(i+1)%t,j
    else:
        i,j=(i-1)%t,(j+1)%t
for u in l:
    print(*u)
```



```python
# 23n2300017735(夏天明BrightSummer)
N = int(input())
mat = [[0]*(2*N-1) for j in range(2*N-1)]
pos = [0, N-1]
val = 1
while val <= (2*N-1)**2:
    mat[pos[0]%(2*N-1)][pos[1]%(2*N-1)] = val
    val += 1
    pos[0] -= 1
    pos[1] += 1
    if mat[pos[0]%(2*N-1)][pos[1]%(2*N-1)] != 0:
        pos[0] += 2
        pos[1] -= 1
for row in mat:
    print(*row)
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



## 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/

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



## 08758: 2的幂次方表示

http://cs101.openjudge.cn/dsapre/08758/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional Problems 部分相应题目



## 09201: Freda的越野跑

http://cs101.openjudge.cn/dsapre/09201/

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
            inv_count += mid - i
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

http://cs101.openjudge.cn/dsapre/20453/

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

http://cs101.openjudge.cn/dsapre/20456/

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





# 61-70

## 20472: 死循环的机器人

http://cs101.openjudge.cn/dsapre/20472/

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

http://cs101.openjudge.cn/dsapre/20576/

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

def postorder(string):    #中缀改后缀
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
        return ['not'] + (['('] + printTree(parsetree.getleftchild()) + [')'] if parsetree.leftChild.getroot() not in ['True', 'False'] else printTree(parsetree.getleftchild()))
    elif parsetree.getroot() == 'and':
        leftpart = ['('] + printTree(parsetree.getleftchild()) + [')'] if parsetree.leftChild.getroot() == 'or' else printTree(parsetree.getleftchild())
        rightpart = ['('] + printTree(parsetree.getrightchild()) + [')'] if parsetree.rightChild.getroot() == 'or' else printTree(parsetree.getrightchild())
        return leftpart + ['and'] + rightpart
    else:
        return [str(parsetree.getroot())]

def main():
    infix = input()
    Tree = buildParseTree(infix)
    print(' '.join(printTree(Tree)))

main()
```



## 20625: 1跟0数量相等的子字串

http://cs101.openjudge.cn/dsapre/20625/

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

http://cs101.openjudge.cn/dsapre/20626/

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

http://cs101.openjudge.cn/dsapre/20644/

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

http://cs101.openjudge.cn/dsapre/20650/

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



## 20741: 两座孤岛最短距离

http://cs101.openjudge.cn/dsapre/20741/

给一个由1跟0组成的方形地图，1代表土地，0代表水域

相邻(上下左右4个方位当作相邻)的1组成孤岛

现在你可以将0转成1，搭建出一个链接2个孤岛的桥

请问最少要将几个0转成1，才能建成链接孤岛的桥。

题目中恰好有2个孤岛(顾答案不会是0)

**输入**

一个正整数n，代表几行输入
n行0跟1字串

**输出**

一个正整数k，代表最短距离

样例输入

```
3
110
000
001
```

样例输出

```
2
```

提示

样例输入中的两个孤岛最短距离为2



dfs + bfs

```python
from collections import deque


class Solution:
    def shortestBridge(self, grid) -> int:
        m, n = len(grid), len(grid[0])
        points = deque()

        def dfs(points, grid, m, n, i, j):
            if i < 0 or i == m or j < 0 or j == n or grid[i][j] == 2:
                return
            if grid[i][j] == 0:
                points.append((i, j))
                return

            grid[i][j] = 2
            dfs(points, grid, m, n, i - 1, j)
            dfs(points, grid, m, n, i + 1, j)
            dfs(points, grid, m, n, i, j - 1)
            dfs(points, grid, m, n, i, j + 1)

        flag = False
        for i in range(m):
            if flag:
                break
            for j in range(n):
                if grid[i][j] == 1:
                    dfs(points, grid, m, n, i, j)
                    flag = True
                    break

        x, y, count = 0, 0, 0
        while points:
            count += 1
            n_points = len(points)
            while n_points > 0:
                point = points.popleft()
                r, c = point[0], point[1]
                for k in range(4):
                    x, y = r + direction[k], c + direction[k + 1]
                    if x >= 0 and y >= 0 and x < m and y < n:
                        if grid[x][y] == 2:
                            continue
                        if grid[x][y] == 1:
                            return count
                        points.append((x, y))
                        grid[x][y] = 2
                n_points -= 1

        return 0


direction = [-1, 0, 1, 0, -1]

n = int(input())
grid = []
for i in range(n):
    row = list(map(int, list(input())))
    grid.append(row)

print(Solution().shortestBridge(grid))
```



只有一个break，(sr,sc)是最右下的陆地点，先入队列。从它开始找是可能最近的桥。

for r in range(n):
    for c in range(n):
        if grid\[r][c] == 1:
            sr, sc = r, c
            break

```python
import collections
def main():
    n=int(input())
    #grid=[[0]*(n+2)]
    grid=[]
    for i in range(n):
        p=list(int(x) for x in input())
        #p.insert(0,0)
        #p.append(0)
        grid.append(p)
    grid.append([0]*(n+2))
    visited = [[False for _ in range(n)] for _ in range(n)]
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    sr, sc = -1, -1
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                sr, sc = r, c
                break
    q = collections.deque()
    q.append((sr, sc))
    visited[sr][sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] == 1 and visited[nr][nc] == False:
                    visited[nr][nc] = True
                    q.append((nr, nc))

    #------------ 计算最短距离。多源bfs
    for r in range(n):
        for c in range(n):
            if visited[r][c] == True and grid[r][c] == 1:
                q.append((r, c))
    step = 0
    while q:
        curLen = len(q)
        for _ in range(curLen):
            r, c = q.popleft()
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n and 0 <= nc < n and visited[nr][nc] == False:
                    visited[nr][nc] = True
                    if grid[nr][nc] == 1:
                        return step
                    q.append((nr, nc))
        step += 1
    return step
print(main())
```



## 20742: 泰波拿契數

http://cs101.openjudge.cn/dsapre/20742/

泰波拿契数列 Tn 定义是

$T_0 = 0, T_1 = 1, T_2 = 1, and T_{n+3} = T_n + T_{n+1} + T_{n+2} \space for \space n >= 0$.

给定n请算出Tn

n的范围:1<=n<=30



**输入**

一个正整数n

**输出**

一个正整数k

样例输入

```
4
```

样例输出

```
4
```

提示

T3=0 + 1 + 1 = 2
T4=1 + 1 + 2 = 4



```python
def tribonacci(n):
    if n == 0:
        return 0
    elif n <= 2:
        return 1
    trib = [0, 1, 1] + [0] * (n - 2)
    for i in range(3, n + 1):
        trib[i] = trib[i - 1] + trib[i - 2] + trib[i - 3]
    return trib[n]

# 读取输入并处理
n = int(input())
print(tribonacci(n))
```



## 20743: 整人的提词本

http://cs101.openjudge.cn/dsapre/20743/

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



## 20744: 土豪购物

http://cs101.openjudge.cn/dsapre/20744/

给一个整数组成的数列，其中每个数字代表商品价值(可能为负)

土豪买东西的方法是 "从第n个到第k个商品我全要了!!!" (n<=k)，

换句话说土豪一定会买下连续的几个商品



买完以后土豪会看心情最多放回去其中一个商品(可以不放回)

但土豪不能空手而归，他至少要带回去一个商品

请问聪明的(?)土豪可以买到最大价值总和为多少的商品?

样例:

商品价值:1,-5,0,3 输出:4 最大价值总和是买[1,-5,0,3]，并放回-5后的总和

商品价值:-2,-2,-2 输出:-2 最大价值总和是买[-2]，不放回的总和(至少要带回去一个商品)



**输入**

一个逗号分隔，由整数组成的商品价值

**输出**

一个整数

样例输入

```
1,-5,0,3
```

样例输出

```
4
```

提示

最大价值总和是买[1,-5,0,3]，并放回-5后的总和



需要考虑两种情况：

1. 不放回商品时的最大连续子数组和（Kadane算法）。
2. 放回一个商品时的最大连续子数组和。

由于我们可以选择放回任何一个商品，因此需要考虑放回每一个商品对最大连续子数组和的影响。我们可以通过两次遍历数组来解决这个问题：

- 第一次遍历从左到右计算以每个元素结尾的最大子数组和。
- 第二次遍历从右到左计算以每个元素开始的最大子数组和。

然后，我们遍历数组，对于每个位置，我们尝试放回该位置的商品，并检查如果放回这个商品后，左边子序列的最大和加上右边子序列的最大和是否会比当前的最大值还要大。

```python
def kadane(nums):
    max_ending_here = max_so_far = nums[0]
    for x in nums[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

def max_sum_shopping(values):
    # 不放回商品的情况下的最大价值总和
    max_without_deletion = kadane(values)

    # 如果整个数列的和都是负的，则土豪只能选择一个价值最大的商品
    if max_without_deletion < 0:
        return max(values)

    # 准备两个数组来存储从左到右和从右到左的最大子数组和
    left_max_sums = [0] * len(values)
    right_max_sums = [0] * len(values)

    # 从左到右的最大子数组和
    current = 0
    for i in range(len(values)):
        current = max(0, current + values[i])
        left_max_sums[i] = current

    # 从右到左的最大子数组和
    current = 0
    for i in range(len(values) - 1, -1, -1):
        current = max(0, current + values[i])
        right_max_sums[i] = current

    # 放回一个商品时的最大价值总和
    max_with_deletion = 0
    for i in range(1, len(values) - 1):
        max_with_deletion = max(max_with_deletion, left_max_sums[i - 1] + right_max_sums[i + 1])

    # 返回放回一个商品和不放回一个商品两种情况下的最大价值
    return max(max_with_deletion, max_without_deletion)

# 读取输入并处理
values_str = input().strip()
values = list(map(int, values_str.split(',')))
print(max_sum_shopping(values))
```





# 71-80

## 20746: 满足合法工时的最少人数

http://cs101.openjudge.cn/dsapre/20746/

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

http://cs101.openjudge.cn/dsapre/21006/

把M个同样的苹果放在N个同样的盘子里，允许有的盘子空着不放，问共有多少种不同的分法？（用K表示）5，1，1和1，5，1 是同一种分法。

**输入**

苹果个数m 和盘子个数n(0<=M，1<=N<=10)

**输出**

不同的放法数目

样例输入

```
7 3
```

样例输出

```
8
```



这是一个经典的组合数学问题，也被称为“球和盒子模型”。使用动态规划来解决这个问题。定义dp\[i][j]为将i个苹果放入j个盘子的方法数。那么，我们有两种情况：

1. 所有的盘子都至少放一个苹果，那么问题就变成了如何将i-j个苹果放入j个盘子。这就是dp\[i-j][j]。

2. 至少有一个盘子是空的，那么问题就变成了如何将i个苹果放入j-1个盘子。这就是dp\[i][j-1]。

因此，我们有$dp[i][j] = dp[i-j][j] + dp[i][j-1]$。我们可以使用一个二维数组来存储dp值，并使用嵌套循环来计算所有的dp值。

以下是Python代码实现：

```python
# GitHub copilot
def count_ways(m, n):
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(m+1):
        dp[i][1] = 1
    for i in range(1, m+1):
        for j in range(2, min(i, n)+1):
            dp[i][j] = dp[i-j][j] + dp[i][j-1]
    return dp[m][n]

# Processing input
try:
    while True:
        m, n = map(int, input().split())
        print(count_ways(m, n))
except EOFError:
    pass
```





```python
# 23n2300011427
m,n=map(int,input().split())
dp=[[0]*(m+1) for i in range(n+1)]
for i in range(1,n+1):
    dp[i][0]=1

dp[1]=[1]*(m+1)
for i in range(1,n+1):
    dp[i][1]=1
for i in range(1,n+1):
    for j in range(1,m+1):
        if i>j:
            dp[i][j]=dp[j][j]
        else:
            dp[i][j]=dp[i-1][j]+dp[i][j-i]
print(dp[n][m])
```



## 22067: 快速堆猪

http://cs101.openjudge.cn/dsapre/22067/

小明有很多猪，他喜欢玩叠猪游戏，就是将猪一头头叠起来。猪叠上去后，还可以把顶上的猪拿下来。小明知道每头猪的重量，而且他还随时想知道叠在那里的猪最轻的是多少斤。

输入

有三种输入
1)push n
n是整数(0<=0 <=20000)，表示叠上一头重量是n斤的新猪
2)pop
表示将猪堆顶的猪赶走。如果猪堆没猪，就啥也不干
3)min
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



辅助栈

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

http://cs101.openjudge.cn/dsapre/22068/

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
ef post_order(pre_order):
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



## 22359: Goldbach Conjecture

http://cs101.openjudge.cn/dsapre/22359/

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





## 22509: 解方程

http://cs101.openjudge.cn/dsapre/22509/



题解在 https://github.com/GMyhf/2020fall-cs101 题集 2020fall_cs101.openjudge.cn_problems.md

的 Optional problems



## 22548: 机智的股民老张

http://cs101.openjudge.cn/dsapre/22548/

股民老张通过某种渠道事先知道了一支股票在未来几天的价格。老张获得了一个数组 a，其中 a[i] 是给定股票在第 i 天的价格。

现在老张希望通过选择某一天购买该股票并选择未来的另一天出售该股票来最大化他的利润。

返回老张可以从这次交易中获得的最大利润。 如果价格一直下跌，老张无法获得任何利润，则返回 0。

**输入**

由空格分开的若干非负整数，数组 a 的长度不超过 100,000，各元素 a[i] 满足 0 <= a[i] <= 10000。

**输出**

一个数，可以从这次交易中获得的最大利润

样例输入

```
7 1 5 3 6 4
```

样例输出

```
5
```

提示

在第二天买入（价格为 1），在第五天买出（价格为 6），因此收益为 5。

相反，如果输入为 7 6 5 4 3 1，则老张怎么买都不可能获得利润，因此返回 0。



```python
*a, = map(int, input().split())
min_price = float('inf')
max_profit = 0

for price in a:
    min_price = min(min_price, price)  # 更新最小值
    max_profit = max(max_profit, price - min_price)  # 更新最大利润

print(max_profit)
```



## 22636: 修仙之路

http://cs101.openjudge.cn/dsapre/22636/

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





## 23451: 交互四则运算计算器_带错误表达式版

http://cs101.openjudge.cn/dsapre/23451/

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





# 81-90

## 23563: 多项式时间复杂度

http://cs101.openjudge.cn/dsapre/23563/

请参看 2020fall_cs101.openjudge.cn_problems.md 的 Basic Exercises部分的相同题目。



## 23568: 幸福的寒假生活

http://cs101.openjudge.cn/dsapre/23568/

请参看 2020fall_cs101.openjudge.cn_problems.md 的 Optional problems部分的相同题目。



## 23570: 特殊密码锁

http://cs101.openjudge.cn/dsapre/23570/

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

http://cs101.openjudge.cn/dsapre/23660/

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

http://cs101.openjudge.cn/dsapre/24375/

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



与 Optional problems的 01011: Sticks 一样的题目，算法说明也在 01011。



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

http://cs101.openjudge.cn/dsapre/24588/

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





## 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/dsapre/24591/

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



## 24676: 共同富裕

http://cs101.openjudge.cn/dsapre/24676/

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

http://cs101.openjudge.cn/dsapre/24677/

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

http://cs101.openjudge.cn/dsapre/24678/

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





# 91-102

## 24684: 直播计票

http://cs101.openjudge.cn/dsapre/24684/

直播间发起了投票活动：在屏幕上列出若干选项，观众通过发送弹幕向自己支持的选项投票。在幕后工作的你需要根据弹幕信息，向直播间的观众们展示哪个选项得票最多。

这里每个选项用一个正整数编号表示。

**输入**

输入只有一行，由若干正整数组成，每个正整数表示这条弹幕是投票给哪个选项的。

输入的正整数个数不超过100,000，且满足最多有100个不同的选项，选项的编号不超过100,000。

**输出**

输出只有一行，为得票最多的选项。若有并列第一的情况出现，则按编号从小到大依次输出所有得票数最多的选项，用空格隔开。

样例输入

```
1 10 2 3 3 10
```

样例输出

```
3 10
```



```python
from collections import defaultdict

# 读取输入并转换成整数列表
votes = list(map(int, input().split()))

# 使用字典统计每个选项的票数
vote_counts = defaultdict(int)
for vote in votes:
    vote_counts[vote] += 1

# 找出得票最多的票数
max_votes = max(vote_counts.values())

# 按编号顺序收集得票最多的选项
winners = sorted([item for item in vote_counts.items() if item[1] == max_votes])

# 输出得票最多的选项，如果有多个则并列输出
print(' '.join(str(winner[0]) for winner in winners))

```



## 24686: 树的重量

http://cs101.openjudge.cn/dsapre/24686/

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

http://cs101.openjudge.cn/dsapre/24687/

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



## 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/dsapre/24750/

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





## 25815: 回文字符串

http://cs101.openjudge.cn/dsapre/25815/

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



## 26572: 多余的括号

http://cs101.openjudge.cn/dsapre/26572/

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
```

样例输出

```
1+11
1+2+3*(4+5)
```

来源: lxp



```python
# 23n2300011031 黄源森23工院 version2
"""
parsing and evaluating mathematical expressions involving addition (+) and multiplication (*)
with a simple form of precedence (multiplication before addition) without
using the built-in eval function.
"""
import re

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def dfs(node):
    if isinstance(node, str):
        return node
    if node.value == '*':
        left = dfs(node.left)
        right = dfs(node.right)
        if isinstance(node.left, Node) and node.left.value == '+':
            left = f'({left})'
        if isinstance(node.right, Node) and node.right.value == '+':
            right = f'({right})'
        return left + '*' + right
    else:
        return dfs(node.left) + node.value + dfs(node.right)

def build_tree(tokens):
    def helper(tokens):
        stack = []
        for token in tokens:
            if token == ')':
                sub_expr = []
                while stack and stack[-1] != '(':
                    sub_expr.append(stack.pop())
                stack.pop()  # Remove the '(' symbol
                stack.append(build_tree(sub_expr[::-1]))
            else:
                stack.append(token)
        return stack

    tokens = helper(tokens)
    # Process multiplication with higher precedence
    while '*' in tokens:
        index = tokens.index('*')
        node = Node('*', tokens[index - 1], tokens[index + 1])
        tokens = tokens[:index - 1] + [node] + tokens[index + 2:]
    # Process addition
    if len(tokens) == 1:
        return tokens[0]
    left_operand = tokens[0]
    for i in range(1, len(tokens), 2):
        left_operand = Node(tokens[i], left_operand, tokens[i + 1])
    return left_operand

while True:
    try:
        expression = input()
        tokens = [token for token in re.split(r"(\D)", expression) if token]
        root = build_tree(tokens)
        print(dfs(root))
    except EOFError:
        break

```



```python
#  23 元培 夏天明
"""
用栈搜索括号，然后暴力枚举尝试去掉每个括号，检验是否改变表达式本身。

但是数据弱了，例如：
(1+1)*1
1+1*1
"""
import re

while True:
    try:
        s = re.split(r"(\D)", input())
    except EOFError:
        break
    pf = eval(''.join(s))
    parenthesis = []
    stack = []
    for i, token in enumerate(s):
        if token == '(':
            stack.append(len(parenthesis))
            parenthesis.append([i])
        elif token == ')':
            parenthesis[stack.pop()].append(i)
    for l, r in parenthesis:
        s[l] = ''
        s[r] = ''
        if pf != eval(''.join(s)):
            s[l] = '('
            s[r] = ')'
    print(''.join(s))
```



## 26573: 康托集的图像表示

http://cs101.openjudge.cn/dsapre/26573/

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



## 27625: AVL树至少有几个结点

http://cs101.openjudge.cn/dsapre/27625/

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

http://cs101.openjudge.cn/dsapre/27626/

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

http://cs101.openjudge.cn/dsapre/27635/

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



## 27637: 括号嵌套二叉树

http://cs101.openjudge.cn/practice/27637/

可以用括号嵌套的方式来表示一棵二叉树。方法如下：'*'表示空的二叉树如果一棵二叉树只有一个结点，则该树就用一个非'*'字符表示，代表其根结点。如果一棵二叉左右子树都非空，则用“树根(左子树,右子树)”的形式表示。树根是一个非'*'字符,左右子树之间用逗号隔开，没有空格。左右子树都用括号嵌套法表示。如果左子树非空而右子树为空，则用“树根(左子树)”形式表示；如果左子树为空而右子树非空，则用“树根(*,右子树)”形式表示。给出一棵树的括号嵌套表示形式，请输出其前序遍历序列、中序遍历序列、后序遍历序列。例如，"A(B(*,C),D(E))"表示的二叉树如图所示

![img](http://media.openjudge.cn/images/upload/1636/1707558029.jpg)

**输入**

第一行是整数n表示有n棵二叉树(n<100) 接下来有n行，每行是1棵二叉树的括号嵌套表示形式

**输出**

对每棵二叉树，输出其前序遍历序列和中序遍历序列

样例输入

```
2
A
A(B(*,C),D(E))
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



```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def parse_tree(s):
    if s == '*':
        return None
    if '(' not in s:
        return TreeNode(s)

    # Find the root value and the subtrees
    root_value = s[0]
    subtrees = s[2:-1]  # Remove the root and the outer parentheses

    # Use a stack to find the comma that separates the left and right subtrees
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

    # Parse the subtrees
    root = TreeNode(root_value)
    root.left = parse_tree(left_subtree)
    root.right = parse_tree(right_subtree) if right_subtree else None
    return root


# Define the traversal functions
def preorder_traversal(root):
    if root is None:
        return ""
    return root.value + preorder_traversal(root.left) + preorder_traversal(root.right)


def inorder_traversal(root):
    if root is None:
        return ""
    return inorder_traversal(root.left) + root.value + inorder_traversal(root.right)


# Input reading and processing
n = int(input().strip())
for _ in range(n):
    tree_string = input().strip()
    tree = parse_tree(tree_string)
    preorder = preorder_traversal(tree)
    inorder = inorder_traversal(tree)
    print(preorder)
    print(inorder)
```



## 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/dsapre/27638/

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







# 题目渐进性分类

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





## 2023期末上机考试（数算B）7题

02774: 木材加工

http://cs101.openjudge.cn/practice/02774/

02766: 最大子矩阵

http://cs101.openjudge.cn/practice/02766/

26573: 康托集的图像表示

http://cs101.openjudge.cn/practice/26573/

26572: 多余的括号

http://cs101.openjudge.cn/practice/26572/

06364: 牛的选举

http://cs101.openjudge.cn/practice/06364

03720: 文本二叉树

http://cs101.openjudge.cn/practice/03720/

05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



## 2022期末测试（校外）

24684: 直播计票

http://cs101.openjudge.cn/practice/24684/

24677: 安全位置

http://cs101.openjudge.cn/practice/24677/

24676: 共同富裕

http://cs101.openjudge.cn/practice/24676/

24678: 任性买房

http://cs101.openjudge.cn/practice/24678/

24686: 树的重量

http://cs101.openjudge.cn/dsapre/24686/

24687: 封锁管控

http://cs101.openjudge.cn/practice/24687/



## 2021模拟考试/2020finaltest

20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/

20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/

20741: 两座孤岛最短距离

http://cs101.openjudge.cn/practice/20741

20746: 满足合法工时的最少人数

http://cs101.openjudge.cn/practice/20746/

20626: 对子数列做XOR运算

http://cs101.openjudge.cn/practice/20626/

20744: 土豪购物

http://cs101.openjudge.cn/practice/20744/



## 2020模拟上机

20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/

20453: 和为k的子数组个数

http://cs101.openjudge.cn/practice/20453/

20456: 统计封闭岛屿的数目

http://cs101.openjudge.cn/practice/20456/

20472: 死循环的机器人

http://cs101.openjudge.cn/practice/20472/

20625: 1跟0数量相等的子字串

http://cs101.openjudge.cn/practice/20625/

20644: 统计全为 1 的正方形子矩阵

http://cs101.openjudge.cn/practice/20644/



## 二叉树的应用

### 20555: evaluate

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



20576: printExp

http://cs101.openjudge.cn/practice/20576/







