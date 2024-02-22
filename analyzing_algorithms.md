# 20240227-Week2-分析算法

Updated 1015 GMT+8 Feb 22, 2024

2024 spring, Complied by Hongfei Yan



**本周发布作业：**

Assignment2, https://github.com/GMyhf/2024spring-cs201

27653: Fraction类

http://cs101.openjudge.cn/2024sp_routine/27653/

04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110

18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B

1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A

18176:2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



作业评分标准

| 标准                                 | 等级                                             | 得分 |
| ------------------------------------ | ------------------------------------------------ | ---- |
| 按时提交                             | 1 得分提交，0.5 得分请假，0 得分未提交           | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 1 得分4或4+题目，0.5 得分2或2+题目，0 得分无源码 | 1 分 |
| AC代码截图                           | 1 得分4或4+题目，0.5 得分2或2+题目，0 得分无截图 | 1 分 |
| 清晰头像、pdf、md/doc                | 1 得分三项全，0.5 得分有二项，0 得分少于二项     | 1 分 |
| 学习总结和收获                       | 1 得分有，0 得分无                               | 1 分 |
| 总得分： 5 ，满分 5                  |                                                  |      |



**Puzzle**

22507:薛定谔的二叉树

http://cs101.openjudge.cn/practice/22507/

假设二叉树的节点里包含一个大写字母，每个节点的字母都不同。
给定二叉树的前序遍历序列和后序遍历序列(长度均不超过20)，请计算二叉树可能有多少种

前序序列或后序序列中出现相同字母则直接认为不存在对应的树

**输入**

多组数据
每组数据一行，包括前序遍历序列和后序遍历序列，用空格分开。
输入数据不保证一定存在满足条件的二叉树。

**输出**

每组数据，输出不同的二叉树可能有多少种

样例输入

```
ABCDE CDBEA
BCD DCB
AB C
AA AA
```

样例输出

```
1
4
0
0
```

来源: 刘宇航



![image-20240222105900057](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240222105900057.png)



# 一、时间复杂度



https://stackoverflow.com/questions/44421828/is-this-sieve-really-on



https://www.geeksforgeeks.org/sieve-eratosthenes-0n-time-complexity/



## Analyzing algorithms

**Analyzing** an algorithm has come to mean predicting the resources that the algorithm requires. Occasionally, resources such as memory, communication bandwidth, or computer hardware are of primary concern, but most often it is computational time that we want to measure. Generally, by analyzing several candidate algorithms for a problem, we can identify a most efficient one. Such analysis may indicate more than one viable candidate, but we can often discard several inferior algorithms in the process.

Before we can analyze an algorithm, we must have a model of the implementation technology that we will use, including a model for the resources of that technology and their costs. For most of this book, we shall assume a generic oneprocessor, **random-access machine (RAM)** model of computation as our implementation technology and understand that our algorithms will be implemented as computer programs. In the RAM model, instructions are executed one after another, with no concurrent operations.
Strictly speaking, we should precisely define the instructions of the RAM model and their costs. To do so, however, would be tedious and would yield little insight into algorithm design and analysis. Yet we must be careful not to abuse the RAM model. For example, what if a RAM had an instruction that sorts? Then we couldsort in just one instruction. Such a RAM would be unrealistic, since real computers do not have such instructions. Our guide, therefore, is how real computers are designed. The RAM model contains instructions commonly found in real computers: arithmetic (such as add, subtract, multiply, divide, remainder, floor, ceiling), data movement (load, store, copy), and control (conditional and unconditional branch, subroutine call and return). Each such instruction takes a constant amount of time.

The data types in the RAM model are integer and floating point (for storing real numbers). Although we typically do not concern ourselves with precision in this book, in some applications precision is crucial. We also assume a limit on the size of each word of data. For example, when working with inputs of size n, we typically assume that integers are represented by c lg n bits for some constant $c \ge 1$. We require $c \ge 1$​ so that each word can hold the value of n, enabling us to index the individual input elements, and we restrict c to be a constant so that the word size does not grow arbitrarily. (If the word size could grow arbitrarily, we could store huge amounts of data in one word and operate on it all in constant time—clearly an unrealistic scenario.)

> 计算机字长（Computer Word Length）是指计算机中用于存储和处理数据的基本单位的位数。它表示计算机能够一次性处理的二进制数据的位数。
>
> 字长的大小对计算机的性能和数据处理能力有重要影响。较大的字长通常意味着计算机能够处理更大范围的数据或执行更复杂的操作。常见的字长包括 8 位、16 位、32 位和 64 位。
>
> 较小的字长可以节省存储空间和资源，适用于简单的计算任务和资源有限的设备。较大的字长通常用于处理更大量级的数据、进行复杂的计算和支持高性能计算需求。
>
> 需要注意的是，字长并不是唯一衡量计算机性能的指标，还有其他因素如处理器速度、缓存大小、操作系统等也会影响计算机的整体性能。

Real computers contain instructions not listed above, and such instructions represent a gray area in the RAM model. For example, is exponentiation a constanttime instruction? In the general case, no; it takes several instructions to compute xy
when x and y are real numbers. In restricted situations, however, exponentiation is a constant-time operation. Many computers have a “shift left” instruction, which in constant time shifts the bits of an integer by k positions to the left. In most computers, shifting the bits of an integer by one position to the left is equivalent to multiplication by 2, so that shifting the bits by k positions to the left is equivalent to multiplication by $2^k$. Therefore, such computers can compute $2^k$ in one constant-time instruction by shifting the integer 1 by k positions to the left, as long as k is no more than the number of bits in a computer word. We will endeavor to avoid such gray areas in the RAM model, but we will treat computation of $2^k$ as a constant-time operation when k is a small enough positive integer.

In the RAM model, we do not attempt to model the memory hierarchy that is common in contemporary computers. That is, we do not model caches or virtual memory. Several computational models attempt to account for memory-hierarchy effects, which are sometimes significant in real programs on real machines. A handful of problems in this book examine memory-hierarchy effects, but for the most part, the analyses in this book will not consider them. Models that include the memory hierarchy are quite a bit more complex than the RAM model, and so they can be difficult to work with. Moreover, RAM-model analyses are usually excellent predictors of performance on actual machines.

Analyzing even a simple algorithm in the RAM model can be a challenge. The mathematical tools required may include combinatorics, probability theory, algebraic dexterity, and the ability to identify the most significant terms in a formula. Because the behavior of an algorithm may be different for each possible input, we need a means for summarizing that behavior in simple, easily understood formulas.

Even though we typically select only one machine model to analyze a given algorithm, we still face many choices in deciding how to express our analysis. We would like a way that is simple to write and manipulate, shows the important characteristics of an algorithm’s resource requirements, and suppresses tedious details.







# 二、



# 三、题目

## 02810: 完美立方

bruteforce, http://cs101.openjudge.cn/practice/02810

## 02808: 校门外的树

implementation, http://cs101.openjudge.cn/practice/02808

## 04146: 数字方格

math, http://cs101.openjudge.cn/practice/04146



作业

assignment2.md,  at https://github.com/GMyhf/2024spring-cs201



# 四、笔试题目

## 选择（每题2分）

Q: 下列不影响算法时间复杂性的因素有（ ）。
A：问题的规模	B：输入值	C：计算结果	D：算法的策略



Q: 有 $n^2$​ 个整数，找到其中最小整数需要比较次数至少为（ ）次。

A:$n$	B: $log_{2}{n}$	C:$n^2-1$	D:$n-1$



假设有 $n^2$ 个整数，我们需要找到其中的最小整数。在最坏的情况下，最小整数可能位于数组的最后一个位置，因此我们需要比较 $n^2 - 1$ 次才能确定最小整数。

具体地说，我们可以进行以下步骤来找到最小整数：

1. 假设第一个整数为当前的最小整数。
2. 依次比较当前最小整数和数组中的其他整数。
3. 如果遇到比当前最小整数更小的整数，将其设为当前最小整数。
4. 重复步骤 2 和 3，直到遍历完所有的 $n^2$ 个整数。

在这个过程中，我们需要进行 $n^2 - 1$ 次比较才能找到最小整数。这是因为第一个整数不需要与自身进行比较，而后续的每个整数都需要与之前的最小整数进行比较。



## 判断（每题1分）

Q: 考虑一个长度为 n 的顺序表中各个位置插入新元素的概率是相同的，则顺序表的插入算法平均时间复杂度为 $O(n) $。

A: Y	B: N



Q: 直接插入排序、冒泡排序、 希尔排序都是在数据正序的情况下比数据在逆序的情况下要快。

A: Y	B: N



Q: 用相邻接矩阵法存储一个图时，在不考虑压缩存储的情况下，所占用的存储空间大小只与图中结点个数有关，而与图的边数无关。

A: Y	B: N



## 填空（每题2分）

Q: 线性表的顺序存储与链式存储是两种常见存储形式；当表元素有序排序进行二分检索时，应采用（ ）存储形式。



Q: 如果只想得到 1000 个元素的序列中最小的前 5 个元素，在冒泡排序、快速排序、堆排序和归并排序中，哪种算法最快？ （ ）



## 简答（每题6分）

Q: 哈夫曼树是进行编码的一种有效方式。设给定五个字符，其相应的权值分别为{4， 8， 6， 9， 18}， 试画出相应的哈夫曼树，并计算它的带权外部路径长度 WPL 。



为了构建哈夫曼树，我们可以按照以下步骤进行：

1. 将给定的字符和权值按照权值从小到大的顺序进行排序。

字符：   A    B    C    D    E
权值：   4    6    8    9    18

2. 从排序后的字符和权值列表中选取权值最小的两个节点，创建一个新节点作为它们的父节点，并将权值设置为这两个节点的权值之和。

字符：   A C    B    D    E
权值：   4 8    6    9   18

字符：         AC    B    D    E
权值：         12    6    9   18

3. 重复步骤 2，直到只剩下一个节点，这个节点就是哈夫曼树的根节点。

字符：         AC   BDE
权值：         12   33

字符：              ABCDE
权值：              45

下面是绘制的哈夫曼树的图形表示：

```
      45
     /  \
   12    33
  /  \  
 A    12
     /  \
    4    8
```

计算带权外部路径长度（Weighted Path Length，WPL）的方法是将每个叶子节点的权值乘以其到根节点的路径长度，然后将所有叶子节点的乘积求和。在这个例子中，WPL 的计算如下：

WPL = (4 * 2) + (6 * 3) + (8 * 3) + (9 * 2) + (18 * 2) = 82

因此，这棵哈夫曼树的带权外部路径长度为 82。





## 算法（每题8分）

阅读下列程序，完成图的深度优先周游算法实现的迷宫探索。已知图采用邻接表表示，Graph 类和 Vertex 类基本定义如下：

```python
import sys													#这个程序运行不起来！！！
sys.setrecursionlimit(10000000)

class Graph:
    def __init__(self):
        self.vertices = {}

    def addVertex(self, key, label):
        self.vertices[key] = Vertex(key, label)

    def getVertex(self, key):
        return self.vertices.get(key)

    def __contains__(self, key):
        return key in self.vertices

    def addEdge(self, f, t, cost=0):
        if f in self.vertices and t in self.vertices:
            self.vertices[f].addNeighbor(t, cost)

    def getVertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, key, label=None):
        self.id = key
        self.label = label
        self.color = "white"
        self.connections = {}

    def addNeighbor(self, nbr, weight=0):
        self.connections[nbr] = weight

    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color

    def getConnections(self):
        return self.connections.keys()

    def getId(self):
        return self.id

    def getLabel(self):
        return self.label


mazelist = [
    "++++++++++++++++++++++",
    "+   +   ++ ++        +",
    "E     +     ++++++++++",
    "+ +    ++  ++++ +++ ++",
    "+ +   + + ++    +++  +",
    "+          ++  ++  + +",
    "+++++ + +      ++  + +",
    "+++++ +++  + +  ++   +",
    "+         + + S+ +   +",
    "+++++ +  + + +     + +",
    "++++++++++++++++++++++",
]


def mazeGraph(mlist, rows, cols):
    mGraph = Graph()
    vstart = None
    for row in range(rows):
        for col in range(cols):
            if mlist[row][col] != "+":
                mGraph.addVertex((row, col), mlist[row][col])
                if mlist[row][col] == "S":
                    vstart = mGraph.getVertex((row, col))   # 等号右侧填空（1分）

    for v in mGraph:
        row, col = v.getId()
        for i in [(-1, 0), (1, 0), (0, -1), (0, +1)]:
            if 0 <= row + i[0] < rows and 0 <= col + i[1] < cols:
                if (row + i[0], col + i[1]) in mGraph:
                    mGraph.addEdge((row, col), (row + i[0], col + i[1])) #括号中两个参数填空（1分）

    return mGraph, vstart


def searchMaze(path, vcurrent, mGraph):
    path.append(vcurrent.getId())
    #print(path)
    if vcurrent.getLabel() != "E":
        done = False
        for nbr in vcurrent.getConnections(): # in 后面部分填空（2分）
            nbr_vertex = mGraph.getVertex(nbr)
            if nbr_vertex.getColor() == "white":
                done = searchMaze(path, nbr_vertex, mGraph) # 参数填空（2分）
                if done:
                    break
        if not done:
            path.pop()  # 这条语句空着，填空（2分）
            vcurrent.setColor("white")
    else:
        done = True
    return done


g, vstart = mazeGraph(mazelist, len(mazelist), len(mazelist[0]))
path = []
searchMaze(path, vstart, g)
print(path)
```





# 参考

Introduction to Algorithms, 3rd Edition (Mit Press) 3rd Edition, by Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, Clifford Stein

https://dahlan.unimal.ac.id/files/ebooks/2009%20Introduction%20to%20Algorithms%20Third%20Ed.pdf

or

https://github.com/calvint/AlgorithmsOneProblems/blob/master/Algorithms/Thomas%20H.%20Cormen,%20Charles%20E.%20Leiserson,%20Ronald%20L.%20Rivest,%20Clifford%20Stein%20Introduction%20to%20Algorithms,%20Third%20Edition%20%202009.pdf



Complexity of Python Operations 数据类型操作时间复杂度

https://www.ics.uci.edu/~pattis/ICS-33/lectures/complexitypython.txt

This is called "static" analysis, because we
do not need to run any code to perform it (contrasted with Dynamic or Empirical
Analysis, when we do run code and take measurements of its execution).