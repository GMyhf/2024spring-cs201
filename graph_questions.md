# 20240409～23-Week8~10 图论

Updated 1756 GMT+8 Apr 14, 2024

2024 spring, Complied by Hongfei Yan



> Logs:
>
> 2024/4/7 打*的章节，可以跳过，可能超纲了。
>
> 2024/4/5 数算重点是树、图、和算法。图这部分重点是算法，因为图的主要表示方式——邻接表，树也使用。
>
> 其中涉及到矩阵存储的图遍历，基本都是计概题目。



> 这可以是笔试判断题：
>
> 计算机存储器线性编址的简单性与程序逻辑的复杂性之间的鸿沟，导致了数据结构的必要性。✅
>
> 取自：李晓明. 为什么会有“数据结构”？计算机教育，2019-01-18 



```mermaid
mindmap
  Graph(Graph)
    Notations{{**NOTATIONS**}}
    	Vertex,Edge
    	Path, Weight
    	DFS, BFS
      
    Representation{{**REPRESENTATION**}}
      Matrix
      Adjacency List
      
    Algorithm{{**ALGORITHM**}}
    	Shortest Path
    		Dijkstra
    		Bellman-Ford, SPFA
    		Floyd-Warshall 
    	Topological Sorting
    	MST(Minimum Spanning Tree)
    		Prim
    		Kruskal
      

```

<center>图的知识图谱</center>



# 一、（Week8）图的概念、表示方法和遍历

图论是数学的一个分支，主要研究图的性质以及图之间的关系。在与数据结构和算法相关的内容中，图论涵盖了以下几个方面：

1. **图的表示**：图可以用不同的数据结构来表示，包括邻接矩阵、邻接表、~~关联矩阵~~等。这些表示方法影响着对图进行操作和算法实现的效率。

2. **图的遍历**：图的遍历是指从图中的某个顶点出发，访问图中所有顶点且不重复的过程。常见的图遍历算法包括深度优先搜索（DFS）和广度优先搜索（BFS）。

3. **最短路径**：最短路径算法用于找出两个顶点之间的最短路径，例如 Dijkstra 算法和 Floyd-Warshall 算法。这些算法在网络路由、路径规划等领域有广泛的应用。

4. **最小生成树**：最小生成树算法用于在一个连通加权图中找出一个权值最小的生成树，常见的算法包括 Prim 算法和 Kruskal 算法。最小生成树在网络设计、电力传输等领域有着重要的应用。

5. ~~**图的匹配**：图的匹配是指在一个图中找出一组边，使得没有两条边有一个公共顶点。匹配算法在任务分配、航线规划等问题中有着广泛的应用。~~

6. **拓扑排序**：拓扑排序算法用于对有向无环图进行排序，使得所有的顶点按照一定的顺序排列，并且保证图中的边的方向符合顺序关系。拓扑排序在任务调度、依赖关系分析等领域有重要的应用。

7. **图的连通性**：图的连通性算法用于判断图中的顶点是否连通，以及找出图中的连通分量。这对于网络分析、社交网络分析等具有重要意义。

8. ~~**图的颜色着色**：图的着色问题是指给图中的顶点赋予不同的颜色，使得相邻的顶点具有不同的颜色。这在调度问题、地图着色等方面有应用。~~

这些内容是图论在数据结构与算法领域的一些重要内容，它们在计算机科学和工程领域有广泛的应用。



## 1 术语和定义

图是更通用的结构；事实上，可以把树看作一种特殊的图。图可以用来表示现实世界中很多有意思的事物，包括道路系统、城市之间的航班、互联网的连接，甚至是计算机专业的一系列必修课。图一旦有了很好的表示方法，就可以用一些标准的图算法来解决那些看起来非常困难的问题。

尽管我们能够轻易看懂路线图并理解其中不同地点之间的关系，但是计算机并不具备这样的能力。不过，我们也可以将路线图看成是一张图，从而使计算机帮我们做一些非常有意思的事情。用过互联网地图网站的人都知道，计算机可以帮助我们找到两地之间最短、最快、最便捷的路线。

计算机专业的学生可能会有这样的疑惑：自己需要学习哪些课程才能获得学位呢？图可以很好地表示课程之间的依赖关系。图1展示了要获得计算机科学学位，所需学习课程的先后顺序。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/CS-Prereqs.png" alt="../_images/CS-Prereqs.png" style="zoom:67%;" />

<center>图1 计算机课程的学习顺序</center>



抽象出来看，**图（Graph）**由**顶点（Vertex）**和**边（Edge）**组成，每条边的两端都必须是图的两个顶点(可以是相同的顶点)。而记号 G(V,E)表示图 G 的顶点集为 V、边集为 E。图 2 是一个抽象出来的图。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240405134926341.png" alt="image-20240405134926341" style="zoom:50%;" />

<center>图2 抽象出来的图</center>



一般来说，图可分为有向图和无向图。有向图的所有边都有方向，即确定了顶点到顶点的一个指向；而无向图的所有边都是双向的，即无向边所连接的两个顶点可以互相到达。在一些问题中，可以把无向图当作所有边都是正向和负向的两条有向边组成，这对解决一些问题很有帮助。图 3是有向图和无向图的举例。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240405134513089.png" alt="image-20240405134513089" style="zoom: 50%;" />

<center> 图3 无向图与有向图 </center>



**顶点Vertex**
顶点又称节点，是图的基础部分。它可以有自己的名字，我们称作“键”。顶点也可以带有附加信息，我们称作“有效载荷”。



**边Edge**
边是图的另一个基础部分。两个顶点通过一条边相连，表示它们之间存在关系。边既可以是单向的，也可以是双向的。如果图中的所有边都是单向的，我们称之为有向图。图1明显是一个有向图，因为必须修完某些课程后才能修后续的课程。



**度Degree**

顶点的度是指和该顶点相连的边的条数。特别是对于有向图来说，顶点的出边条数称为该顶点的出度，顶点的入边条数称为该顶点的入度。例如图 3 的无向图中，V1的度为 2,V5的度为 4；有向图例子中，V2的出度为 1、入度为 2。



**权值Weight**

顶点和边都可以有一定属性，而量化的属性称为权值，顶点的权值和边的权值分别称为点权和边权。权值可以根据问题的实际背景设定，例如点权可以是城市中资源的数目，边权可以是两个城市之间来往所需要的时间、花费或距离。



有了上述定义之后，再来正式地定义**图Graph**。图可以用G来表示，并且G = (V, E)。其中，V是一个顶点集合，E是一个边集合。每一条边是一个二元组(v, w)，其中w, v∈V。可以向边的二元组中再添加一个元素，用于表示权重。子图s是一个由边e和顶点v构成的集合，其中e⊂E且v⊂V。

图4 展示了一个简单的带权有向图。我们可以用6个顶点和9条边的两个集合来正式地描述这个图：

$V = \left\{ V0,V1,V2,V3,V4,V5 \right\}$



$\begin{split}E = \left\{ \begin{array}{l}(v0,v1,5), (v1,v2,4), (v2,v3,9), (v3,v4,7), (v4,v0,1), \\
             (v0,v5,2),(v5,v4,8),(v3,v5,3),(v5,v2,1)
             \end{array} \right\}\end{split}$



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/digraph.png" alt="../_images/digraph.png" style="zoom: 67%;" />



<center>图4 简单的带权有向图</center>



图4中的例子还体现了其他两个重要的概念。

**路径Path**
路径是由边连接的顶点组成的序列。路径的正式定义为$w_1, w_2, ···, w_n$，其中对于所有的1≤i≤n-1，有$(w_i, w_{i+1})∈E$。无权重路径的长度是路径上的边数，有权重路径的长度是路径上的边的权重之和。以图4为例，从 V3到 V1的路径是顶点序列(V3, V4, V0, V1)，相应的边是{(v3, v4,7), (v4, v0,1), (v0, v1,5)}。

**环Cycle**
环是有向图中的一条起点和终点为同一个顶点的路径。例如，图4中的路径(V5, V2, V3, V5)就是一个环。没有环的图被称为无环图，没有环的有向图被称为有向无环图，简称为DAG。接下来会看到，DAG能帮助我们解决很多重要的问题。



### 1.1 编程题目

晴问题目练习网址，https://sunnywhy.com/sfbj

#### sy374: 无向图的度 简单

https://sunnywhy.com/sfbj/10/1/374

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），求每个顶点的度。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

在一行中输出n个整数，表示编号为从`0`到`n-1`的顶点的度。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 2
0 1
0 2
```

输出

```
2 1 1
```

解释

对应的无向图如下图所示，`0`号顶点的度为`2`，`1`号和`2`号顶点的度为`1`。

![无向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/602b5ae4-6958-4a1a-8d5a-00e9d888e3c1.png)





为了求解每个顶点的度，我们可以创建一个列表来存储每个顶点的度，初始值都为0。然后，对于每条边，我们将边的两个端点的度都加1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
degrees = [0] * n
for _ in range(m):
    u, v = map(int, input().split())
    degrees[u] += 1
    degrees[v] += 1

print(' '.join(map(str, degrees)))
```

这段代码首先读取输入，然后创建一个列表来存储每个顶点的度。然后，它遍历每条边，将边的两个端点的度都加1。最后，它输出每个顶点的度。





#### sy375: 有向图的度 简单

https://sunnywhy.com/sfbj/10/1/375

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），求每个顶点的入度和出度。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出行，每行为编号从`0`到`n-1`的一个顶点的入度和出度，中间用空格隔开。

样例1

输入

```
3 3
0 1
0 2
2 1
```

输出

```
0 2
2 0
1 1
```

解释

对应的有向图如下图所示。

`0`号顶点有`0`条入边，`2`条出边，因此入度为`0`，出度为`2`；

`1`号顶点有`2`条入边，`0`条出边，因此入度为`2`，出度为`0`；

`2`号顶点有`1`条入边，`1`条出边，因此入度为`1`，出度为`1`。

![有向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/21cec140-e555-4cb5-9ca9-290e373a782f.png)





为了求解每个顶点的入度和出度，我们可以创建两个列表来分别存储每个顶点的入度和出度，初始值都为0。然后，对于每条边，我们将起点的出度加1，终点的入度加1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
in_degrees = [0] * n
out_degrees = [0] * n
for _ in range(m):
    u, v = map(int, input().split())
    out_degrees[u] += 1
    in_degrees[v] += 1

for i in range(n):
    print(in_degrees[i], out_degrees[i])
```

这段代码首先读取输入，然后创建两个列表来存储每个顶点的入度和出度。然后，它遍历每条边，将边的起点的出度加1，终点的入度加1。最后，它输出每个顶点的入度和出度。





## 2 图的表示方法


图的抽象数据类型由下列方法定义。

❏ Graph() 新建一个空图。
❏ addVertex(vert) 向图中添加一个顶点实例。
❏ addEdge(fromVert, toVert) 向图中添加一条有向边，用于连接顶点fromVert和toVert。
❏ addEdge(fromVert, toVert, weight) 向图中添加一条带权重weight的有向边，用于连接顶点fromVert和toVert。
❏ getVertex(vertKey) 在图中找到名为vertKey的顶点。
❏ getVertices() 以列表形式返回图中所有顶点。
❏ in 通过 vertex in graph 这样的语句，在顶点存在时返回True，否则返回False。

根据图的正式定义，可以通过多种方式在Python中实现图的抽象数据类型（ADT）。在使用不同的表达方式来实现图的抽象数据类型时，需要做很多取舍。有两种非常著名的图实现，它们分别是邻接矩阵 **adjacency matrix** 和邻接表**adjacency list**。本节会解释这两种实现，并且用 Python 类来实现邻接表。



> dict套list, dict套queue，dict套dict，真好用啊！
>
> 确实，Python 中的字典（`dict`）是非常灵活和强大的数据结构，可以用于构建各种复杂的数据结构。
>
> 1. **Dict套List**：
>    - 可以用字典来表示多个对象的集合，其中每个对象都有一个唯一的键，而值则是一个列表，存储相关的数据。
>    - 例如，可以使用字典来表示学生和他们的成绩列表：`{"张三": [90, 85, 95], "李四": [88, 92, 90]}`。
>
> 2. **Dict套Queue**：
>    - 可以用字典来实现队列的功能，其中字典的键可以表示队列中的元素，而值则是用于实现队列操作的相关信息。
>    - 例如，可以使用字典来实现一个任务队列，其中键是任务名称，值是任务的优先级或其他相关信息。
>    - 例如，27925:小组队列，http://cs101.openjudge.cn/practice/27925/
>    
> 3. **Dict套Dict**：
>    - 可以用字典来表示更复杂的数据结构，例如树、图等。其中外层字典的键可以表示节点或顶点，而值则是另一个字典，存储该节点或顶点的属性以及与之相关联的其他节点或顶点。
>    - 例如，可以使用字典来表示图，其中外层字典的键是节点，内层字典的键是该节点与其他节点之间的边，值是边的权重或其他相关信息。
>
> 使用字典来构建这些复杂的数据结构，使得数据的组织和操作更加方便和高效，同时也提高了代码的可读性和可维护性。因此，dict套list、dict套queue、dict套dict等形式的数据结构在Python中确实是非常好用的。
>
> 
>
> **dict的value如果是list/set，是邻接表。dici嵌套dict 是 字典树/前缀树/Trie**
>
> 是的，你提到的两种数据结构分别是邻接表和字典树（前缀树，Trie）。
>
> 1. **邻接表**：在图论中，邻接表是一种表示图的常见方式之一。如果你使用字典（`dict`）来表示图的邻接关系，并且将每个顶点的邻居顶点存储为列表（`list`），那么就构成了邻接表。例如：
>
> ```python
> graph = {
>     'A': ['B', 'C'],
>     'B': ['A', 'D'],
>     'C': ['A', 'D'],
>     'D': ['B', 'C']
> }
> ```
>
> ## 27928: 遍历树
>
> http://cs101.openjudge.cn/practice/27928/
>
> 请你对输入的树做遍历。遍历的规则是：遍历到每个节点时，按照该节点和所有子节点的值从小到大进行遍历，例如：
>
> ```
>         7
>     /   |   \
>   10    3     6
> ```
>
> 对于这个树，你应该先遍历值为3的子节点，然后是值为6的子节点，然后是父节点7，最后是值为10的子节点。
>
> 本题中每个节点的值为互不相同的正整数，最大不超过9999999。
>
> **输入**
>
> 第一行：节点个数n (n<500)
>
> 接下来的n行：第一个数是此节点的值，之后的数分别表示它的所有子节点的值。每个数之间用空格隔开。如果没有子节点，该行便只有一个数。
>
> **输出**
>
> 输出遍历结果，一行一个节点的值。
>
> 样例输入
>
> ```
> sample1 input:
> 4
> 7 10 3 6
> 10
> 6
> 3
> 
> sample1 output:
> 3
> 6
> 7
> 10
> ```
>
> 样例输出
>
> ```
> sample2 input:
> 6
> 10 3 1
> 7
> 9 2 
> 2 10
> 3 7
> 1
> 
> sample2 output:
> 2
> 1
> 3
> 7
> 10
> 9
> ```
>
> 来源
>
> 2024spring zht
>
> 
>
> ```python
> # 李思哲 物理学院
> class TreeNode:
>     def __init__(self, value):
>         self.value = value
>         self.children = []
> 
> 
> def traverse_print(root, nodes):
>     if root.children == []:
>         print(root.value)
>         return
>     pac = {root.value: root}
>     for child in root.children:
>         pac[child] = nodes[child]
>     for value in sorted(pac.keys()):
>         if value in root.children:
>             traverse_print(pac[value], nodes)
>         else:
>             print(root.value)
> 
> 
> n = int(input())
> nodes = {}
> children_list = []
> for i in range(n):
>     info = list(map(int, input().split()))
>     nodes[info[0]] = TreeNode(info[0])
>     for child_value in info[1:]:
>         nodes[info[0]].children.append(child_value)
>         children_list.append(child_value)
> root = nodes[[value for value in nodes.keys() if value not in children_list][0]]
> traverse_print(root, nodes)
> 
> ```
>
> 
>
> 2. **字典树（前缀树，Trie）**：字典树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。如果你使用嵌套的字典来表示字典树，其中每个字典代表一个节点，键表示路径上的字符，而值表示子节点，那么就构成了字典树。例如：
>
> ```python
> trie = {
>     'a': {
>         'p': {
>             'p': {
>                 'l': {
>                     'e': {'is_end': True}
>                 }
>             }
>         }
>     },
>     'b': {
>         'a': {
>             'l': {
>                 'l': {'is_end': True}
>             }
>         }
>     },
>     'c': {
>         'a': {
>             't': {'is_end': True}
>         }
>     }
> }
> ```
>
> 这样的表示方式使得我们可以非常高效地搜索和插入字符串，特别是在大型数据集上。
>
> 例如，04089:电话号码，http://cs101.openjudge.cn/practice/04089/
>
> 
>
> **神奇的dict**
>
> 字典（dict）是Python中非常强大和灵活的数据结构之一，它可以用来存储键值对，是一种可变容器模型，可以存储任意数量的 Python 对象。
>
> 字典在 Python 中被广泛用于各种场景，例如：
>
> 1. **哈希映射**：字典提供了一种快速的键值查找机制，可以根据键快速地检索到相应的值。这使得字典成为了哈希映射（Hash Map）的理想实现。
>
> 2. **符号表**：在编程语言的实现中，字典常常被用作符号表，用来存储变量名、函数名等符号和它们的关联值。
>
> 3. **配置文件**：字典可以用来表示配置文件中的键值对，例如JSON文件就是一种常见的字典格式。
>
> 4. **缓存**：字典常常用于缓存中，可以将计算结果与其输入参数关联起来，以便后续快速地检索到相同参数的计算结果。
>
> 5. **图的表示**：如前文所述，字典可以用来表示图的邻接关系，是一种常见的图的表示方式。
>
> 由于其灵活性和高效性，字典在Python中被广泛应用于各种场景，并且被称为是Python中最常用的数据结构之一。



### 2.1 邻接矩阵

要实现图，最简单的方式就是使用二维矩阵。在矩阵实现中，每一行和每一列都表示图中的一个顶点。第v行和第w列交叉的格子中的值表示从顶点v到顶点w的边的权重。如果两个顶点被一条边连接起来，就称它们是相邻的。图5展示了图4对应的邻接矩阵。格子中的值表示从顶点v到顶点w的边的权重。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/digraph.png" alt="../_images/digraph.png" style="zoom: 67%;" />



<center>图4 简单的带权有向图</center>



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/adjMat.png" alt="../_images/adjMat.png" style="zoom:67%;" />

Figure 5: An Adjacency Matrix Representation for a Graph



邻接矩阵的优点是简单。对于小图来说，邻接矩阵可以清晰地展示哪些顶点是相连的。但是，图5中的绝大多数单元格是空的，我们称这种矩阵是“稀疏”的。对于存储稀疏数据来说，矩阵并不高效。

邻接矩阵适用于表示有很多条边的图。但是，“很多条边”具体是什么意思呢？要填满矩阵，共需要多少条边？由于每一行和每一列对应图中的每一个顶点，因此填满矩阵共需要|V|^2^条边。当每一个顶点都与其他所有顶点相连时，矩阵就被填满了。在现实世界中，很少有问题能够达到这种连接度。



### 2.2 邻接表

为了实现稀疏连接的图，更高效的方式是使用邻接表。在邻接表实现中，我们为图对象的所有顶点保存一个主列表，同时为每一个顶点对象都维护一个列表，其中记录了与它相连的顶点。在对Vertex类的实现中，我们使用字典（而不是列表），字典的键是顶点，值是权重。图6展示了图4所对应的邻接表



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/adjlist.png" alt="../_images/adjlist.png" style="zoom:67%;" />

Figure 6: An Adjacency List Representation of a Graph

邻接表的优点是能够紧凑地表示稀疏图。此外，邻接表也有助于方便地找到与某一个顶点相连的其他所有顶点。



### *2.3 关联矩阵

关联矩阵是一种图的表示方法，通常用于表示有向图。在关联矩阵中，行代表顶点，列代表边，如果顶点与边相连，则在对应的位置填上1，否则填上0。

虽然关联矩阵和邻接表都可以用来表示图，但它们在内存消耗和操作效率上有所不同。关联矩阵的存储空间复杂度为O(V*E)，其中V是顶点数，E是边数。但是，邻接表的存储空间复杂度为O(V+E)，它通常比关联矩阵更节省空间。

另外，对于某些图的操作，如查找两个顶点之间是否存在边，邻接表的操作效率更高。而对于其他操作，如计算图的闭包或者判断两个图是否同构等，关联矩阵可能更方便。

因此，关联矩阵和邻接表都有各自的优势和适用场景，选择使用哪种数据结构取决于具体的应用需求和对内存和操作效率的考虑。

> 图的闭包是指对于一个有向图或无向图，将所有顶点对之间的可达路径都加入到图中的过程。闭包可以用于分析图中的传递性关系。
>
> 判断两个图是否同构是指确定两个图是否具有相同的结构，即它们的顶点和边的连接关系是否一致。同构性是图的一个重要性质，可以用于在不同图之间进行匹配、比较和分类。
>
> 图的闭包和判断两个图是否同构是两个不同的概念，它们在图论中具有不同的应用和研究方向。
>
> 图的闭包可以通过图的邻接矩阵或邻接表来计算。对于有向图，可以使用传递闭包算法，通过矩阵乘法或深度优先搜索来确定顶点对之间的可达性。对于无向图，可以使用传递闭包算法或者深度优先搜索来计算。
>
> 判断两个图是否同构是一个复杂的问题，没有简单的算法可以解决。通常使用图同构性的定义和算法来进行判断，其中一种常用的方法是通过图的特征向量或特征值来比较两个图的结构。其他方法包括基于图的同构性的哈希算法和子图同构性的算法。
>
> 判断两个图是否同构是一个重要的图论问题，它在许多领域中都有应用，包括网络分析、化学和生物信息学等。
>
> 当涉及到图的闭包和判断两个图是否同构时，以下是一些示例：
>
> 1. 图的闭包：
> 考虑一个有向图，表示人际关系网络。假设有以下关系：A认识B，B认识C，C认识D。闭包操作将在图中添加所有可达的路径。在这种情况下，闭包操作将在图中添加A到D的边，因为A通过B和C可以到达D。闭包操作后的图将包含A、B、C、D四个顶点，并存在A到D的边。
>
> 2. 判断两个图是否同构：
> 考虑两个无向图G1和G2。G1有三个顶点{A, B, C}和两条边{A-B, B-C}。G2有三个顶点{X, Y, Z}和两条边{X-Y, Y-Z}。通过比较两个图的边连接关系，我们可以发现G1和G2具有相同的结构。因此，可以判断G1和G2是同构的。
>
> 
>
> 尽管图的闭包和同构问题在理论上具有重要性，并且在某些应用领域中有实际意义，但在常见的编程题目中，它们并不是经常出现的主题。更常见的图相关编程题目包括广度优先搜索、深度优先搜索、拓扑排序、最短路径算法（如Dijkstra算法和Floyd-Warshall算法）、最小生成树算法（如Prim算法和Kruskal算法）等。
>
> 
>
> 闭包和最小生成树是两个不同的概念？
>
> 闭包是指对于一个有向图或无向图，将所有顶点对之间的可达路径都加入到图中的过程。闭包操作后的图包含了原图中所有顶点和它们之间的可达路径。闭包操作可以用于分析图中的传递性关系。
>
> 最小生成树是指在一个连通无向图中，选择一棵包含所有顶点且边权重之和最小的树。最小生成树可以用来找到连接图中所有顶点的最短路径。
>
> 闭包和最小生成树是针对不同类型的图的不同操作。闭包适用于有向图和无向图，而最小生成树适用于无向图。闭包操作是为了添加路径，以展示顶点之间的可达性，而最小生成树是为了找到一棵最小权重的树，以连接所有顶点。
>
> 因此，闭包和最小生成树是不同的概念，在图论中具有不同的应用和意义。



下面是一个用 Python 实现的关联矩阵示例：

```python
class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.adj_matrix = self.create_adj_matrix()

    def create_adj_matrix(self):
        # Create an empty adjacency matrix
        adj_matrix = [[0] * len(self.edges) for _ in range(len(self.vertices))]

        # Fill adjacency matrix based on edges
        for i, vertex in enumerate(self.vertices):
            for j, edge in enumerate(self.edges):
                if vertex in edge:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def display_adj_matrix(self):
        for row in self.adj_matrix:
            print(row)


# Example usage
if __name__ == "__main__":
    vertices = ['A', 'B', 'C', 'D']
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]

    graph = Graph(vertices, edges)
    print("Adjacency Matrix:")
    graph.display_adj_matrix()
```

在这个示例中，定义了一个 `Graph` 类来表示图。在初始化时，传入顶点列表和边列表。然后，使用 `create_adj_matrix` 方法来创建关联矩阵。最后，调用 `display_adj_matrix` 方法来显示关联矩阵。

注意：这个示例假设了顶点和边的表示是字符串形式，可以根据需要调整表示方式。





### 2.4 编程题目

#### sy376: 无向图的邻接矩阵 简单

https://sunnywhy.com/sfbj/10/2/376

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），将其按邻接矩阵的方式存储（存在边的位置填充`1`，不存在边的位置填充`0`），然后输出整个邻接矩阵。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出n行n列，表示邻接矩阵。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 2
0 1
0 2
```

输出

```
0 1 1
1 0 0
1 0 0
```

解释

对应的无向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`1`条出边，到达`0`号顶点；

`2`号顶点有`1`条出边，到达`0`号顶点。

![无向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/602b5ae4-6958-4a1a-8d5a-00e9d888e3c1.png)





为了将无向图按邻接矩阵的方式存储，我们可以创建一个n*n的二维列表，初始值都为0。然后，对于每条边，我们将边的两个端点对应的位置填充为1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_matrix = [[0]*n for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_matrix[u][v] = 1
    adjacency_matrix[v][u] = 1

for row in adjacency_matrix:
    print(' '.join(map(str, row)))
```

这段代码首先读取输入，然后创建一个n*n的二维列表来存储邻接矩阵。然后，它遍历每条边，将边的两个端点对应的位置填充为1。最后，它输出整个邻接矩阵。





#### sy377: 有向图的邻接矩阵 简单

https://sunnywhy.com/sfbj/10/2/377

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），将其按邻接矩阵的方式存储（存在边的位置填充`1`，不存在边的位置填充`0`），然后输出整个邻接矩阵。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出n行n列，表示邻接矩阵。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 3
0 1
0 2
2 1
```

输出

```
0 1 1
0 0 0
0 1 0
```

解释

对应的有向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`0`条出边；

`2`号顶点有`1`条出边，到达`1`号顶点。

![有向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/21cec140-e555-4cb5-9ca9-290e373a782f.png)





为了将有向图按邻接矩阵的方式存储，我们可以创建一个n*n的二维列表，初始值都为0。然后，对于每条边，我们将边的起点和终点对应的位置填充为1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_matrix = [[0]*n for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_matrix[u][v] = 1

for row in adjacency_matrix:
    print(' '.join(map(str, row)))
```

这段代码首先读取输入，然后创建一个n*n的二维列表来存储邻接矩阵。然后，它遍历每条边，将边的起点和终点对应的位置填充为1。最后，它输出整个邻接矩阵。



#### sy378: 无向图的邻接表 简单

https://sunnywhy.com/sfbj/10/2/378

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），将其按邻接表的方式存储，然后输出整个邻接表。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出行，按顺序给出编号从`0`到`n-1`的顶点的所有出边，每行格式如下：

```text
id(k) v_1 v_2 ... v_k
```

其中表示当前顶点的编号，表示该顶点的出边数量，、、...、表示条出边的终点编号（按边输入的顺序输出）。行末不允许有多余的空格。

样例1

输入

```
3 2
0 1
0 2
```

输出

```
0(2) 1 2
1(1) 0
2(1) 0
```

解释

对应的无向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`1`条出边，到达`0`号顶点；

`2`号顶点有`1`条出边，到达`0`号顶点。

![无向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/602b5ae4-6958-4a1a-8d5a-00e9d888e3c1.png)





为了将无向图按邻接表的方式存储，我们可以创建一个列表，其中每个元素都是一个列表，表示一个顶点的所有邻接顶点。然后，对于每条边，我们将边的两个端点添加到对方的邻接列表中。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

for i in range(n):
    num = len(adjacency_list[i])
    if num == 0:
        print(f"{i}({num})")
    else:
        print(f"{i}({num})", ' '.join(map(str, adjacency_list[i])))
```

这段代码首先读取输入，然后创建一个列表来存储邻接表。然后，它遍历每条边，将边的两个端点添加到对方的邻接列表中。最后，它输出整个邻接表。





#### sy379: 有向图的邻接表 简单

https://sunnywhy.com/sfbj/10/2/379

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），将其按邻接表的方式存储，然后输出整个邻接表。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出行，按顺序给出编号从`0`到`n-1`的顶点的所有出边，每行格式如下：

```text
id(k) v_1 v_2 ... v_k
```

其中id表示当前顶点的编号，k表示该顶点的出边数量，v1、v2、...、vk表示k条出边的终点编号（按边输入的顺序输出）。行末不允许有多余的空格。

样例1

输入

```
3 3
0 1
0 2
2 1
```

输出

```
0(2) 1 2
1(0)
2(1) 1
```

解释

对应的有向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`0`条出边；

`2`号顶点有`1`条出边，到达`1`号顶点。

![有向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/21cec140-e555-4cb5-9ca9-290e373a782f.png)





为了将有向图按邻接表的方式存储，我们可以创建一个列表，其中每个元素都是一个列表，表示一个顶点的所有邻接顶点。然后，对于每条边，我们将边的终点添加到起点的邻接列表中。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)

for i in range(n):
    num = len(adjacency_list[i])
    if num == 0:
        print(f"{i}({num})")
    else:
        print(f"{i}({num})", ' '.join(map(str, adjacency_list[i])))
```

这段代码首先读取输入，然后创建一个列表来存储邻接表。然后，它遍历每条边，将边的终点添加到起点的邻接列表中。最后，它输出整个邻接表。





### 2.5 图的类实现（笔试）

在Python中，通过字典可以轻松地实现邻接表。我们要创建两个类：Graph类存储包含所有顶点的主列表，Vertex类表示图中的每一个顶点。
Vertex使用字典connectedTo来记录与其相连的顶点，以及每一条边的权重。代码清单7-1展示了Vertex类的实现，其构造方法简单地初始化id（它通常是一个字符串），以及字典connectedTo。addNeighbor方法添加从一个顶点到另一个的连接。getConnections方法返回邻接表中的所有顶点，由connectedTo来表示。getWeight方法返回从当前顶点到以参数传入的顶点之间的边的权重。

代码清单7-1 Vertex类

```python
class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]
```



Graph类的实现如代码清单7-2所示，其中包含一个将顶点名映射到顶点对象的字典。在图6中，该字典对象由灰色方块表示。Graph类也提供了向图中添加顶点和连接不同顶点的方法。getVertices方法返回图中所有顶点的名字。此外，我们还实现了`__iter__`方法，从而使遍历图中的所有顶点对象更加方便。总之，这两个方法使我们能够根据顶点名或者顶点对象本身遍历图中的所有顶点。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/adjlist.png" alt="../_images/adjlist.png" style="zoom:67%;" />

Figure 6: An Adjacency List Representation of a Graph



代码清单7-2 Graph类

```python
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,weight=0):
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

下面的Python会话使用Graph类和Vertex类创建了如图6所示的图。首先创建6个顶点，依次编号为0～5。然后打印顶点字典。注意，对每一个键，我们都创建了一个Vertex实例。接着，添加将顶点连接起来的边。最后，用一个嵌套循环验证图中的每一条边都已被正确存储。请按照图6的内容检查会话的最终结果。



```
>>> g = Graph()
>>> for i in range(6):
...    g.addVertex(i)
>>> g.vertList
{0: <adjGraph.Vertex instance at 0x41e18>,
 1: <adjGraph.Vertex instance at 0x7f2b0>,
 2: <adjGraph.Vertex instance at 0x7f288>,
 3: <adjGraph.Vertex instance at 0x7f350>,
 4: <adjGraph.Vertex instance at 0x7f328>,
 5: <adjGraph.Vertex instance at 0x7f300>}
>>> g.addEdge(0,1,5)
>>> g.addEdge(0,5,2)
>>> g.addEdge(1,2,4)
>>> g.addEdge(2,3,9)
>>> g.addEdge(3,4,7)
>>> g.addEdge(3,5,3)
>>> g.addEdge(4,0,1)
>>> g.addEdge(5,4,8)
>>> g.addEdge(5,2,1)
>>> for v in g:
...    for w in v.getConnections():
...        print("( %s , %s )" % (v.getId(), w.getId()))
...
( 0 , 5 )
( 0 , 1 )
( 1 , 2 )
( 2 , 3 )
( 3 , 4 )
( 3 , 5 )
( 4 , 0 )
( 5 , 4 )
( 5 , 2 )
```



上面类方式定义顶点和图，要求掌握，**数算B-2021笔试出现在算法部分**。在机考中，也可以直接使用二维列表或者字典来表示邻接表。



**19943: 图的拉普拉斯矩阵**

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





## 3 图的遍历

### 3.1 宽度优先搜索

**Algorithm for BFS**

How to implement Breadth First Search algorithm in Python 

https://www.codespeedy.com/breadth-first-search-algorithm-in-python/

BFS is one of the traversing algorithm used in graphs. This algorithm is implemented using a queue data structure. In this algorithm, the main focus is on the vertices of the graph. Select a starting node or vertex at first, mark the starting node or vertex as visited and store it in a queue. Then visit the vertices or nodes which are adjacent to the starting node, mark them as visited and store these vertices or nodes in a queue. Repeat this process until all the nodes or vertices are completely visited.

**Advantages of BFS**

1. It can be useful in order to find whether the graph has connected components or not.
2. It always finds or returns the **shortest path** if there is more than one path between two vertices.



**Disadvantages of BFS**

1. The execution time of this algorithm is very slow because the time complexity of this algorithm is exponential.
2. This algorithm is not useful when large graphs are used.



**Implementation of BFS in Python ( Breadth First Search )**

**Source Code: BFS in Python**

```python
graph = {'A': ['B', 'C', 'E'],
         'B': ['A','D', 'E'],
         'C': ['A', 'F', 'G'],
         'D': ['B'],
         'E': ['A', 'B','D'],
         'F': ['C'],
         'G': ['C']}


def bfs(graph, initial):
    visited = []
    queue = [initial]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]

            for neighbour in neighbours:
                queue.append(neighbour)
    return visited

print(bfs(graph,'A'))
```



Explanation:

1. Create a graph.
2. Initialize a starting node.
3. Send the graph and initial node as parameters to the bfs function.
4. Mark the initial node as visited and push it into the queue.
5. Explore the initial node and add its neighbours to the queue and remove the initial node from the queue.
6. Check if the neighbours node of a neighbouring node is already visited.
7. If not, visit the neighbouring node neighbours and mark them as visited.
8. Repeat this process until all the nodes in a graph are visited and the queue becomes empty.

Output:

```
['A', 'B', 'C', 'E', 'D', 'F', 'G']
```



Breadth First Search (BFS), algorithm for traversing or searching graphs

O(|V| + |E|) time complexity, |V| number of nodes, |E| number of edges

Applications:

Shortest path between two nodes (unweighted Graph)

Ford-Fulkson algorithm (Maximum Flow in a network)

#### 3.1.1 词梯问题

我们从词梯问题开始学习图算法。考虑这样一个任务：将单词FOOL转换成SAGE。在解决词梯问题时，必须每次只替换一个字母，并且每一步的结果都必须是一个单词，而不能是不存在的词。词梯问题由《爱丽丝梦游仙境》的作者刘易斯·卡罗尔于1878年提出。下面的单词转换序列是样例问题的一个解。

FOOL
POOL
POLL
POLE
PALE
SALE
SAGE

词梯问题有很多变体，例如在给定步数内完成转换，或者必须用到某个单词。在本节中，我们研究从起始单词转换到结束单词所需的**最小步数**。

由于主题是图，因此我们自然会想到使用图算法来解决这个问题。以下是大致步骤：

❏ 用图表示单词之间的关系；
❏ 用一种名为宽度优先搜索的图算法找到从起始单词到结束单词的最短路径。



##### 1 构建词梯图

第一个问题是如何用图来表示大的单词集合。如果两个单词的区别仅在于有一个不同的字母，就用一条边将它们相连。如果能创建这样一个图，那么其中的任意一条连接两个单词的路径就是词梯问题的一个解。图1展示了一个小型图，可用于解决从FOOL到SAGE的词梯问题。注意，它是无向图，并且边没有权重。



![../_images/wordgraph.png](https://raw.githubusercontent.com/GMyhf/img/main/img/wordgraph.png)

Figure 1: A Small Word Ladder Graph



创建这个图有多种方式。假设有一个单词列表，其中每个单词的长度都相同。首先，为每个单词创建顶点。为了连接这些顶点，可以将每个单词与列表中的其他所有单词进行比较。如果两个单词只相差一个字母，就可以在图中创建一条边，将它们连接起来。对于只有少量单词的情况，这个算法还不错。但是，假设列表中有3933个单词，将一个单词与列表中的其他所有单词进行比较，时间复杂度为O( n^2 )。对于3933个单词来说，这意味着要进行1500多万次比较。

采用下述方法，可以更高效地构建这个关系图。假设有数目巨大的桶，每一个桶上都标有一个长度为4的单词，但是某一个字母被下划线代替。图2展示了一些例子，如POP_。当处理列表中的每一个单词时，将它与桶上的标签进行比较。使用下划线作为通配符，我们将POPE和POPS放入同一个桶中。一旦将所有单词都放入对应的桶中之后，我们就知道，同一个桶中的单词一定是相连的。



![../_images/wordbuckets.png](https://raw.githubusercontent.com/GMyhf/img/main/img/wordbuckets.png)

Figure 2: Word Buckets for Words That are Different by One Letter



在Python中，可以通过字典来实现上述方法。字典的键就是桶上的标签，值就是对应的单词列表。一旦构建好字典，就能利用它来创建图。首先为每个单词创建顶点，然后在字典中对应同一个键的单词之间创建边。代码清单1展示了构建图所需的Python代码。

代码清单1 为词梯问题构建单词关系

```python
from pythonds3.graphs import Graph

def build_graph(filename):
    buckets = {}
    the_graph = Graph()
    with open(filename, "r", encoding="utf8") as file_in:
        all_words = file_in.readlines()
    # all_words = ["bane", "bank", "bunk", "cane", "dale", "dunk", "foil", "fool", "kale",
    #              "lane", "male", "mane", "pale", "pole", "poll", "pool", "quip",
    #              "quit", "rain", "sage", "sale", "same", "tank", "vain", "wane"
    #              ]

    # create buckets of words that differ by 1 letter
    for line in all_words:
        word = line.strip()
        for i, _ in enumerate(word):
            bucket = f"{word[:i]}_{word[i + 1:]}"
            buckets.setdefault(bucket, set()).add(word)

    # connect different words in the same bucket
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words - {word1}:
                the_graph.add_edge(word1, word2)

    return the_graph	
```



这是我们在本节中遇到的第一个实际的图问题，你可能会好奇这个图的稀疏程度如何。本例中的单词列表文件 vocabulary.txt 包含 3,933个单词。如果使用邻接矩阵表示，就会有15,468,489个单元格（3933 ＊3933 = 15,468,489）。用buildGraph函数创建的图一共有42,600条边（用邻接表来表示图）。因此，只有0.27%的单元格被填充。这显然是一个非常稀疏的矩阵。





##### 2 BFS实现词梯问题

完成图的构建之后，就可以编写能帮我们找到最短路径的图算法。我们使用的算法叫作宽度优先搜索（breadth first search，以下简称BFS）。BFS是最简单的图搜索算法之一，也是后续要介绍的其他重要图算法的原型。

给定图 G 和起点 s , BFS通过边来访问在 G 中与 s 之间存在路径的顶点。BFS的一个重要特性是，它会在访问完所有与 s 相距为 k 的顶点之后再去访问与 s 相距为 k+1 的顶点。为了理解这种搜索行为，可以想象BFS以每次生成一层的方式构建一棵树。它会在访问任意一个孙节点之前将起点的所有子节点都添加进来。

为了记录进度，BFS会将顶点标记成白色、灰色或黑色。在构建时，所有顶点都被初始化成白色。白色代表该顶点没有被访问过。当顶点第一次被访问时，它就会被标记为灰色；当BFS完成对该顶点的访问之后，它就会被标记为黑色。这意味着一旦顶点变为黑色，就没有白色顶点与之相连。灰色顶点仍然可能与一些白色顶点相连，这意味着还有额外的顶点可以访问。



来看看bfs函数如何构建对应于图1的宽度优先搜索树。从顶点fool开始，将所有与之相连的顶点都添加到树中。相邻的顶点有pool、foil、foul，以及cool。它们都被添加到队列中，作为之后要访问的顶点。



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

    # def __lt__(self,o):
    #     return self.id < o.id

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




def build_graph(filename):
    buckets = {}
    the_graph = Graph()
    with open(filename, "r", encoding="utf8") as file_in:
        all_words = file_in.readlines()
    # all_words = ["bane", "bank", "bunk", "cane", "dale", "dunk", "foil", "fool", "kale",
    #              "lane", "male", "mane", "pale", "pole", "poll", "pool", "quip",
    #              "quit", "rain", "sage", "sale", "same", "tank", "vain", "wane"
    #              ]

    # create buckets of words that differ by 1 letter
    for line in all_words:
        word = line.strip()
        for i, _ in enumerate(word):
            bucket = f"{word[:i]}_{word[i + 1:]}"
            buckets.setdefault(bucket, set()).add(word)

    # connect different words in the same bucket
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words - {word1}:
                the_graph.add_edge(word1, word2)

    return the_graph


#g = build_graph("words_small")
g = build_graph("vocabulary.txt")
print(len(g))


def bfs(start):
    start.distnce = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()  # 取队首作为当前顶点
        for neighbor in current.get_neighbors():   # 遍历当前顶点的邻接顶点
            if neighbor.color == "white":
                neighbor.color = "gray"
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = "black" # 当前顶点已经处理完毕，设黑色

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



#bfs(g.getVertex("fool"))

# 以FOOL为起点，进行广度优先搜索, 从FOOL到SAGE的最短路径,
# 并为每个顶点着色、赋距离和前驱。
bfs(g.get_vertex("FOOL"))


# 回溯路径
def traverse(starting_vertex):
    ans = []
    current = starting_vertex
    while (current.previous):
        ans.append(current.key)
        current = current.previous
    ans.append(current.key)

    return ans


# ans = traverse(g.get_vertex("sage"))
ans = traverse(g.get_vertex("SAGE")) # 从SAGE开始回溯，逆向打印路径，直到FOOL
print(*ans[::-1])
"""
3867
FOOL TOOL TOLL TALL SALL SALE SAGE
"""
```

代码及数据在，https://github.com/GMyhf/2024spring-cs201/tree/main/code



Q: "vocabulary.txt"是 3933行单词，**但是build_graph，只有3867个顶点？**检查了vocabulary.txt，没有重复的。

A: 建图过程中，如果桶里只有一个单词，就没有加入顶点集合。





#### 3.1.2 分析宽度优先搜索

在学习其他图算法之前，让我们先分析BFS的性能。

```python
def bfs(start):
    start.distnce = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()  # 取队首作为当前顶点
        for neighbor in current.get_neighbors():   # 遍历当前顶点的邻接顶点
            if neighbor.color == "white":
                neighbor.color = "gray"
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = "black" # 当前顶点已经处理完毕，设黑色
```



BFS 算法主体是两个循环的嵌套，while-for。while 循环对图中每个顶点最多只执行一次，时间复杂度是 O(|V|)，因为只有白色顶点才能被访问并添加到队列中。嵌套在 while 中的 for，由于每条边只有在其起始顶点u出队的时候才会被检查一次，而每个顶点最多出队1次，所以边最多被检查1次，一共是 O(|E|)。因此两个循环总的时间复杂度为 O(|V|+|E|)。

词梯问题还包括两个部分算法。建立 BFS 树之后，回溯顶点到起始顶点的过程，最多为 O(|V|)，因为最坏情况是整个图是一条长链。另外，创建单词关系图也需要时间，时间是 O(|V|+|E|) 的，因为每个顶点和边都只被处理一次。





### 3.2 深度优先搜索

**Algorithm for DFS**

https://www.codespeedy.com/depth-first-search-algorithm-in-python/

This algorithm is a recursive algorithm which follows the concept of backtracking and implemented using stack data structure. But, what is backtracking.

Backtracking:-

It means whenever a tree or a graph is moving forward and there are no nodes along the existing path, the tree moves backwards along the same path which it went forward in order to find new nodes to traverse. This process keeps on iterating until all the unvisited nodes are visited.

How stack is implemented in DFS:-

1. Select a starting node, mark the starting node as visited and push it into the stack.
2. Explore any one of adjacent nodes of the starting node which are unvisited.
3. Mark the unvisited node as visited and push it into the stack.
4. Repeat this process until all the nodes in the tree or graph are visited.
5. Once all the nodes are visited, then pop all the elements in the stack until the stack becomes empty.

 

 Implementation of DFS in Python

**Source Code: DFS in Python**

```python
import sys

def ret_graph():
    return {
        'A': {'B':5.5, 'C':2, 'D':6},
        'B': {'A':5.5, 'E':3},
        'C': {'A':2, 'F':2.5},
        'D': {'A':6, 'F':1.5},
        'E': {'B':3, 'J':7},
        'F': {'C':2.5, 'D':1.5, 'K':1.5, 'G':3.5},
        'G': {'F':3.5, 'I':4},
        'H': {'J':2},
        'I': {'G':4, 'J':4},
        'J': {'H':2, 'I':4},
        'K': {'F':1.5}
    }

start = 'A'                 
dest = 'J'                  
visited = []                
stack = []                  
graph = ret_graph()
path = []


stack.append(start)                  
visited.append(start)                
while stack:                         
    curr = stack.pop()            
    path.append(curr)
    for neigh in graph[curr]:        
        if neigh not in visited:       
            visited.append(neigh)       
            stack.append(neigh)         
            if neigh == dest :            
                print("FOUND:", neigh)
                print(path)
                sys.exit(0)
print("Not found")
print(path)
```

Explanation:

1. First, create a graph in a function.
2. Intialize a starting node and destination node.
3. Create a list for the visited nodes and stack for the next node to be visited.
4. Call the graph function.
5. Initially, the stack is empty.Push the starting node into the stack (stack.append(start) ).
6. Mark the starting node as visited (visited.append(start) ).
7. Repeat this process until all the neighbours are visited in the stack till the destination node is found.
8. If the destination node is found exit the while loop.
9. If the destination node is not present then “Not found” is printed.
10. Finally, print the path from starting node to the destination node.





#### 3.2.1 骑士周游问题

骑士周游问题是经典问题，用它来说明DFS算法。为了解决骑士周游问题，取一块国际象棋棋盘和一颗骑士棋子（马）。目标是找到一系列走法，使得骑士对棋盘上的每一格刚好都只访问一次。这样的一个移动序列被称为“周游路径”。多年来，骑士周游问题吸引了众多棋手、数学家和计算机科学家。对于8×8的棋盘，周游数的上界是1.305×10^35，但死路更多。很明显，解决这个问题需要聪明人或者强大的计算能力，抑或兼具二者。

尽管人们研究出很多种算法来解决骑士周游问题，但是图搜索算法是其中最好理解和最易编程的一种。我们再一次通过两步来解决这个问题：

❏ 用图表示骑士在棋盘上的合理走法；
❏ 使用图算法找到一条长度为rows × columns-1的路径，满足图中的每一个顶点都只被访问一次。

##### 1 构建骑士周游图

为了用图表示骑士周游问题，将棋盘上的每一格表示为一个顶点，同时将骑士的每一次合理走法表示为一条边。图1展示了骑士的合理走法以及在图中对应的边。



![../_images/knightmoves.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202404110007738.png)

图1 骑士的合理走法以及在图中对应的边



用List 1中的函数来构建n × n棋盘对应的完整图。`knight_graph`函数将整个棋盘遍历了一遍。当它访问棋盘上的每一格时，都会调用辅助函数`gen_legal_moves`来创建一个列表，用于记录从这一格开始的所有合理走法。之后，所有的合理走法都被转换成图中的边。另一个辅助函数`pos_to_node_id`将棋盘上的行列位置转换成与图1中顶点编号相似的线性顶点数。

**Listing 1 knight_graph函数**

```python
from pythonds3.graphs import Graph

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
```



在List2中，`gen_legal_moves`函数接受骑士在棋盘上的位置，并且生成8种可能的走法。并确认走法是合理的。

**Listing 2 gen_legal_moves函数**

```python
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
```



图2 展示了在8×8的棋盘上所有合理走法所对应的完整图，其中一共有336条边。注意，与棋盘中间的顶点相比，边缘顶点的连接更少。可以看到，这个图也是非常稀疏的。如果图是完全相连的，那么会有4096条边。由于本图只有336条边，因此邻接矩阵的填充率只有8.2%。还是稀疏图。

<img src="https://runestone.academy/ns/books/published/pythonds/_images/bigknight.png" alt="../_images/bigknight.png" style="zoom:67%;" />

Figure 2: All Legal Moves for a Knight on an 8×8 Chessboard



##### 2 实现骑士周游

用来解决骑士周游问题的搜索算法叫作深度优先搜索（depth first search，以下简称DFS）。与BFS每次构建一层不同，DFS通过尽可能深地探索分支来构建搜索树。下面介绍DFS的两种实现：第1种每个顶点仅访问一次，用于解决骑士周游问题；第2种更通用，它在构建搜索树时允许其中的顶点被多次访问。可以作为其他的图算法的基础。

DFS正是为找到由63条边构成的路径所需的算法。我们会看到，当DFS遇到死路时（无法找到下一个合理走法），它会回退到树中倒数第2深的顶点，以继续移动。

在list 3中，`knight_tour`函数接受4个参数：n是搜索树的当前深度；path是到当前为止访问过的顶点列表；u是希望在图中访问的顶点；limit是路径上的顶点总数。`knight_tour`函数是递归的。当被调用时，它首先检查基本情况。如果有一条包含64个顶点的路径，就返回True，以表示找到了一次成功的周游。如果路径不够长，则通过选择一个新的访问顶点并对其递归调用`knightTour`来进行更深一层的探索。

DFS也使用颜色来记录已被访问的顶点。未访问的顶点是白色的，已被访问的则是灰色的。如果一个顶点的所有相邻顶点都已被访问过，但是路径长度仍然没有达到64，就说明遇到了死路。如果遇到死路，就必须回溯。当从`knight_tour`返回False时，就会发生回溯。在宽度优先搜索中，我们使用了队列来记录将要访问的顶点。由于深度优先搜索是递归的，因此我们隐式地使用一个栈来回溯。当从`knight_tour`调用返回False时，仍然在循环中，并且会查看`neighbors`中的下一个顶点。



**List 3 knight_tour函数**

```python
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
```

第 5 行是最重要的一行。这一行保证接下来要访问的顶点有最少的合理走法。你可能认为这样做非常影响性能；为什么不选择合理走法最多的顶点呢？

> 选择合理走法最多的顶点作为下一个访问顶点的问题在于，它会使骑士在周游的前期就访问位于棋盘中间的格子。当这种情况发生时，骑士很容易被困在棋盘的一边，而无法到达另一边的那些没访问过的格子。首先访问合理走法最少的顶点，则可使骑士优先访问棋盘边缘的格子。这样做保证了骑士能够尽早访问难以到达的角落，并且在需要的时候通过中间的格子跨越到棋盘的另一边。我们称利用这类知识来加速算法为启发式技术。人类每天都在使用启发式技术做决定，启发式搜索也经常被用于人工智能领域。本例用到的启发式技术被称作 **Warnsdorff 算法**，以纪念在 1823 年提出该算法的数学家 H. C. Warnsdorff。



我们通过一个例子来看看`knight_tour`的运行情况，可以参照图3来追踪搜索的变化。这个例子假设在list 3中第6行对`getConnections`方法的调用将顶点按照字母顺序排好。首先调用`knight_tour(0, path, A, 6)`。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202404110022563.png" alt="image-20240411002152657" style="zoom: 50%;" />

图3 利用knightTour函数找到路径

`knight_tour`函数从顶点A开始访问。与A相邻的顶点是B和D。按照字母顺序，B在D之前，因此DFS选择B作为下一个要访问的顶点，如图3b所示。对B的访问从递归调用`knight_tour`开始。B与C和D相邻，因此`knight_tour`接下来会访问C。但是，<u>C没有白色的相邻顶点（如图3c所示），因此是死路。此时，将C的颜色改回白色</u>。`knight_tour`的调用返回False，也就是将搜索回溯到顶点B，如图3d所示。接下来要访问的顶点是D，因此`knight_tour`进行了一次递归调用来访问它。从顶点D开始，`knight_tour`可以继续进行递归调用，直到再一次访问顶点C。但是，这一次，检验条件n < limit失败了，因此我们知道遍历完了图中所有的顶点。此时返回True，以表明对图进行了一次成功的遍历。当返回列表时，path包含[A, B, D, E, F, C]。其中的顺序就是每个顶点只访问一次所需的顺序。

图4展示了在8×8的棋盘上周游的完整路径。存在多条周游路径，其中有一些是对称的。通过一些修改之后，可以实现循环周游，即起点和终点在同一个位。



![../_images/completeTour.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202404110023421.png)

Figure 4: A Complete Tour of the Board



##### 3 分析骑士周游（Warnsdorff 算法）

在学习深度优先搜索的通用版本之前，我们探索骑士周游问题中的最后一个有趣的话题：性能。具体地说，`knight_tour`对用于选择下一个访问顶点的方法非常敏感。例如，利用速度正常的计算机，可以在1.5秒之内针对5×5的棋盘生成一条周游路径。但是，如果针对8×8的棋盘，会怎么样呢？可能需要等待半个小时才能得到结果！

如此耗时的原因在于，目前实现的骑士周游问题算法是一种$O(k^N)$的指数阶算法，其中 N是棋盘上的格子数，k是一个较小的常量。图5有助于理解搜索过程。树的根节点代表搜索过程的起点。从起点开始，算法生成并且检测骑士能走的每一步。如前所述，合理走法的数目取决于骑士在棋盘上的位置。若骑士位于四角，只有2种合理走法；若位于与四角相邻的格子中，则有3种合理走法；若在棋盘中央，则有8种合理走法。图5展示了棋盘上的每一格所对应的合理走法数目。在树的下一层，对于骑士当前位置来说，又有2～8种不同的合理走法。待检查位置的数目对应搜索树中的节点数目。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202404110025553.png" alt="../_images/8arrayTree.png" style="zoom: 67%;" />

Figure 5: A Search Tree for the Knight’s Tour



![../_images/moveCount.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202404110026225.png)

图6 每个格子对应的合理走法数目

我们已经看到，在高度为N的二叉树中，节点数为 $2^{N+1}-1$；至于子节点可能多达8个而非2个的树，其节点数会更多。由于每一个节点的分支数是可变的，因此可以使用平均分支因子来估计节点数。需要注意的是，这个算法是指数阶算法：$k^{N+1}-1$，其中k是棋盘的平均分支因子。让我们看看它增长得有多快。对于5× 5的棋盘，搜索树有25层（若把顶层记为第0层，则N = 24），平均分支因子k = 3.8。因此，搜索树中的节点数是$3.8^{25}-1$或者$3.12×10^{14}$。对于6×6的棋盘，k =4.4，搜索树有$1.5×10^{23}$个节点。对于8×8的棋盘，k = 5.25，搜索树有$1.3×10^{46}$个节点。由于这个问题有很多个解，因此不需要访问搜索树中的每一个节点。但是，需要访问的节点的小数部分只是一个常量乘数，它并不能改变该问题的指数特性。

幸运的是，有办法针对8×8的棋盘在1秒内得到一条周游路径。list 4展示了加速搜索过程的代码。在`order_by_avail`函数中，第10行是最重要的一行。这一行保证接下来要访问的顶点有最少的合理走法。你可能认为这样做非常影响性能；为什么不选择合理走法最多的顶点呢？

**List4 选择下一个要访问的顶点至关重要**

```python
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
```



选择合理走法最多的顶点作为下一个访问顶点的问题在于，它会使骑士在周游的前期就访问位于棋盘中间的格子。当这种情况发生时，骑士很容易被困在棋盘的一边，而无法到达另一边的那些没访问过的格子。首先访问合理走法最少的顶点，则可使骑士优先访问棋盘边缘的格子。这样做保证了骑士能够尽早访问难以到达的角落，并且在需要的时候通过中间的格子跨越到棋盘的另一边。我们称<u>利用这类知识来加速算法为**启发式技术**</u>。人类每天都在使用启发式技术做决定，启发式搜索也经常被用于人工智能领域。本例用到的启发式技术被称作Warnsdorff算法，以纪念在1823年提出该算法的数学家H. C. Warnsdorff。

> Warnsdorff 算法是一种用于解决骑士周游问题的启发式算法。该算法的主要思想是优先选择下一步可行路径中具有最少可选路径的顶点（优先选择子节点中可行节点少的），从而尽可能地减少搜索空间。



骑士周游程序在，https://github.com/GMyhf/2024spring-cs201/blob/main/code/KnightTour.py

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



#### 3.2.4 通用深度优先搜索（depth first forest）

骑士周游是深度优先搜索的一种特殊情况，它需要创建没有分支的最深深度优先搜索树。通用的深度优先搜索其实更简单，它的目标是尽可能深地搜索，尽可能多地连接图中的顶点，并且在需要的时候进行分支。

有时候深度优先搜索会创建多棵深度优先搜索树，称之为**深度优先森林（depth first forest）**。和宽度优先搜索类似，深度优先搜索也利用前驱连接来构建树。此外，深度优先搜索还会使用Vertex类中的两个额外的实例变量：`发现时间`记录算法在第一次访问顶点时的步数，`结束时间`记录算法在顶点被标记为黑色时的步数。在学习之后会发现，顶点的`发现时间`和`结束时间`提供了一些有趣的特性，后续算法会用到这些特性。

深度优先搜索的实现如代码list 5所示。由于`dfs函数`和`dfsvisit`辅助函数使用一个变量来记录调用`dfsvisit`的时间，因此我们选择将代码作为Graph类的一个子类中的方法来实现。该实现继承Graph类，并且增加了time实例变量，以及dfs和dfsvisit两个方法。注意第10行，dfs方法遍历图中所有的顶点，并对白色顶点调用`dfsvisit`方法。之所以遍历所有的顶点，而不是简单地从一个指定的顶点开始搜索，是因为这样做能够确保深度优先森林中的所有顶点都在考虑范围内，而不会有被遗漏的顶点。`for vertex in self`这条语句可能看上去不太正确，但是此处的self是DFSGraph类的一个实例，遍历一个图实例中的所有顶点其实是一件非常自然的事情。

**List5 实现通用深度优先搜索**

```python
# https://github.com/psads/pythonds3
from pythonds3.graphs import Graph

class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0                   #不是物理世界，而是算法执行步数

    def dfs(self):
        for vertex in self:
            vertex.color = "white"      #颜色初始化
            vertex.previous = -1
        for vertex in self:             #从每个顶点开始遍历
            if vertex.color == "white":
                self.dfs_visit(vertex)  #第一次运行后还有未包括的顶点
                                        # 则建立森林

    def dfs_visit(self, start_vertex):
        start_vertex.color = "gray"
        self.time = self.time + 1       #记录算法的步骤
        start_vertex.discovery_time = self.time
        for next_vertex in start_vertex.get_neighbors():
            if next_vertex.color == "white":
                next_vertex.previous = start_vertex
                self.dfs_visit(next_vertex)     #深度优先递归访问
        start_vertex.color = "black"
        self.time = self.time + 1
        start_vertex.closing_time = self.time
```



在接下来的两个例子中，我们会看到为何记录深度优先森林十分重要。

从`start_vertex`开始，`dfs_visit`方法尽可能深地探索所有相邻的白色顶点。如果仔细观察`df_svisit`的代码并且将其与bfs比较，应该注意到二者几乎一样，除了内部for循环的最后一行，`dfs_visit`通过递归地调用自己来继续进行下一层的搜索，bfs则将顶点添加到队列中，以供后续搜索。有趣的是，bfs使用队列，dfsvisit则使用栈。我们没有在代码中看到栈，但是它其实隐式地存在于`dfs_visit`的递归调用中。

图7展示了在小型图上应用深度优先搜索算法的过程。图中，虚线表示被检查过的边，但是其一端的顶点已经被添加到深度优先搜索树中。在代码中，这是通过检查另一端的顶点是否不为白色来完成的。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202404110100894.png" alt="image-20240411010044844" style="zoom: 50%;" />

图7 构建深度优先搜索树



搜索从图中的顶点A开始。由于所有顶点一开始都是白色的，因此算法会访问A。访问顶点的第一步是将其颜色设置为灰色，以表明正在访问该顶点，并将其发现时间设为1。由于A有两个相邻顶点（B和D），因此它们都需要被访问。我们按照字母顺序来访问顶点。

接下来访问顶点B，将它的颜色设置为灰色，并把发现时间设置为2。B也与两个顶点（C和D）相邻，因此根据字母顺序访问C。
访问C时，搜索到达某个分支的终点。在将C标为灰色并且把发现时间设置为3之后，算法发现C没有相邻顶点。这意味着对C的探索完成，因此将它标为黑色，并将完成时间设置为4。图7d展示了搜索至这一步时的状态。

由于C是一个分支的终点，因此需要返回到B，并且继续探索其余的相邻顶点。唯一的待探索顶点就是D，它把搜索引到E。E有两个相邻顶点，即B和F。正常情况下，应该按照字母顺序来访问这两个顶点，但是由于B已经被标记为灰色，因此算法自知不应该访问B，因为如果这么做就会陷入死循环。因此，探索过程跳过B，继续访问F。

F只有C这一个相邻顶点，但是C已经被标记为黑色，因此没有后续顶点需要探索，也即到达另一个分支的终点。从此时起，算法一路回溯到起点，同时为各个顶点设置完成时间并将它们标记为黑色，如图7h～图7l所示。

每个顶点的发现时间和结束时间都体现了括号特性，这意味着深度优先搜索树中的任一节点的子节点都有比该节点更晚的发现时间和更早的结束时间。图8展示了通过深度优先搜索算法构建的树。



![../_images/dfstree.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202404110102281.png)

图8 最终的深度优先搜索



#### 3.2.5 分析深度优先搜索

一般来说，深度优先搜索的运行时间如下。在`dfs函数`中有两个循环，每个都是|V|次，所以是O(|V|)，这是由于它们针对图中的每个顶点都只执行一次。在`dfs_visit`中，循环针对当前顶点的邻接表中的每一条边都执行一次，且仅在顶点是白色时被递归调用，因此循环最多会对图中的每一条边执行一次，也就是O(|E|)。因此，深度优先搜索算法的时间复杂度是O(|V|+|E|)，与BFS一样。



> 深度优先搜索树（林）的性质
>
> 1. **顶点大小的定义：** 对于图 G 中的任意两个顶点 u 和 v，当且仅当顶点 u 的结束时间（fin）小于顶点 v 的结束时间（fin）时，称顶点 u 小于顶点 v，即符号化表示为：$∀ u, v \in G; u < v \equiv u.fin < v.fin$。
>
> 2. **子树大小的定义：** 对于深度优先搜索树（林）中的任意两棵不相交的子树 t1 和 t2，当且仅当 t1 中的任意节点的结束时间都早于 t2 中的任意节点的开始时间时，称 t1 小于 t2，即符号化表示为：$t1 < t2 \equiv ∀u \in t1, v \in t2; u < v$。
>
> 3. **节点间的互斥关系：** 如果 t1 小于 t2，且节点 u 属于 t1，节点 v 属于 t2，则 $u \rightarrow v$ 不存在。
>
> 4. **节点间的关系限定：** 如果顶点 u 小于顶点 v，并且它们在同一棵树中，则只有两种可能情况：
>    - 顶点 v 是顶点 u 的祖先。
>    - 顶点 u 和 v 具有共同的祖先 t。其中，顶点 u 属于 t 的子树 t1，顶点 v 属于 t 的子树 t2。这种情况下，t1 的结束时间早于 t2 的开始时间。
>



### 3.3 编程题目

#### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



#### 28050: 骑士周游

http://cs101.openjudge.cn/practice/28050/



#### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



#### 02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/practice/02754



#### 02698: 八皇后问题解输出

dfs and similar, http://cs101.openjudge.cn/practice/02698



#### sy321: 迷宫最短路径

https://sunnywhy.com/sfbj/8/2/321

现有一个 n*m 大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格，且只能移动到平地上。假设左上角坐标是(1,1)，行数增加的方向为增长的方向，列数增加的方向为增长的方向，求从迷宫左上角到右下角的最少步数的路径。

**输入**

第一行两个整数$n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来 n 行，每行 m 个整数（值为`0`或`1`），表示迷宫。

**输出**

从左上角的坐标开始，输出若干行（每行两个整数，表示一个坐标），直到右下角的坐标。

数据保证最少步数的路径存在且唯一。

样例1

输入

```
3 3
0 1 0
0 0 0
0 1 0
```

输出

```
1 1
2 1
2 2
2 3
3 3
```

解释

假设左上角坐标是(1,)，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角的最少步数的路径为：(1,1)=>(2,1)=>(2,2)=>(2,3)=>(3,3)。



in_queue 数组，结点是否已入过队

```python
from collections import deque

MAX_DIRECTIONS = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid_move(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not in_queue[x][y]

def bfs(start_x, start_y):
    queue = deque()
    queue.append((start_x, start_y))
    in_queue[start_x][start_y] = True
    while queue:
        x, y = queue.popleft()
        if x == n - 1 and y == m - 1:
            return
        for i in range(MAX_DIRECTIONS):
            next_x = x + dx[i]
            next_y = y + dy[i]
            if is_valid_move(next_x, next_y):
                prev[next_x][next_y] = (x, y)
                in_queue[next_x][next_y] = True
                queue.append((next_x, next_y))

def print_path(pos):
    prev_position = prev[pos[0]][pos[1]]
    if prev_position == (-1, -1):
        print(pos[0] + 1, pos[1] + 1)
        return
    print_path(prev_position)
    print(pos[0] + 1, pos[1] + 1)

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

in_queue = [[False] * m for _ in range(n)]
prev = [[(-1, -1)] * m for _ in range(n)]

bfs(0, 0)
print_path((n - 1, m - 1))

```



#### sy380: 无向图的连通块 简单

https://sunnywhy.com/sfbj/10/3/380

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），求图中的连通块个数。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出一个整数，表示图中的连通块个数。

样例1

输入

```
6 5
0 1
0 3
3 5
2 4
1 5
```

输出

```
2
```

解释

对应的无向图如下图所示，共有两个连通块。

![无向图的连通块.png](https://raw.githubusercontent.com/GMyhf/img/main/img/d95ea531-1e5b-44e7-8d9c-8908a4bb1115.png)





要解决这个问题，我们可以使用深度优先搜索（DFS）或广度优先搜索（BFS）来找出图中的所有连通块。我们首先创建一个邻接列表来表示图，然后对每个未访问的节点进行DFS或BFS，每次搜索都会找到一个连通块。搜索过程中，我们需要标记已访问的节点，以防止重复访问。

以下是使用DFS的Python代码：

```python
def dfs(node, visited, adjacency_list):
    visited[node] = True
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adjacency_list)

n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

visited = [False] * n
connected_components = 0
for i in range(n):
    if not visited[i]:
        dfs(i, visited, adjacency_list)
        connected_components += 1

print(connected_components)
```

在这段代码中，我们首先读取顶点数和边数，然后读取每条边的信息并在邻接列表中添加相应的边。然后我们初始化一个布尔数组`visited`来跟踪已访问的节点，并初始化连通块的数量为0。接着，我们遍历每个节点，如果它还未被访问，我们就从它开始进行DFS，并将连通块的数量加1。最后，我们输出连通块的数量。



#### sy381: 无向连通图 简单

https://sunnywhy.com/sfbj/10/3/381

现有一个共个顶点、条边的无向图（假设顶点编号为从`0`到`n-1`），判断其是否是连通图。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果是连通图，那么输出`Yes`，否则输出`No`。

样例1

输入

```
6 5
0 1
0 3
3 5
2 4
1 5
```

输出

```
No
```

解释

对应的无向图如下图所示，不是连通图。

![无向图的连通块.png](https://raw.githubusercontent.com/GMyhf/img/main/img/d95ea531-1e5b-44e7-8d9c-8908a4bb1115.png)

样例2

输入

```
6 6
0 1
0 3
3 5
2 4
1 5
3 4
```

输出

```
Yes
```

解释

对应的无向图如下图所示，是连通图。

![无向连通图_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/7c186721-b047-4cc0-b3d9-329d17c53d82.png)





要判断一个图是否是连通图，我们可以使用深度优先搜索（DFS）或广度优先搜索（BFS）来遍历图。如果在遍历结束后，所有的节点都被访问过，那么这个图就是连通图。否则，就不是连通图。

以下是使用DFS的Python代码：

```python
def dfs(node, visited, adjacency_list):
    visited[node] = True
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adjacency_list)

n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

visited = [False] * n
dfs(0, visited, adjacency_list)

if all(visited):
    print("Yes")
else:
    print("No")
```

在这段代码中，我们首先读取顶点数和边数，然后读取每条边的信息并在邻接列表中添加相应的边。然后我们初始化一个布尔数组`visited`来跟踪已访问的节点，并从第一个节点开始进行DFS。最后，我们检查是否所有的节点都被访问过，如果是，那么输出`Yes`，否则输出`No`。



#### sy382: 有向图判环 中等

https://sunnywhy.com/sfbj/10/3/382

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），如果从图中一个顶点出发，沿着图中的有向边前进，最后能回到这个顶点，那么就称其为图中的一个环。判断图中是否有环。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果图中有环，那么输出`Yes`，否则输出`No`。

样例1

输入

```
4 4
1 0
0 3
3 2
2 1
```

输出

```
Yes
```

解释

对应的有向图如下图所示，存在`0->3->2->1->0`的环。

![有向图判环_样例1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/c2072489-936c-40c8-8e89-7ddd87d76705.png)

样例2

输入

```
4 4
1 0
0 3
2 3
2 1
```

输出

```
No
```

解释

对应的有向图如下图所示，图中不存在环。

![有向图判环_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/cd48cf44-20f5-4c21-aaae-f1df3fcc3ba8.png)





在这个问题中，需要检查给定的有向图是否包含一个环。可以使用深度优先搜索（DFS）来解决这个问题。在DFS中，从一个节点开始，然后访问它的每一个邻居。如果在访问过程中，遇到了一个已经在当前路径中的节点，那么就存在一个环。**可以使用一个颜色数组来跟踪每个节点的状态：未访问（0），正在访问（1），已访问（2）。**

以下是解决这个问题的Python代码：

```python
def has_cycle(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    color = [0] * n

    def dfs(node):
        if color[node] == 1:
            return True
        if color[node] == 2:
            return False

        color[node] = 1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = 2
        return False

    for i in range(n):
        if dfs(i):
            return "Yes"
    return "No"

# 接收数据
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
print(has_cycle(n, edges))
```

在这个函数中，我们首先构建了一个邻接列表来表示图。然后，我们对每个节点执行深度优先搜索。如果在搜索过程中，我们遇到了一个正在访问的节点，那么就存在一个环。如果我们遍历完所有的节点都没有找到环，那么就返回"No"。



#### sy383: 最大权值连通块 中等

https://sunnywhy.com/sfbj/10/3/383

现有一个共个顶点、条边的无向图（假设顶点编号为从`0`到`n-1`），每个顶点有各自的权值。我们把一个连通块中所有顶点的权值之和称为这个连通块的权值。求图中所有连通块的最大权值。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$​），分别表示顶点数和边数；

第二行个用空格隔开的正整数（每个正整数不超过`100`），表示个顶点的权值。

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出一个整数，表示连通块的最大权值。

样例1

输入

```
6 5
2 3 4 1 3 2
0 1
0 3
3 5
2 4
1 5
```

输出

```
8
```

解释

对应的无向图如下图所示，左边连通块的权值为，右边连通块的权值为，因此最大权值为`8`。

![最大权值连通块.png](https://raw.githubusercontent.com/GMyhf/img/main/img/0b6c3a49-118e-4432-bc17-e8d2f72d4410.png)





需要找到给定无向图中所有连通块的最大权值。使用深度优先搜索（DFS）来解决这个问题。在DFS中，从一个节点开始，然后访问它的每一个邻居。可以使用一个visited数组来跟踪每个节点是否已经被访问过。对于每个连通块，可以计算其权值之和，并更新最大权值。

以下是解决这个问题的Python代码：

```python
def max_weight(n, m, weights, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    max_weight = 0

    def dfs(node):
        visited[node] = True
        total_weight = weights[node]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor)
        return total_weight

    for i in range(n):
        if not visited[i]:
            max_weight = max(max_weight, dfs(i))

    return max_weight

# 接收数据
n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
print(max_weight(n, m, weights, edges))
```

在这段代码中，首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`和每个顶点的权值，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，我们调用`max_weight`函数并打印结果。





#### sy384: 无向图的顶点层号

https://sunnywhy.com/sfbj/10/3/384

现有一个共n个顶点、m条边的无向连通图（假设顶点编号为从`0`到`n-1`）。我们称从s号顶点出发到达其他顶点经过的最小边数称为各顶点的层号。求图中所有顶点的层号。

**输入**

第一行三个整数n、m、s（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1$​），分别表示顶点数、边数、起始顶点编号；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出n个整数，分别为编号从`0`到`n-1`的顶点的层号。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6 6 0
0 1
0 3
3 5
2 4
1 5
3 4
```

输出

```
0 1 3 1 2 2
```

解释

对应的无向图和顶点层号如下图所示。

![无向图的顶点层号.png](https://raw.githubusercontent.com/GMyhf/img/main/img/a896296d-23e0-4420-9e70-c4e979a78e89.png)





需要找到从给定的起始顶点到图中所有其他顶点的最短路径长度，这也被称为顶点的层号。可以使用广度优先搜索（BFS）来解决这个问题。在BFS中，从起始节点开始，然后访问它的所有邻居，然后再访问这些邻居的邻居，依此类推。我们可以使用一个队列来跟踪待访问的节点，并使用一个距离数组来记录从起始节点到每个节点的最短距离。

以下是解决这个问题的Python代码：

```python
from collections import deque

def bfs(n, m, s, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    distance = [-1] * n
    distance[s] = 0

    queue = deque([s])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance

# 接收数据
n, m, s = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
distances = bfs(n, m, s, edges)
print(' '.join(map(str, distances)))
```

在这段代码中，我们首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`和起始顶点`s`，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，我们调用`bfs`函数并打印结果。





#### sy385: 受限层号的顶点数 中等

https://sunnywhy.com/sfbj/10/3/385

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`）。我们称从s号顶点出发到达其他顶点经过的最小边数称为各顶点的层号。求层号不超过的顶点个数。

**输入**

第一行四个整数n、m、s、k（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le k \le 100$​），分别表示顶点数、边数、起始顶点编号；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出一个整数，表示层号不超过的顶点个数。

样例1

输入

```
6 6 0 2
0 1
0 3
3 5
4 2
3 4
5 2
```

输出

```
5
```

解释

对应的有向图和顶点层号如下图所示，层号不超过`2`的顶点有`5`个。

![受限层号的顶点数.png](https://raw.githubusercontent.com/GMyhf/img/main/img/4f4d78cc-db0d-4a94-b195-b8fcc14e1d36.png)





需要找到从给定的起始顶点到图中所有其他顶点的最短路径长度（也被称为顶点的层号），并计算层号不超过k的顶点个数。可以使用广度优先搜索（BFS）来解决这个问题。在BFS中，从起始节点开始，然后访问它的所有邻居，然后再访问这些邻居的邻居，依此类推。可以使用一个队列来跟踪待访问的节点，并使用一个距离数组来记录从起始节点到每个节点的最短距离。

以下是解决这个问题的Python代码：

```python
from collections import deque

def bfs(n, m, s, k, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)  # 只按照输入的方向添加边

    distance = [-1] * n
    distance[s] = 0

    queue = deque([s])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return sum(1 for d in distance if d <= k and d != -1)

# 接收数据
n, m, s, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
count = bfs(n, m, s, k, edges)
print(count)
```

在这段代码中，首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`、起始顶点`s`和层号上限`k`，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，调用`bfs`函数并打印结果。



### 3.4 笔试题目（类图 + dfs）

#### 1 数算B-2021笔试最后一个算法题目（8分）

图的深度优先周游算法实现的迷宫探索。图采用邻接表表示，给出了Graph类和Vertex类的基本定义。从题面看基本上与书上提供的G Graph的ADT实现一样，稍作更改。但是这样不一定得到的是最短路径？

在走迷宫时，通常会使用深度优先搜索（DFS）或广度优先搜索（BFS），具体选择哪种搜索算法取决于问题的要求和对性能的考虑。

1. **DFS（深度优先搜索）**：
   - DFS 适合于解决能够表示成树形结构的问题，包括迷宫问题。DFS 会尽可能地沿着迷宫的一条路径向前探索，直到不能再前进为止，然后回溯到前一个位置，再尝试其他路径。在解决迷宫问题时，DFS 可以帮助我们快速地找到一条通路（如果存在），但不一定能够找到最短路径。
   - DFS 的优点是实现简单，不需要额外的数据结构来保存搜索状态。

2. **BFS（广度优先搜索）**：
   - BFS 适合于解决需要找到最短路径的问题，因为它会逐层地搜索所有可能的路径，从起点开始，一层一层地向外扩展。在解决迷宫问题时，BFS 可以找到最短路径（如果存在），但相对于 DFS 来说，它可能需要更多的空间来存储搜索过程中的状态信息。
   - BFS 的优点是能够找到最短路径，并且保证在找到解之前不会漏掉任何可能的路径。

综上所述，如果你只需要找到一条通路而不关心路径的长度，可以选择使用 DFS。但如果你需要找到最短路径，或者需要在可能的路径中选择最优解，那么应该选择 BFS。



![image-20240407223151702](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240407223151702.png)





#### 2 参考答案

<u>下面是我给的答案，请同学自己完成上面算法填空，然后来对照，看答案是否正确。</u>

阅读下列程序，完成图的深度优先周游算法实现的迷宫探索。已知图采用邻接表表示，Graph 类和 Vertex 类基本定义如下：

```python
class Graph:
    def __init__(self):
        self.vertices = {}

    def addVertex(self, key, label): #添加节点，id 为key，附带数据 label
        self.vertices[key] = Vertex(key, label)

    def getVertex(self, key): # 返回 id 为 key 的节点
        return self.vertices.get(key)

    def __contains__(self, key): # 判断 key 节点是否在图中
        return key in self.vertices

    def addEdge(self, f, t, cost=0): # 添加从节点 id==f 到 id==t 的边
        if f in self.vertices and t in self.vertices:
            self.vertices[f].addNeighbor(t, cost)

    def getVertices(self): # 返回所有的节点 key
        return self.vertices.keys()

    def __iter__(self): # 迭代每一个节点对象
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, key, label=None): # 缺省颜色为"white“
        self.id = key
        self.label = label
        self.color = "white"
        self.connections = {}

    def addNeighbor(self, nbr, weight=0): # 添加到节点 nbr 的边
        self.connections[nbr] = weight

    def setColor(self, color): # 设置节点颜色标记
        self.color = color

    def getColor(self): # 返回节点颜色标记
        return self.color

    def getConnections(self): # 返回节点的所有邻接节点列表
        return self.connections.keys()

    def getId(self): # 返回节点的 id
        return self.id

    def getLabel(self): # 返回节点的附带数据 label
        return self.label

#https://github.com/Yuqiu-Yang/problem_solving_with_algorithms_and_data_structures_using_python/blob/master/ch7/ch4_maze2.txt
mazelist = [
    "++++++++++++++++++++++",
    "+   +   ++ ++        +",
    "E     +     ++++++++++",
    "+ +    ++  ++++ +++ ++",
    "+ +   + + ++    +++  +",
    "+          ++  ++  + +",
    "+++++ + +      ++  + +",
    "+++++ +++  + +  ++   +",
    "+          + + S+ +  +",
    "+++++ +  + + +     + +",
    "++++++++++++++++++++++",
]

def mazeGraph(mlist, rows, cols): # 从 mlist 创建图，迷宫有 rows 行 cols 列
    mGraph = Graph()
    vstart = None
    for row in range(rows):
        for col in range(cols):
            if mlist[row][col] != "+":
                mGraph.addVertex((row, col), mlist[row][col])
                if mlist[row][col] == "S":
                    vstart = mGraph.getVertex((row, col)) # 等号右侧填空（1分）

    for v in mGraph:
        row, col = v.getId()
        for i in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= row + i[0] < rows and 0 <= col + i[1] < cols:
                if (row + i[0], col + i[1]) in mGraph:
                    mGraph.addEdge((row, col), (row + i[0], col + i[1])) #括号中两个参数填空（1分）

    return mGraph, vstart # 返回图对象，和开始节点


def searchMaze(path, vcurrent, mGraph): # 从 vcurrent 节点开始 DFS 搜索迷宫，path 保存路径
    path.append(vcurrent.getId())
    vcurrent.setColor("gray")
    if vcurrent.getLabel() != "E":
        done = False
        for nbr in vcurrent.getConnections(): # in 后面部分填空（2分）
            nbr_vertex = mGraph.getVertex(nbr)
            if nbr_vertex.getColor() == "white":
                done = searchMaze(path, nbr_vertex, mGraph) # 参数填空（2分）
                if done:
                    break
        if not done:
            path.pop() # 这条语句空着，填空（2分）
            vcurrent.setColor("white")
    else:
        done = True
    return done # 返回是否成功找到通路


g, vstart = mazeGraph(mazelist, len(mazelist), len(mazelist[0]))
path = []
searchMaze(path, vstart, g)
print(path)

# [(8, 15), (7, 15), (7, 14), (6, 14), (5, 14), (4, 14), (4, 13), (5, 13), (6, 13), (6, 12), (6, 11), (6, 10), (5, 10), (5, 9), (4, 9), (3, 9), (2, 9), (2, 8), (2, 7), (1, 7), (1, 6), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (4, 4), (3, 4), (2, 4), (2, 3), (1, 3), (1, 2), (2, 2), (2, 1), (2, 0)]
```







# 二、（Week9~10）图的算法

## 5 图的算法

### 5.1 拓扑排序

##### 5.1.1 煎松饼

为了展示计算机科学家可以将几乎所有问题都转换成图问题，让我们来考虑如何制作一批松饼。配方十分简单：一个鸡蛋、一杯松饼粉、一勺油，以及3/4杯牛奶。为了制作松饼，需要加热平底锅，并将所有原材料混合后倒入锅中。当出现气泡时，将松饼翻面，继续煎至底部变成金黄色。在享用松饼之前，还会加热一些枫糖浆。图7-18用图的形式展示了整个过程。



![../_images/pancakes.png](https://raw.githubusercontent.com/GMyhf/img/main/img/pancakes.png)



图1 松饼的制作步骤

制作松饼的难点在于知道先做哪一步。从图1可知，可以首先加热平底锅或者混合原材料。我们借助拓扑排序这种图算法来确定制作松饼的步骤。

拓扑排序根据有向无环图生成一个包含所有顶点的线性序列，使得如果图G中有一条边为(v, w)，那么顶点v排在顶点w之前。在很多应用中，有向无环图被用于表明事件优先级。制作松饼只是其中一个例子，其他例子还包括软件项目调度、优化数据库查询的优先级表，以及矩阵相乘。

拓扑排序是对深度优先搜索的一种简单而强大的改进，其算法如下。
(1) 对图`g`调用`dfs(g)`。之所以调用深度优先搜索函数，是因为要计算每一个顶点的结束时间。
(2) 基于结束时间，将顶点按照递减顺序存储在列表中。
(3) 将有序列表作为拓扑排序的结果返回。

图2 展示了`dfs`根据如图1所示的松饼制作步骤构建的深度优先森林。



![../_images/pancakesDFS.png](https://raw.githubusercontent.com/GMyhf/img/main/img/pancakesDFS.png)

图2 根据松饼制作步骤构建的深度优先森林



图3 展示了拓扑排序结果。现在，我们明确地知道了制作松饼所需的步骤



![../_images/pancakesTS.png](https://raw.githubusercontent.com/GMyhf/img/main/img/pancakesTS.png)

图3 对有向无环图的拓扑排序结果



##### 5.1.2 实现煎松饼（depth first forest）

最新MakingPancake_DepthFirstForest.py 在 https://github.com/GMyhf/2024spring-cs201/tree/main/code

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
        self.discovery = 0
        self.finish = None

    def __lt__(self, o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def setDiscovery(self, dtime):
        self.discovery = dtime

    def setFinish(self, ftime):
        self.finish = ftime

    def getFinish(self):
        return self.finish

    def getDiscovery(self):
        return self.discovery

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.discovery) + ":fin " + str(
            self.finish) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"


class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.topologicalList = []

    def dfs(self):
        for aVertex in self:
            aVertex.color = "white"
            aVertex.predecessor = -1
        for aVertex in self:
            if aVertex.color == "white":
                self.dfsvisit(aVertex)

    def dfsvisit(self, startVertex):
        startVertex.color = "gray"
        self.time += 1
        startVertex.setDiscovery(self.time)
        for nextVertex in startVertex.get_neighbors():
            if nextVertex.color == "white":
                nextVertex.previous = startVertex
                self.dfsvisit(nextVertex)
        startVertex.color = "black"
        self.time += 1
        startVertex.setFinish(self.time)

    def topologicalSort(self):
        self.dfs()
        temp = list(self.vertices.values())
        temp.sort(key = lambda x: x.getFinish(), reverse = True)
        print([(x.key,x.finish) for x in temp])
        self.topologicalList = [x.key for x in temp]
        return self.topologicalList

# Creating the graph
g = DFSGraph()

g.add_vertex('cup_milk')    # 3/4杯牛奶
g.add_vertex('egg')         # 一个鸡蛋
g.add_vertex('tbl_oil')     # 1勺油

g.add_vertex('heat_griddle')    # 加热平底锅
g.add_vertex('mix_ingredients') # 混合材料——1杯松饼粉
g.add_vertex('pour_batter')   # 倒入1/4杯松饼粉
g.add_vertex('turn_pancake')    # 出现气泡时翻面
g.add_vertex('heat_syrup')  # 加热枫糖浆
g.add_vertex('eat_pancake') # 开始享用

# Adding edges based on dependencies
g.add_edge('cup_milk', 'mix_ingredients')
g.add_edge('mix_ingredients', 'pour_batter')
g.add_edge('pour_batter', 'turn_pancake')
g.add_edge('turn_pancake', 'eat_pancake')

g.add_edge('mix_ingredients', 'heat_syrup')
g.add_edge('heat_syrup', 'eat_pancake')

g.add_edge('heat_griddle', 'pour_batter')
g.add_edge('tbl_oil', 'mix_ingredients')
g.add_edge('egg', 'mix_ingredients')



# Getting topological sort of the tasks
topo_order = g.topologicalSort()
print("Topological Sort of the Pancake Making Process:")
print(topo_order)

"""
Output:
函数 topologicalSort 中的调试信息
[('eat_pancake', 18), ('heat_syrup', 16), ('turn_pancake', 14), ('pour_cup', 12), ('cup_mix', 10), ('cup_milk', 8), ('heat_griddle', 6), ('tbl_oil', 4), ('egg', 2)]

Topological Sort of the Pancake Making Process:
['heat_griddle', 'tbl_oil', 'egg', 'cup_milk', 'cup_mix', 'heat_syrup', 'pour_cup', 'turn_pancake', 'eat_pancake']
"""
```





##### 5.1.3 Kahn算法

拓扑排序（Topological Sorting）是对有向无环图（DAG）进行排序的一种算法。它将图中的顶点按照一种线性顺序进行排列，使得对于任意的有向边 (u, v)，顶点 u 在排序中出现在顶点 v 的前面。

拓扑排序可以用于解决一些依赖关系的问题，例如任务调度、编译顺序等。

下面是拓扑排序的一种常见算法，称为**Kahn算法**：

1. 计算每个顶点的入度（Indegree），即指向该顶点的边的数量。

2. 初始化一个空的结果列表 `result` 和一个队列 `queue`。

3. 将所有入度为 0 的顶点加入队列 `queue`。

4. 当队列 `queue` 不为空时，执行以下步骤：
   - 从队列中取出一个顶点 `u`。
   - 将 `u` 添加到 `result` 列表中。
   - 对于顶点 `u` 的每个邻接顶点 `v`，减少 `v` 的入度值。
   - 如果顶点 `v` 的入度变为 0，则将 `v` 加入队列 `queue`。

5. 如果 `result` 列表的长度等于图中顶点的数量，则拓扑排序成功，返回结果列表 `result`；否则，图中存在环，无法进行拓扑排序。

下面是一个使用 Kahn 算法进行拓扑排序的示例代码：

```python
from collections import deque, defaultdict

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()

    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.append(u)

    # 执行拓扑排序
    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None

# 示例调用代码
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['F'],
    'F': []
}

sorted_vertices = topological_sort(graph)
if sorted_vertices:
    print("Topological sort order:", sorted_vertices)
else:
    print("The graph contains a cycle.")

# Output:
# Topological sort order: ['A', 'B', 'C', 'D', 'E', 'F']
```

在上述代码中，`graph` 是一个字典，用于表示有向图的邻接关系。它的键表示顶点，值表示一个列表，表示从该顶点出发的边所连接的顶点。

你可以将你的有向图表示为一个邻接矩阵或邻接表，并将其作为参数传递给 `topological_sort` 函数。如果存在拓扑排序，函数将返回一个列表，按照拓扑排序的顺序包含所有顶点。如果图中存在环，函数将返回 `None`，表示无法进行拓扑排序。



##### 5.1.4 实现煎松饼（Kahn）

最新MakingPancake_Kahn.py 在 https://github.com/GMyhf/2024spring-cs201/tree/main/code

```python
from collections import deque, defaultdict

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()

    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.append(u)

    # 执行拓扑排序
    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None

# 示例调用代码
graph = {
    'cup_milk': ['mix_ingredients'],
    'mix_ingredients': ['pour_batter', 'heat_syrup'],
    'pour_batter': ['turn_pancake'],
    'turn_pancake': ['eat_pancake'],
    'heat_syrup': ['eat_pancake'],
    'heat_griddle': ['pour_batter'],
    'tbl_oil': ['mix_ingredients'],
    'egg': ['mix_ingredients'],
    'eat_pancake': []
}


sorted_vertices = topological_sort(graph)
if sorted_vertices:
    print("Topological sort order:", sorted_vertices)
else:
    print("The graph contains a cycle.")

"""
#Depth First Forest ouput:
#['heat_griddle', 'tbl_oil', 'egg', 'cup_milk', 'mix_ingredients', 'heat_syrup', 'pour_batter', 'turn_pancake', 'eat_pancake']

# Kahn ouput:
Topological sort order: ['cup_milk', 'heat_griddle', 'tbl_oil', 'egg', 'mix_ingredients', 'pour_batter', 'heat_syrup', 'turn_pancake', 'eat_pancake']

"""
```



### 5.2 强连通单元（SCC）

接下来将注意力转向规模庞大的图。我们将以互联网主机与各个网页构成的图为例，学习其他几种算法。首先讨论网页。

在互联网上，各种网页形成一张大型的有向图，谷歌和必应等搜索引擎正是利用了这一事实。要将互联网转换成一张图，我们将网页当作顶点，将超链接当作连接顶点的边。图1展示了以路德学院计算机系的主页作为起点的网页连接图的一小部分。由于这张图的规模庞大，因此我们限制网页与起点页之间的链接数不超过10个。



![../_images/cshome.png](https://raw.githubusercontent.com/GMyhf/img/main/img/cshome.png)

图1 以路德学院计算机系的主页作为起点的网页连接图

仔细研究图1，会有一些非常有趣的发现。首先，图中的很多网页来自路德学院的其他网站。其次，一些链接指向爱荷华州的其他学校。最后，一些链接指向其他文理学院。由此可以得出这样的结论：网络具有一种基础结构，使得在某种程度上相似的网页相互聚集。

通过一种叫作强连通单元的图算法，可以找出图中高度连通的顶点簇。对于图G，强连通单元C为最大的顶点子集$C \subset V$ ，其中对于每一对顶点$v, w \in C$，都有一条从v到w的路径和一条从w到v的路径。

We formally define a **strongly connected component (SCC) **, C, of a graph G, as the largest subset of vertices $C \subset V$ such that for every pair of vertices $v, w \in C$ we have a path from v to w and a path from w to v. 

图2展示了一个包含3个强连通单元的简单图。不同的强连通单元通过不同的阴影来表现。

![../_images/scc1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/scc1.png)

图2 含有3个强连通单元的有向图





![../_images/scc2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/scc2.png)
图7-23 简化后的有向图

定义强连通单元之后，就可以把强连通单元中的所有顶点组合成单个顶点，从而将图简化。图3是图2的简化版。

利用深度优先搜索，我们可以再次创建强大高效的算法。在学习强连通单元算法之前，还要再看一个定义。图G的转置图被定义为$G^T$，其中所有的边都与图G的边反向。这意味着，如果在图G中有一条由A到B的边，那么在$G^T$中就会有一条由B到A的边。图4展示了一个简单图及其转置图。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240413153931736.png" alt="image-20240413153931736" style="zoom: 33%;" />

图4 图G及其转置图

再次观察图4。注意，图4a中有2个强连通单元，图4b中也是如此。
以下是计算强连通单元的算法。
(1) 对图G调用dfs，以计算每一个顶点的结束时间。
(2) 计算图$G^T$。
(3) 对图$G^T$​调用dfs，但是在主循环中，按照结束时间的递减顺序访问顶点。
(4) 第3步得到的深度优先森林中的每一棵树都是一个强连通单元。输出每一棵树中的顶点的id。

以图2为例，让我们来逐步分析。图5a展示了用深度优先搜索算法对原图计算得到的发现时间和结束时间，图5b展示了用深度优先搜索算法在转置图上得到的发现时间和结束时间。



![../_images/scc1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/scc1.png)

图2 含有3个强连通单元的有向图

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240413154118511.png" alt="image-20240413154118511" style="zoom:50%;" />

图5 计算强连通单元

最后，图6展示了由强连通单元算法在第3步生成的森林，其中有3棵树。我们没有提供强连通单元算法的Python代码



![../_images/sccforest.png](https://runestone.academy/ns/books/published/pythonds/_images/sccforest.png)

图6 由强连通单元算法生成的森林 SCCs



### 5.3 最短路径

当我们浏览网页、发送电子邮件，或者从校园的另一处登录实验室里的计算机时，在后台发生了很多事，信息从一台计算机传送到另一台计算机。深入地研究信息在多台计算机之间的传送过程，是计算机网络课程的主要内容。本节将适当地讨论互联网的运作机制，并以此介绍另一个非常重要的图算法。

图1从整体上展示了互联网的通信机制。当我们使用浏览器访问某一台服务器上的网页时，访问请求必须通过路由器从本地局域网传送到互联网上，并最终到达该服务器所在局域网对应的路由器。然后，被请求的网页通过相同的路径被传送回浏览器。在图1中，标有“互联网”的云图标中有众多额外的路由器，它们的工作就是协同将信息从一处传送到另一处。如果你的计算机支持traceroute命令，可以利用它亲眼看到许多路由器。List 1展示了traceroute命令的执行结果：在路德学院的Web服务器和明尼苏达大学的邮件服务器之间有13个路由器。



![../_images/Internet.png](https://raw.githubusercontent.com/GMyhf/img/main/img/Internet.png)

图1 互联网通信概览



List1 服务器之间的路由器

```
1  192.203.196.1
2  hilda.luther.edu (216.159.75.1)
3  ICN-Luther-Ether.icn.state.ia.us (207.165.237.137)
4  ICN-ISP-1.icn.state.ia.us (209.56.255.1)
5  p3-0.hsa1.chi1.bbnplanet.net (4.24.202.13)
6  ae-1-54.bbr2.Chicago1.Level3.net (4.68.101.97)
7  so-3-0-0.mpls2.Minneapolis1.Level3.net (64.159.4.214)
8  ge-3-0.hsa2.Minneapolis1.Level3.net (4.68.112.18)
9  p1-0.minnesota.bbnplanet.net (4.24.226.74)
10  TelecomB-BR-01-V4002.ggnet.umn.edu (192.42.152.37)
11  TelecomB-BN-01-Vlan-3000.ggnet.umn.edu (128.101.58.1)
12  TelecomB-CN-01-Vlan-710.ggnet.umn.edu (128.101.80.158)
13  baldrick.cs.umn.edu (128.101.80.129)(N!)  88.631 ms (N!)


Routers from One Host to the Next over the Internet
```



互联网上的每一个路由器都与一个或多个其他的路由器相连。如果在不同的时间执行traceroute命令，极有可能看到信息在不同的路由器间流动。这是由于一对路由器之间的连接存在着一定的成本，成本大小取决于流量、时间段以及众多其他因素。至此，你应该能够理解为何可以用带权重的图来表示路由器网络。

图2展示了一个小型路由器网络对应的带权图。我们要解决的问题是为给定信息找到权重最小的路径。这个问题并不陌生，因为它和我们之前用宽度优先搜索解决过的问题十分相似，只不过现在考虑的是路径的总权重，而不是路径的长度。需要注意的是，如果所有的权重都相等，那么两个问题就没有区别。



![../_images/routeGraph.png](https://raw.githubusercontent.com/GMyhf/img/main/img/routeGraph.png)

图2 小型路由器网络对应的带权图



#### 5.3.1 Dijkstra算法

> **Breadth First Search** explores equally in all directions. This is an incredibly useful algorithm, not only for regular path finding, but also for procedural map generation, flow field pathfinding, distance maps, and other types of map analysis.
> **Dijkstra’s Algorithm** (also called Uniform Cost Search) lets us prioritize which paths to explore. Instead of exploring all possible paths equally, it favors lower cost paths. We can assign lower costs to encourage moving on roads, higher costs to avoid enemies, and more. When movement costs vary, we use this instead of Breadth First Search.



Dijkstra算法可用于确定最短路径，它是一种循环算法，可以提供从一个顶点到其他所有顶点的最短路径。这与宽度优先搜索非常像。

为了记录从起点到各个终点的总开销，要利用Vertex类中的实例变量distance。该实例变量记录从起点到当前顶点的最小权重路径的总权重。Dijkstra算法针对图中的每个顶点都循环一次，但循环顺序是由一个优先级队列控制的。用来决定顺序的正是dist。在创建顶点时，将distance设为一个非常大的值。理论上可以将distance设为无穷大，但是实际一般将其设为一个大于所有可能出现的实际距离的值。

Dijkstra算法的实现如代码清单List 2所示。当程序运行结束时，distance和previous都会被设置成正确的值。

> 2024/4/14 说明：教材目前已经有第3版了。我尽量按照第3版做课件，例程与老版相比有稍微改动。

List1 Dijkstra算法的Python实现

```python
# https://github.com/psads/pythonds3
from pythonds3.graphs import PriorityQueue
def dijkstra(graph,start):
    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in graph])
    while pq:
        distance, current_v = pq.delete()
        for next_v in current_v.getneighbors():
            new_distance = current_v.distance + current_v.get_neighbor(next_v) # + get_weight
            if new_distance < next_v.distance:
                next_v.distance = new_distance
                next_v.previous = current_v
                pq.change_priority(next_v,new_distance)
```



```python
from pythonds3.trees.binary_heap import BinaryHeap
class PriorityQueue(BinaryHeap):
    def change_priority(self, search_key: Any, new_priority: Any) -> None:
        key_to_move = -1
        for i, (_, key) in enumerate(self._heap):
            if key == search_key:
                key_to_move = i
                break
        if key_to_move > -1:
            self._heap[key_to_move] = (new_priority, search_key)
            self._perc_up(key_to_move)

    def __contains__(self, search_key: Any) -> bool:
        for _, key in self._heap:
            if key == search_key:
                return True
        return False
```



Dijkstra算法使用了优先级队列。你应该记得，"树"那一章讲过如何用堆实现优先级队列。不过，当时的简单实现和用于Dijkstra算法的实现有几个不同点。首先，PriorityQueue类存储了键-值对的二元组。这对于Dijkstra算法来说非常重要，因为优先级队列中的键必须与图中顶点的键相匹配。其次，二元组中的值被用来确定优先级，对应键在优先级队列中的位置。在Dijkstra算法的实现中，我们使用了顶点的距离作为优先级，这是因为我们总希望访问距离最小的顶点。另一个不同点是增加了`change_priority`方法（第14行）。当到一个顶点的距离减少并且该顶点已在优先级队列中时，就调用这个方法，从而将该顶点移向优先级队列的头部。

让我们对照图3 来理解如何针对每一个顶点应用Dijkstra算法。从顶点u开始，与u相邻的3个顶点分别是v、w和x。由于到v、w和x的初始距离都是`sys.maxint`，因此从起点到它们的新开销就是直接开销。更新这3个顶点的开销，同时将它们的前驱顶点设置成 u，并将它们添加到优先级队列中。我们使用距离作为优先级队列的键。此时，算法运行的状态如图3a所示。



![image-20240413161937819](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240413161937819.png)

图3 Dijkstra算法的应用过程

下一次while循环检查与x相邻的顶点。之所以x是第2个被访问的顶点，是因为它到起点的开销最小，因此排在了优先级队列的头部。与x相邻的有u、v、w和y。对于每一个相邻顶点，检查经由x到它的距离是否比已知的距离更短。显然，对于y来说确实如此，因为它的初始距离是`sys.maxsize`；对于u和v来说则不然，因为它们的距离分别为0和2。但是，我们发现经过x到w的距离比直接从u到w的距离要短。因此，将到达w的距离更新为更短的值，并且将w的前驱顶点从u改为x。图3b展示了此时的状态。

下一步检查与v相邻的顶点。这一步没有对图做任何改动，因此我们继续检查顶点y。此时，我们发现经由y到达w和z的距离都更短，因此相应地调整它们的距离及前驱顶点。最后检查w和z，发现不需要做任何改动。由于优先级队列为空，因此退出。

非常重要的一点是，Dijkstra算法只适用于边的权重均为正的情况。如果图2中有一条边的权重为负，那么Dijkstra算法永远不会退出。

除了Dijkstra算法，还有其他一些算法被用于寻找最短路径。Dijkstra算法的问题是需要有完整的图，这意味着每一个路由器都要知道整个互联网的路由器连接情况，而事实并非如此。Dijkstra算法的一些变体允许每个路由器在运行时才发现图，例如“距离向量”路由算法。



#### 5.3.2 分析Dijkstra算法

最后，我们来分析Dijkstra算法的时间复杂度。开始时，要将图中的每一个顶点都添加到优先级队列中，这个操作的时间复杂度是$O(|V|)$。优先级队列构建完成之后，`while`循环针对每一个顶点都执行一次，这是由于一开始所有顶点都被添加到优先级队列中，并且只在循环时才被移除。在循环内部，每次对`pq.delete`的调用都是$O(|V| \log(|V|))$。综合起来考虑，循环和delMin调用的总时间复杂度是$O(|V| \log(|V|))$。for循环对图中的每一条边都执行一次，并且循环内部的`change_priority`调用为O(|E| log|V|)。因此，总的时间复杂度为$O((|V|+|E|) \log(|V|))$。



#### *5.3.3 Bellman-Ford算法

在图论中，有两种常见的方法用于求解最短路径问题：**Dijkstra算法**和**Bellman-Ford算法**。这两种算法各有优劣，选择哪种算法取决于图的特性和问题要求。如果图中没有负权边，并且只需要求解单源最短路径，Dijkstra算法通常是一个较好的选择。如果图中存在负权边或需要检测负权回路，或者需要求解所有节点对之间的最短路径，可以使用Bellman-Ford算法。

**Bellman-Ford算法**：Bellman-Ford算法用于解决单源最短路径问题，与Dijkstra算法不同，它可以处理带有负权边的图。算法的基本思想是通过松弛操作逐步更新节点的最短路径估计值，直到收敛到最终结果。具体步骤如下：

- 初始化一个距离数组，用于记录源节点到所有其他节点的最短距离。初始时，源节点的距离为0，其他节点的距离为无穷大。
- 进行V-1次循环（V是图中的节点数），每次循环对所有边进行松弛操作。如果从节点u到节点v的路径经过节点u的距离加上边(u, v)的权重比当前已知的从源节点到节点v的最短路径更短，则更新最短路径。
- 检查是否存在负权回路。如果在V-1次循环后，仍然可以通过松弛操作更新最短路径，则说明存在负权回路，因此无法确定最短路径。

Bellman-Ford算法的时间复杂度为O(V*E)，其中V是图中的节点数，E是图中的边数。





#### *5.3.4 多源最短路径Floyd-Warshall算法

求解所有顶点之间的最短路径可以使用**Floyd-Warshall算法**，它是一种多源最短路径算法。Floyd-Warshall算法可以在有向图或无向图中找到任意两个顶点之间的最短路径。

算法的基本思想是通过一个二维数组来存储任意两个顶点之间的最短距离。初始时，这个数组包含图中各个顶点之间的直接边的权重，对于不直接相连的顶点，权重为无穷大。然后，通过迭代更新这个数组，逐步求得所有顶点之间的最短路径。

具体步骤如下：

1. 初始化一个二维数组`dist`，用于存储任意两个顶点之间的最短距离。初始时，`dist[i][j]`表示顶点i到顶点j的直接边的权重，如果i和j不直接相连，则权重为无穷大。

2. 对于每个顶点k，在更新`dist`数组时，考虑顶点k作为中间节点的情况。遍历所有的顶点对(i, j)，如果通过顶点k可以使得从顶点i到顶点j的路径变短，则更新`dist[i][j]`为更小的值。

   `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`

3. 重复进行上述步骤，对于每个顶点作为中间节点，进行迭代更新`dist`数组。最终，`dist`数组中存储的就是所有顶点之间的最短路径。

Floyd-Warshall算法的时间复杂度为O(V^3)，其中V是图中的顶点数。它适用于解决稠密图（边数较多）的最短路径问题，并且可以处理负权边和负权回路。

以下是一个使用Floyd-Warshall算法求解所有顶点之间最短路径的示例代码：

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

在上述代码中，`graph`是一个字典，用于表示图的邻接关系。它的键表示起始顶点，值表示一个字典，其中键表示终点顶点，值表示对应边的权重。

你可以将你的图表示为一个邻接矩阵或邻接表，并将其作为参数传递给`floyd_warshall`函数。函数将返回一个二维数组，其中`dist[i][j]`表示从顶点i到顶点j的最短路径长度。



### 5.4 最小生成树(MST) 

#### 5.4.1 Prim算法

在学习最后一个图算法之前，先考虑网络游戏设计师和互联网广播服务提供商面临的问题。他们希望高效地把信息传递给所有人。这在网络游戏中非常重要，因为所有玩家都可以据此知道其他玩家的最近位置。互联网广播也需要做到这一点，以让所有听众都接收到所需数据。图1展示了上述广播问题。



![../_images/bcast1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/bcast1.png)

图1 广播问题



为了更好地理解上述问题，我们先来看看如何通过蛮力法求解。你稍后会看到，本节提出的解决方案为何优于蛮力法。假设互联网广播服务提供商要向所有收听者播放一条消息，最简单的方法是保存一份包含所有收听者的列表，然后向每一个收听者单独发送消息。以图1为例，若采用上述解法，则每一条消息都需要有4份副本。假设使用开销最小的路径，让我们来看看每一个路由器需要处理多少次相同的消息。

从广播服务提供商发出的所有消息都会经过路由器A，因此A能够看到每一条消息的所有副本。路由器C只能看到一份副本，而由于路由器B和D在收听者1、2、3的最短路径上，因此它们能够看到每一条消息的3份副本。考虑到广播服务提供商每秒会发送数百条消息，这样做会导致流量剧增。

一种蛮力法是广播服务提供商针对每条消息只发送一份副本，然后由路由器来正确地发送。最简单的方法就是`不受控泛洪法（uncontrolled flooding）`，策略如下：每一条消息都设有**存活时间（time to live, ttl）**，它大于或等于广播服务提供商和最远的收听者之间的距离；每一个路由器都接收到消息的一份副本，并且将消息发送给所有的相邻路由器。在消息被发送时，它的`ttl`递减，直到变为0。不难发现，不受控泛洪法产生的不必要消息比第一种方法更多。

解决广播问题的关键在于构建一棵权重最小的`生成树`。我们对最小权重`生成树`的正式定义如下：对于图G=(V, E)，最小生成树T是E的无环子集，并且连接V 中的所有顶点。

图2展示了简化的广播图，并且突出显示了形成最小生成树的所有边。为了解决广播问题，广播服务提供商只需向网络中发送一条消息副本。每一个路由器向属于生成树的相邻路由器转发消息，其中不包括刚刚向它发送消息的路由器。在图2的例子中，A把消息转发给B, B把消息转发给C和D, D转发给E, E转发给F, F转发给G。每一个路由器都只看到任意消息的一份副本，并且所有的收听者都接收到了消息。



![../_images/mst1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/mst1.png)

图2 广播图中的最小生成树



上述思路对应的算法叫作Prim算法。由于每一步都选择代价最小的下一步，因此Prim算法属于一种“贪婪算法”。在这个问题中，代价最小的下一步是选择权重最小的边。接下来实现Prim算法。

构建生成树的基本思想如下：

```python
While T is not yet a spanning tree
   Find an edge that is safe to add to the tree
   Add the new edge to T
```

难点在于，如何找到“可以安全添加到树中的边”。我们这样定义安全的边：它的一端是生成树中的顶点，另一端是还不在生成树中的顶点。这保证了构建的树不会出现循环。

Prim算法的Python实现如代码List1所示。与Dijkstra算法类似，Prim算法也使用了优先级队列来选择下一个添加到图中的顶点。



List1 Prim算法的Python实现

```python
# https://github.com/psads/pythonds3
from pythonds3.graphs import PriorityQueue

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



图3展示了将Prim算法应用于示例生成树的过程。以顶点A作为起点，将A到其他所有顶点的距离都初始化为无穷大。检查A的相邻顶点后，可以更新从A到B和C的距离，因为实际的距离小于无穷大。更新距离之后，B和C被移到优先级队列的头部。并且，它们的前驱顶点被设置为A。注意，我们还没有把B和C添加到生成树中。只有在从优先级队列中移除时，顶点才会被添加到生成树中。


![image-20240413164314867](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240413164314867.png)

图3 Prim算法的应用过程

由于到B的距离最短，因此接下来检查B的相邻顶点。检查后发现，可以更新D和E。接下来处理优先级队列中的下一个顶点C。与C相邻的唯一一个还在优先级队列中的顶点是F，因此更新到F的距离，并且调整F在优先级队列中的位置。

现在检查与D相邻的顶点，发现可以将到E的距离从6减少为4。修改距离的同时，把E的前驱顶点改为D，以此准备将E添加到生成树中的另一个位置。Prim算法正是通过这样的方式将每一个顶点都添加到生成树中。



#### 5.4.2 Kruskal's Algorithm:

- Approach: Kruskal's algorithm sorts all the edges in the graph by their weights and then iteratively adds the edges with the minimum weight as long as they do not create a cycle in the MST.
- Suitable for: Kruskal's algorithm is often used when the graph is sparse or when the number of edges is much smaller than the number of vertices. It is efficient for finding the MST in such cases.
- Connectivity: Kruskal's algorithm may produce a forest of MSTs initially, and then it merges them into a single MST.

Key similarities and connections between Prim's and Kruskal's algorithms:

- Both algorithms find the minimum spanning tree of a graph.
- They are both greedy algorithms that make locally optimal choices in each step to achieve the overall minimum weight.
- The resulting MSTs produced by both algorithms have the same total weight.

In summary, you can choose between Prim's algorithm and Kruskal's algorithm based on the characteristics of the graph, such as density or sparsity, and the specific requirements of your problem.



Kruskal算法是一种用于解决最小生成树（Minimum Spanning Tree，简称MST）问题的贪心算法。给定一个连通的带权无向图，Kruskal算法可以找到一个包含所有顶点的最小生成树，即包含所有顶点且边权重之和最小的树。

以下是Kruskal算法的基本步骤：

1. 将图中的所有边按照权重从小到大进行排序。

2. 初始化一个空的边集，用于存储最小生成树的边。

3. 重复以下步骤，直到边集中的边数等于顶点数减一或者所有边都已经考虑完毕：

   - 选择排序后的边集中权重最小的边。
   - 如果选择的边不会导致形成环路（即加入该边后，两个顶点不在同一个连通分量中），则将该边加入最小生成树的边集中。

4. 返回最小生成树的边集作为结果。

Kruskal算法的核心思想是通过不断选择权重最小的边，并判断是否会形成环路来构建最小生成树。算法开始时，每个顶点都是一个独立的连通分量，随着边的不断加入，不同的连通分量逐渐合并为一个连通分量，直到最终形成最小生成树。

实现Kruskal算法时，一种常用的数据结构是并查集（Disjoint Set）。并查集可以高效地判断两个顶点是否在同一个连通分量中，并将不同的连通分量合并。

下面是一个使用Kruskal算法求解最小生成树的示例代码：

```python
class DisjointSet:
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1


def kruskal(graph):
    num_vertices = len(graph)
    edges = []

    # 构建边集
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    # 按照权重排序
    edges.sort(key=lambda x: x[2])

    # 初始化并查集
    disjoint_set = DisjointSet(num_vertices)

    # 构建最小生成树的边集
    minimum_spanning_tree = []

    for edge in edges:
        u, v, weight = edge
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            minimum_spanning_tree.append((u, v, weight))

    return minimum_spanning_tree
```

在上述代码中，`graph` 是一个二维矩阵，表示带权无向图的邻接矩阵。`graph[i][j]` 表示顶点 i 和顶点 j 之间的边的权重。

Kruskal算法的时间复杂度为 O(ElogE)，其中 E 是边的数量。排序边集的时间复杂度为 O(ElogE)，并查集操作的时间复杂度为 O(Eα(V))，其中 α 是 Ackermann 函数的反函数，近似为常数。因此，总体上来说，Kruskal算法的时间复杂度可以近似为 O(ElogE)。



### 5.5 Dijkstra 和 Prim实现

#### 5.5.1 通常的Dijkstra实现

使用 `heapq` 来实现 Dijkstra 算法的完整 Python 代码。这个实现包括了图的类表示，顶点类，以及 Dijkstra 算法的具体逻辑。

通过维护一个 `visited` 集合，我们可以确保每个顶点只被处理一次。

这个实现也包括了输出每个顶点的最短距离和从起始点到每个顶点的具体路径。

使用 `heapq` 模块确实是一个好选择，因为它通常比 `PriorityQueue` 更为高效。

> `PriorityQueue，即 Python 标准库中 `queue.PriorityQueue`，其底层实际上是基于 `heapq`实现的。就像`heapq`，`PriorityQueue` 也没有直接支持修改队列中元素优先级的内置方法。`PriorityQueue`提供线程安全的队列操作，适用于多线程程序，但它并不支持如`decrease-key` 操作，这种操作在很多图算法中非常有用。
>
> 要实现修改优先级的功能，你可以采用与 `heapq` 类似的策略：将新的优先级作为一个新的条目加入到队列中，并通过一种机制（比如标记或记录）忽略或移除旧的条目。这种方法在 `PriorityQueue` 中同样适用，但要注意它的线程安全性和性能影响。

**Dijkstra算法**：Dijkstra算法用于解决单源最短路径问题，即从给定源节点到图中所有其他节点的最短路径。算法的基本思想是通过不断扩展离源节点最近的节点来逐步确定最短路径。具体步骤如下：

- 初始化一个距离数组，用于记录源节点到所有其他节点的最短距离。初始时，源节点的距离为0，其他节点的距离为无穷大。
- 选择一个未访问的节点中距离最小的节点作为当前节点。
- 更新当前节点的邻居节点的距离，如果通过当前节点到达邻居节点的路径比已知最短路径更短，则更新最短路径。
- 标记当前节点为已访问。
- 重复上述步骤，直到所有节点都被访问或者所有节点的最短路径都被确定。

Dijkstra算法的时间复杂度为O(V^2)，其中V是图中的节点数。当使用优先队列（如最小堆）来选择距离最小的节点时，可以将时间复杂度优化到O((V+E)logV)，其中E是图中的边数。

Dijkstra.py 程序在 https://github.com/GMyhf/2024spring-cs201/tree/main/code

```python
import heapq
import sys

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = sys.maxsize
        self.pred = None

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def __lt__(self, other):
        return self.distance < other.distance

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        self.numVertices += 1
        return newVertex

    def getVertex(self, n):
        return self.vertList.get(n)

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

def dijkstra(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        currentDist, currentVert = heapq.heappop(pq)    # 当一个顶点的最短路径确定后（也就是这个顶点
                                                        # 从优先队列中被弹出时），它的最短路径不会再改变。
        if currentVert in visited:
            continue
        visited.add(currentVert)

        for nextVert in currentVert.getConnections():
            newDist = currentDist + currentVert.getWeight(nextVert)
            if newDist < nextVert.distance:
                nextVert.distance = newDist
                nextVert.pred = currentVert
                heapq.heappush(pq, (newDist, nextVert))

# 创建图和边
g = Graph()
g.addEdge('A', 'B', 4)
g.addEdge('A', 'C', 2)
g.addEdge('C', 'B', 1)
g.addEdge('B', 'D', 2)
g.addEdge('C', 'D', 5)
g.addEdge('D', 'E', 3)
g.addEdge('E', 'F', 1)
g.addEdge('D', 'F', 6)

# 执行 Dijkstra 算法
print("Shortest Path Tree:")
dijkstra(g, g.getVertex('A'))

# 输出最短路径树的顶点及其距离
for vertex in g.vertList.values():
    print(f"Vertex: {vertex.id}, Distance: {vertex.distance}")

# 输出最短路径到每个顶点
def printPath(vert):
    if vert.pred:
        printPath(vert.pred)
        print(" -> ", end="")
    print(vert.id, end="")

print("\nPaths from Start Vertex 'A':")
for vertex in g.vertList.values():
    print(f"Path to {vertex.id}: ", end="")
    printPath(vertex)
    print(", Distance: ", vertex.distance)

"""
Shortest Path Tree:
Vertex: A, Distance: 0
Vertex: B, Distance: 3
Vertex: C, Distance: 2
Vertex: D, Distance: 5
Vertex: E, Distance: 8
Vertex: F, Distance: 9

Paths from Start Vertex 'A':
Path to A: A, Distance:  0
Path to B: A -> C -> B, Distance:  3
Path to C: A -> C, Distance:  2
Path to D: A -> C -> B -> D, Distance:  5
Path to E: A -> C -> B -> D -> E, Distance:  8
Path to F: A -> C -> B -> D -> E -> F, Distance:  9
"""
```



#### 5.5.2 通常的Prim实现

Prim's algorithm and Kruskal's algorithm are both used to find the minimum spanning tree (MST) of a connected, weighted graph. However, they have different approaches and are suitable for different scenarios. Here are the key differences and the typical use cases for each algorithm:

Prim's Algorithm:

- Approach: Prim's algorithm starts with a single vertex and gradually grows the MST by iteratively adding the edge with the minimum weight that connects a vertex in the MST to a vertex outside the MST.

- Suitable for: Prim's algorithm is often used when the graph is dense or when the number of edges is close to the number of vertices. It is efficient for finding the MST in such cases.

- Connectivity: Prim's algorithm always produces a connected MST.

  

通过维护一个 `visited` 集合，我们可以确保每个顶点只被处理一次。

```python
import sys
import heapq

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = sys.maxsize
        self.pred = None

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def __lt__(self, other):
        return self.distance < other.distance

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        self.numVertices += 1
        return newVertex

    def getVertex(self, n):
        return self.vertList.get(n)

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)
        self.vertList[t].addNeighbor(self.vertList[f], cost)

def prim(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        currentDist, currentVert = heapq.heappop(pq)
        if currentVert in visited:
            continue
        visited.add(currentVert)

        for nextVert in currentVert.getConnections():
            weight = currentVert.getWeight(nextVert)
            if nextVert not in visited and weight < nextVert.distance:
                nextVert.distance = weight
                nextVert.pred = currentVert
                heapq.heappush(pq, (weight, nextVert))

# 创建图和边
g = Graph()
g.addEdge('A', 'B', 4)
g.addEdge('A', 'C', 3)
g.addEdge('C', 'B', 1)
g.addEdge('C', 'D', 2)
g.addEdge('D', 'B', 5)
g.addEdge('D', 'E', 6)

# 执行 Prim 算法
print("Minimum Spanning Tree:")
prim(g, g.getVertex('A'))

# 输出最小生成树的边
for vertex in g.vertList.values():
    if vertex.pred:
        print(f"{vertex.pred.id} -> {vertex.id} Weight:{vertex.distance}")

"""
Minimum Spanning Tree:
C -> B Weight:1
A -> C Weight:3
C -> D Weight:2
D -> E Weight:6
"""
```



#### 5.5.3 书上Dijkstra实现，不敢恭维

在 `heapq` 中，直接改变一个元素的优先级并重新排序堆并不是直接支持的功能，因为 `heapq` 模块提供的是一个简单的堆实现，而不是一个优先级队列。然而，你可以通过一种变通的方法来模拟这个功能。

一个常用的技巧是将被更新优先级的元素重新插入到堆中，但标记原有的元素为失效。具体到 Dijkstra 算法中，这意味着当你发现到某个顶点的更短路径时，你将这个顶点与新的距离一起推入堆中，并通过一个字典或集合来跟踪最新的有效状态。

下面的代码展示了这种技术的具体实现。Book_Dijkstra.py在https://github.com/GMyhf/2024spring-cs201/tree/main/code

```python
import sys
from heapq import heappop, heappush, heapify

class Vertex:
    def __init__(self, key):
        self.key = key
        self.neighbors = {}
        self.distance = sys.maxsize
        self.previous = None
        self.color = None

    def get_neighbor(self, other):
        return self.neighbors.get(other, None)

    def set_neighbor(self, other, weight=0):
        self.neighbors[other] = weight

    def __repr__(self):
        return f"Vertex({self.key})"

    def __str__(self):
        return (
            f"{self.key} connected to: "
            + f"{[x.key for x in self.neighbors]}"
        )

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_key(self):
        return self.key

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.key == other.key
        return False

    def __lt__(self, other):
        if isinstance(other, Vertex):
            return self.distance < other.distance
        return False

    def __hash__(self):
        return hash(self.key)


class Graph:
    def __init__(self):
        self.vertices = {}

    def set_vertex(self, key):
        self.vertices[key] = Vertex(key)

    def get_vertex(self, key):
        return self.vertices.get(key, None)

    def __contains__(self, key):
        return key in self.vertices

    def add_edge(self, from_vert, to_vert, weight=0):
        if from_vert not in self.vertices:
            self.set_vertex(from_vert)
        if to_vert not in self.vertices:
            self.set_vertex(to_vert)
        self.vertices[from_vert].set_neighbor(
            self.vertices[to_vert], weight
        )

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())


# print("\n---Graph---\n")
# g = Graph()
# for i in range(6):
#     g.set_vertex(i)
# print(g.vertices)
# g.add_edge(0, 1, 5)
# g.add_edge(0, 5, 2)
# g.add_edge(1, 2, 4)
# g.add_edge(2, 3, 9)
# g.add_edge(3, 4, 7)
# g.add_edge(3, 5, 3)
# g.add_edge(4, 0, 1)
# g.add_edge(5, 4, 8)
# g.add_edge(5, 2, 1)
# for v in g:
#     for w in v.get_neighbors():
#         print(f"({v.get_key()}, {w.get_key()})")



"""
这个实现使用了一个字典 visited 来跟踪每个顶点的最短已知距离。
如果在优先队列中发现一个顶点的旧记录，通过检查 visited 中记录的距离来决定是否忽略它。
这样可以确保每个顶点的最新距离总是被正确处理，即便它被多次推入堆中。
"""
def dijkstra(graph, start):
    pq = [(v.distance, v) for v in graph]
    start.distance = 0
    heapify(pq)
    visited = {}
    while pq:
        distance, current_v = heappop(pq)
        if current_v in visited and visited[current_v] < distance:
            continue
        for next_v in current_v.get_neighbors():
            new_distance = (
                current_v.distance
                + current_v.get_neighbor(next_v)
            )
            if new_distance < next_v.distance:
                next_v.distance = new_distance
                next_v.previous = current_v
                heappush(pq, (next_v.distance, next_v))
                print("".join(f"{v.distance % 1000:<5d}" for v in graph))

print("\n---Dijkstra's---\n")
g = Graph()
vertices = ["u", "v", "w", "x", "y", "z"]
for v in vertices:
    g.set_vertex(v)
g.add_edge("u", "v", 2)
g.add_edge("u", "w", 5)
g.add_edge("u", "x", 1)
g.add_edge("v", "u", 2)
g.add_edge("v", "w", 3)
g.add_edge("v", "x", 1)
g.add_edge("w", "u", 5)
g.add_edge("w", "v", 3)
g.add_edge("w", "x", 3)
g.add_edge("w", "y", 1)
g.add_edge("w", "z", 5)
g.add_edge("x", "u", 1)
#g.add_edge("x", "v", 2)
g.add_edge("x", "v", 1)
g.add_edge("x", "w", 3)
g.add_edge("x", "y", 1)
g.add_edge("y", "w", 1)
g.add_edge("y", "x", 1)
g.add_edge("y", "z", 1)
g.add_edge("z", "w", 5)
g.add_edge("z", "y", 1)
print("".join(f"{v:5s}" for v in vertices))
dijkstra(g, g.get_vertex("u"))
print(
    "".join(
        f"{g.get_vertex(v).distance:<5d}"
        for v in vertices
    )
)

"""
---Dijkstra's---

u    v    w    x    y    z    
0    2    807  807  807  807  
0    2    5    807  807  807  
0    2    5    1    807  807  
0    2    4    1    807  807  
0    2    4    1    2    807  
0    2    3    1    2    807  
0    2    3    1    2    3    
0    2    3    1    2    3    
"""

```



#### 5.5.4 书上Prim实现，不敢恭维

Book_Prim.py在https://github.com/GMyhf/2024spring-cs201/tree/main/code

```python
import sys
import heapq

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = sys.maxsize
        self.pred = None

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def __lt__(self, other):
        return self.distance < other.distance

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        self.numVertices += 1
        return newVertex

    def getVertex(self, n):
        return self.vertList.get(n)

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)
        self.vertList[t].addNeighbor(self.vertList[f], cost)

def prim(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        currentDist, currentVert = heapq.heappop(pq)
        if currentVert in visited:
            continue
        visited.add(currentVert)

        for nextVert in currentVert.getConnections():
            weight = currentVert.getWeight(nextVert)
            if nextVert not in visited and weight < nextVert.distance:
                nextVert.distance = weight
                nextVert.pred = currentVert
                heapq.heappush(pq, (weight, nextVert))

# 创建图和边
g = Graph()
g.addEdge('A', 'B', 4)
g.addEdge('A', 'C', 3)
g.addEdge('C', 'B', 1)
g.addEdge('C', 'D', 2)
g.addEdge('D', 'B', 5)
g.addEdge('D', 'E', 6)

# 执行 Prim 算法
print("Minimum Spanning Tree:")
prim(g, g.getVertex('A'))

# 输出最小生成树的边
for vertex in g.vertList.values():
    if vertex.pred:
        print(f"{vertex.pred.id} -> {vertex.id} Weight:{vertex.distance}")

"""
Minimum Spanning Tree:
C -> B Weight:1
A -> C Weight:3
C -> D Weight:2
D -> E Weight:6
"""
```



### 5.6 小结

介绍了图的抽象数据类型，以及一些实现方式。如果能将一个问题用图表示出来，那么就可以利用图算法加以解决。对于解决下列问题，图非常有用。

❏ 利用宽度优先搜索找到无权重的最短路径。
❏ 利用Dijkstra算法求解带权重的最短路径。
❏ 利用深度优先搜索来探索图。
❏ 利用强连通单元来简化图。
❏ 利用拓扑排序为任务排序。
❏ 利用最小生成树广播消息。





### *5.7 关键路径

在数据结构中，关键路径算法通常与有向加权图（有向图中每条边都有一个权重）相关。一种常用的关键路径算法是**AOV 网络关键路径算法**（Activity On Vertex Network Critical Path Algorithm），它适用于没有环路的有向加权图。

以下是 AOV 网络关键路径算法的基本步骤：

1. 根据项目的活动和依赖关系，构建有向加权图。图的顶点表示活动，边表示活动之间的依赖关系，边的权重表示活动的持续时间。

2. 对图进行拓扑排序，以确定活动的执行顺序。拓扑排序可以使用 Kahn 算法来实现。

3. 初始化两个数组：`earliest_start_time` 和 `latest_finish_time`，分别用于存储每个顶点的最早开始时间和最晚完成时间。

4. 从拓扑排序的第一个顶点开始，按照拓扑排序的顺序遍历每个顶点。

   - 对于当前顶点 u，计算其最早开始时间 `earliest_start_time[u]`，即前面所有依赖顶点的最晚完成时间中的最大值加上 u 的持续时间。

5. 从拓扑排序的最后一个顶点开始，按照逆拓扑排序的顺序遍历每个顶点。

   - 对于当前顶点 v，计算其最晚完成时间 `latest_finish_time[v]`，即后面所有依赖顶点的最早开始时间中的最小值减去 v 的持续时间。

6. 对于每条边 (u, v)，计算其总时差（Total Float）：

   - 总时差等于 `latest_finish_time[v] - earliest_start_time[u] - edge_weight(u, v)`。

7. 找到总时差为 0 的边，这些边构成了关键路径。关键路径上的活动是项目的关键活动，任何关键活动的延迟都会导致项目延迟。

关键路径算法可以使用图的邻接表或邻接矩阵来表示有向加权图，并以此作为输入进行计算。通过计算关键路径，可以确定项目的关键活动和项目的最长完成时间，有助于项目管理和资源分配。

请注意，这里介绍的是一种常见的关键路径算法，其他算法和技术也可用于求解关键路径问题，具体选择取决于实际情况和需求。



### 5.8 编程题目



#### sy403: 先导课程 中等

https://sunnywhy.com/sfbj/10/6/403

现有n门课程（假设课程编号为从`0`到`n-1`），课程之间有依赖关系，即可能存在两门课程，必须学完其中一门才能学另一门。现在给出个依赖关系，问能否把所有课程都学完。

注：能同时学习多门课程时总是先学习编号最小的课程。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果能学完所有课程，那么输出一行`Yes`，然后在第二行输出学习课程编号的顺序，编号之间用空格隔开，行末不允许有多余的空格；如果不能学完所有课程，那么输出一行`No`，然后在第二行输出不能学习的课程门数。

样例1

输入

```
4 5
0 1
0 2
0 3
1 2
3 2
```

输出

```
Yes
0 1 3 2
```

解释

对应的依赖关系如下图所示。由于每次选择编号最小的顶点，因此学习顺序为`0 1 3 2`。

![拓扑排序.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b4214671-2ee9-421c-8e1e-26c8f0de224e.png)

样例2

输入

```
4 4
0 1
1 2
2 3
3 1
```

输出

```
No
3
```

解释

对应的依赖关系如下图所示。编号为`0`的课程可以直接学习；由于编号`1`、`2`、`3`的课程互相依赖，因此无法学习的课程数为`3`。

![先导课程_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/9579516f-e705-451f-a6d1-6d15314b8a99.png)





这个问题可以通过使用拓扑排序来解决。拓扑排序是对有向无环图的顶点的一种排序，它使得如果存在一条从顶点 u 到顶点 v 的路径，那么在排序中 u 一定在 v 的前面。如果在进行拓扑排序的过程中，发现存在没有被访问的顶点，但是已经没有入度为0的顶点，那么就说明图中存在环，也就是存在不能学习的课程。

以下是使用拓扑排序判断有向图是否存在环并输出学习课程顺序的Python代码：

```python
from collections import defaultdict

def courseSchedule(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue = [i for i in range(n) if indegree[i] == 0]
    queue.sort()
    result = []

    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
        queue.sort()

    if len(result) == n:
        return "Yes", result
    else:
        return "No", n - len(result)

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]
res, courses = courseSchedule(n, edges)
print(res)
if res == "Yes":
    print(*courses)
else:
    print(courses)
```

在这段代码中，我们首先构建了一个图，并计算了每个顶点的入度。然后我们将所有入度为0的顶点添加到队列中，并对队列进行排序。然后在循环中，我们每次从队列中取出一个顶点，将其添加到结果中，然后遍历其所有的邻接顶点，将邻接顶点的入度减1，如果邻接顶点的入度变为0，那么就将其添加到队列中。最后，我们检查访问的顶点数是否等于总的顶点数，如果等于，那么说明可以学完所有课程，否则说明存在不能学习的课程。



#### sy386: 最短距离 简单

https://sunnywhy.com/sfbj/10/4/386

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从s号城市出发到达t号城市的最短距离。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

输出一个整数，表示最短距离。如果无法到达，那么输出`-1`。

样例1

输入

```
6 6 0 2
0 1 2
0 2 5
0 3 1
2 3 2
1 2 1
4 5 1
```

输出

```
3
```

解释

对应的无向图如下图所示。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`；
2. `0->2`：距离为`5`；
3. `0->1->2`：距离为`3`。

因此最短距离为`3`。

![最短距离.png](https://raw.githubusercontent.com/GMyhf/img/main/img/1123ea31-976a-43fb-bc9d-11eec6ce0f26.png)

样例2

输入

```
6 6 0 5
0 1 2
0 2 5
0 3 1
2 3 2
1 2 1
4 5 1
```

输出

```
-1
```

解释

和第一个样例相同的图，终点换成了`5`号顶点，显然从`0`号无法到达`5`号。



需要找到从给定的起始城市到目标城市的最短距离。可以使用Dijkstra算法来解决这个问题。Dijkstra算法是一种用于在图中找到最短路径的算法。它从起始节点开始，然后逐步扩展到所有可达的节点，每次选择当前最短的路径进行扩展。

以下是使用 Python 实现 Dijkstra 算法来解决这个问题的示例代码：

```python
import heapq

def dijkstra(n, edges, s, t):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    pq = [(0, s)]  # (distance, node)
    visited = set()
    distances = [float('inf')] * n
    distances[s] = 0

    while pq:
        dist, node = heapq.heappop(pq)
        if node == t:
            return dist
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    return -1

# Read input
n, m, s, t = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem and print the result
result = dijkstra(n, edges, s, t)
print(result)
```

这段代码实现了 Dijkstra 算法来求解从起点到终点的最短路径。首先构建了一个图，然后使用优先队列来选择下一个要探索的节点，并在探索过程中更新最短距离。最后返回从起点到终点的最短距离。

这个版本的Dijkstra算法使用了一个集合`visited`来记录已经访问过的节点，这样可以避免对同一个节点的重复处理。当我们从优先队列中取出一个节点时，如果这个节点已经在`visited`集合中，那么我们就跳过这个节点，处理下一个节点。这样可以提高算法的效率。

此外，这个版本的Dijkstra算法还在找到目标节点`t`时就立即返回结果，而不是等到遍历完所有节点。这是因为Dijkstra算法保证了每次从优先队列中取出的节点就是当前距离最短的节点，所以当我们找到目标节点`t`时，就已经找到了从起始节点`s`到`t`的最短路径，无需再继续搜索。

这个版本的Dijkstra算法的时间复杂度仍然是O((V+E)logV)，其中V是顶点数，E是边数。这是因为每个节点最多会被加入到优先队列中一次（当找到一条更短的路径时），并且每条边都会被处理一次（在遍历节点的邻居时）。优先队列的插入和删除操作的时间复杂度都是O(logV)，所以总的时间复杂度是O((V+E)logV)。



Dijkstra 算法是一种经典的图算法，它综合运用了多种技术，包括邻接表、集合、优先队列（堆）、贪心算法和动态规划的思想。例题：最短距离，https://sunnywhy.com/sfbj/10/4/386

- 邻接表：Dijkstra 算法通常使用邻接表来表示图的结构，这样可以高效地存储图中的节点和边。
- 集合：在算法中需要跟踪已经访问过的节点，以避免重复访问，这一般使用集合（或哈希集合）来实现。
- 优先队列（堆）：Dijkstra 算法中需要选择下一个要探索的节点，通常使用优先队列（堆）来维护当前候选节点的集合，并确保每次都能快速找到距离起点最近的节点。
- 贪心算法：Dijkstra 算法每次选择距离起点最近的节点作为下一个要探索的节点，这是一种贪心策略，即每次做出局部最优的选择，期望最终能达到全局最优。
- 动态规划：Dijkstra 算法通过不断地更新节点的最短距离来逐步得到从起点到各个节点的最短路径，这是一种动态规划的思想，即将原问题拆解成若干子问题，并以最优子结构来解决。

综合运用这些技术，Dijkstra 算法能够高效地求解单源最短路径问题，对于解决许多实际问题具有重要意义。



#### sy396: 最小生成树-Prim算法 简单

https://sunnywhy.com/sfbj/10/5/396

现有一个共个顶点、条边的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权。在图中寻找一棵树，使得这棵树包含图上所有顶点、所有边都是图上的边，且树上所有边的边权之和最小。使用Prim算法求出这个边权之和的最小值。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

输出一个整数，表示最小的边权之和。如果不存在这样的树，那么输出`-1`。

样例1

输入

```
4 5
0 1 3
0 2 2
0 3 3
2 3 1
1 2 1
```

输出

```
4
```

解释

对应的无向图如下图所示。加粗的部分即为最小生成树，其边权之和为1+1+2=4。

![最小生成树-Prim算法.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292321529.png)

样例2

输入

```
3 1
0 1 1
```

输出

```
-1
```

解释

由于此图不连通，因此不存在最小生成树。



以下是使用 Prim 算法求解的 Python 代码：

```python
import heapq

def prim(graph, n):
    visited = [False] * n
    min_heap = [(0, 0)]  # (weight, vertex)
    min_spanning_tree_cost = 0

    while min_heap:
        weight, vertex = heapq.heappop(min_heap)

        if visited[vertex]:
            continue

        visited[vertex] = True
        min_spanning_tree_cost += weight

        for neighbor, neighbor_weight in graph[vertex]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (neighbor_weight, neighbor))

    return min_spanning_tree_cost if all(visited) else -1

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]

    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    min_spanning_tree_cost = prim(graph, n)
    print(min_spanning_tree_cost)

if __name__ == "__main__":
    main()

```





#### sy397: 最小生成树-Kruskal算法 简单

https://sunnywhy.com/sfbj/10/5/397

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权。在图中寻找一棵树，使得这棵树包含图上所有顶点、所有边都是图上的边，且树上所有边的边权之和最小。使用Kruskal算法求出这个边权之和的最小值。

**输入**

第一行两个整数n、m（$1 \le n \le 10^4,0 \le m \le 10^5$），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权。数据保证不会有重边。

**输出**

输出一个整数，表示最小的边权之和。如果不存在这样的树，那么输出`-1`。

样例1

输入

```
4 5
0 1 3
0 2 2
0 3 3
2 3 1
1 2 1
```

输出

```
4
```

解释

对应的无向图如下图所示。加粗的部分即为最小生成树，其边权之和为1+1+2=4。

![最小生成树-Prim算法.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292321529.png)





Kruskal算法是一种用于寻找最小生成树的算法。它的基本思想是按照边的权值从小到大的顺序选择边，如果这条边连接的两个顶点不在同一连通分量中，则选择这条边，否则放弃这条边。重复这个过程，直到图中所有的顶点都在同一连通分量中。

在实现Kruskal算法时，我们需要使用并查集来维护图中的连通分量，以便于快速判断两个顶点是否在同一连通分量中。

以下是Kruskal算法的Python实现：

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
        px, py = self.find(x), self.find(y)
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1

def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    res = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            res += w
    if len(set(uf.find(i) for i in range(n))) > 1:
        return -1
    return res

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))
print(kruskal(n, edges))
```

这段代码首先定义了一个并查集类`UnionFind`，然后定义了`kruskal`函数来实现Kruskal算法。在`kruskal`函数中，我们首先创建一个并查集实例，然后按照边的权值对边进行排序。然后遍历每一条边，如果这条边连接的两个顶点不在同一连通分量中，就将这两个顶点合并到同一连通分量中，并将这条边的权值加到结果中。最后，如果图中的所有顶点都在同一连通分量中，就返回结果，否则返回-1。



可以看到，kruskal 算法的时间复杂度主要来源于对边进行排序，因此其时间复杂度是O(ElogE)，其中E为图的边数。显然 kruskal 适合顶点数较多、边数较少的情况，这和 prim算法恰好相反。于是可以根据题目所给的数据范围来选择合适的算法，即**如果是稠密图(边多)，则用 prim 算法;如果是稀疏图(边少)，则用 kruskal 算法**。



#### 20106: 走山路

bfs + heap, Dijkstra, http://cs101.openjudge.cn/routine/20106/



#### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/



#### 04116: 拯救行动

bfs, http://cs101.openjudge.cn/practice/04116/



#### 04129: 变换的迷宫

bfs, http://cs101.openjudge.cn/practice/04129



#### 02802: 小游戏

bfs, http://cs101.openjudge.cn/practice/02802/ 



#### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930



# 三、笔试题目

2023年有考到KMP，冒泡排序的优化。

2022年5个大题：图Dijkstra，二叉树，排序，单链表，二叉树。

2021年6个大题：森林dfs、bfs，哈夫曼树，二叉树建堆，图prim，二叉树遍历，图走迷宫。







## 选择（30分，每题2分）

**Q:** 



## 判断（10分，每题1分）

对填写"Y"，错填写"N"

**Q:** （Y）



## 填空（20分，每题2分）

**Q:** 



## 简答（24分，每题6分）

**Q:** 







# 参考

Problem Solving with Algorithms and Data Structures using Python, 3ed.

https://runestone.academy/ns/books/published/pythonds3/index.html?mode=browsing



https://github.com/psads/pythonds3



https://github.com/Yuqiu-Yang/problem_solving_with_algorithms_and_data_structures_using_python

https://github.com/wesleyjtann/Problem-Solving-with-Algorithms-and-Data-Structures-Using-Python



晴问题目练习网址，https://sunnywhy.com/sfbj
