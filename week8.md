# 20240409-Week8



Updated 1436 GMT+8 Apr 4, 2024

2024 spring, Complied by Hongfei Yan



**Logs：**





# 一、基数排序

基数排序是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。



```python
def radixSort(arr):
    digit = 0
    max_digit = 1
    max_value = max(arr)
    # 找出列表中最大的位数
    while 10 ** max_digit < max_value:
        max_digit = max_digit + 1

    while digit < max_digit:
        temp = [[] for i in range(10)]
        for i in arr:
            # 求出每一个元素的个、十、百位的值
            t = int((i / 10 ** digit) % 10)
            temp[t].append(i)

        coll = []
        for bucket in temp:
            for i in bucket:
                coll.append(i)

        arr = coll
        digit = digit + 1

    return arr

arr = [ 170, 45, 75, 90, 802, 24, 2, 66]
ans = radixSort(arr)
print(*ans)
```



这个程序是一个实现基数排序（Radix Sort）的函数。基数排序是一种非比较型的排序算法，它根据数字的位数来对数字进行排序。

下面是对程序的解读：

1. `radixSort` 函数接受一个整数列表 `arr` 作为输入，并返回排序后的列表。
2. 在函数中，首先找出列表中的最大值 `max_value`，以确定需要排序的数字的最大位数。
3. 然后，通过 `digit` 变量来表示当前处理的位数，初始化为 1。在每次迭代中，`digit` 的值会乘以 10，以处理下一个更高位的数字。
4. 在每次迭代中，创建一个包含 10 个空列表的临时列表 `temp`，用于存储每个数字在当前位数上的分组情况。
5. 对于列表中的每个数字 `i`，计算其在当前位数上的值 `t`（通过取整除和取模操作），然后将数字 `i` 存入对应的桶中。
6. 在填充完所有桶之后，将桶中的数字按照顺序取出，重新放入原始列表 `arr` 中。这样就完成了对当前位数的排序。
7. 继续迭代，直到处理完所有位数为止。
8. 最后，返回排序后的列表 `arr`。

通过基数排序，可以有效地对整数列表进行排序，时间复杂度为 O(d * (n + k))，其中 d 是最大位数，n 是数字个数，k 是基数（这里是 10）。



**Complexity Analysis of Radix Sort**

Time Complexity:

- Radix sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping the keys by the individual digits which share the same significant position and value. It has a time complexity of O(d \* (n + b)), where d is the number of digits, n is the number of elements, and b is the base of the number system being used.
- In practical implementations, radix sort is often faster than other comparison-based sorting algorithms, such as quicksort or merge sort, for large datasets, especially when the keys have many digits. However, its time complexity grows linearly with the number of digits, and so it is not as efficient for small datasets.

Auxiliary Space:

- Radix sort also has a space complexity of O(n + b), where n is the number of elements and b is the base of the number system. This space complexity comes from the need to create buckets for each digit value and to copy the elements back to the original array after each digit has been sorted.





# 二、神奇的dict

dict的value如果是list，是邻接表。dici嵌套dict 是 字典树/前缀树/Trie。

是的，你提到的两种数据结构分别是邻接表和字典树（前缀树，Trie）。

1. **邻接表**：在图论中，邻接表是一种表示图的常见方式之一。如果你使用字典（`dict`）来表示图的邻接关系，并且将每个顶点的邻居顶点存储为列表（`list`），那么就构成了邻接表。例如：

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}
```

2. **字典树（前缀树，Trie）**：字典树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。如果你使用嵌套的字典来表示字典树，其中每个字典代表一个节点，键表示路径上的字符，而值表示子节点，那么就构成了字典树。例如：

```python
trie = {
    'a': {
        'p': {
            'p': {
                'l': {
                    'e': {'is_end': True}
                }
            }
        }
    },
    'b': {
        'a': {
            'l': {
                'l': {'is_end': True}
            }
        }
    },
    'c': {
        'a': {
            't': {'is_end': True}
        }
    }
}
```

这样的表示方式使得我们可以非常高效地搜索和插入字符串，特别是在大型数据集上。



神奇的dict

字典（dict）是Python中非常强大和灵活的数据结构之一，它可以用来存储键值对，是一种可变容器模型，可以存储任意数量的 Python 对象。

字典在 Python 中被广泛用于各种场景，例如：

1. **哈希映射**：字典提供了一种快速的键值查找机制，可以根据键快速地检索到相应的值。这使得字典成为了哈希映射（Hash Map）的理想实现。

2. **符号表**：在编程语言的实现中，字典常常被用作符号表，用来存储变量名、函数名等符号和它们的关联值。

3. **配置文件**：字典可以用来表示配置文件中的键值对，例如JSON文件就是一种常见的字典格式。

4. **缓存**：字典常常用于缓存中，可以将计算结果与其输入参数关联起来，以便后续快速地检索到相同参数的计算结果。

5. **图的表示**：如前文所述，字典可以用来表示图的邻接关系，是一种常见的图的表示方式。

由于其灵活性和高效性，字典在Python中被广泛应用于各种场景，并且被称为是Python中最常用的数据结构之一。



## Algorithm for BFS

How to implement Breadth First Search algorithm in Python 

https://www.codespeedy.com/breadth-first-search-algorithm-in-python/

BFS is one of the traversing algorithm used in graphs. This algorithm is implemented using a queue data structure. In this algorithm, the main focus is on the vertices of the graph. Select a starting node or vertex at first, mark the starting node or vertex as visited and store it in a queue. Then visit the vertices or nodes which are adjacent to the starting node, mark them as visited and store these vertices or nodes in a queue. Repeat this process until all the nodes or vertices are completely visited.

**Advantages of BFS**

1. It can be useful in order to find whether the graph has connected components or not.
2. It always finds or returns the shortest path if there is more than one path between two vertices.

 

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



## Algorithm for DFS

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
5.  Initially, the stack is empty.Push the starting node into the stack (stack.append(start) ).
6. Mark the starting node as visited (visited.append(start) ).
7. Repeat this process until all the neighbours are visited in the stack till the destination node is found.
8. If the destination node is found exit the while loop.
9. If the destination node is not present then “Not found” is printed.
10. Finally, print the path from starting node to the destination node.



# 三、晴问基础题目

栈和队列（20题）

搜索专题（15题）

树专题（46题）

Todo 图算法专题（33题）





# 四、笔试题目

2023年有考到KMP，冒泡排序的优化。

2022年5个大题：图Dijkstra，二叉树，排序，单链表，二叉树。

2021年6个大题：森林dfs、bfs，哈夫曼树，二叉树建堆，图prim，二叉树遍历，图走迷宫。







# 参考

Python数据结构与算法分析(第2版)，布拉德利·米勒 戴维·拉努姆/吕能,刁寿钧译，出版时间:2019-09

Brad Miller and David Ranum, Problem Solving with Algorithms and Data Structures using Python, https://runestone.academy/ns/books/published/pythonds/index.html



https://github.com/wesleyjtann/Problem-Solving-with-Algorithms-and-Data-Structures-Using-Python



数据结构（C语言版 第2版） (严蔚敏)
