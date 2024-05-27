# 目录

基础数据结构和算法

排序

线性表及其算法：链表、栈、队列；调度场、表达式之间的转换、单调栈

树：并查集、AVL、Huffman编码树、堆、字典树

图：最短路、最小生成树、拓扑排序

# Python内置接口

## collections

### deque

`deque`（双端队列）是一个从头部和尾部都能快速增删元素的容器。这种数据结构非常适合用于需要快速添加和弹出元素的场景，如队列和栈。

1. 添加元素

- **`append(x)`**：在右端添加一个元素 `x`。时间复杂度为 O(1)。
- **`appendleft(x)`**：在左端添加一个元素 `x`。时间复杂度为 O(1)。

2. 移除元素

- **`pop()`**：移除并返回右端的元素。如果没有元素，将引发 `IndexError`。时间复杂度为 O(1)。
- **`popleft()`**：移除并返回左端的元素。如果没有元素，将引发 `IndexError`。时间复杂度为 O(1)。

3. 扩展

- **`extend(iterable)`**：在右端依次添加 `iterable` 中的元素。整体操作的时间复杂度为 O(k)，其中 `k` 是 `iterable` 的长度。
- **`extendleft(iterable)`**：在左端依次添加 `iterable` 中的元素。注意，添加的顺序会是 `iterable` 元素的逆序。整体操作的时间复杂度为 O(k)，其中 `k` 是 `iterable` 的长度。

4. 其他操作

- **`rotate(n=1)`**：向右旋转队列 `n` 步。如果 `n` 是负数，则向左旋转。这个操作的时间复杂度为 O(k)，其中 `k` 是 `n` 的绝对值，但实际上因为只涉及到指针移动，所以非常快。
- **`clear()`**：移除所有的元素，使其长度为 0。时间复杂度为 O(n)，其中 `n` 是 `deque` 中元素的数量。
- **`remove(value)`**：移除找到的第一个值为 `value` 的元素。这个操作在最坏情况下的时间复杂度为 O(n)，因为可能需要遍历整个 `deque`。

5. 访问元素

- 对于 `deque`，虽然可以通过索引访问，如 `d[0]` 或 `d[-1]`，但这不是 `deque` 设计的主要用途，且访问中间元素的时间复杂度为 O(n)。因此，如果你需要频繁地从随机位置访问数据，`deque` 可能不是最佳选择。

```python
from collections import deque

# 初始化deque
d = deque([1, 2, 3])

# 添加元素
d.append(4)  # deque变为[1, 2, 3, 4]
d.appendleft(0)  # deque变为[0, 1, 2, 3, 4]

# 移除元素
d.pop()  # 返回 4, deque变为[0, 1, 2, 3]
d.popleft()  # 返回 0, deque变为[1, 2, 3]

# 扩展
d.extend([4, 5])  # deque变为[1, 2, 3, 4, 5]
d.extendleft([0])  # deque变为[0, 1, 2, 3, 4, 5]

# 旋转
d.rotate(1)  # deque变为[5, 0, 1, 2, 3, 4]
d.rotate(-2)  # deque变为[1, 2, 3, 4, 5, 0]

# 清空
d.clear()  # deque变为空
```

### Counter, defaultdict, namedtuple, OrderedDict

1. `Counter`

`Counter` 是一个用于计数可哈希对象的字典子类。它是一个集合，其中元素的存储形式为字典键值对，键是元素，值是元素计数。

```python
from collections import Counter

# 创建 Counter 对象
cnt = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])

# 访问计数
print(cnt['blue'])    # 输出: 3
print(cnt['red'])     # 输出: 2

# 更新计数
cnt.update(['blue', 'red', 'blue'])
print(cnt['blue'])    # 输出: 5

# 计数的常见方法
print(cnt.most_common(2))  # 输出 [('blue', 5), ('red', 3)]
```

2. `defaultdict`

`defaultdict` 是另一种字典子类，它提供了一个默认值，用于字典所尝试访问的键不存在时返回。

```python
from collections import defaultdict

# 使用 lambda 来指定默认值为 0
d = defaultdict(lambda: 0)

d['key1'] = 5
print(d['key1'])  # 输出: 5
print(d['key2'])  # 输出: 0，因为 key2 不存在，返回默认值 0
```

3. `namedtuple`

`namedtuple` 生成可以使用名字来访问元素内容的元组子类。

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)

print(p.x + p.y)  # 输出: 33
print(p[0] + p[1])  # 输出: 33  # 还可以像普通元组那样用索引访问
```

4. `OrderedDict`

`OrderedDict` 是一个字典子类，它保持了元素被添加的顺序，这在某些情况下非常有用。

```python
from collections import OrderedDict

od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3

for key in od:
    print(key, od[key])
# 输出:
# z 1
# y 2
# x 3
```

## permutations

在 Python 中，`permutations` 是 `itertools` 模块中的一个非常有用的函数，用于生成输入可迭代对象的所有可能排列。排列是将一组元素组合成一定顺序的所有可能方式。例如，集合 [1, 2, 3] 的全排列包括 [1, 2, 3]、[1, 3, 2]、[2, 1, 3] 等。

使用 `itertools.permutations`

`itertools.permutations(iterable, r=None)` 函数接收两个参数：

- `iterable`：要排列的数据集。
- `r`：可选参数，指定生成排列的长度。如果 `r` 未指定，则默认值等于 `iterable` 的长度，即生成全排列。

返回值是一个迭代器，生成元组，每个元组是一个可能的排列。

示例代码

下面是使用 `itertools.permutations` 的一些示例：

1. 生成全排列

```python
import itertools

data = [1, 2, 3]
permutations_all = list(itertools.permutations(data))

# 输出所有排列
for perm in permutations_all:
    print(perm)
```

输出：

```python
(1, 2, 3)
(1, 3, 2)
(2, 1, 3)
(2, 3, 1)
(3, 1, 2)
(3, 2, 1)
```

2. 生成长度为 `r` 的排列

如果你只想生成一部分元素的排列，可以设置 `r` 的值。

```python
import itertools

data = [1, 2, 3, 4]
permutations_r = list(itertools.permutations(data, 2))

# 输出长度为2的排列
for perm in permutations_r:
    print(perm)
```

输出：

```python
(1, 2)
(1, 3)
(1, 4)
(2, 1)
(2, 3)
(2, 4)
(3, 1)
(3, 2)
(3, 4)
(4, 1)
(4, 2)
(4, 3)
```

注意事项

- `itertools.permutations` 生成的排列是 **不重复的**，即使输入的元素中有重复，输出的每个排列仍然是唯一的。
- 生成的排列是按照字典序排列的，基于输入 `iterable` 的顺序。
- 由于排列的数量非常快地随着 `n`（元素总数）和 `r`（排列的长度）的增加而增加，生成非常大的排列集可能会消耗大量的内存和计算资源。例如，10个元素的全排列总共有 10! (即 3,628,800) 种可能，这在实际应用中可能是不切实际的。

使用 `itertools.permutations` 可以有效地处理排列问题，是解决许多算法问题的有力工具。

## heapq

`heapq` 模块是 Python 的标准库之一，提供了基于堆的优先队列算法的实现。堆是一种特殊的完全二叉树，满足父节点的值总是小于或等于其子节点的值（在最小堆的情况下）。这个属性使堆成为实现优先队列的理想数据结构。

基本操作

`heapq` 模块提供了一系列函数来管理堆，但它只提供了“最小堆”的实现。以下是一些主要功能及其用法：

1. `heapify(x)`

- **用途**：将列表 `x` 原地转换为堆。

- 示例

  ```
  import heapq
  data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
  heapq.heapify(data)
  print(data)  # 输出将是堆，但可能不是完全排序的
  ```

2. `heappush(heap, item)`

- **用途**：将 `item` 加入到堆 `heap` 中，并保持堆的不变性。

- 示例

  ```
  heap = []
  heapq.heappush(heap, 3)
  heapq.heappush(heap, 1)
  heapq.heappush(heap, 4)
  print(heap)  # 输出最小元素总是在索引0
  ```

3. `heappop(heap)`

- **用途**：弹出并返回 `heap` 中最小的元素，保持堆的不变性。

- 示例

  ```
  print(heapq.heappop(heap))  # 返回1
  print(heap)  # 剩余的堆
  ```

4. `heapreplace(heap, item)`

- **用途**：弹出堆中最小的元素，并将新的 `item` 插入堆中，效率高于先 `heappop()` 后 `heappush()`。

- 示例

  ```
  heapq.heapreplace(heap, 7)
  print(heap)
  ```

5. `heappushpop(heap, item)`

- **用途**：先将 `item` 压入堆中，然后弹出并返回堆中最小的元素。

- 示例

  ```
  result = heapq.heappushpop(heap, 0)
  print(result)  # 输出0
  print(heap)  # 剩余的堆
  ```

6. `nlargest(n, iterable, key=None)` 和 `nsmallest(n, iterable, key=None)`

- **用途**：从 `iterable` 数据中找出最大的或最小的 `n` 个元素。

- 示例

  ```
  data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
  print(heapq.nlargest(3, data))  # 输出[9, 6, 5]
  print(heapq.nsmallest(3, data))  # 输出[1, 1, 2]
  ```

应用场景

`heapq` 通常用于需要快速访问最小（或最大）元素的场景，但不需要对整个列表进行完全排序。它广泛应用于数据处理、实时计算、优先级调度等领域。例如，任务调度、Dijkstra 最短路径算法、Huffman 编码树生成等都会用到堆结构。

注意事项

- 如需实现最大堆功能，可以通过对元素取反来实现。将所有元素取负后使用 `heapq`，然后再取负回来即可。
- 堆操作的时间复杂度一般为 O(log n)，适合处理大数据集。
- `heapq` 只能保证列表中的第一个元素是最小的，其他元素的排序并不严格。

## queue

Python 的 `queue` 模块提供了多种队列类型，主要用于线程间的通信和数据共享。这些队列都是线程安全的，设计用来在生产者和消费者线程之间进行数据交换。除了已经提到的 `LifoQueue` 之外，`queue` 模块还提供了以下几种有用的队列类型：

1. `Queue`

这是标准的先进先出（FIFO）队列。元素从队列的一端添加，并从另一端被移除。这种类型的队列特别适用于任务调度，保证了任务被处理的顺序。

- **`put(item, block=True, timeout=None)`**：将 `item` 放入队列中。如果可选参数 `block` 设为 `True`，并且 `timeout` 是一个正数，则在超时前会阻塞等待可用的槽位。
- **`get(block=True, timeout=None)`**：从队列中移除并返回一个元素。如果可选参数 `block` 设为 `True`，并且 `timeout` 是一个正数，则在超时前会阻塞等待元素。
- **`empty()`**：判断队列是否为空。
- **`full()`**：判断队列是否已满。
- **`qsize()`**：返回队列中的元素数量。注意，这个大小只是近似值，因为在返回值和队列实际状态间可能存在时间差。

2. `PriorityQueue`

基于优先级的队列，队列中的每个元素都有一个优先级，优先级最低的元素（注意是最“低”）最先被移除。这是通过将元素存储为 `(priority_number, data)` 对来实现的。

- 优先级可以是任何可排序的类型，通常是数字，其中较小的值具有较高的优先级。

3. `SimpleQueue`

在 Python 3.7 及以后版本中引入了 `SimpleQueue`，它是一个简单的先进先出队列，没有大小限制，不像 `Queue`，它没有任务跟踪或其他复杂的功能，通常性能更好。

- **`put(item)`**：将 `item` 放入队列。
- **`get()`**：从队列中移除并返回一个元素。
- **`empty()`**：判断队列是否为空。

4.`LifoQueue` 

在 Python 中，LIFO（后进先出）队列可以通过标准库中的 `queue` 模块实现，其中 `LifoQueue` 类提供了一个基于 LIFO 原则的队列实现。LIFO 队列通常被称为堆栈（stack），因为它遵循“后进先出”的原则，即最后一个添加到队列中的元素将是第一个被移除的元素。

`LifoQueue` 提供了以下几个主要的方法：

- **`put(item)`**: 将 `item` 元素放入队列中。
- **`get()`**: 从队列中移除并返回最顶端的元素。
- **`empty()`**: 检查队列是否为空。
- **`full()`**: 检查队列是否已满。
- **`qsize()`**: 返回队列中的元素数量。

示例代码

下面是如何使用 `queue.LifoQueue` 的一个简单示例：

```
import queue

# 创建一个 LIFO 队列
lifo_queue = queue.LifoQueue()

# 添加元素
lifo_queue.put('a')
lifo_queue.put('b')
lifo_queue.put('c')

# 依次取出元素
print(lifo_queue.get())  # 输出 'c'
print(lifo_queue.get())  # 输出 'b'
print(lifo_queue.get())  # 输出 'a'
```

注意事项

- `LifoQueue` 是线程安全的，这意味着它可以安全地用于多线程环境。
- 如果 `LifoQueue` 初始化时指定了最大容量，`put()` 方法在队列满时默认会阻塞，直到队列中有空闲位置。如果需要，可以用 `put_nowait()` 方法来避免阻塞，但如果队列满了，这会抛出 `queue.Full` 异常。
- 类似地，`get()` 方法在队列为空时会阻塞，直到队列中有元素可以取出。`get_nowait()` 方法也可以用来避免阻塞，但如果队列空了，会抛出 `queue.Empty` 异常。

示例代码

下面是一个使用 `PriorityQueue` 的例子：

```
import queue

# 创建一个优先级队列
pq = queue.PriorityQueue()

# 添加元素及其优先级
pq.put((3, 'Low priority'))
pq.put((1, 'High priority'))
pq.put((2, 'Medium priority'))

# 依次取出元素
while not pq.empty():
    print(pq.get()[1])  # 输出元素的数据部分
```

使用场景

- **`Queue`**: 适用于任务调度，如在多线程下载文件时管理下载任务。
- **`LifoQueue`**: 适用于需要后进先出逻辑的场景，比如回溯算法。
- **`PriorityQueue`**: 用于需要处理优先级任务的场景，如操作系统的任务调度。
- **`SimpleQueue`**: 适用于需要快速操作且不需要额外功能的场景，比如简单的数据传递任务。

这些队列因其线程安全的特性，特别适合用于多线程程序中，以确保数据的一致性和完整性。



# 基础数据结构和算法

## 二分查找

### 04135:月度开销

http://cs101.openjudge.cn/2024sp_routine/04135/

模拟插板，但是二分

```python
n, m = map(int, input().split())
L = list(int(input()) for x in range(n))


def check(x):
    num, cut = 1, 0
    for i in range(n):
        if cut + L[i] > x:
            num += 1
            cut = L[i]  # 在L[i]左边插一个板，L[i]属于新的fajo月
        else:
            cut += L[i]
    return num <= m


maxmax = sum(L)
minmax = max(L)
while minmax < maxmax:
    middle = (maxmax + minmax) // 2
    if check(middle):  # 表明这种插法可行，那么看看更小的插法可不可以
        maxmax = middle
    else:
        minmax = middle + 1  # 这种插法不可行，改变minmax看看下一种插法可不可以
print(maxmax)
```

### 08210:河中跳房子

http://cs101.openjudge.cn/2024sp_routine/08210/

# 排序

## 归并排序（Merge Sort）

基础知识

时间复杂度：

- **最坏情况**: *O*(*n*log*n*)
- **平均情况**: *O*(*n*log*n*)
- **最优情况**: O*(*n*log*n*)

- **空间复杂度**: O(n) — 需要额外的内存空间来存储临时数组。

- **稳定性**: 稳定 — 相同元素的相对顺序在排序后不会改变。

代码示例

应用

- **计算逆序对数**：在一个数组中，如果前面的元素大于后面的元素，则这两个元素构成一个逆序对。归并排序可以在排序过程中修改并计算逆序对的总数。这通过在归并过程中，每当右侧的元素先于左侧的元素被放置到结果数组时，记录左侧数组中剩余元素的数量来实现。
- **排序链表**：归并排序在链表排序中特别有用，因为它可以实现在链表中的有效排序而不需要额外的空间，这是由于链表的节点可以通过改变指针而不是实际移动节点来重新排序。



### **OJ02299:Ultra-QuickSort**

http://cs101.openjudge.cn/2024sp_routine/02299/

与**20018:蚂蚁王国的越野跑**（http://cs101.openjudge.cn/2024sp_routine/20018/）类似。

算需要交换多少次来得到一个排好序的数组，其实就是算逆序对。

```python
d = 0


def merge(arr, l, m, r):
    """对l到m和m到r两段进行合并"""
    global d
    n1, n2 = m - l + 1, r - m  # L1和L2的长
    L1, L2 = arr[l:m + 1], arr[m + 1:r + 1]
    # L1和L2均为有序序列
    i, j, k = 0, 0, l  # i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i < n1 and j < n2:
        if L1[i] <= L2[j]:
            arr[k] = L1[i]
            i += 1
        else:
            arr[k] = L2[j]
            d += n1 - i  # 精髓所在
            j += 1
        k += 1
    while i < n1:
        arr[k] = L1[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = L2[j]
        j += 1
        k += 1


def mergesort(arr, l, r):
    """对arr的l到r一段进行排序"""
    if l < r:  # 递归结束条件，很重要
        m = (l + r) // 2
        mergesort(arr, l, m)
        mergesort(arr, m + 1, r)
        merge(arr, l, m, r)


while True:
    n = int(input())
    if n == 0:
        break
    array = []
    for b in range(n):
        array.append(int(input()))
    d = 0
    mergesort(array, 0, n - 1)
    print(d)
```

## 快速排序（Quick Sort）

时间复杂度

- **最坏情况**: �(�2)*O*(*n*2) — 通常发生在已经排序的数组或基准选择不佳的情况下。
- **平均情况**: �(�log⁡�)*O*(*n*log*n*)
- **最优情况**: �(�log⁡�)*O*(*n*log*n*) — 适当的基准可以保证分割平衡。

- **空间复杂度**: �(log⁡�)*O*(log*n*) — 主要是递归的栈空间。

- **稳定性**: 不稳定 — 基准点的选择和划分过程可能会改变相同元素的相对顺序。

应用：k-th元素

## 堆排序（Heap Sort）

时间复杂度

- **最坏情况**: �(�log⁡�)*O*(*n*log*n*)
- **平均情况**: �(�log⁡�)*O*(*n*log*n*)
- **最优情况**: �(�log⁡�)*O*(*n*log*n*)

- **空间复杂度**: �(1)*O*(1) — 堆排序是原地排序算法，不需要额外的存储空间。

- **稳定性**: 不稳定 — 堆的维护过程可能会改变相同元素的原始相对顺序。

# 线性表

## 单调队列

例题

### 26978:滑动窗口最大值

http://cs101.openjudge.cn/2024sp_routine/26978/

https://leetcode.cn/problems/sliding-window-maximum/solutions/543426/hua-dong-chuang-kou-zui-da-zhi-by-leetco-ki6m/

## 单调栈

### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/

# 树

## 树的四种遍历及其相互转化

### 25145:猜二叉树

http://cs101.openjudge.cn/2024sp_routine/25145/

### 25140:根据后序表达式建立表达式树

http://cs101.openjudge.cn/2024sp_routine/25140/

## AVL树

### 27625:AVL树至少有几个结点

http://cs101.openjudge.cn/2024sp_routine/27625/

## 并查集

并查集（Union-Find 或 Disjoint Set Union，简称DSU）是一种处理不交集合的合并及查询问题的数据结构。它支持两种操作：

1. **Find**: 确定某个元素属于哪一个子集。这个操作可以用来判断两个元素是否属于同一个子集。
2. **Union**: 将两个子集合并成一个集合。

使用场景

并查集常用于处理一些元素分组情况，可以动态地连接和判断连接，广泛应用于网络连接、图的连通分量、最小生成树等问题。

核心思想

并查集通过数组或者特殊结构存储每个元素的父节点信息。初始时，每个元素的父节点是其自身，表示每个元素自成一个集合。通过路径压缩和按秩合并等优化策略，可以提高并查集的效率。

- **路径压缩**：在执行Find操作时，使得路径上的所有点直接指向根节点，这样可以减少后续操作的时间复杂度。
- **按秩合并**：在执行Union操作时，总是将较小的树连接到较大的树的根节点上，这样可以避免树过深，影响操作效率。

### 代码示例

```python
class UnionFind:
    # 初始化
    def __init__(self, size):
        # 将每个节点的上级设置为自己
        self.parent = list(range(size))
        # 每个节点的秩都是0
        self.rank = [0] * size
	
    # 查找
    def find(self, p):
        if self.parent[p] != p:
            # 这一步进行了路径压缩。
            # 如果不进行路径压缩，这一步是 return self.find(self.parent[p])
            self.parent[p] = self.find(self.parent[p])  
        return self.parent[p]
	
    # 合并
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # 按秩合并，总是将较小的树连接到较大的树的根节点上
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                # 如果两个节点的秩相等，就无所谓
                self.parent[rootQ] = rootP
                # 但这时需要把连接后较大的节点的秩+1
                self.rank[rootP] += 1
	
    # 是否属于同一集合
    def connected(self, p, q):
        return self.find(p) == self.find(q)
```

例题

### OJ02524:宗教信仰

http://cs101.openjudge.cn/dsapre/02524/

最基本的应用，只是最后多了一步看看有多少个集合。

```python
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size + 1)]
        self.rank = [0] * (size + 1)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_parent = self.find(x)
        y_parent = self.find(y)
        if x_parent != y_parent:
            if self.rank[x_parent] > self.rank[y_parent]:
                self.parent[y_parent] = x_parent
            elif self.rank[x_parent] < self.rank[y_parent]:
                self.parent[x_parent] = y_parent
            else:
                self.parent[y_parent] = x_parent
                self.rank[x_parent] += 1


n_case = 0
while True:
    n_case += 1
    n, m = map(int, input().split())
    if m == 0 and n == 0:
        break
    uf = UnionFind(n)
    for i in range(m):
        a, b = map(int, input().split())
        uf.union(a, b)
    cnt = set([uf.find(i) for i in uf.parent])  # 这一步是多的
    print(f'Case {n_case}:', len(cnt) - 1)
```

### OJ18250:冰阔落 I

http://cs101.openjudge.cn/2024sp_routine/18250/

这题一开始WA，后来检查，发现原因是按秩合并时，parent[x]不一定更新了。虽然最后用self.find(x)又压缩了一次，仍然可能指向的不是最深的节点。好在此题数据小，无需按秩合并。

```python
class DJS:
    def __init__(self, size):
        self.parent = [i for i in range(size + 1)]
        self.rank = [0 for _ in range(size + 1)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        self.parent[root_b] = root_a
        """
        if root_b != root_a:
            if self.rank[root_b] == self.rank[root_a]:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            elif self.rank[root_b] > self.rank[root_a]:
                self.parent[root_a] = root_b
            else:
                self.parent[root_b] = root_a
        """

    def check(self, a, b):
        if self.find(a) == self.find(b):
            print('Yes')
        else:
            print('No')


while True:
    try:
        n, m = map(int, input().split())
    except EOFError:
        break
    d = DJS(n)
    for _ in range(m):
        x, y = map(int, input().split())
        d.check(x, y)
        d.union(x, y)
    cnt = 0
    ans = []
    for i in range(1, n + 1):
        if d.find(i) == i:
            cnt += 1
            ans.append(i)
    print(len(ans))
    print(*ans)

```

### OJ01703:发现它，抓住它

这题一开始没想出来，因为给出的条件是某两个节点属于不同的集合，而非相同的集合。但是由于一共只有两个集合，所以可以创建一个长度为2n的数组，parent[x]是和x同类的，parent[x+n]是和x不同的。

思路很新颖，值得学习。

http://cs101.openjudge.cn/2024sp_routine/01703/

```python
class DJS:
    def __init__(self, size):
        self.parent = [i for i in range(size + 1)]
        self.rank = [0 for _ in range(size + 1)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_b != root_a:
            if self.rank[root_b] == self.rank[root_a]:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            elif self.rank[root_b] > self.rank[root_a]:
                self.parent[root_a] = root_b
            else:
                self.parent[root_b] = root_a

    def check(self, a, b):
        if self.find(a) == self.find(b):
            print('Yes')
        else:
            print('No')


for _ in range(int(input())):
    n, m = map(int, input().split())
    d = DJS(2 * n)
    for _ in range(m):
        info = input().split()
        a, b = map(int, info[1:])
        if info[0] == 'A':
            if d.find(a) == d.find(b) or d.find(a + n) == d.find(b + n):
                print('In the same gang.')
            elif d.find(a + n) == d.find(b) or d.find(a) == d.find(b + n):
                print('In different gangs.')
            else:
                print('Not sure yet.')
        else:
            d.union(a, b + n)
            d.union(a + n, b)
```

### OJ01182:食物链

http://cs101.openjudge.cn/2024sp_routine/01182/

和上一题很像

```python
n, k = map(int, input().split())
cnt = 0
ds = []  # 本身, 被x吃, 吃x

for i in range(3 * n + 1):
    ds.append(i)


def find(a):
    # print(a)
    if ds[a] != a:
        ds[a] = find(ds[a])
    return ds[a]


def union(a, b):
    root_a, root_b = find(a), find(b)
    ds[root_a] = root_b


def check(d, a, b):
    if d == 1:
        return find(a + n) == find(b) or find(b + n) == find(a)
    else:
        return find(a) == find(b) or find(b) == find(a + 2 * n)


for _ in range(k):
    d, x, y = map(int, input().split())
    if x > n or y > n or check(d, x, y):
        cnt += 1
        continue
    if d == 1:
        for i in range(3):
            union(x + i * n, y + i * n)
    elif d == 2:
        union(y, x + n)
        union(x, y + 2 * n)
        union(x + 2 * n, y + n)
print(cnt)
```

## 拓扑排序

拓扑排序是对有向无环图（DAG，Directed Acyclic Graph）的顶点进行排序的一种方法，使得对于图中的每条有向边 UV（从顶点 U 指向顶点 V），U 在排序中都出现在 V 之前。拓扑排序不是唯一的，一个有向无环图可能有多个有效的拓扑排序。

拓扑排序常用的算法包括基于 DFS（深度优先搜索）的方法和基于 BFS（广度优先搜索，也称为Kahn算法）的方法。

作用：检测是否有环

### 代码示例

```python
from collections import deque, defaultdict


def topological_sort(vertices, edges):
    # 计算所有顶点的入度
    in_degree = {v: 0 for v in vertices}
    graph = defaultdict(list)
	
    # u->v
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1  # v的入度+1

    # 将所有入度为0的顶点加入队列
    queue = deque([v for v in vertices if in_degree[v] == 0])
    sorted_order = []

    while queue:
        u = queue.popleft()
        sorted_order.append(u)

        # 对于每一个相邻顶点，减少其入度
        for v in graph[u]:
            in_degree[v] -= 1
            # 如果入度减为0，则加入队列
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_order) != len(vertices):
        return None  # 存在环，无法进行拓扑排序
    return sorted_order


# 示例使用
vertices = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [('A', 'D'), ('F', 'B'), ('B', 'D'), ('F', 'A'), ('D', 'C')]
result = topological_sort(vertices, edges)
if result:
    print("拓扑排序结果:", result)
else:
    print("图中有环，无法进行拓扑排序")
```

例题

### OJ04084:拓扑排序

http://cs101.openjudge.cn/2024sp_routine/04084/

拓扑排序，但是要求“同等条件下，编号小的顶点在前”，不得不把普通队列转换成一个优先队列了。

```python
from collections import deque, defaultdict
import heapq

def topo_sort(g, nv):
    ans = []
    deg = {v: 0 for v in range(1, nv+1)}
    child = {v: [] for v in range(1, nv+1)}
    for u, v in g:
        # u->v
        if v not in deg:
            deg[v] = 1
        else:
            deg[v] += 1
        if u not in child:
            child[u] = [v]
        else:
            child[u].append(v)
    q = [v for v in deg.keys() if deg[v] == 0]
    heapq.heapify(q)
    while q:
        now = heapq.heappop(q)
        ans.append(now)
        for i in child[now]:
            deg[i] -= 1
            if deg[i] == 0:
                heapq.heappush(q, i)

    return ans
v, a = map(int, input().split())
g = []
for _ in range(a):
    x, y = map(int, input().split())
    g.append([x, y])
for i in topo_sort(g, v):
    print('v' + str(i), end=' ')
```

### OJ01094:Sorting It All Out

http://cs101.openjudge.cn/dsapre/01094/

此题要求每给出一条边就进行一次拓扑排序。首先判断给出的图有没有环，若拓扑排序后有顶点入度不为0，则有环。然后判断拓扑排序是否唯一，若同一时间队列长度大于1，则给出的条件不足以唯一确定拓扑排序。

```python
from collections import deque


def topo_sort(g, nv):
    ans = []
    deg = {chr(i): 0 for i in range(65, 65 + nv)}
    child = {chr(i): [] for i in range(65, 65 + nv)}
    for u, v in g:
        # u->v
        if v in child[u]:
            continue
        deg[v] += 1
        child[u].append(v)
    q = deque([v for v in deg.keys() if deg[v] == 0])
    not_determined = False
    while q:
        not_determined = len(q) >= 2 or not_determined
        now = q.popleft()
        ans.append(now)
        for i in child[now]:
            deg[i] -= 1
            if deg[i] == 0:
                q.append(i)
    loop = False
    for k, v in deg.items():
        if v != 0:
            loop = True
            break
    return ans, loop, not_determined


while True:
    v, a = map(int, input().split())
    if v == 0 and a == 0:
        break
    g = []
    sorted_seq = None
    end = False
    for _ in range(a):
        x, y = map(str, input().split('<'))
        if end:
            continue
        g.append([x, y])
        sorted_seq, loop, not_determined = topo_sort(g, v)
        if loop:
            print(f'Inconsistency found after {_ + 1} relations.')
            end = True
        elif not not_determined:
            print(f'Sorted sequence determined after {_ + 1} relations: ', end='')
            for qq in sorted_seq:
                print(qq, end='')
            print('.')
            end = True
    if end:
        continue
    print('Sorted sequence cannot be determined.')
```

### OJ09202:舰队、海域出击！

http://cs101.openjudge.cn/2024sp_routine/09202/

检测有向图有没有环，拓扑排序，也就是Kahn算法。

```python
from collections import deque
def topo(g, deg, v):
    q = deque([x for x in deg.keys() if deg[x] == 0])
    cnt = 0
    while q:
        now = q.popleft()
        cnt += 1
        for next in g[now]:
            deg[next] -= 1
            if deg[next] == 0:
                q.append(next)
    if cnt == v:
        print('No')
    else:
        print('Yes')
for _ in range(int(input())):
    v, m = map(int, input().split())
    g = {a: [] for a in range(1, v + 1)}
    deg = {a: 0 for a in range(1, v + 1)}
    for _ in range(m):
        x, y = map(int, input().split())
        deg[y] += 1
        g[x].append(y)
    topo(g, deg, v)
```

# 图

## Dijkstra

### 代码示例

```python
import heapq


def dijkstra(graph, start):
    # 初始化距离字典，所有顶点距离为无穷大，起始点距离为0
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    # 优先队列，用于存储每个顶点及其对应的距离，并按距离自动排序
    priority_queue = [(0, start)]

    while priority_queue:
        # 获取当前距离最小的顶点
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 遍历当前顶点的邻接点
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # 如果计算的距离小于已知距离，更新距离
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


# 示例图
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# 测试算法
start_vertex = 'A'
distances = dijkstra(graph, start_vertex)
print(f"Distances from {start_vertex}: {distances}")
```

例题

### **OJ05443:兔子与樱花**

http://cs101.openjudge.cn/dsapre/05443/

模板题目，额外的一点是需要记录路径

```python
import heapq

def dijkstra(adjacency, start):
    # 初始化，将其余所有顶点到起始点的距离都设为inf（无穷大）
    distances = {vertex: float('inf') for vertex in adjacency}
    # 初始化，所有点的前一步都是None
    previous = {vertex: None for vertex in adjacency}
    # 起点到自身的距离为0
    distances[start] = 0
    # 优先队列
    pq = [(0, start)]

    while pq:
        # 取出优先队列中，目前距离最小的
        current_distance, current_vertex = heapq.heappop(pq)
        # 剪枝，如果优先队列里保存的距离大于目前更新后的距离，则可以跳过
        if current_distance > distances[current_vertex]:
            continue

        # 对当前节点的所有邻居，如果距离更优，将他们放入优先队列中
        for neighbor, weight in adjacency[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # 这一步用来记录每个节点的前一步
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous

def shortest_path_to(adjacency, start, end):
    # 逐步访问每个节点上一步
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

### OJ07735:道路

http://cs101.openjudge.cn/practice/07735/

dijkstra，但是有点区别，加入优先队列的条件不是距离更短，而是金币够用，但是优先队列的比较仍然是用距离比的

```python
import heapq

k, n, r = int(input()), int(input()), int(input())


def dij(g, s, e):
    dis = {v: float('inf') for v in range(1, n + 1)}
    dis[s] = 0
    q = [(0, s, 0)]
    heapq.heapify(q)
    while q:
        d, now, fee = heapq.heappop(q)
        if now == n:
            return d
        for neighbor, distance, c in g[now]:
            if fee + c <= k:
                dis[neighbor] = distance + d
                heapq.heappush(q, (distance + d, neighbor, fee + c))
    return -1


g = {v: [] for v in range(1, n + 1)}
for _ in range(r):
    s, e, m, j = map(int, input().split())
    g[s].append((e, m, j))
p = dij(g, 1, n)
print(p)
```
